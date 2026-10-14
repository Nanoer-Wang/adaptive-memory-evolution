"""Independent public-fact B1 generator. Evaluation labels never enter inputs."""
from __future__ import annotations
import hashlib
import json
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def generate(stories, vectors, world_seed, trajectory_seed, stable_fraction=.5,
             regime='rapid', capacity=8, phases=6):
    if stable_fraction not in (.2, .5, .8) or regime not in ('stationary', 'slow', 'rapid'):
        raise ValueError('Unsupported B1 condition')
    if not 1 <= capacity <= 40 or phases < 2:
        raise ValueError('Sanity generator needs 1..40 initial slots and >=2 phases')
    world = np.random.default_rng(world_seed)
    schedule = np.random.default_rng(trajectory_seed)
    # Split-specific public facts, without loading reserved formal test facts.
    pools = {}
    for story in stories.values():
        if story['split'] == 'train':
            for e in story['events']:
                if e['type'] == 'observation':
                    pools.setdefault(e['key'], {})[e['value']] = e['text']
    subjects = sorted(k for k, v in pools.items() if len(v) >= 2)
    if not subjects:
        raise ValueError('Public facts with multiple observed values required')
    permutation = list(map(int, world.permutation(80)))
    core, distractors = permutation[:40], permutation[40:]
    topics = {key: n // 10 for keys in (core, distractors) for n, key in enumerate(keys)}
    stable = set()
    role_rng = np.random.default_rng(world_seed + 900)
    for topic in range(4):
        stable.update(map(int, role_rng.choice(core[topic*10:(topic+1)*10], int(10*stable_fraction), replace=False)))
    scope = np.random.default_rng(world_seed + 901).normal(size=(80, 32))
    scope /= np.linalg.norm(scope, axis=1, keepdims=True)
    subject_for = {key: subjects[i % len(subjects)] for i, key in enumerate(permutation)}
    current = {}; versions = {}; catalog = {}
    for key in permutation:
        subject = subject_for[key]; values = sorted(pools[subject])
        value = values[int(world.integers(len(values)))]; current[key] = value; versions[key] = 0
        catalog[key] = dict(topic=topics[key], stable=key in stable, core=key in core,
                            public_subject=subject)

    def vector(key, text=None):
        source = np.array(vectors[text]) if text else np.mean([vectors[t] for t in pools[subject_for[key]].values()], axis=0)
        v = source + .35 * scope[key]
        return (v / max(np.linalg.norm(v), 1e-8)).tolist()

    def observation(key, time, change=False):
        if change:
            values = sorted(pools[subject_for[key]])
            current[key] = values[(values.index(current[key]) + 1) % len(values)]
        versions[key] += 1
        text = pools[subject_for[key]][current[key]]
        return dict(kind='observe', time=float(time), key=key, topic=topics[key],
                    version=versions[key], item_id=f'{key}:{versions[key]}', content=text,
                    vector=vector(key, text))

    # Initial records are a common supplied snapshot, not selected by each policy.
    initial_keys = list(map(int, world.permutation(core)[:capacity]))
    initial = [observation(k, 0.) for k in initial_keys]
    tape = []; now = 0.; current_phase = -1

    def emit(inp, evaluation=None):
        inp = dict(inp, event_id=len(tape))
        tape.append(dict(input=inp, evaluation=dict(phase=current_phase, **(evaluation or {}))))

    # Exactly one initial observation per core key, in a shared random order.
    for key in schedule.permutation([k for k in core if k not in initial_keys]):
        now += float(schedule.uniform(.5, 1.5)); emit(observation(int(key), now))
    focus_sequence = [0] * phases if regime == 'stationary' else [0, 1, 0, 2, 1, 3] * ((phases + 5)//6)
    revision_probability = {'stationary': 0., 'slow': .1, 'rapid': .5}[regime]
    for current_phase in range(phases):
        focus = focus_sequence[current_phase]
        now += 2.
        topic_keys = [k for k in core if topics[k] == focus]
        emit(dict(kind='context', time=now, topic=focus,
                  vector=np.mean([vector(k) for k in topic_keys], axis=0).tolist()),
             dict(shift=current_phase > 0 and focus != focus_sequence[current_phase-1]))
        # Importance shifts do not rewrite the truth. Revisions are separate observations.
        for key in schedule.permutation([k for k in core if k not in stable]):
            if schedule.random() < revision_probability:
                now += float(schedule.uniform(.5, 1.5)); emit(observation(int(key), now, True))
        for key in schedule.choice(distractors, 6, replace=False):
            now += float(schedule.uniform(.5, 1.5)); emit(observation(int(key), now, versions[int(key)] > 0))
        stable_pool = [k for k in topic_keys if k in stable]
        volatile_pool = [k for k in topic_keys if k not in stable]
        # 20 queries per role, deliberately separate from item ratio. Balanced cyclic
        # sampling avoids accidental per-item popularity within each role.
        queues = []
        for pool in (stable_pool, volatile_pool):
            sequence = []
            while len(sequence) < 20:
                sequence.extend(map(int, schedule.permutation(pool)))
            queues.append(sequence[:20])
        for j in range(20):
            for role in schedule.permutation(2):
                key = queues[int(role)][j]; now += float(schedule.uniform(.5, 1.5))
                emit(dict(kind='query', time=now, key=key, topic=topics[key], vector=vector(key),
                          content=f'Where is scoped subject {key}?'),
                     dict(target_id=f'{key}:{versions[key]}', stable=key in stable,
                          target_query=key not in stable,
                          shifted=current_phase > 0 and focus != focus_sequence[current_phase-1]))
    row = dict(world_seed=world_seed, trajectory_seed=trajectory_seed, stable_fraction=stable_fraction,
               regime=regime, phases=phases, core_keys=core, distractor_keys=distractors,
               catalog={str(k): v for k,v in catalog.items()}, initial_memory=initial, events=tape,
               query_sequence=[r['input']['key'] for r in tape if r['input']['kind']=='query'],
               importance_sequence=focus_sequence[:phases], source_split='public_babi_train',
               purpose='sanity only; not independent scientific confirmation')
    row['environment_sha256'] = digest(row)
    return row


def audit_environment(row):
    public = [e['input'] for e in row['events']]
    assert all(not {'stable', 'target_id', 'phase', 'shifted', 'future'} & set(e) for e in public)
    observations = row['initial_memory'] + [e for e in public if e['kind']=='observe']
    ids = [e['item_id'] for e in observations]
    assert len(ids) == len(set(ids)), 'Main mini uses unique observed versions for equal overflow opportunities'
    catalog = {int(k): v for k, v in row['catalog'].items()}
    stable = {k for k, v in catalog.items() if v['stable']}
    assert len(stable) == round(40*row['stable_fraction'])
    for k in stable:
        records = [e for e in observations if e['key']==k]
        assert len(records)==1 and records[0]['version']==1, 'Stable fact was changed or refreshed'
    for topic in range(4):
        assert sum(catalog[k]['topic']==topic for k in stable)==round(10*row['stable_fraction'])
    queries = [e for e in row['events'] if e['input']['kind']=='query']
    assert sum(e['evaluation']['stable'] for e in queries)==len(queries)//2
    assert [e['time'] for e in public]==sorted(e['time'] for e in public)
    known = {e['key']: e['item_id'] for e in row['initial_memory']}
    for row_event in row['events']:
        e = row_event['input']
        if e['kind']=='observe': known[e['key']] = e['item_id']
        if e['kind']=='query': assert row_event['evaluation']['target_id']==known[e['key']]
    return dict(passed=True, queries=len(queries), observation_opportunities=sum(e['kind']=='observe' for e in public),
                stable_queries=len(queries)//2, stable_fact_count=len(stable), core_fact_count=40,
                stable_observations=sum(e['key'] in stable for e in observations),
                nonstable_observations=sum(e['key'] not in stable for e in observations),
                topic_role_balance=True, initial_sha256=digest(row['initial_memory']),
                input_sha256=digest(public), query_sha256=digest(row['query_sequence']),
                unique_observed_versions=True)
