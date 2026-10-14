"""Freeze existing evidence and materialize staged validation plans.

This tool deliberately does not train models or imply that future benchmark
adapters are implemented. Readiness is explicit in each generated batch plan.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import atomic_json

CONFIG = ROOT / 'configs/ame_validation_batches.json'
OUT = ROOT / 'runs/ame-validation-v2'
REQUIRED = {'fixed_decay', 'frequency', 'lru', 'random', 'gru', 'linear', 'adaptive_evolution'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate(config):
    assert REQUIRED <= set(config['methods']), 'Required baseline missing'
    common = config['common']
    seeds = []
    for split in ('train', 'val', 'test'):
        spec = common['world_seeds'][split]
        seeds.append(set(range(spec['start'], spec['start'] + spec['count'])))
    assert not any(seeds[i] & seeds[j] for i in range(3) for j in range(i)), 'Seed overlap'
    assert not set(common['smoke_world_seeds']).intersection(set.union(*seeds)), 'Smoke seed overlap'
    ids = [b['id'] for b in config['batches']]
    assert len(ids) == len(set(ids)), 'Duplicate batch'
    for index, batch in enumerate(config['batches']):
        assert set(batch['depends_on']) <= set(ids[:index]), 'Dependency must precede batch'
        for capacity in batch.get('capacities', []):
            assert 0 < capacity < common['core_keys'] + common['distractor_keys'], 'No pressure at capacity'
    for fraction in common['stable_fractions']:
        count = fraction * common['core_keys']
        assert 0 < count < common['core_keys'] and abs(count - round(count)) < 1e-8, 'Invalid exact ratio'
    assert common['controller_input_dim'] == 40 and common['gru_hidden_dim'] == 8
    assert common['adaptive_head_parameters'] == 123, 'Do not increase adaptive model complexity'
    assert set(range(1, 9)) == {e for b in config['batches'] for e in b['experiments']}


def local_path(relative):
    target = (ROOT / relative).resolve()
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError('Artifact path escapes workspace')
    return target


def verify_manifest(manifest, root=ROOT):
    root = Path(root).resolve()
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise ValueError(f'Frozen reference missing or changed: {relative}')
    archive = (root / manifest['archive']).resolve()
    if not archive.is_relative_to(root) or sha(archive) != manifest['archive_sha256']:
        raise ValueError('Frozen archive missing or changed')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(manifest['files']), 'Archive membership mismatch'
        for relative, expected in manifest['files'].items():
            with z.open(relative) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert digest == expected, f'Archived bytes differ: {relative}'


def freeze(config):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'frozen_reference.json'
    if path.exists():
        verify(config)
        print('Frozen reference already exists and verifies; no overwrite.')
        return
    reference = config['frozen_reference']
    archive = ROOT / 'artifacts/frozen/ame-evolving-memory-v1.zip'
    if archive.exists():
        raise FileExistsError('Unregistered archive exists; do not overwrite evidence')
    files = [local_path(reference['source_archive'])]
    run = local_path(reference['run'])
    files += sorted(p for p in run.rglob('*') if p.is_file())
    files += [ROOT / n for n in ('README.md', 'docs/Algorithm.md', 'docs/Experiment_Design.md', 'EXPERIMENT_PROGRESS_2026-10-13.md')]
    assert all(p.is_file() and p.resolve().is_relative_to(ROOT) for p in files)
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), reference=reference,
                    archive=archive.relative_to(ROOT).as_posix(), archive_sha256=sha(archive),
                    archive_bytes=archive.stat().st_size, files=hashes,
                    policy='No .env, API credentials or unrelated datasets. Evidence files remain unchanged; documentation is recoverable in archive.')
    # Documents will be updated for v2; archived document hashes remain recorded
    # separately, so later verification checks immutable evidence and source only.
    manifest['mutable_document_paths'] = [p.relative_to(ROOT).as_posix() for p in files if p.suffix == '.md']
    atomic_json(path, manifest)
    verify_manifest(manifest)
    print(f'Frozen {len(files)} files; compressed archive {archive.stat().st_size / 1024**2:.2f} MiB.')


def verify(config):
    manifest = read(OUT / 'frozen_reference.json')
    # Live documentation may evolve, but the archive must preserve every old byte.
    archive = local_path(manifest['archive'])
    assert sha(archive) == manifest['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == set(manifest['files'])
        for relative, digest in manifest['files'].items():
            with z.open(relative) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
            if relative not in manifest.get('mutable_document_paths', []):
                assert sha(local_path(relative)) == digest, f'Frozen evidence changed: {relative}'
    assert manifest['reference'] == config['frozen_reference']
    print('Frozen implementation, live v1 evidence and archived documentation verified.')


def prepare(config, batch_id):
    verify(config)
    batch = next(b for b in config['batches'] if b['id'] == batch_id)
    digest = sha(CONFIG)
    # Version by complete configuration, so a later plan edit never rewrites an old plan.
    folder = OUT / 'plans' / digest[:16] / batch_id
    folder.mkdir(parents=True, exist_ok=True)
    cells = []
    if batch_id in ('B1', 'B2'):
        for ratio in batch['ratios']:
            for regime in batch['regimes']:
                for capacity in batch['capacities']:
                    for method in config['methods']:
                        for policy_seed in config['common']['policy_seeds']:
                            cells.append(dict(stable_fraction=ratio, regime=regime, capacity=capacity,
                                              method=method, policy_seed=policy_seed,
                                              action='reuse_B1' if capacity in batch.get('reuse_capacities', []) else 'new_evaluation'))
    readiness = dict(batch=batch_id, plan_ready=True, training_started=False, api_calls=0,
                     implementation_ready=batch_id == 'B0',
                     state='frozen' if batch_id == 'B0' else 'implementation_required',
                     reason='Planning and freeze commands are executable. New benchmark generator, baseline adapters and training runner must be implemented and smoke-tested before execution. Legacy24-key runner cannot execute this plan.',
                     dependencies=batch['depends_on'], config_sha256=digest,
                     primary_cells=len(cells),
                     new_primary_cells=sum(c['action'] == 'new_evaluation' for c in cells),
                     test_worlds_per_cell=config['common']['world_seeds']['test']['count'],
                     excludes='Ablations, retention sweeps and matched-admission controls are additional cells, not included in this primary count.')
    if batch_id == 'B1' and (OUT/'B1-sanity-review/protocol_diagnosis.json').exists():
        readiness.update(state='sanity_complete_protocol_revision_required', sanity_implementation_ready=True,
                         implementation_ready=False, formal_training_ready=False,
                         reason='Five requested adapters and independent mini runner are implemented and checked. Sanity detected distractor-order bias and uninformative within-block adaptation lag. Revise and recheck the protocol before formal B1; remaining formal baseline adapters and training runner are not claimed complete.')
    if batch_id == 'B1' and (OUT/'B1-interleaved-review/trace_review.json').exists():
        readiness.update(state='interleaved_sanity_complete_review_pending', sanity_implementation_ready=True,
                         implementation_ready=False, formal_training_ready=False,
                         reason='Controlled distinct-key pressure and interleaved observation/query/feedback mini passed. Capacity8 evaluated,16 unit-tested. Await review; no formal training. Replacement-matched attribution and remaining formal adapters/training runner are not claimed complete.')
    atomic_json(folder / 'configuration.json', config)
    atomic_json(folder / 'batch.json', batch)
    atomic_json(folder / 'primary_cells.json', cells)
    atomic_json(folder / 'readiness.json', readiness)
    print(json.dumps(dict(path=str(folder), **readiness), ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'verify', 'status', 'prepare'])
    parser.add_argument('--batch', choices=[f'B{i}' for i in range(8)], default='B1')
    args = parser.parse_args(argv)
    config = read(CONFIG); validate(config)
    if args.action == 'freeze': freeze(config)
    elif args.action == 'verify': verify(config)
    elif args.action == 'prepare': prepare(config, args.batch)
    else:
        for batch in config['batches']:
            print(batch['id'], batch['name'], '->', batch['status'])
        print('No new validation experiments have been run by this planning tool.')


if __name__ == '__main__': main()
