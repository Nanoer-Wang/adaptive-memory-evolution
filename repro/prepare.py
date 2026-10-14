"""Restore frozen inputs, or download hash-checked public benchmark assets.

This script never trains, generates LLM answers, or changes the memory policy.
Run from a fresh checkout. Existing mismatched inputs are never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_verified(path, data, expected):
    assert hashlib.sha256(data).hexdigest() == expected, f'Hash mismatch: {path.name}'
    if path.exists():
        assert sha(path) == expected, f'Existing file differs; use a fresh checkout: {path}'
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def download(url, path, expected):
    if path.exists():
        assert sha(path) == expected, f'Existing input differs: {path}'
        return
    assert url.startswith('https://'), 'Only HTTPS upstream sources are accepted'
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.download')
    req = urllib.request.Request(url, headers={'User-Agent': 'AME-reproducibility/1.0'})
    with urllib.request.urlopen(req, timeout=120) as response, temp.open('wb') as out:
        shutil.copyfileobj(response, out)
    assert sha(temp) == expected, f'Upstream checksum mismatch: {path.name}'
    temp.replace(path)
    print('Verified', path.relative_to(ROOT), flush=True)


def install_bootstrap(manifest):
    for entry in manifest['files']:
        # A later B8 reproduction legitimately has a newly registered protocol.
        dst = ROOT / entry['path']
        if entry['path'].endswith('/B8/protocol.json') and dst.exists():
            if (dst.parent / 'prepared.json').exists():
                continue
        write_verified(dst, (ROOT/'repro/bootstrap'/entry['path']).read_bytes(), entry['sha256'])
    for path, expected in manifest['core_hashes'].items():
        assert sha(ROOT/path) == expected, f'Frozen policy changed: {path}'


def synthetic(sources, manifest):
    meta = sources['babi']
    missing = [n for n,h in meta['files'].items() if not (ROOT/'data/babi'/n).exists()]
    if missing:
        archive = ROOT/'data/downloads/babi_tasks_1-20_v1-2.tar.gz'
        download(meta['url'], archive, meta['archive_sha256'])
        with tarfile.open(archive, 'r:gz') as tar:
            for name, expected in meta['files'].items():
                candidates = [m for m in tar.getmembers() if m.name.endswith('/en/'+name)]
                assert len(candidates) == 1, name
                # Read only the two named members; never extract arbitrary paths.
                write_verified(ROOT/'data/babi'/name, tar.extractfile(candidates[0]).read(), expected)
    for name, expected in meta['files'].items():
        assert sha(ROOT/'data/babi'/name) == expected
    from src.data.babi_events import stories, read_key
    datasets = {}
    for split in ['train', 'test']:
        original = stories(ROOT/f'data/babi/qa1_single-supporting-fact_{split}.txt', split)
        for index, story in enumerate(original):
            target = 'test' if split == 'test' else ('train' if index < 120 else 'val')
            if target == 'val' and index >= 160:
                continue
            events = []
            for event in story:
                if isinstance(event, dict):
                    events.append(dict(type='query', **event, key=json.dumps(read_key(event['question']))))
                else:
                    events.append(dict(type='observation', line=event.turn, text=event.quote,
                                       key=json.dumps(event.key), value=event.value))
            datasets[f'{split}-{index}'] = dict(split=target, events=events)
    # Match atomic_json's historical platform line endings as well as content.
    serialized = json.dumps(datasets, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    data = serialized.encode('utf-8')
    if hashlib.sha256(data).hexdigest() != manifest['stories_sha256']:
        data = serialized.replace('\n', '\r\n').encode('utf-8')
    write_verified(ROOT/'runs/liquid-consolidation-v2/stories.json', data, manifest['stories_sha256'])
    print('Synthetic inputs and three frozen heads verified; no training or API calls.')


def semantic(sources):
    meta = sources['memops']
    p = ROOT/'data/memops/provenance.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    for entry in meta['files']:
        download(entry['url'], ROOT/'data/memops/updates'/entry['file'], entry['sha256'])
    encoder()


def encoder():
    meta = read(ROOT/'artifacts/encoders/minilm/source.json')
    for name, entry in meta['files'].items():
        download(entry['url'], ROOT/'artifacts/encoders/minilm'/name, entry['sha256'])


def language(sources, manifest):
    for name, filename in [('longmemeval','longmemeval_s_cleaned.json'), ('locomo','locomo10.json')]:
        meta = sources[name]
        download(meta['url'], ROOT/'data'/name/filename, meta['sha256'])
    for name, meta in sources['memorybench']['files'].items():
        download(meta['url'], ROOT/'data/memorybench'/name, meta['sha256'])
    encoder()
    # C batches read the historic B8 protocol as an input ledger. A NEW language
    # run must register its own source hash. Remove ONLY our exact bootstrap copy.
    dst = ROOT/'runs/ame-validation-v3/B8/protocol.json'
    entry = next(r for r in manifest['files'] if r['path'].endswith('/B8/protocol.json'))
    if dst.exists() and sha(dst) == entry['sha256']:
        if any((dst.parent/n).exists() for n in ('prepared.json','cases.json','histories.json.gz')):
            raise RuntimeError('Existing B8 evidence found. Use a fresh checkout for a new language run.')
        dst.unlink()
    print('Language data ready. Complete B2-B7 before running validation_language.py --stage prepare.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--semantic', action='store_true')
    parser.add_argument('--language', action='store_true')
    args = parser.parse_args()
    manifest = read(ROOT/'repro/bootstrap_manifest.json')
    sources = read(ROOT/'repro/upstream_sources.json')
    install_bootstrap(manifest)
    if args.synthetic or args.semantic or args.language:
        synthetic(sources, manifest)
    if args.semantic:
        semantic(sources)
    if args.language:
        language(sources, manifest)
    if not (args.synthetic or args.semantic or args.language):
        print('Bootstrap verified. Add --synthetic for public bAbI input preparation.')


if __name__ == '__main__':
    main()
