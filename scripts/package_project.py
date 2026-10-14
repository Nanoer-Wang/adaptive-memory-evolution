"""Create a verified portable project archive, excluding credentials and runtime debris."""
import hashlib
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / 'exports'
COMPACT_EXCLUSIONS = {
    'artifacts/content_vectors.sqlite3': 'Regenerable local MiniLM vector cache; recreated automatically on use.',
    'runs/semantic-gate-v1/features.npz': 'Old experiment feature matrix; regenerate before retraining this historical experiment.',
    'runs/semantic-gate-v2/features.npz': 'Old experiment feature matrix; regenerate before retraining this historical experiment.',
}


def omitted_dir(name):
    return (name.startswith('.') or name in {'__pycache__', 'exports', 'node_modules'}
            or name.endswith('.egg-info') or name.startswith(('pytest-', 'test-tmp', 'test-temp')))


def omitted_file(name):
    return ((name.startswith('.env') and name != '.env.example')
            or name.endswith(('.pyc', '.pyo', '.lock', '.sqlite3-wal', '.sqlite3-shm', '.pem', '.key'))
            or name == 'rate_limit.sqlite3')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compact', action='store_true', help='Exclude three regenerable feature caches; enforce size below512MB decimal.')
    args = parser.parse_args()
    EXPORTS.mkdir(exist_ok=True)
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    suffix = '_compact' if args.compact else ''
    target = EXPORTS / f'LCM_Final_Project_{now:%Y-%m-%d_%H%M%S}{suffix}.zip'
    secrets = [v.encode() for k, v in dotenv_values(ROOT / '.env').items()
               if v and len(v) >= 8 and any(s in k.upper() for s in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))]
    files = []
    for folder, dirs, names in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if not omitted_dir(d))
        for name in sorted(names):
            p = Path(folder) / name
            if args.compact and p.relative_to(ROOT).as_posix() in COMPACT_EXCLUSIONS:
                continue
            if not omitted_file(name) and not p.is_symlink():
                files.append(p)
    manifest = dict(created_at=now.isoformat(),project='LCM_Final_Project',
                    latest_report='runs/bidirectional-fallback-v1/index.html',
                    excluded=['API credentials (.env)', 'virtual environments', 'test temporary directories',
                              'Python caches', 'package installation metadata', 'lock files', 'rate limiter runtime state',
                              'previous exports'],
                    included='Current code/config/tests/docs, public datasets, all research runs and models, encoder weights, source snapshots, consistent response/vector database backups.',
                    files=[])
    if args.compact:
        manifest['compact_exclusions'] = COMPACT_EXCLUSIONS
        manifest['size_limit_bytes'] = 512_000_000
        manifest['included'] = 'Current code/config/tests/docs, public datasets, all research reports and models, encoder weights, source snapshots, API response backups. Only three regenerable feature caches additionally excluded.'
    overlap = max([len(s) for s in secrets] + [1]) - 1

    def inspect(path, relative):
        digest = hashlib.sha256(); previous = b''; size = 0
        with path.open('rb') as source:
            while chunk := source.read(8 * 1024 * 1024):
                if any(secret in previous + chunk for secret in secrets):
                    raise RuntimeError(f'Credential detected; packaging stopped: {relative}')
                previous = chunk[-overlap:] if overlap else b''
                digest.update(chunk); size += len(chunk)
        if path.suffix.lower() == '.zip':
            with zipfile.ZipFile(path) as nested:
                for name in nested.namelist():
                    if Path(name).name.startswith('.env') and Path(name).name != '.env.example':
                        raise RuntimeError(f'Credential file in nested archive: {relative}')
                    if any(secret in nested.read(name) for secret in secrets):
                        raise RuntimeError(f'Credential detected in nested archive: {relative}')
        return dict(path=relative,size=size,sha256=digest.hexdigest())

    with tempfile.TemporaryDirectory(prefix='package-stage-', dir=EXPORTS) as temporary:
        stage = Path(temporary).resolve(); assert stage.parent == EXPORTS.resolve()
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            for i, original in enumerate(files, 1):
                relative = original.relative_to(ROOT).as_posix(); source = original
                if original.suffix == '.sqlite3':
                    source = stage / f'{i}.sqlite3'
                    with closing(sqlite3.connect(original.as_uri()+'?mode=ro', uri=True)) as db:
                        with closing(sqlite3.connect(source)) as backup:
                            db.backup(backup)
                item = inspect(source, relative)
                archive.write(source, 'LCM_Final_Project/' + relative)
                manifest['files'].append(item)
                if i % 1000 == 0: print(f'Packed {i}/{len(files)} files', flush=True)
            note = ('Project snapshot. Read README.md and open runs/bidirectional-fallback-v1/index.html.\n'
                    'API credentials are intentionally excluded. Copy .env.example to .env and configure your own key.\n'
                    'Recreate the Python environment using the included requirement files. Models, datasets, research results and reusable databases are included.\n'
                    'PACKAGE_MANIFEST.json lists archived file sizes and SHA256 checksums. Database files are consistent SQLite backups.\n')
            if args.compact:
                note += ('Compact edition: public data, trained models and latest experiment records are retained.\n'
                         'Excluded content_vectors.sqlite3 is recreated automatically using the included encoder; first use is slower, without extra API calls for embeddings.\n'
                         'Old semantic-gate-v1/v2 features.npz matrices are excluded; regenerate them before retraining those historical experiments.\n'
                         'The original project files were not deleted. See compact_exclusions in PACKAGE_MANIFEST.json.\n')
            archive.writestr('LCM_Final_Project/PACKAGE_README.txt', note)
            archive.writestr('LCM_Final_Project/PACKAGE_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    print('Archive created; verifying every archived file...', flush=True)
    if args.compact:
        assert target.stat().st_size < 512_000_000, 'Archive exceeds512MB size limit'
    with zipfile.ZipFile(target) as archive:
        for item in manifest['files']:
            digest = hashlib.sha256()
            with archive.open('LCM_Final_Project/' + item['path']) as source:
                while chunk := source.read(8 * 1024 * 1024): digest.update(chunk)
            assert digest.hexdigest() == item['sha256'], item['path']
        assert 'LCM_Final_Project/.env' not in archive.namelist()
    digest = hashlib.sha256()
    with target.open('rb') as source:
        while chunk := source.read(8 * 1024 * 1024): digest.update(chunk)
    checksum = digest.hexdigest()
    target.with_suffix('.zip.sha256').write_text(f'{checksum}  {target.name}\n', encoding='utf-8')
    result = dict(zip=str(target),files=len(files)+2,bytes=target.stat().st_size,sha256=checksum,verified=True)
    target.with_suffix('.package.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':main()
