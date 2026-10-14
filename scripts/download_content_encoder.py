"""Restore the pinned public CPU encoder; verify each downloaded file."""
import hashlib
import json
import urllib.request
from src.config import resolve


def main():
    root = resolve('artifacts/encoders/minilm')
    manifest = json.loads((root / 'source.json').read_text(encoding='utf-8'))
    entries = manifest['files']
    if isinstance(entries, dict):
        entries = [dict(value, name=name) for name, value in entries.items()]
    for entry in entries:
        name = entry.get('name') or entry.get('local_name') or entry.get('path', '').split('/')[-1]
        target = root / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == entry['sha256']:
            continue
        temporary = target.with_suffix(target.suffix + '.download')
        urllib.request.urlretrieve(entry['url'], temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Hash mismatch: {name}')
        temporary.replace(target)
        print(f'Verified {name}', flush=True)


if __name__ == '__main__':
    main()
