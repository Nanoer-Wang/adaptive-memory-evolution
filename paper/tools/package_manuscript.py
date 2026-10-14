"""Package only the finished manuscript, source, and selected analysis evidence."""
from pathlib import Path
import hashlib,json,zipfile
P=Path(__file__).resolve().parents[1]
files=[P/f for f in ['main.tex','main.pdf','references.bib','neurips_2026.sty',
                     'README.txt','AUTHOR_NOTES_ZH.txt','build.ps1']]
for directory in ['sections','tables','figures','evidence']:
    files.extend(f for f in (P/directory).rglob('*') if f.is_file())
files.extend(sorted((P/'tools').glob('*.py')))
files.append(P/'tools/tectonic_release.json')
assert all(f.is_file() for f in files)
assert all('.env' not in f.parts for f in files)
out=P/'AME_Manuscript_Source.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for f in sorted(set(files)):z.write(f,Path('paper')/f.relative_to(P))
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
report={'archive':out.name,'files':len(set(files)), 'bytes':out.stat().st_size,
        'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'excludes':['API credentials','full raw trajectory archives','portable compiler binary','review PNGs','build intermediates']}
(P/'package_manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
