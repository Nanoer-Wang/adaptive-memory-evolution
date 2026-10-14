"""Render the manuscript and inventory text; run with pypdf, pypdfium2, Pillow."""
from pathlib import Path
import json, re
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image, ImageDraw

p = Path(__file__).resolve().parents[1]
out = p / 'review'
out.mkdir(exist_ok=True)
src = p / 'build/main.pdf'
reader = PdfReader(src)
doc = pdfium.PdfDocument(src)
inventory = []
thumbs = []
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    inventory.append({'page': i+1, 'words': len(text.split()), 'start': text[:140], 'unresolved_reference': '??' in text})
    rendered = doc[i].render(scale=1.7).to_pil().convert('RGB')
    rendered.save(out / f'page-{i+1:02d}.png')
    rendered.thumbnail((330, 435))
    thumb = Image.new('RGB', (350, 462), '#e6e6e6')
    thumb.paste(rendered, ((350-rendered.width)//2, 20))
    ImageDraw.Draw(thumb).text((12, 4), f'PAGE {i+1}', fill='black')
    thumbs.append(thumb)
for start in range(0, len(thumbs), 6):
    sheet = Image.new('RGB', (1050,924), 'white')
    for j, t in enumerate(thumbs[start:start+6]): sheet.paste(t, ((j%3)*350,(j//3)*462))
    sheet.save(out/f'contact-{start//6+1:02d}.png')
(out/'page_inventory.json').write_text(json.dumps(inventory, indent=2),encoding='utf-8')
(out/'pdf_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{pg.extract_text()}' for i,pg in enumerate(reader.pages)),encoding='utf-8')
aux=(p/'build/main.aux').read_text()
main_page=re.search(r'newlabel\{main:end\}\{\{[^}]+\}\{(\d+)\}',aux)
print(json.dumps({'pages':len(reader.pages), 'main_pages':int(main_page[1]) if main_page else None,'unresolved_pages':[r['page'] for r in inventory if r['unresolved_reference']]}))
