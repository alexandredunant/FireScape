import csv, json, re, sys
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader

sys.stdout.reconfigure(encoding='utf-8')
root = Path(__file__).resolve().parents[1]
out = root / 'tmp' / 'source_audit'
out.mkdir(exist_ok=True)
tex = (root / 'review_paper_v5.tex').read_text(encoding='utf-8')
refs = list(csv.DictReader((root / 'papers' / 'FireScape_review_reference_list.csv').open(encoding='utf-8-sig')))
counts = Counter(k.strip() for m in re.finditer(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}', tex) for k in m[1].split(','))

def extract(row):
    path = root / 'papers' / row['pdf_in_folder']
    result = {'key': row['bibkey'], 'file': row['pdf_in_folder'], 'citations':counts[row['bibkey']]}
    if not row['pdf_in_folder'] or not path.is_file():
        result['status']='MISSING'; return result
    try:
        reader=PdfReader(path)
        pages=[p.extract_text() or '' for p in reader.pages]
        text='\n\n'.join(f'PAGE {i+1}\n{t}' for i,t in enumerate(pages))
        (out / (row['bibkey'] + '.txt')).write_text(text,encoding='utf-8')
        result.update(status='OK', pages=len(pages), chars=len(text), doi_in_text=row['doi'].lower() in text.lower().replace('\n','').replace(' ',''))
    except Exception as e: result.update(status='ERROR',error=str(e))
    return result

with ThreadPoolExecutor(max_workers=6) as pool: results=list(pool.map(extract,refs))
(out/'audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('Cited keys',len(counts),'mapped references',len(refs),'PDF status',dict(Counter(r['status'] for r in results)))
print('Unmapped citations',sorted(set(counts)-{r['bibkey'] for r in refs}))
print('Missing/error',[r for r in results if r['status']!='OK'])
print('DOI not found',[(r['key'],r['file']) for r in results if r['status']=='OK' and not r['doi_in_text']])
print('Most cited',counts.most_common(20))
print('Very long sentences:')
for n,line in enumerate(tex.splitlines(),1):
    if not line or line.startswith('\\') or ' & ' in line: continue
    clean=re.sub(r'\\cite\w*(?:\[[^]]*\])*\{[^}]+\}','',line)
    for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])',clean):
        if len(s.split())>75: print(n,len(s.split()),s[:250])
