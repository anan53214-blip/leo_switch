from pathlib import Path
import csv, json, subprocess, hashlib
from PIL import Image
from pypdf import PdfReader

root = Path('D:/program/LEO_switch')
build = root / 'tmp/pptx_report_20261030'
assets = build / 'assets'
assets.mkdir(parents=True, exist_ok=True)
poppler = Path('C:/Users/19704/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe')
mapping = {'系统架构图.drawio':'system', '算法框架图10_23':'framework', 'algorithm_convergence_u30':'algorithms', 'convergence_u30':'parameters', 'figure1_line_metrics':'task_metrics', 'figure2_bar_metrics':'service_metrics'}
meta = {}
for stem, name in mapping.items():
    source = root / 'paper' / (stem + '.pdf')
    subprocess.run([str(poppler),'-singlefile','-r','260','-png',str(source),str(assets/name)], check=True, capture_output=True)
    with Image.open(assets/(name+'.png')) as im:
        meta[name] = {'width':im.width,'height':im.height,'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (assets/(name+'.txt')).write_text(PdfReader(source).pages[0].extract_text(), encoding='utf-8')
rows = list(csv.DictReader((root/'results/multiseed_0907/multiuser_summary.csv').open(encoding='utf-8-sig')))
metrics = ['avg_success_delay','deadline_violation_rate','energy_per_successful_task','blocked_user_seconds','total_energy','handovers_per_user_minute','task_success_rate']
print('Columns relevant:', [k for k in rows[0] if any(k==m or k==m+'_mean' for m in metrics)])
print('Method keys:', sorted({r['method'] for r in rows}))
data = {}
for method in ['han_pdqn','pdqn']:
    rs = [r for r in rows if r['method']==method]
    data[method] = {m:sum(float(r[m]) for r in rs)/len(rs) for m in metrics}
delta = {m:100*(data['pdqn'][m]-data['han_pdqn'][m])/data['pdqn'][m] for m in metrics}
(build/'evidence.json').write_text(json.dumps({'assets':meta,'averages':data,'reductions_pct':delta},indent=2,ensure_ascii=False),encoding='utf-8')
draft = Path('C:/Users/19704/Desktop/leo_switch.pdf')
(build/'manuscript.txt').write_text('\n'.join(p.extract_text() for p in PdfReader(draft).pages), encoding='utf-8')
print(json.dumps({'averages':data,'reductions_pct':delta},indent=2))
