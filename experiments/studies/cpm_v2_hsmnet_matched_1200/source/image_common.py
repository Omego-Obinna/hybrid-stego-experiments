import csv,hashlib,json,os,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get('CPM_IMAGE_RUN','/home/omego/Documents/Hybrid_Stego_Eva/revision_20261017/runs/cpm_v2_matched_image_1200'))
REV=ROOT.parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False));q.replace(p)
def config():
    c=json.loads((ROOT/'run_config.json').read_text())
    for p,h in c['source_hashes'].items():assert sha(ROOT/p)==h,('source changed',p)
    for p,h in c['input_hashes'].items():assert sha(ROOT/p)==h,('input changed',p)
    return c
def status(**kw):
    try:s=json.loads((ROOT/'status.json').read_text())
    except FileNotFoundError:s={}
    s.update(kw,updated_at=now(),run_id=ROOT.name);save(ROOT/'status.json',s)
    save(REV/'status/hsmcomparison.json',dict(s,panel='hsm_only',scope='Author-prioritised HSMNet five-method comparison',generative_status_file=str(REV/'runs/cpm_v2_native_generative/status.json')))
def records():return [json.loads(l) for l in (ROOT/'image_results.jsonl').open()]
def complete_blocks(rows):
    methods=['cpm','suniward','mipod','wow','hill'];groups={}
    for r in rows:groups.setdefault((r['cover_id'],r['payload_bpp']),{})[r['method']]=r
    return {k:v for k,v in groups.items() if set(v)==set(methods) and all(r['valid_image'] for r in v.values())}
def append(f,r):f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def image_metrics(a,b):
    from skimage.metrics import structural_similarity
    d=b.astype(np.int16)-a.astype(np.int16);mse=float(np.mean(d.astype(float)**2))
    return dict(mse=mse,psnr_db=10*np.log10(255**2/mse) if mse else None,ssim=float(structural_similarity(a,b,data_range=255)),modified_pixels=int(np.count_nonzero(d)),pixel_change_max=int(np.max(abs(d))),pixels_over_one=int(np.count_nonzero(abs(d)>1)))
def worker_response(s):
    for line in reversed(s.splitlines()):
        try:return json.loads(line)
        except json.JSONDecodeError:continue
    return {}
