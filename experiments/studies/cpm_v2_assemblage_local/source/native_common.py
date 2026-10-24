import hashlib,json,os,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(os.environ['CPM_NATIVE_RUN']);REV=ROOT.parents[1]
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False));q.replace(p)
def append(f,d):f.write(json.dumps(d,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def config():
    c=json.loads((ROOT/'run_config.json').read_text())
    for p,h in c['source_hashes'].items():assert sha(ROOT/p)==h,('source modified',p)
    for p,h in c['input_hashes'].items():assert sha(ROOT/p)==h,('input modified',p)
    return c
def status(**kw):
    s=json.loads((ROOT/'status.json').read_text());s.update(kw,updated_at=now());save(ROOT/'status.json',s);save(REV/'status'/('generative.json' if 'generative' in ROOT.name else 'distributed.json'),s)
def execute(cmd,folder,label,timeout=1200):
    import subprocess
    t=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,env=os.environ.copy());(folder/f'{label}.stdout.txt').write_text(r.stdout);(folder/f'{label}.stderr.txt').write_text(r.stderr);return r,time.perf_counter()-t
def response(s):
    for l in reversed(s.splitlines()):
        try:return json.loads(l)
        except json.JSONDecodeError:continue
    return {}
