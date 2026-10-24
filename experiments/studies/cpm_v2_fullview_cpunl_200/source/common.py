from datetime import datetime, timezone
import base64, hashlib, json
from pathlib import Path
import numpy as np
import epoch_protocol as ep

def now(): return datetime.now(timezone.utc).isoformat()
def sha(data): return hashlib.sha256(data).hexdigest()
def save(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)
def read_jsonl(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def status(root, **fields):
    p=Path(root)/'status.json';data=json.loads(p.read_text()) if p.exists() else {}
    data.update(fields,updated_at=now());save(p,data)
    if not data.get('development_only',False):
        target=Path(root).parent.parent/'status/fullview.json'
        if target.parent.exists():save(target,data)
    return data
def check_sources(root):
    cfg=json.loads((root/'run_config.json').read_text())
    for name,digest in cfg['source_sha256'].items(): assert sha((root/'source'/name).read_bytes())==digest,name
    for name,digest in cfg.get('input_sha256',{}).items(): assert sha((root/name).read_bytes())==digest,name
    if 'private_manifest_sha256' in cfg:
        private=root/'secrets/private_manifest.json';assert sha(private.read_bytes())==cfg['private_manifest_sha256']
        for item in json.loads(private.read_text()):assert sha(Path(item['path']).read_bytes())==item['sha256']
    return cfg
def frame_bytes(e):
    common=dict(version=ep.VERSION.decode(),session=e.sid.hex(),epoch=e.epoch,header=e.hdr.hex())
    result=[]
    for i,n,c in [(1,e.nonce_a,e.gamma1),(2,e.nonce_b,e.gamma2),(3,e.nonce_c,e.stego)]:
        row=dict(common,channel=f'C{i}',nonce=n.hex(),content=base64.b64encode(c).decode())
        if i==3:row.update(codec=ep.CODEC.decode(),payload_bits=e.payload_bits,tag=e.tag.hex())
        result.append(json.dumps(row,sort_keys=True).encode())
    return result
def query_message(index,bits,chain):
    if index==0:return bytes((bits+7)//8)
    return ep.truncate(hashlib.shake_256(b'CPUNL-adaptive-public-query-v1'+chain).digest((bits+7)//8),bits)
def next_chain(chain,message,frames):
    response=ep.enc([(f'C{i+1}'.encode(),b) for i,b in enumerate(frames)]) if frames else b'PUBLIC-ORACLE-FAILURE'
    return hashlib.sha256(chain+ep.enc([(b'query',message),(b'response',response)])).digest()
def pixel_summary(original,stego):
    assert original.shape==stego.shape==(512,512)
    delta=stego.astype(np.int16)-original.astype(np.int16);v,n=np.unique(delta,return_counts=True)
    return dict(max_abs_pixel_change=int(np.abs(delta).max()),pixels_exceeding_one_level=int((np.abs(delta)>1).sum()),
        pixel_change_histogram={str(int(x)):int(c) for x,c in zip(v,n)})
def worker_response(stdout):
    try:return json.loads([s for s in stdout.splitlines() if s.strip()][-1])
    except (ValueError,IndexError):return {}
