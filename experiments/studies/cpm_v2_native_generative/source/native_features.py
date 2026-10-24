import json
from pathlib import Path
import numpy as np,png
from PIL import Image
from native_common import *
def pixels(path):
    path=Path(path)
    if path.suffix=='.png':
        w,h,it,meta=png.Reader(str(path)).asDirect();a=np.vstack(list(it)).reshape(h,w,meta['planes']);assert not meta.get('alpha',False);depth=meta['bitdepth'];a=a.astype(float)/(2**depth-1)
    else:
        with Image.open(path) as im:assert im.mode=='L';a=np.asarray(im,dtype=float)[...,None]/255
    if a.shape[2]==1:a=np.repeat(a,3,axis=2)
    assert a.shape[2]==3 and np.isfinite(a).all();return a
def features(a):
    out=[];q=np.rint(np.clip(a,0,1)*65535).astype(np.uint16)
    for ch in range(3):
        x=a[:,:,ch];out.extend([x.mean(),x.std(),*np.quantile(x,[.01,.05,.25,.5,.75,.95,.99])])
        rs=[x[:,1:]-x[:,:-1],x[1:]-x[:-1],x[1:,1:]-x[:-1,:-1]]
        for r in rs:out.extend([r.mean(),r.std(),abs(r).mean(),np.sqrt(np.mean(r*r)),*np.quantile(r,[.05,.25,.5,.75,.95])])
        for r in rs:out.extend(np.histogram(np.clip(r,-.25,.25),bins=32,range=(-.25,.25))[0]/r.size)
        for bit in range(8):
            p=((q[:,:,ch]>>bit)&1).astype(float);out.extend([p.mean(),np.mean(p[:,1:]*p[:,:-1])-p.mean()**2])
        phase=q[:,:,ch]%257;out.extend(np.histogram(phase,bins=16,range=(0,257))[0]/phase.size)
    for i,j in [(0,1),(0,2),(1,2)]:
        d=a[:,:,i]-a[:,:,j];out.extend([d.mean(),d.std(),abs(d).mean(),np.mean((a[:,:,i]-a[:,:,i].mean())*(a[:,:,j]-a[:,:,j].mean()))])
    out=np.asarray(out,dtype=np.float64);assert out.shape==(504,) and np.isfinite(out).all();return out
def main():
    cfg=config();assert not (ROOT/'feature_dataset.npz').exists();rows=[json.loads(l) for l in (ROOT/'generation_results.jsonl').open()];groups={}
    for r in rows:groups.setdefault(r['case_id'],{})[r['method']]=r
    selected={k:v for k,v in groups.items() if len(v)==2 and all(r['exact_recovery'] for r in v.values())};data=[];meta=[]
    for k,rr in sorted(selected.items()):
        for method in ['cpm','pulsar']:
            r=rr[method]
            for label,which in [(0,'benign'),(1,'stego')]:
                path=r[which+'_path'];assert sha(path)==r[which+'_sha256'];data.append(features(pixels(path)));meta.append(dict(case_id=k,group_id=r['group_id'],method=method,label=label,partition=r['partition'],message_bytes=r['message_bytes'],source_path=path,source_sha256=r[which+'_sha256']))
        if len(meta)%80==0:status(phase='features',stage='Extracting fixed native-precision content features',features_completed=len(meta),features_total=4*len(selected))
    np.savez_compressed(ROOT/'feature_dataset.npz',features=np.stack(data));save(ROOT/'feature_metadata.json',meta)
    counts={str(n):{part:sum(v['cpm']['message_bytes']==n and v['cpm']['partition']==part for v in selected.values()) for part in ['train','validation','test']} for n in cfg['payload_bytes']}
    evaluable={n:all(v[p]>=cfg['minimum_complete'][p] for p in v) for n,v in counts.items()}
    save(ROOT/'feature_completion.json',dict(completed_at=now(),counts=counts,evaluable=evaluable,dimensions=504,feature_sha256=sha(ROOT/'feature_dataset.npz'),metadata_sha256=sha(ROOT/'feature_metadata.json'),all_attempts=len(rows),complete_groups=len(selected)))
if __name__=='__main__':main()
