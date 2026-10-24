import concurrent.futures,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import sealwatch as sw
from image_common import *
def extract(item):
    key,p,h=item;assert sha(p)==h,('image hash',p)
    out=ROOT/'features'/f'{h}.npy'
    if out.exists():
        a=np.load(out,allow_pickle=False);assert a.shape==(34671,) and np.isfinite(a).all()
        return key,str(out)
    a=sw.tools.flatten(sw.srm.extract_from_file(str(p))).astype(np.float32)
    assert a.shape==(34671,) and np.isfinite(a).all(),(key,a.shape)
    np.save(out,a,allow_pickle=False);return key,str(out)
def main():
    cfg=config();assert not (ROOT/'feature_completion.json').exists()
    blocks=complete_blocks(records());items={}
    for (cover,pay),rr in blocks.items():
        c=rr['cpm'];items['cover_'+str(cover)]=(c['cover_path'],c['cover_sha256'])
        for m,r in rr.items():items[f'{m}_{cover}_{pay}']=(r['stego_path'],r['stego_sha256'])
    (ROOT/'features').mkdir(exist_ok=True);mapping={};start=time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['feature_workers']) as pool:
        fut=[pool.submit(extract,(k,p,h)) for k,(p,h) in items.items()]
        for f in concurrent.futures.as_completed(fut):
            k,p=f.result();mapping[k]=p
            if len(mapping)%25==0:
                status(phase='features',stage='Extracting native 512-pixel SRM features',features_completed=len(mapping),features_total=len(items),phase_elapsed_sec=time.monotonic()-start)
                print(len(mapping),'/',len(items),flush=True)
    save(ROOT/'feature_manifest.json',mapping)
    save(ROOT/'feature_hashes.json',{p:sha(p) for p in sorted(set(mapping.values()))})
    save(ROOT/'feature_completion.json',dict(completed_at=now(),images=len(mapping),dimensions=34671,extractor='sealwatch 2025.9 srm.extract_from_file',manifest_sha256=sha(ROOT/'feature_manifest.json'),feature_hash_manifest_sha256=sha(ROOT/'feature_hashes.json')))
    status(phase='srm_training',stage='Training matched SRM ensemble wardens; no test scoring yet')
if __name__=='__main__':main()
