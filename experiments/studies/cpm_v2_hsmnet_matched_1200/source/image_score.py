import json,sys,time
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from torch.utils.data import DataLoader
from image_common import *
from image_hsm_train import Pairs,HSMNet,evaluate
def main():
    cfg=config();assert not (ROOT/'test_scoring_started.json').exists(),'A held-out selection has already been used; do not rerun'
    hs=json.loads((ROOT/'hsm_selection.json').read_text())
    save(ROOT/'test_scoring_started.json',dict(started_at=now(),hsm_selection_sha256=sha(ROOT/'hsm_selection.json'),config_sha256=sha(ROOT/'run_config.json'),detectors=['HSMNet']))
    blocks=complete_blocks(records());out=ROOT/'predictions';out.mkdir();panels=[]
    torch.set_num_threads(1)
    for s in hs['panels']:
        pay=s['payload_bpp'];dataset=Pairs(blocks,pay,'test');loader=DataLoader(dataset,batch_size=cfg['hsmnet']['paired_batch_size'],shuffle=False,num_workers=2,pin_memory=True);scores=[]
        for seed,path in zip(cfg['hsmnet']['seeds'],s['paths']):
            model=HSMNet().cuda();ck=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);ev=evaluate(model,loader);scores.append(ev['scores'])
            with (out/f'hsm_{pay}_{seed}.npz').open('xb') as f:np.savez_compressed(f,**ev)
            del model;torch.cuda.empty_cache()
        avg=np.mean(scores,axis=0)*s['orientation']
        for m in cfg['methods']:
            use=ev['methods']==m;p=out/f'hsm_{m}_{pay}_ensemble.npz'
            with p.open('xb') as f:np.savez_compressed(f,cover_ids=ev['cover_ids'][use],labels=ev['labels'][use],scores=avg[use])
            panels.append(dict(detector='HSMNet equal-method mixture',method=m,payload_bpp=pay,prediction_path=str(p),threshold=s['threshold'],test_covers=int(use.sum()/2)))
            status(phase='heldout_scoring',stage='Scoring the frozen HSMNet models once',test_panels_completed=len(panels),test_panels_total=15)
    save(ROOT/'test_predictions.json',dict(scored_at=now(),panels=panels,prediction_hashes={str(p):sha(p) for p in sorted(out.glob('*.npz'))}))
    status(phase='verification',stage='Held-out predictions saved; independent checkpoint and artifact verification')
if __name__=='__main__':main()
