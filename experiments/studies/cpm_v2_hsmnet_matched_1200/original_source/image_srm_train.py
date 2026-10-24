import json,time
from pathlib import Path
import joblib,numpy as np,sealwatch as sw
from sklearn.metrics import roc_auc_score
from image_common import *
def choose_threshold(y,s):
    # Maximise validation balanced accuracy; deterministic threshold tie rule.
    from sklearn.metrics import roc_curve
    f,t,th=roc_curve(y,s);v=(t+1-f)/2;valid=np.flatnonzero(np.isfinite(th));i=valid[np.argmax(v[valid])]
    return float(th[i])
def main():
    cfg=config();assert not (ROOT/'srm_selection.json').exists()
    blocks=complete_blocks(records());fm=json.loads((ROOT/'feature_manifest.json').read_text());models=ROOT/'models_srm';models.mkdir()
    selections=[];done=0;start=time.monotonic()
    for pay in cfg['payloads']:
        ids={p:sorted(k[0] for k,v in blocks.items() if k[1]==pay and v['cpm']['partition']==p) for p in ['train','validation']}
        xc={p:np.stack([np.load(fm[f'cover_{i}'],allow_pickle=False) for i in ids[p]]) for p in ids}
        for m in cfg['methods']:
            xs={p:np.stack([np.load(fm[f'{m}_{i}_{pay}'],allow_pickle=False) for i in ids[p]]) for p in ids}
            y=np.r_[np.zeros(len(ids['validation'])),np.ones(len(ids['validation']))];candidates=[]
            for d in cfg['srm']['d_sub_candidates']:
                scores=[];paths=[]
                for seed in cfg['srm']['seeds']:
                    trainer=sw.ensemble_classifier.FldEnsembleTrainer(xc['train'],xs['train'],seed=seed,L=cfg['srm']['L'],d_sub=d,verbose=0)
                    model,history=trainer.train();path=models/f'{m}_{pay}_{d}_{seed}.joblib';joblib.dump(model,path);paths.append(str(path))
                    score=model.predict_confidence(np.r_[xc['validation'],xs['validation']]);scores.append(score)
                    np.save(models/f'{m}_{pay}_{d}_{seed}_validation.npy',score,allow_pickle=False)
                    save(models/f'{m}_{pay}_{d}_{seed}_training.json',history)
                    done+=1;status(phase='srm_training',stage=f'SRM validation-only selection: {m}, {pay:.2f} bpp',models_completed=done,models_total=90,phase_elapsed_sec=time.monotonic()-start)
                    print('SRM',done,'/90',m,pay,d,seed,flush=True)
                avg=np.mean(scores,axis=0);orient=1 if roc_auc_score(y,avg)>=.5 else -1;s=avg*orient
                candidates.append(dict(d_sub=d,paths=paths,validation_auc=float(roc_auc_score(y,s)),orientation=orient,threshold=choose_threshold(y,s)))
            best=max(candidates,key=lambda r:r['validation_auc']);selections.append(dict(method=m,payload_bpp=pay,**best,candidates=candidates,validation_ids=ids['validation']))
    save(ROOT/'srm_selection.json',dict(selected_at=now(),test_scored=False,panels=selections,rule=cfg['srm']['selection']))
    status(phase='hsm_training',stage='SRM models fixed; training published HSMNet architecture on the same covers')
if __name__=='__main__':main()
