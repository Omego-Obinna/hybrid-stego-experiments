import json
from pathlib import Path
import joblib,numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score,roc_curve
from native_common import *
from native_features import features
def threshold(y,s):
    f,t,h=roc_curve(y,s);v=(t+1-f)/2;i=np.flatnonzero(np.isfinite(h));return float(h[i[np.argmax(v[i])]])
def control():
    rng=np.random.default_rng(881);data=[];ys=[]
    for i in range(120):
        raw=rng.integers(0,65536,size=(64,64,3),dtype=np.uint16)
        for y in [0,1]:data.append(features(((raw & np.uint16(65534))|y).astype(float)/65535));ys.append(y)
    x=np.stack(data);y=np.array(ys);out={}
    for family in ['logistic','extra_trees']:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=4000)) if family=='logistic' else ExtraTreesClassifier(n_estimators=200,min_samples_leaf=2,random_state=901,n_jobs=1)
        model.fit(x[:160],y[:160]);auc=roc_auc_score(y[160:],model.predict_proba(x[160:])[:,1]);out[family]=float(auc);assert auc>=.95,'Fixed feature-map known-leak control failed'
    save(ROOT/'known_leak_control.json',dict(auc=out,feature_map='same 504 native statistics',native_images_used=False))
def main():
    cfg=config();control();assert not (ROOT/'selection.json').exists() and not (ROOT/'heldout_started.json').exists();meta=json.loads((ROOT/'feature_metadata.json').read_text());fc=json.loads((ROOT/'feature_completion.json').read_text());assert sha(ROOT/'feature_dataset.npz')==fc['feature_sha256'];x=np.load(ROOT/'feature_dataset.npz',allow_pickle=False)['features'];models=ROOT/'models';models.mkdir();selection=[]
    for size in cfg['payload_bytes']:
        if not fc['evaluable'][str(size)]:continue
        candidates={};validation={}
        for method in ['cpm','pulsar']:
            ix={p:np.array([i for i,r in enumerate(meta) if r['message_bytes']==size and r['method']==method and r['partition']==p]) for p in ['train','validation']};y={p:np.array([meta[i]['label'] for i in ix[p]]) for p in ix};cands=[]
            for c in cfg['classifier']['logistic_C']:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=4000));model.fit(x[ix['train']],y['train']);s=model.predict_proba(x[ix['validation']])[:,1];path=models/f'{method}_{size}_lr_{c}.joblib';joblib.dump(model,path);orient=1 if roc_auc_score(y['validation'],s)>=.5 else -1;cands.append(dict(family='logistic',C=c,paths=[str(path)],orientation=orient,validation_auc=float(roc_auc_score(y['validation'],s*orient)),threshold=threshold(y['validation'],s*orient)))
                np.save(models/f'{method}_{size}_lr_{c}_validation.npy',s)
            best=max(cands,key=lambda r:r['validation_auc']);scores=[];paths=[]
            for seed in cfg['classifier']['seeds']:
                m=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=2,max_features=.7,random_state=seed,n_jobs=1);m.fit(x[ix['train']],y['train']);s=m.predict_proba(x[ix['validation']])[:,1];scores.append(s);path=models/f'{method}_{size}_et_{seed}.joblib';joblib.dump(m,path);paths.append(str(path));np.save(models/f'{method}_{size}_et_{seed}_validation.npy',s)
            avg=np.mean(scores,axis=0);orient=1 if roc_auc_score(y['validation'],avg)>=.5 else -1;et=dict(family='extra_trees',paths=paths,orientation=orient,validation_auc=float(roc_auc_score(y['validation'],avg*orient)),threshold=threshold(y['validation'],avg*orient));candidates[method]=dict(logistic=best,extra_trees=et);validation[method]=cands+[et]
        family=max(['logistic','extra_trees'],key=lambda f:np.mean([candidates[m][f]['validation_auc'] for m in candidates]))
        for method in candidates:selection.append(dict(method=method,message_bytes=size,primary_family=family,families=candidates[method],all_candidates=validation[method]))
        status(phase='training',stage=f'Validation-only native source selection at {size} application bytes',models_completed=len(selection),models_total=6)
    save(ROOT/'selection.json',dict(selected_at=now(),panels=selection,heldout_used=False));save(ROOT/'heldout_started.json',dict(started_at=now(),selection_sha256=sha(ROOT/'selection.json')))
    out=ROOT/'predictions';out.mkdir();panels=[];checks=[]
    for sel in selection:
        ix=np.array([i for i,r in enumerate(meta) if r['message_bytes']==sel['message_bytes'] and r['method']==sel['method'] and r['partition']=='test']);y=np.array([meta[i]['label'] for i in ix]);groups=np.array([meta[i]['group_id'] for i in ix])
        for family,c in sel['families'].items():
            scores=[]
            for j,path in enumerate(c['paths']):
                s=joblib.load(path).predict_proba(x[ix])[:,1];scores.append(s);p=out/f'{sel["method"]}_{sel["message_bytes"]}_{family}_{j}.npz';np.savez_compressed(p,scores=s,feature_indices=ix);checks.append(dict(model_path=path,prediction_path=str(p)))
            s=np.mean(scores,axis=0)*c['orientation'];p=out/f'{sel["method"]}_{sel["message_bytes"]}_{family}_ensemble.npz';np.savez_compressed(p,scores=s,labels=y,group_ids=groups)
            panels.append(dict(method=sel['method'],message_bytes=sel['message_bytes'],family=family,primary=family==sel['primary_family'],prediction_path=str(p),threshold=c['threshold'],test_groups=len(y)//2))
    save(ROOT/'test_predictions.json',dict(scored_at=now(),panels=panels,checkpoint_checks=checks,prediction_hashes={str(p):sha(p) for p in sorted(out.glob('*.npz'))}))
    status(phase='verification',stage='Saved native content predictions; independent model checks next')
if __name__=='__main__':main()
