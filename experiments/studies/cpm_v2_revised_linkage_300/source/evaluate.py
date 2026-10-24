#!/usr/bin/env python3
"""Fixed models, validation-only selection, then held-out scoring and CIs."""
import argparse,json,time,warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from common import sha,save,now,status
from metrics import point,bootstrap_metrics,self_check

def model(name,seed,trees=400):
    if name=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=3000,
        class_weight='balanced',solver='lbfgs',random_state=seed))
    return ExtraTreesClassifier(n_estimators=trees,min_samples_leaf=2,max_features='sqrt',
        class_weight='balanced',random_state=seed,n_jobs=3)

def view_indices(schema,view):
    names=schema['names']
    prefixes={'image_only':('img_',),'text_only':('gamma_',),'cross_only':('cross_',),
              'text_image':('img_','gamma_','cross_')}[view]
    return np.array([i for i,n in enumerate(names) if n.startswith(prefixes)],dtype=int)

def synthetic_features(meta):
    eye=np.eye(12,dtype=np.float32)
    true=eye[meta['true_pair_index'].to_numpy(int)]
    candidate=eye[meta['candidate_pair_index'].to_numpy(int)]
    return np.concatenate([true,candidate,np.abs(true-candidate),true*candidate],axis=1)

def quick_controls():
    rows=[]
    for partition in ('train','val','test'):
        for cover in range(3):
            for true in range(12):
                for candidate in range(12):rows.append(dict(partition=partition,true_pair_index=true,
                    candidate_pair_index=candidate,label=int(true==candidate)))
    meta=pd.DataFrame(rows);train=meta.partition=='train';test=meta.partition=='test'
    clf=model('logistic',13);clf.fit(synthetic_features(meta[train]),meta.loc[train,'label'])
    auc=roc_auc_score(meta.loc[test,'label'],clf.predict_proba(synthetic_features(meta[test]))[:,1])
    assert auc>.99
    return dict(**self_check(),synthetic_known_leak_detected=True,synthetic_auc=float(auc))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();cfg=json.loads((root/'run_config.json').read_text())
    assert not (root/'primary_selection.json').exists(),'Never rerun selection against a used test set'
    out=root/'evaluation';out.mkdir()
    schema=json.loads((root/'feature_schema.json').read_text())
    assert sha((root/'feature_dataset.npz').read_bytes())==schema['feature_sha256']
    assert sha((root/'feature_metadata.csv.gz').read_bytes())==schema['metadata_sha256']
    data=np.load(root/'feature_dataset.npz');X,y=data['X'],data['y']
    meta=pd.read_csv(root/'feature_metadata.csv.gz',dtype={'cover_id':str})
    assert len(meta)==len(X)==len(y) and np.array_equal(meta.label.to_numpy(),y)
    for a,b in (('train','val'),('train','test'),('val','test')):
        assert not (set(meta.loc[meta.partition==a,'cover_id'])&set(meta.loc[meta.partition==b,'cover_id']))
    save(out/'preflight_controls.json',quick_controls())
    validation=[];selections={};models_done=0;modeldir=out/'models';modeldir.mkdir()
    start=time.perf_counter()
    # Training + validation phase: no test scoring or metric-based changes here.
    for payload in cfg['payloads']:
        train_idx=np.flatnonzero((meta.payload_bpp==payload)&(meta.partition=='train'))
        val_idx=np.flatnonzero((meta.payload_bpp==payload)&(meta.partition=='val'))
        assert len(train_idx)>0 and len(val_idx)>0
        for view in cfg['views']:
            cols=view_indices(schema,view)
            for classifier in cfg['classifiers']:
                tag=f'p{int(round(payload*100)):03d}_{view}_{classifier}'
                scores=[];convergence=[]
                for seed in cfg['model_seeds']:
                    status(root,phase='training',stage=f'{payload:.2f} bpp · {view} · {classifier} · seed {seed}',
                        models_completed=models_done,models_total=72,eta_sec=None)
                    clf=model(classifier,seed)
                    with warnings.catch_warnings(record=True) as records:
                        warnings.simplefilter('always',ConvergenceWarning)
                        clf.fit(X[np.ix_(train_idx,cols)],y[train_idx])
                    convergence += [str(w.message) for w in records if issubclass(w.category,ConvergenceWarning)]
                    score=clf.predict_proba(X[np.ix_(val_idx,cols)])[:,1];scores.append(score)
                    joblib.dump(dict(estimator=clf,columns=cols,seed=seed,tag=tag),modeldir/f'{tag}_seed{seed}.joblib',compress=3)
                    models_done+=1
                assert not convergence,f'Unresolved convergence warning: {tag}: {convergence}'
                seed_scores=np.stack(scores);mean=seed_scores.mean(axis=0)
                raw_auc=roc_auc_score(y[val_idx],mean);orientation=-1 if raw_auc<.5 else 1
                oriented=1-mean if orientation<0 else mean
                result=dict(payload_bpp=payload,view=view,classifier=classifier,tag=tag,orientation=orientation,
                    raw_validation_auc=float(raw_auc),validation_auc=float(roc_auc_score(y[val_idx],oriented)),
                    validation_covers=meta.iloc[val_idx].cover_id.nunique(),feature_count=len(cols))
                validation.append(result)
                np.savez_compressed(out/f'{tag}_validation.npz',row_indices=val_idx,seed_scores=seed_scores,
                                    ensemble_scores=oriented)
        candidates=[r for r in validation if r['payload_bpp']==payload and r['view']=='text_image']
        # max returns the first family on exact ties: the predeclared logistic first.
        selections[str(payload)]=max(candidates,key=lambda r:r['validation_auc'])
    pd.DataFrame(validation).to_csv(out/'validation_models.csv',index=False)
    selection=dict(frozen_at=now(),selection_rule=cfg['primary_selection'],selections=selections,
        validation_file_sha256=sha((out/'validation_models.csv').read_bytes()),test_scored_at_selection=False)
    save(root/'primary_selection.json',selection)
    selection_digest=sha((root/'primary_selection.json').read_bytes())
    status(root,phase='test_scoring',stage='Validation selection frozen; scoring held-out test covers',models_completed=72)
    selected_test=[c['cover_id'] for c in __import__('csv').DictReader((root/'cover_manifest.csv').open()) if c['partition']=='test']
    assert len(selected_test)==100
    rng=np.random.default_rng(cfg['uncertainty']['seed'])
    draws=rng.integers(0,len(selected_test),size=(cfg['uncertainty']['bootstrap_replicates'],len(selected_test)))
    counts=np.stack([np.bincount(draw,minlength=len(selected_test)) for draw in draws])
    np.save(out/'test_cover_bootstrap_counts.npy',counts)
    summary=[];seed_metrics=[];primary=[];controls=[]
    for spec in validation:
        pay,view,classifier,tag=(spec[k] for k in ('payload_bpp','view','classifier','tag'))
        idx=np.flatnonzero((meta.payload_bpp==pay)&(meta.partition=='test'));testmeta=meta.iloc[idx].reset_index(drop=True)
        scores=[]
        for seed in cfg['model_seeds']:
            saved=joblib.load(modeldir/f'{tag}_seed{seed}.joblib')
            scores.append(saved['estimator'].predict_proba(X[np.ix_(idx,saved['columns'])])[:,1])
        raw=np.stack(scores);oriented=raw if spec['orientation']>0 else 1-raw
        ensemble=oriented.mean(axis=0)
        np.savez_compressed(out/f'{tag}_test.npz',row_indices=idx,seed_scores=oriented,ensemble_scores=ensemble)
        result=point(y[idx],ensemble,testmeta)
        boot=bootstrap_metrics(y[idx],ensemble,testmeta,selected_test,counts)
        for k,s in zip(cfg['model_seeds'],oriented):
            seed_metrics.append(dict(payload_bpp=pay,view=view,classifier=classifier,seed=k,**point(y[idx],s,testmeta)))
        record=dict(payload_bpp=pay,view=view,classifier=classifier,orientation=spec['orientation'],
            test_covers=testmeta.cover_id.nunique(),test_images=len(testmeta)//12,candidate_rows=len(testmeta),
            primary=(view=='text_image' and classifier==selections[str(pay)]['classifier']),**result)
        for name,values in boot.items():
            lo,hi=np.quantile(values,[.025,.975]);record.update({name+'_ci_low':float(lo),name+'_ci_high':float(hi)})
        lo,hi=np.quantile(boot['auc'],[.05/6,1-.05/6])
        record.update(auc_familywise_ci_low=float(lo),auc_familywise_ci_high=float(hi))
        summary.append(record)
        if record['primary']:primary.append(record)
        if view in ('image_only','text_only'):
            assert np.isclose(result['auc'],.5,atol=1e-8),'Marginal-balance control failed'
            assert np.isclose(result['balanced_accuracy'],.5,atol=1e-8)
        status(root,phase='test_scoring',stage=f'Test panel {len(summary)}/24 checked',test_panels_completed=len(summary),test_panels_total=24)
    # Deliberately encoded synthetic agreement is a separate positive control.
    for pay in cfg['payloads']:
        train=np.flatnonzero((meta.payload_bpp==pay)&(meta.partition=='train'))
        test=np.flatnonzero((meta.payload_bpp==pay)&(meta.partition=='test'))
        clf=model('logistic',13);clf.fit(synthetic_features(meta.iloc[train]),y[train])
        pred=clf.predict_proba(synthetic_features(meta.iloc[test]))[:,1]
        auc=roc_auc_score(y[test],pred);assert auc>.99
        controls.append(dict(payload_bpp=pay,control='synthetic_pair_agreement',auc=float(auc),
            interpretation='Engineered signal checks pipeline functionality, not sensitivity to every possible real leakage'))
    assert len(primary)==3 and sha((root/'primary_selection.json').read_bytes())==selection_digest
    pd.DataFrame(summary).to_csv(out/'test_metrics.csv',index=False)
    pd.DataFrame(seed_metrics).to_csv(out/'test_metrics_by_seed.csv',index=False)
    pd.DataFrame(primary).to_csv(out/'primary_metrics.csv',index=False)
    save(out/'controls.json',controls)
    save(out/'evaluation_completion.json',dict(completed_at=now(),panels=24,models=72,
        primary_selection_sha256=selection_digest,primary=primary,all_controls_passed=True,
        elapsed_sec=time.perf_counter()-start,pointwise_ci_level=.95,
        primary_auc_interval='98.3333% per payload; Bonferroni familywise 95% over three selected payloads',
        conditional_inference='Fixed trained ensembles, observed complete cover blocks and fixed 12-pair bank'))
    status(root,phase='verification',stage='Test evaluation finished; independently validating saved predictions')
    print(json.dumps(dict(panels=24,primary_panels=3,models=72,all_controls_passed=True)))

if __name__=='__main__':main()
