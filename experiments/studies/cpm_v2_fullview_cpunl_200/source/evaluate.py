import argparse,json,time,warnings
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from common import check_sources,sha,save,status,now
from metrics import auc_kernel,weighted_auc,self_check

def model(family,seed,cfg):
    h=cfg['hyperparameters']
    if family=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=h['logistic_C'],max_iter=h['max_iter'],class_weight='balanced',solver='lbfgs',random_state=seed))
    return ExtraTreesClassifier(n_estimators=h['trees'],min_samples_leaf=h['min_samples_leaf'],max_features=h['max_features'],class_weight='balanced',random_state=seed,n_jobs=3)
def point(y,score):
    pred=score>=.5;tpr=float(pred[y==1].mean());fpr=float(pred[y==0].mean())
    auc=float(roc_auc_score(y,score));assert 1/24-1e-12<=auc<=23/24+1e-12
    return dict(auc=auc,balanced_accuracy=(tpr+1-fpr)/2,tpr=tpr,fpr=fpr,signed_gap=tpr-fpr,absolute_gap=abs(tpr-fpr))
def bootstrap(y,score,groups,counts):
    n=counts.shape[1];kernel,pos,neg=auc_kernel(y,score,groups,n);auc=weighted_auc(kernel,pos,neg,counts)
    for i in range(5):assert abs(auc[i]-roc_auc_score(y,score,sample_weight=counts[i,groups]))<1e-12
    pred=score>=.5;tp=np.bincount(groups,weights=((y==1)&pred).astype(float),minlength=n);fp=np.bincount(groups,weights=((y==0)&pred).astype(float),minlength=n)
    tpr=(counts@tp)/(counts@pos);fpr=(counts@fp)/(counts@neg)
    return dict(auc=auc,balanced_accuracy=(tpr+1-fpr)/2,signed_gap=tpr-fpr,absolute_gap=np.abs(tpr-fpr))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();cfg=check_sources(root)
    assert not cfg['development_only'],'Development fixtures never select recorded models'
    assert not (root/'primary_selection.json').exists(),'Never repeat a used test selection'
    controls=json.loads((root/'controls.json').read_text());assert controls['known_leak_detected']
    out=root/'evaluation';out.mkdir(exist_ok=False);(out/'models').mkdir()
    schema=json.loads((root/'feature_schema.json').read_text());X=np.load(root/'observer_features.npy',mmap_mode='r');y=np.load(root/'observer_labels.npy');meta=pd.read_csv(root/'observer_metadata.csv.gz',dtype={'cover_id':str})
    assert sha((root/'observer_features.npy').read_bytes())==schema['feature_sha256'] and np.array_equal(y,meta.label)
    for p1,p2 in [('train','val'),('train','test'),('val','test')]:assert not set(meta.loc[meta.partition==p1,'cover_id'])&set(meta.loc[meta.partition==p2,'cover_id'])
    save(out/'metric_method_check.json',self_check());validation=[];selection={};done=0
    for pay in cfg['payloads']:
        train=np.flatnonzero((meta.payload_bpp==pay)&(meta.partition=='train'));val=np.flatnonzero((meta.payload_bpp==pay)&(meta.partition=='val'))
        for view in cfg['views']:
            cols=np.asarray(schema['columns'][view]);trainX=np.asarray(X[np.ix_(train,cols)]);valX=np.asarray(X[np.ix_(val,cols)])
            for family in cfg['classifiers']:
                tag=f'p{round(pay*100):03d}_{view}_{family}';scores=[]
                for seed in cfg['model_seeds']:
                    status(root,phase='training',stage=f'{pay:.2f} bpp · {view} · {family} · seed {seed}',models_completed=done,models_total=90,eta_sec=None)
                    clf=model(family,seed,cfg)
                    with warnings.catch_warnings(record=True) as warns:
                        warnings.simplefilter('always',ConvergenceWarning);clf.fit(trainX,y[train])
                    unresolved=[str(x.message) for x in warns if issubclass(x.category,ConvergenceWarning)];assert not unresolved,unresolved
                    scores.append(clf.predict_proba(valX)[:,1]);joblib.dump(dict(estimator=clf,columns=cols),out/'models'/f'{tag}_seed{seed}.joblib',compress=3);done+=1
                raw=np.mean(scores,axis=0);rawauc=roc_auc_score(y[val],raw);orientation=-1 if rawauc<.5 else 1;oriented=raw if orientation==1 else 1-raw
                row=dict(payload_bpp=pay,view=view,classifier=family,tag=tag,orientation=orientation,raw_validation_auc=float(rawauc),validation_auc=float(roc_auc_score(y[val],oriented)))
                validation.append(row);np.savez_compressed(out/f'{tag}_validation.npz',indices=val,raw_seed_scores=np.stack(scores),ensemble=oriented)
        candidates=[x for x in validation if x['payload_bpp']==pay and x['view']==cfg['primary_view']];selection[str(pay)]=max(candidates,key=lambda r:r['validation_auc'])
    assert done==90;pd.DataFrame(validation).to_csv(out/'validation_metrics.csv',index=False)
    save(root/'primary_selection.json',dict(frozen_at=now(),test_scored_at_selection=False,selections=selection,rule=cfg['selection_rule'],validation_sha256=sha((out/'validation_metrics.csv').read_bytes())))
    digest=sha((root/'primary_selection.json').read_bytes());status(root,phase='test_scoring',stage='Validation selection frozen; scoring full-view held-out comparisons',models_completed=done,test_panels_completed=0)
    covermeta=pd.read_csv(root/'cover_manifest.csv',dtype={'cover_id':str});testcovers=list(covermeta.loc[covermeta.partition=='test','cover_id']);assert len(testcovers)==60
    rng=np.random.default_rng(cfg['uncertainty']['seed']);draws=rng.integers(0,60,(cfg['uncertainty']['replicates'],60));counts=np.stack([np.bincount(x,minlength=60) for x in draws]);np.save(out/'bootstrap_cover_counts.npy',counts)
    allrows=[];seedrows=[];primary=[]
    for i,spec in enumerate(validation):
        pay,view,family,tag=(spec[k] for k in ['payload_bpp','view','classifier','tag']);idx=np.flatnonzero((meta.payload_bpp==pay)&(meta.partition=='test'));scores=[]
        for seed in cfg['model_seeds']:
            saved=joblib.load(out/'models'/f'{tag}_seed{seed}.joblib');raw=saved['estimator'].predict_proba(np.asarray(X[np.ix_(idx,saved['columns'])]))[:,1]
            score=raw if spec['orientation']==1 else 1-raw;scores.append(score);seedrows.append(dict(payload_bpp=pay,view=view,classifier=family,seed=seed,**point(y[idx],score)))
        seedscore=np.stack(scores);score=seedscore.mean(axis=0);np.savez_compressed(out/f'{tag}_test.npz',indices=idx,seed_scores=seedscore,ensemble=score)
        groups=np.asarray([testcovers.index(str(x)) for x in meta.iloc[idx].cover_id]);boot=bootstrap(y[idx],score,groups,counts)
        row=dict(**spec,test_covers=meta.iloc[idx].cover_id.nunique(),rows=len(idx),primary=view==cfg['primary_view'] and family==selection[str(pay)]['classifier'],**point(y[idx],score))
        for metric,values in boot.items():row[metric+'_ci_low'],row[metric+'_ci_high']=map(float,np.quantile(values,[.025,.975]))
        if row['primary']:
            alpha=cfg['uncertainty']['primary_alpha'];row['auc_adjusted_ci_low'],row['auc_adjusted_ci_high']=map(float,np.quantile(boot['auc'],[alpha/2,1-alpha/2]));primary.append(dict(row))
        if view=='metadata_only':assert abs(row['auc']-.5)<1e-10
        allrows.append(row);status(root,test_panels_completed=i+1,stage=f'Checked full-view test panel {i+1}/30')
    assert len(allrows)==30 and len(primary)==3
    pd.DataFrame(allrows).to_csv(out/'test_metrics.csv',index=False);pd.DataFrame(seedrows).to_csv(out/'seed_metrics.csv',index=False);pd.DataFrame(primary).to_csv(out/'primary_metrics.csv',index=False)
    save(out/'completion.json',dict(completed_at=now(),models=done,panels=30,primary_panels=3,primary_selection_sha256=digest,all_controls_passed=True))
    status(root,phase='verification',stage='Checking all saved full-view model predictions',checkpoints_checked=0)
    print(json.dumps(dict(models=done,panels=30,primary=primary)))

if __name__=='__main__':main()
