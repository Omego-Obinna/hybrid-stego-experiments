"""Cover-weighted AUC and fractional-tie ranking, checked against sklearn."""
import numpy as np
from sklearn.metrics import roc_auc_score

def auc_kernel(y,score,group,n_groups):
    y=np.asarray(y,dtype=int);group=np.asarray(group,dtype=int)
    _,indices=np.unique(np.asarray(score,dtype=float),return_inverse=True)
    size=(int(indices.max())+1)*n_groups
    pos=np.bincount(indices[y==1]*n_groups+group[y==1],minlength=size).reshape(-1,n_groups).astype(float)
    neg=np.bincount(indices[y==0]*n_groups+group[y==0],minlength=size).reshape(-1,n_groups).astype(float)
    kernel=pos.T@(np.cumsum(neg,axis=0)-.5*neg)
    return kernel,pos.sum(axis=0),neg.sum(axis=0)

def weighted_auc(kernel,pos,neg,counts):
    counts=np.asarray(counts,dtype=float)
    numerator=np.einsum('bi,ij,bj->b',counts,kernel,counts,optimize=True)
    denominator=(counts@pos)*(counts@neg)
    assert np.all(denominator>0)
    return numerator/denominator

def rank_items(metadata,scores):
    # Each image's twelve candidate rows are contiguous and share epoch/cover.
    assert len(scores)%12==0
    values=np.asarray(scores).reshape(-1,12)
    labels=metadata['label'].to_numpy().reshape(-1,12)
    assert np.all(labels.sum(axis=1)==1)
    true=(values*labels).sum(axis=1)
    tie=np.isclose(values,true[:,None],rtol=0,atol=1e-12)
    above=((values>true[:,None])&~tie).sum(axis=1)
    ties=tie.sum(axis=1);assert np.all(ties>=1)
    result={f'top{k}':np.clip((k-above)/ties,0,1) for k in (1,3,5)}
    result['mean_rank']=above+(ties+1)/2
    result['mrr']=np.array([np.mean(1/np.arange(a+1,a+t+1)) for a,t in zip(above,ties)])
    result['epoch']=metadata['epoch'].to_numpy().reshape(-1,12)[:,0]
    result['cover_id']=metadata['cover_id'].astype(str).to_numpy().reshape(-1,12)[:,0]
    return result

def point(y,score,metadata):
    y=np.asarray(y);score=np.asarray(score);prediction=score>=.5
    tpr=float(prediction[y==1].mean());tnr=float((~prediction[y==0]).mean())
    ranking=rank_items(metadata,score)
    return dict(auc=float(roc_auc_score(y,score)),balanced_accuracy=(tpr+tnr)/2,
        fpr=1-tnr,fnr=1-tpr,**{k:float(ranking[k].mean()) for k in ('top1','top3','top5','mrr','mean_rank')})

def bootstrap_metrics(y,score,metadata,test_cover_ids,counts):
    covermap={str(c):i for i,c in enumerate(test_cover_ids)}
    group=np.array([covermap[str(c)] for c in metadata['cover_id']])
    g=len(test_cover_ids);kernel,pos,neg=auc_kernel(y,score,group,g)
    auc=weighted_auc(kernel,pos,neg,counts)
    assert np.isclose(weighted_auc(kernel,pos,neg,np.ones((1,g)))[0],roc_auc_score(y,score),atol=1e-12)
    # Spot-check cluster-weighted AUC with a separate library implementation.
    for k in range(min(5,len(counts))):
        expected=roc_auc_score(y,score,sample_weight=counts[k,group])
        assert np.isclose(auc[k],expected,atol=1e-12)
    pred=np.asarray(score)>=.5;y=np.asarray(y)
    tp=np.bincount(group,weights=((y==1)&pred).astype(float),minlength=g)
    tn=np.bincount(group,weights=((y==0)&~pred).astype(float),minlength=g)
    bacc=.5*((counts@tp)/(counts@pos)+(counts@tn)/(counts@neg))
    output=dict(auc=auc,balanced_accuracy=bacc)
    ranking=rank_items(metadata,score)
    rankgroup=np.array([covermap[str(c)] for c in ranking['cover_id']])
    denom=np.bincount(rankgroup,minlength=g)
    for metric in ('top1','top3','top5','mrr','mean_rank'):
        sums=np.bincount(rankgroup,weights=ranking[metric],minlength=g)
        output[metric]=(counts@sums)/(counts@denom)
    return output

def self_check():
    rng=np.random.default_rng(991)
    group=np.repeat(np.arange(4),12);y=np.tile(np.r_[1,np.zeros(11)],4)
    scores=rng.integers(0,5,len(y))/4
    kernel,pos,neg=auc_kernel(y,scores,group,4)
    counts=np.array([[1,1,1,1],[2,0,1,1],[0,0,0,4],[3,1,0,0]])
    values=weighted_auc(kernel,pos,neg,counts)
    for c,v in zip(counts,values):assert np.isclose(v,roc_auc_score(y,scores,sample_weight=c[group]),atol=1e-12)
    import pandas as pd
    meta=pd.DataFrame(dict(label=y,epoch=np.repeat(np.arange(4),12),cover_id=group))
    rank=rank_items(meta,np.full(len(y),.5))
    assert np.allclose(rank['top1'],1/12) and np.allclose(rank['mean_rank'],6.5)
    assert np.allclose(rank['mrr'],np.mean(1/np.arange(1,13)))
    return dict(weighted_auc_matches_sklearn=True,fractional_tie_baselines_correct=True)

if __name__=='__main__':print(self_check())
