"""Bounded, fixed HSMNet pilot on saved development images only."""
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(os.environ['CPM_HSM_PILOT_RUN'])
sys.path.insert(0, str(ROOT/'source/HSMNet/src'))
from models.HSMNet import HSMNet


def now(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def status(**kw):
    p=ROOT/'status.json';s=json.loads(p.read_text()) if p.exists() else {}
    s.update(kw,updated_at=now());save(p,s);save(ROOT.parents[1]/'status/hsmpilot.json',s)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


class Pairs(Dataset):
    def __init__(self, groups, payload, partition, seed=0):
        self.rows=[groups[k] for k in sorted(groups) if k[1]==payload and groups[k]['cpm']['pilot_partition']==partition]
        self.train=partition=='train';self.seed=seed;self.epoch=0;self.methods=['cpm','suniward']
    def __len__(self): return len(self.rows) if self.train else 2*len(self.rows)
    def __getitem__(self, index):
        if self.train: i=index;method=self.methods[(index+self.epoch+self.seed)%2]
        else: i,m=divmod(index,2);method=self.methods[m]
        r=self.rows[i][method]
        with Image.open(r['cover_path']) as im: a=np.asarray(im,dtype=np.float32).copy()
        with Image.open(r['stego_path']) as im: b=np.asarray(im,dtype=np.float32).copy()
        assert a.shape==b.shape==(512,512)
        x=np.stack([a,b])[:,None]
        if self.train:
            rng=np.random.default_rng(self.seed*1000000+self.epoch*10000+index)
            x=np.rot90(x,int(rng.integers(4)),axes=(-2,-1))
            if rng.integers(2): x=x[...,::-1]
        return torch.from_numpy(x.copy()),torch.tensor([0,1]),method,r['cover_id']


def evaluate(model, loader):
    model.eval();scores=[];labels=[];ids=[];methods=[];losses=[]
    with torch.inference_mode():
        for x,y,ms,cs in loader:
            x=x.flatten(0,1).cuda();y=y.flatten().cuda();out=model(x)
            scores.extend((out[:,1]-out[:,0]).cpu().numpy());labels.extend(y.cpu().numpy())
            losses.extend(torch.nn.functional.cross_entropy(out,y,reduction='none').cpu().numpy())
            ids.extend(np.repeat(cs.numpy(),2));methods.extend(np.repeat(ms,2))
    return dict(scores=np.asarray(scores),labels=np.asarray(labels),cover_ids=np.asarray(ids),
                methods=np.asarray(methods),loss=float(np.mean(losses)))


def control(cfg):
    """Separate strong synthetic texture control; no native pilot observation."""
    seed_all(cfg['control']['seed']);rng=np.random.default_rng(cfg['control']['seed'])
    a=np.clip(rng.normal(127,12,(48,1,512,512)),0,255).astype(np.float32)
    board=((np.indices((512,512)).sum(axis=0)%2)*2-1).astype(np.float32)
    b=np.clip(a+48*board,0,255)
    model=HSMNet().cuda();opt=torch.optim.Adamax(model.parameters(),lr=.001)
    history=[]
    for epoch in range(5):
        model.train();losses=[]
        for start in range(0,32,2):
            x=np.stack([a[start:start+2],b[start:start+2]],axis=1).reshape(4,1,512,512)
            y=torch.tensor([0,1,0,1],device='cuda');opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(torch.from_numpy(x).cuda()),y)
            assert torch.isfinite(loss);loss.backward();opt.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)));print('synthetic control epoch',epoch+1,history[-1],flush=True)
    model.eval();scores=[]
    with torch.inference_mode():
        for start in range(32,48,2):
            x=np.stack([a[start:start+2],b[start:start+2]],axis=1).reshape(4,1,512,512)
            out=model(torch.from_numpy(x).cuda());scores.extend((out[:,1]-out[:,0]).cpu().numpy())
    labels=np.tile([0,1],16);auc=float(roc_auc_score(labels,scores))
    np.savez_compressed(ROOT/'synthetic_control_predictions.npz',scores=np.asarray(scores),labels=labels)
    save(ROOT/'synthetic_control.json',dict(auc=auc,training_loss=history,
         scope='32 synthetic texture pairs for training, 16 independent pairs for control evaluation; fixed checkerboard bias, not native-image security or detector completeness.'))
    assert auc>=.95,('Strong synthetic control did not pass before native fitting',auc)
    del model,opt,a,b;torch.cuda.empty_cache()


def threshold(y,scores):
    f,t,h=roc_curve(y,scores);ix=np.flatnonzero(np.isfinite(h));return float(h[ix[np.argmax((t+1-f)[ix])]])


def table(name, rows):
    with (ROOT/name).open('x',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    assert not (ROOT/'pipeline_started.json').exists()
    cfg=json.loads((ROOT/'run_config.json').read_text())
    for name,digest in cfg['source_hashes'].items(): assert sha(ROOT/name)==digest,name
    for name,digest in cfg['input_hashes'].items(): assert sha(ROOT/name)==digest,name
    gpu_lock=(ROOT.parents[1]/'comparison_gpu.lock').open('a')
    status(state='running',phase='gpu_wait',stage='Waiting for shared GPU for the fixed HSMNet pilot')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX)
    assert torch.cuda.is_available();torch.set_num_threads(1)
    save(ROOT/'pipeline_started.json',dict(started_at=now(),pid=os.getpid(),config_sha256=sha(ROOT/'run_config.json')))
    rows=json.loads((ROOT/'saved_image_records.json').read_text());cohort=json.loads((ROOT/'cohort.json').read_text())
    assert len(rows)==1800 and len(cohort)==300 and all(c['full_partition']=='train' for c in cohort)
    assert not set(c['cover_id'] for c in cohort)&set(cfg['excluded_srm_pilot_covers'])
    groups={}
    for r in rows: groups.setdefault((r['cover_id'],r['payload_bpp']),{})[r['method']]=r
    groups={k:g for k,g in groups.items() if set(g)==set(cfg['methods']) and all(r['valid_image'] for r in g.values())}
    part={c['cover_id']:c['pilot_partition'] for c in cohort};counts={}
    for pay in cfg['payloads']:
        counts[str(pay)]={p:sum(k[1]==pay and part[k[0]]==p for k in groups) for p in cfg['split_counts']}
        for p,n in cfg['minimum_complete_counts'].items(): assert counts[str(pay)][p]>=n,(pay,p,counts)
    image_hashes={}
    for group in groups.values():
        for r in group.values():
            image_hashes[r['cover_path']]=r['cover_sha256'];image_hashes[r['stego_path']]=r['stego_sha256']
    status(state='running',phase='audit',stage='Hash audit of saved native images; full study paused')
    for path,digest in image_hashes.items(): assert sha(path)==digest,path
    save(ROOT/'image_hashes.json',image_hashes)
    save(ROOT/'cohort_accounting.json',dict(selected_source_groups=300,saved_method_records=1800,
         valid_saved_images=sum(r['valid_image'] for r in rows),common_valid_counts=counts,
         full_study_validation_or_test_covers_used=0,srm_pilot_covers_used=0,native_images_generated=0,failed_cases_replaced=0))
    status(phase='control',stage='Separate synthetic HSMNet texture control before native fitting')
    control(cfg)
    (ROOT/'models').mkdir();selections=[];models_done=0;started=time.monotonic()
    for pay in cfg['payloads']:
        val=DataLoader(Pairs(groups,pay,'validation'),batch_size=2,shuffle=False,num_workers=2,pin_memory=True)
        paths=[];validation=[]
        for seed in cfg['seeds']:
            seed_all(seed);model=HSMNet().cuda()
            opt=torch.optim.Adamax(model.parameters(),lr=cfg['lr'],eps=1e-8,weight_decay=0)
            scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=cfg['milestones'],gamma=.1)
            train=Pairs(groups,pay,'train',seed);best=float('inf')
            best_path=ROOT/'models'/f'{pay}_{seed}_best.pt';last_path=ROOT/'models'/f'{pay}_{seed}_last.pt'
            with (ROOT/'models'/f'{pay}_{seed}_history.jsonl').open('x') as log:
                for epoch in range(cfg['epochs']):
                    train.epoch=epoch;generator=torch.Generator().manual_seed(seed+10000*epoch)
                    loader=DataLoader(train,batch_size=2,shuffle=True,generator=generator,num_workers=2,pin_memory=True)
                    model.train();loss_sum=0.;n=0;correct=0
                    for bi,(x,y,_,_) in enumerate(loader):
                        x=x.flatten(0,1).cuda();y=y.flatten().cuda();opt.zero_grad(set_to_none=True)
                        out=model(x);loss=torch.nn.functional.cross_entropy(out,y);assert torch.isfinite(loss)
                        loss.backward();opt.step();loss_sum+=float(loss.detach())*len(y);n+=len(y)
                        correct+=int((out.argmax(1)==y).sum())
                        if bi%20==0:
                            status(phase='training',stage=f'HSMNet pilot {pay:.2f} bpp, seed {seed}, epoch {epoch+1}/{cfg["epochs"]}',
                              payload_bpp=pay,seed=seed,epoch=epoch+1,epochs=cfg['epochs'],batch=bi+1,batches=len(loader),
                              models_completed=models_done,models_total=6,elapsed_sec=time.monotonic()-started)
                    scheduler.step();history=dict(epoch=epoch+1,training_loss=loss_sum/n,training_accuracy=correct/n,
                                                     elapsed_sec=time.monotonic()-started)
                    if epoch==0 or (epoch+1)%5==0:
                        ev=evaluate(model,val);history['validation_loss']=ev['loss']
                        if ev['loss']<best:
                            best=ev['loss'];torch.save(dict(state_dict=model.state_dict(),epoch=epoch+1,
                                validation_loss=best,payload_bpp=pay,seed=seed),best_path)
                    torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                                    epoch=epoch+1,payload_bpp=pay,seed=seed),last_path)
                    log.write(json.dumps(history)+'\n');log.flush();os.fsync(log.fileno());print(pay,seed,history,flush=True)
            checkpoint=torch.load(best_path,map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state_dict'])
            ev=evaluate(model,val);np.savez_compressed(ROOT/'models'/f'{pay}_{seed}_validation.npz',**ev)
            paths.append(str(best_path));validation.append(ev);models_done+=1
            status(models_completed=models_done);del model,opt,scheduler;torch.cuda.empty_cache()
        assert np.array_equal(validation[0]['labels'],validation[1]['labels'])
        average=np.mean([v['scores'] for v in validation],axis=0);y=validation[0]['labels']
        orientation=1 if roc_auc_score(y,average)>=.5 else -1
        selections.append(dict(payload_bpp=pay,paths=paths,orientation=orientation,
             threshold=threshold(y,average*orientation),validation_auc=float(roc_auc_score(y,average*orientation)),
             rule='Same two-seed mixture observer for both methods; checkpoint minimum joint validation loss, common orientation/threshold validation only.'))
    save(ROOT/'selection.json',dict(selected_at=now(),panels=selections,test_used=False))
    assert not (ROOT/'heldout_started.json').exists()
    save(ROOT/'heldout_started.json',dict(started_at=now(),selection_sha256=sha(ROOT/'selection.json')))
    (ROOT/'predictions').mkdir();checks=[];panels=[]
    status(phase='heldout_scoring',stage='One held-out HSMNet pilot evaluation; every choice frozen')
    for sel in selections:
        pay=sel['payload_bpp'];loader=DataLoader(Pairs(groups,pay,'test'),batch_size=2,shuffle=False,num_workers=2,pin_memory=True)
        predicted=[]
        for seed,path in zip(cfg['seeds'],sel['paths']):
            seed_all(seed);model=HSMNet().cuda();ck=torch.load(path,map_location='cpu',weights_only=True)
            model.load_state_dict(ck['state_dict']);ev=evaluate(model,loader)
            prediction=ROOT/'predictions'/f'{pay}_{seed}.npz';np.savez_compressed(prediction,**ev)
            checks.append(dict(payload_bpp=pay,seed=seed,model_path=path,prediction_path=str(prediction),model_sha256=sha(path),prediction_sha256=sha(prediction)))
            predicted.append(ev);del model;torch.cuda.empty_cache()
        for key in ['labels','cover_ids','methods']: assert np.array_equal(predicted[0][key],predicted[1][key])
        average=np.mean([v['scores'] for v in predicted],axis=0)*sel['orientation']
        for method in cfg['methods']:
            ix=predicted[0]['methods']==method;path=ROOT/'predictions'/f'{method}_{pay}_ensemble.npz'
            np.savez_compressed(path,scores=average[ix],labels=predicted[0]['labels'][ix],cover_ids=predicted[0]['cover_ids'][ix])
            panels.append(dict(method=method,payload_bpp=pay,prediction_path=str(path),threshold=sel['threshold'],test_groups=int(ix.sum()/2)))
    save(ROOT/'test_predictions.json',dict(scored_at=now(),checkpoint_checks=checks,panels=panels))
    status(phase='verification',stage='Independent saved-checkpoint prediction verification')
    checked=0
    for item in checks:
        assert sha(item['model_path'])==item['model_sha256'] and sha(item['prediction_path'])==item['prediction_sha256']
        seed_all(item['seed']);model=HSMNet().cuda();model.load_state_dict(torch.load(item['model_path'],map_location='cpu',weights_only=True)['state_dict'])
        loader=DataLoader(Pairs(groups,item['payload_bpp'],'test'),batch_size=2,shuffle=False,num_workers=2,pin_memory=True)
        actual=evaluate(model,loader);saved=np.load(item['prediction_path'],allow_pickle=False)
        assert np.allclose(actual['scores'],saved['scores'],rtol=0,atol=1e-6)
        for key in ['labels','cover_ids','methods']: assert np.array_equal(actual[key],saved[key])
        checked+=1;status(checkpoints_checked=checked);del model;torch.cuda.empty_cache()
    assert checked==6
    ids=sorted(c['cover_id'] for c in cohort if c['pilot_partition']=='test');index={c:i for i,c in enumerate(ids)}
    weights=np.random.default_rng(cfg['bootstrap_seed']).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=cfg['bootstrap_draws'])
    metrics=[];boots={}
    for panel in panels:
        z=np.load(panel['prediction_path'],allow_pickle=False);y=z['labels'];s=z['scores'];g=z['cover_ids']
        delta=s[y==1,None]-s[None,y==0];concordance=(delta>0).astype(float)+.5*(delta==0)
        wp=weights[:,[index[int(c)] for c in g[y==1]]];wn=weights[:,[index[int(c)] for c in g[y==0]]]
        draws=np.einsum('bi,ij,bj->b',wp,concordance,wn,optimize=True)/(wp.sum(1)*wn.sum(1))
        auc=float(roc_auc_score(y,s));lo,hi=np.quantile(draws,[.025,.975]);accepted=s>=panel['threshold']
        gap=float(accepted[y==1].mean()-accepted[y==0].mean());metrics.append(dict(method=panel['method'],payload_bpp=panel['payload_bpp'],
          test_groups=panel['test_groups'],auc=auc,ci_low=float(lo),ci_high=float(hi),balanced_accuracy=(1+gap)/2,acceptance_gap=gap))
        boots[panel['method'],panel['payload_bpp']]=draws
    contrasts=[]
    for pay in cfg['payloads']:
        rr={r['method']:r for r in metrics if r['payload_bpp']==pay}
        lo,hi=np.quantile(boots['cpm',pay]-boots['suniward',pay],[.05/6,1-.05/6])
        contrasts.append(dict(payload_bpp=pay,cpm_auc_minus_suniward=rr['cpm']['auc']-rr['suniward']['auc'],
                             adjusted_ci_low=float(lo),adjusted_ci_high=float(hi)))
    for path,digest in image_hashes.items(): assert sha(path)==digest
    for name,digest in cfg['source_hashes'].items(): assert sha(ROOT/name)==digest
    for name,digest in cfg['input_hashes'].items(): assert sha(ROOT/name)==digest
    table('detectability.csv',metrics);table('paired_auc_contrasts.csv',contrasts)
    save(ROOT/'verified_summary.json',dict(passed=True,checked_at=now(),selected_checkpoint_predictions_checked=checked,
       image_records_retained=1800,cohorts=counts,full_study_validation_or_test_covers_used=0,srm_pilot_covers_used=0,
       original_sources_and_input_image_hashes_checked=True,native_images_generated=0))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'font.size':10,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(5.7,3.2),layout='constrained')
    for method,color,marker in [('cpm','#176B87','o'),('suniward','#B76A22','s')]:
        rr=[r for r in metrics if r['method']==method];y=np.array([r['auc'] for r in rr]);lo=np.array([r['ci_low'] for r in rr]);hi=np.array([r['ci_high'] for r in rr])
        ax.errorbar([r['payload_bpp'] for r in rr],y,yerr=[y-lo,hi-y],color=color,marker=marker,capsize=3,
                    label='CPM' if method=='cpm' else 'S-UNIWARD')
    ax.axhline(.5,color='.5',ls='--',lw=.8);ax.set(xlabel='Application payload (bpp)',ylabel='HSMNet pilot held-out AUC',ylim=(0,1.025))
    ax.set_xticks(cfg['payloads']);ax.legend(frameon=False);ax.grid(alpha=.2)
    fig.savefig(ROOT/'pilot_detectability.png',dpi=300);fig.savefig(ROOT/'pilot_detectability.pdf');plt.close(fig)
    lines=['# Short HSMNet detectability pilot','',
      'Fixed short neural-detector study on 300 fresh development covers, disjoint from the SRM pilot and taken exclusively from the full study training pool. The full SRM/HSMNet study remains paused; its validation/test covers are untouched. Saved CPM and recorded S-UNIWARD images are reused without new embedding.', '',
      'Official HSMNet architecture, native 512x512 grayscale inputs in [0,255], paired batch size2, random paired D4 training augmentation. Fixed 200/40/60 source-cover split; 20 epochs and seeds801/802 per payload. One equal-method cycling mixture detector per payload/seed; the same observer scores both methods. Adamax lr0.001, milestones12/17, gamma0.1; best joint validation cross-entropy at epochs1/5/10/15/20. Only common score orientation/threshold uses validation. This is a reduced training budget from random initialization, not reproduction of published HSMNet scores.', '',
      '| Method | bpp | Test groups | AUC [pointwise 95% CI] | Balanced accuracy |',
      '|---|---:|---:|---|---:|']
    for row in metrics: lines.append(f"| {row['method']} | {row['payload_bpp']:.2f} | {row['test_groups']} | {row['auc']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['balanced_accuracy']:.4f} |")
    lines+=['','| bpp | CPM minus S-UNIWARD AUC | Three-payload adjusted interval |','|---:|---:|---|']
    for row in contrasts:lines.append(f"| {row['payload_bpp']:.2f} | {row['cpm_auc_minus_suniward']:.4f} | [{row['adjusted_ci_low']:.4f}, {row['adjusted_ci_high']:.4f}] |")
    lines+=['',f'All {checked} selected checkpoint prediction sets reproduced. Source-cover bootstrap uses {cfg["bootstrap_draws"]} shared draws; common-valid counts and all 1800 saved records are retained in cohort_accounting.json. No failed case was retried or replaced. The synthetic texture control checks only strong artificial-signal sensitivity.', '',
      'Training/validation histories and selected epochs remain saved. Near-chance scores in this small, short, randomly initialized neural run may reflect insufficient training and cannot establish security, equivalence or performance of a fully trained HSMNet. Pilot results must not change the frozen full study. The HSMNet and SRM pilots have different cohorts and budgets; do not pool their scores or claim a paired cross-detector ranking.', '',
      '![HSMNet pilot](pilot_detectability.png)']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');save(ROOT/'completion.json',dict(completed_at=now(),verified=True,report_sha256=sha(ROOT/'RESULTS.md')))
    status(state='completed',phase='completed',stage='HSMNet pilot completed and checkpoint-verified; final review pending',
           models_completed=6,models_total=6,report_ready=True)


if __name__=='__main__':
    try: main()
    except BaseException as error:
        (ROOT/'pipeline_failure.txt').write_text(traceback.format_exc())
        status(state='failed',stage='HSMNet pilot stopped; preserve evidence and investigate',last_error=str(error))
        raise
