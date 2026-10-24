import json,os,random,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from image_common import *
sys.path.insert(0,str(ROOT/'source/HSMNet/src'))
from models.HSMNet import HSMNet
from image_srm_train import choose_threshold
from sklearn.metrics import roc_auc_score
class Pairs(Dataset):
    def __init__(self,blocks,pay,part,seed=0):
        self.rr=[blocks[k] for k in sorted(blocks) if k[1]==pay and blocks[k]['cpm']['partition']==part]
        self.train=part=='train';self.seed=seed;self.epoch=0;self.methods=['cpm','suniward','mipod','wow','hill']
    def __len__(self):return len(self.rr) if self.train else len(self.rr)*5
    def __getitem__(self,index):
        if self.train:
            i=index;m=self.methods[(index+self.epoch+self.seed)%5]
        else:i,mi=divmod(index,5);m=self.methods[mi]
        r=self.rr[i][m]
        with Image.open(r['cover_path']) as im:a=np.asarray(im,dtype=np.float32).copy()
        with Image.open(r['stego_path']) as im:b=np.asarray(im,dtype=np.float32).copy()
        x=np.stack([a,b])[:,None]
        if self.train:
            rng=np.random.default_rng(self.seed*1000000+self.epoch*10000+index);k=int(rng.integers(4));x=np.rot90(x,k,axes=(-2,-1))
            if rng.integers(2):x=x[...,::-1]
        return torch.from_numpy(x.copy()),torch.tensor([0,1]),m,r['cover_id']
def evaluate(model,loader):
    model.eval();scores=[];labels=[];losses=[];ids=[];methods=[]
    with torch.inference_mode():
        for x,y,ms,cs in loader:
            x=x.flatten(0,1).cuda();y=y.flatten().cuda();o=model(x);s=(o[:,1]-o[:,0]).detach().cpu().numpy();l=torch.nn.functional.cross_entropy(o,y,reduction='none').detach().cpu().numpy()
            scores.extend(s);labels.extend(y.cpu().numpy());losses.extend(l);ids.extend(np.repeat(cs.numpy(),2));methods.extend(np.repeat(ms,2))
    return dict(scores=np.asarray(scores),labels=np.asarray(labels),loss=float(np.mean(losses)),cover_ids=np.asarray(ids),methods=np.asarray(methods))
def main():
    cfg=config();hc=cfg['hsmnet'];assert torch.cuda.is_available();torch.set_num_threads(1)
    assert not (ROOT/'hsm_selection.json').exists();blocks=complete_blocks(records());d=ROOT/'models_hsm';d.mkdir();selections=[]
    for pay in cfg['payloads']:
        val=Pairs(blocks,pay,'validation');vl=DataLoader(val,batch_size=hc['paired_batch_size'],shuffle=False,num_workers=2,pin_memory=True)
        paths=[];vs=[]
        for seed in hc['seeds']:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            model=HSMNet().cuda();opt=torch.optim.Adamax(model.parameters(),lr=hc['lr'],eps=hc['eps'],weight_decay=hc['weight_decay'])
            scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=hc['milestones'],gamma=hc['gamma'])
            train=Pairs(blocks,pay,'train',seed);best=float('inf');best_path=d/f'{pay}_{seed}_best.pt';last=d/f'{pay}_{seed}_last.pt';hist=d/f'{pay}_{seed}_history.jsonl';t0=time.monotonic()
            with hist.open('x') as log:
                for epoch in range(hc['epochs']):
                    train.epoch=epoch;g=torch.Generator().manual_seed(seed+10000*epoch)
                    loader=DataLoader(train,batch_size=hc['paired_batch_size'],shuffle=True,generator=g,num_workers=2,pin_memory=True)
                    model.train();total=0.;n=0
                    for bi,(x,y,_,_) in enumerate(loader):
                        x=x.flatten(0,1).cuda();y=y.flatten().cuda();opt.zero_grad(set_to_none=True);o=model(x);loss=torch.nn.functional.cross_entropy(o,y);loss.backward();opt.step();total+=float(loss.detach())*len(y);n+=len(y)
                        if bi%50==0:status(phase='hsm_training',stage=f'HSMNet {pay:.2f} bpp · seed {seed} · epoch {epoch+1}/{hc["epochs"]}',hsm_payload=pay,hsm_seed=seed,epoch=epoch+1,epochs=hc['epochs'],batch=bi+1,batches=len(loader),phase_elapsed_sec=time.monotonic()-t0)
                    scheduler.step();row=dict(epoch=epoch+1,training_loss=total/n,elapsed_sec=time.monotonic()-t0)
                    if (epoch+1)%hc['validation_every']==0 or epoch==0:
                        ev=evaluate(model,vl);row['validation_loss']=ev['loss']
                        if ev['loss']<best:
                            best=ev['loss'];torch.save(dict(state_dict=model.state_dict(),epoch=epoch+1,validation_loss=best,payload_bpp=pay,seed=seed),best_path)
                    torch.save(dict(state_dict=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch+1,payload_bpp=pay,seed=seed),last)
                    append(log,row);print(json.dumps(dict(payload=pay,seed=seed,**row)),flush=True)
            ck=torch.load(best_path,map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);ev=evaluate(model,vl);np.savez_compressed(d/f'{pay}_{seed}_validation.npz',**ev);paths.append(str(best_path));vs.append(ev['scores']);del model;torch.cuda.empty_cache()
        avg=np.mean(vs,axis=0);orient=1 if roc_auc_score(ev['labels'],avg)>=.5 else -1;threshold=choose_threshold(ev['labels'],orient*avg)
        selections.append(dict(payload_bpp=pay,paths=paths,orientation=orient,threshold=threshold,validation_auc=float(roc_auc_score(ev['labels'],orient*avg)),selection='mean validation cross-entropy across methods; no method-specific checkpoint selection'))
    save(ROOT/'hsm_selection.json',dict(selected_at=now(),test_scored=False,panels=selections))
    status(phase='heldout_scoring',stage='All model choices frozen; preparing one held-out scoring pass')
if __name__=='__main__':main()
