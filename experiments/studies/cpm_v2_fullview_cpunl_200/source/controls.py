"""Separate valid-tag known-leak fixtures, using the same public feature maps."""
import argparse,io,json,secrets,tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import epoch_protocol as ep
from common import check_sources,save,now,sha
from public_features import gamma_public,image_public,content_vector,challenge_vector,full_vector,columns

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();cfg=check_sources(root)
    from evaluate import model
    bank=json.loads((root/'parameter_bank.json').read_text());g=[gamma_public(x) for x in bank]
    X=[];y=[];split=[];tags_checked=0;cols=None
    rng=np.random.default_rng(75133)
    fixture=root/'synthetic_control';fixture.mkdir(exist_ok=False)
    # This independent fixture is not a CPM recovery/fidelity trial.
    for group in range(12):
        master=secrets.token_bytes(32);sid=secrets.token_bytes(16);epochs=[];ips=[]
        keyfile=fixture/f'private_control_key_{group}.bin';keyfile.write_bytes(master);keyfile.chmod(0o600)
        for b in range(12):
            pixels=rng.integers(0,256,(512,512),dtype=np.uint8);pixels.ravel()[:8]=np.frombuffer(g[b]['digest'],dtype=np.uint8)
            path=fixture/f'group{group}_carrier{b}.pgm';Image.fromarray(pixels).save(path,format='PPM')
            gamma1=Path(bank[b]['gamma1_path']).read_bytes();gamma2=Path(bank[b]['gamma2_path']).read_bytes()
            e=ep.Epoch(sid,4,ep.header(master,sid,4),ep.mac(master,b'a',ep.uint(4))[:16],gamma1,
                ep.mac(master,b'b',ep.uint(4))[:16],gamma2,ep.mac(master,b'c',ep.uint(4))[:16],52429,path.read_bytes())
            epochs.append(ep.authenticate(master,e));ips.append(image_public(path))
        history=[]
        for h in range(4):
            b=h%12;e=ep.authenticate(master,replace(epochs[b],epoch=h,hdr=ep.header(master,sid,h),
                nonce_a=ep.mac(master,b'a',ep.uint(h))[:16],nonce_b=ep.mac(master,b'b',ep.uint(h))[:16],nonce_c=ep.mac(master,b'c',ep.uint(h))[:16]))
            history.append(dict(epoch=e,query=bytes(6554),content=content_vector(g[b],ips[b]),vector=challenge_vector(e,g[b],ips[b])))
        state=ep.ReceiverState()
        for h in history:
            e=h['epoch'];state.accepted.add(e.sid.hex()+':'+str(e.epoch));state.nonces.update([e.nonce_a.hex(),e.nonce_b.hex(),e.nonce_c.hex()])
        for label,a,b in [(1,i,i) for i in range(12)]+[(0,i,j) for i in range(12) for j in range(12)]:
            e=ep.authenticate(master,replace(epochs[b],gamma1=Path(bank[a]['gamma1_path']).read_bytes(),gamma2=Path(bank[a]['gamma2_path']).read_bytes()))
            ep.verify(master,e,sid,state);tags_checked+=1
            if cols is None:cols=columns(e,g[a],ips[b],history)
            X.append(full_vector(e,g[a],ips[b],history));y.append(label);split.append('train' if group<8 else 'test')
    X=np.asarray(X,dtype=np.float32);y=np.asarray(y);split=np.asarray(split);train=split=='train';test=split=='test';result=[]
    for family in cfg['classifiers']:
        scores=[]
        for seed in cfg['model_seeds']:
            clf=model(family,seed,cfg);clf.fit(X[train],y[train]);scores.append(clf.predict_proba(X[test])[:,1])
        mean=np.mean(scores,axis=0);auc=float(roc_auc_score(y[test],mean));assert .90<auc<=23/24+1e-12,(family,auc)
        result.append(dict(classifier=family,auc=auc))
    # Any function of the exactly matched metadata has chance AUC.
    arbitrary=X[test][:,cols['metadata_only']]@np.arange(1,len(cols['metadata_only'])+1)
    assert abs(roc_auc_score(y[test],arbitrary)-.5)<1e-12
    np.savez_compressed(fixture/'feature_fixture.npz',X=X,y=y,split=split)
    save(root/'controls.json',dict(checked_at=now(),same_public_feature_extractor=True,all_synthetic_tags_valid=True,tags_checked=tags_checked,
        implanted_signal='SHA256(canonical public parameter pair) prefix in first eight pixels',models=result,known_leak_detected=True,
        metadata_balance_passed=True,independent_diagonal_retained=True,maximum_auc_due_to_diagonal_overlap=23/24,
        limitation='Separate synthetic coupling only; not a proof of sensitivity to every real leakage or a native CPM recovery sample'))
    print(json.dumps(dict(known_leak_controls_passed=True,models=result)))

if __name__=='__main__':main()
