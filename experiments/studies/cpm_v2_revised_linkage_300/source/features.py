#!/usr/bin/env python3
"""Audit source/data, select complete blocks, extract fixed content-only views."""
import argparse,csv,json,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from common import sha,save,status,read_jsonl
import historical_features as hf

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    root=args.root.resolve();cfg=json.loads((root/'run_config.json').read_text())
    for name,digest in cfg['source_sha256'].items():assert sha((root/'source'/name).read_bytes())==digest
    sys.path.insert(0,str(root/'source'));import epoch_protocol as ep
    rows=read_jsonl(root/'generation_results.jsonl');cases=json.loads((root/'case_manifest.json').read_text())
    assert len(rows)==len(cases) and len({r['epoch'] for r in rows})==len(cases)
    byepoch={c['epoch']:c for c in cases};bank=json.loads((root/'parameter_bank.json').read_text())
    for pair in bank:
        for j in (1,2):assert sha(Path(pair[f'gamma{j}_path']).read_bytes())==pair[f'gamma{j}_sha256']
    for case in cases:
        assert sha(Path(case['message_path']).read_bytes())==case['message_sha256']
    valid=[];failures=[];headers={};nonces=set();pair_map={r['pair_id']:r for r in bank}
    for row in rows:
        case=byepoch[row['epoch']];folder=root/'epochs'/f'epoch_{row["epoch"]:05d}'
        assert all(row[k]==case[k] for k in ('cover_id','partition','payload_bpp','pair_id','message_sha256','application_payload_bits'))
        if not row['exact_recovery']:
            assert row['status']=='sender_preflight_failed'
            r=json.loads((folder/'sender.stdout.txt').read_text())
            assert r['reason']=='Sender preflight failed: STC did not recover the masked bytes'
            assert not (folder/'frames').exists() and not (folder/'recovered.bin').exists()
            failures.append(row);continue
        assert (folder/'recovered.bin').read_bytes()==Path(case['message_path']).read_bytes()
        assert sha(Path(row['stego_path']).read_bytes())==row['stego_sha256']
        e=ep.read_frames(folder/'frames');assert sha(ep.transcript(e))==row['transcript_sha256']
        assert e.epoch==case['epoch'] and e.sid.hex()==cfg['session_hex'] and e.payload_bits==case['application_payload_bits']
        pair=pair_map[case['pair_id']]
        assert sha(e.gamma1)==pair['gamma1_sha256'] and sha(e.gamma2)==pair['gamma2_sha256']
        fresh={n.hex() for n in (e.nonce_a,e.nonce_b,e.nonce_c)}
        assert len(fresh)==3 and not (nonces&fresh);nonces|=fresh
        assert e.stego==Path(row['stego_path']).read_bytes()
        state=json.loads((folder/'receiver_state.json').read_text())
        assert state['accepted']==[cfg['session_hex']+':'+str(e.epoch)]
        assert set(state['nonces'])=={n.hex() for n in (e.nonce_a,e.nonce_b,e.nonce_c)}
        answer=json.loads((folder/'receiver.stdout.txt').read_text())
        assert not answer['original_cover_used'] and not answer['sender_mask_used']
        assert answer['message_sha256']==row['message_sha256']
        headers[row['epoch']]=e.hdr.hex();valid.append(row)
    grouped={}
    for r in valid:grouped.setdefault((r['cover_id'],r['payload_bpp']),[]).append(r)
    complete={k:v for k,v in grouped.items() if len(v)==12 and len({r['pair_id'] for r in v})==12}
    if not args.smoke:
        for payload in cfg['payloads']:
            for partition,minimum in cfg['primary_min_complete_cover_counts'].items():
                count=sum(v[0]['partition']==partition and k[1]==payload for k,v in complete.items())
                assert count>=minimum,f'Too few complete {partition} blocks at {payload}: {count}'
    included=sorted([r for v in complete.values() for r in v],key=lambda r:r['epoch'])
    excluded=[r for r in valid if (r['cover_id'],r['payload_bpp']) not in complete]
    counts=[]
    for pay in cfg['payloads']:
        for part in cfg['cover_counts']:
            attempted=[r for r in rows if r['partition']==part and r['payload_bpp']==pay]
            good=[r for r in valid if r['partition']==part and r['payload_bpp']==pay]
            used=[r for r in included if r['partition']==part and r['payload_bpp']==pay]
            counts.append(dict(payload_bpp=pay,partition=part,attempted=len(attempted),valid=len(good),
                failed=len(attempted)-len(good),primary_images=len(used),complete_covers=len(used)//12))
    audit=dict(all_generation_artifact_checks_passed=True,counts=counts,attempted=len(rows),
        recovered=len(valid),preflight_failures=len(failures),wrong_accepted_plaintexts=0,
        complete_blocks=len(complete),primary_images=len(included),valid_images_excluded_in_incomplete_blocks=len(excluded),
        excluded_valid_epochs=[r['epoch'] for r in excluded],failure_epochs=[r['epoch'] for r in failures],
        intentional_header_control='Within each complete block, the true pair frames match the C3 epoch/header; other candidate frames do not',
        observed_unique_nonces=len(nonces),
        header_control_correct=all(len({headers[r['epoch']] for r in v})==12 for v in complete.values()))
    assert audit['header_control_correct']
    save(root/'generation_audit.json',audit)
    status(root,phase='features',stage='Extracting fixed content-only features',features_total=len(included),features_completed=0,
        complete_blocks=len(complete),generation_audit=audit)
    gamma=[hf.gamma_features(r['pair_id'],r['gamma1_path'],r['gamma2_path'],32) for r in bank]
    gamma_numeric=[{k:float(v) for k,v in g.items() if k.startswith('gamma_')} for g in gamma]
    gamma_names=list(gamma_numeric[0]);assert all(list(g)==gamma_names for g in gamma_numeric)
    pairindex={r['pair_id']:i for i,r in enumerate(bank)}
    gamma_array=np.array([[g[k] for k in gamma_names] for g in gamma_numeric],dtype=np.float64)
    features=[]
    def extract(r):
        values=hf.image_features(r['stego_path'],32)
        return r,{k:float(v) for k,v in values.items() if k.startswith('img_')}
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i,(row,img) in enumerate(pool.map(extract,included),1):
            features.append((row,img))
            if i%100==0 or i==len(included):status(root,features_completed=i,feature_elapsed_sec=time.perf_counter()-start)
    img_names=list(features[0][1]);cross_names=list(hf.cross_features(features[0][1],gamma_numeric[0],32))
    names=img_names+gamma_names+cross_names
    assert all(k.startswith(('img_','gamma_','cross_')) for k in names)
    assert not any(any(bad in k for bad in ('path','sha256','true_pair','header','nonce','epoch','label','session')) for k in names)
    n=len(features)*12;X=np.empty((n,len(names)),dtype=np.float32);y=np.zeros(n,dtype=np.uint8)
    metadata=[];offset=0
    for row,img in features:
        image_values=[img[k] for k in img_names]
        for qi,pair in enumerate(bank):
            cross=hf.cross_features(img,gamma_numeric[qi],32)
            X[offset]=image_values+gamma_array[qi].tolist()+[cross[k] for k in cross_names]
            y[offset]=int(pair['pair_id']==row['pair_id'])
            metadata.append(dict(epoch=row['epoch'],cover_id=row['cover_id'],partition=row['partition'],
                payload_bpp=row['payload_bpp'],true_pair_index=pairindex[row['pair_id']],candidate_pair_index=qi,label=int(y[offset])))
            offset+=1
    assert np.isfinite(X).all() and offset==n
    assert all(y[i:i+12].sum()==1 for i in range(0,n,12))
    np.savez_compressed(root/'feature_dataset.npz',X=X,y=y)
    pd.DataFrame(metadata).to_csv(root/'feature_metadata.csv.gz',index=False,compression='gzip')
    save(root/'feature_schema.json',dict(names=names,image_names=img_names,gamma_names=gamma_names,cross_names=cross_names,
        pair_ids=[r['pair_id'] for r in bank],rows=n,images=len(features),hash_dimension=32,
        feature_sha256=sha((root/'feature_dataset.npz').read_bytes()),metadata_sha256=sha((root/'feature_metadata.csv.gz').read_bytes())))
    status(root,phase='training',stage='Feature extraction complete; training fixed classifier families')
    print(json.dumps(dict(images=len(features),candidate_rows=n,features=len(names),failures=len(failures))))

if __name__=='__main__':main()
