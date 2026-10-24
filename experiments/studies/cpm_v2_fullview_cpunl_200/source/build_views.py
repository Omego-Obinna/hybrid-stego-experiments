import argparse,csv,hashlib,json,time
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import epoch_protocol as ep
from common import check_sources,read_jsonl,sha,save,status,now,frame_bytes,query_message,next_chain,pixel_summary,worker_response
from public_features import gamma_public,image_public,content_vector,challenge_vector,full_vector,columns

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();cfg=check_sources(root)
    rows=read_jsonl(root/'generation_results.jsonl');cases=json.loads((root/'case_manifest.json').read_text());blocks=json.loads((root/'blocks.json').read_text());bank=json.loads((root/'parameter_bank.json').read_text())
    assert len(rows)==len(cases)==cfg['total_attempts'] and len({x['case_id'] for x in rows})==len(cases)
    byid={x['case_id']:x for x in rows};epochs={};images={};originals={};failures=[];deviations=[];complete=[];histories={};keys=set()
    g=[gamma_public(b) for b in bank]
    for b in bank:
        for j in [1,2]:assert len(Path(b[f'gamma{j}_path']).read_bytes())==65536 and sha(Path(b[f'gamma{j}_path']).read_bytes())==b[f'gamma{j}_sha256']
    for bi,block in enumerate(blocks):
        master=Path(block['master_file']).read_bytes();assert len(master)==32 and sha(master) not in keys;keys.add(sha(master))
        cid=block['cover_id']
        if cid not in originals:
            assert sha(Path(block['cover_path']).read_bytes())==block['cover_sha256']
            with Image.open(block['cover_path']) as im: originals[cid]=np.asarray(im,dtype=np.int16).copy()
        history=[];chain=bytes(32);state=ep.ReceiverState();context=None
        for caseid in block['case_ids']:
            case=cases[caseid];r=byid[caseid];folder=root/'cases'/case['case_uid'];message=(folder/'message.bin').read_bytes()
            assert all(case[k]==r[k] for k in case)
            expected=query_message(case['role_index'],block['payload_bits'],chain) if case['role']=='history' else bytes((block['payload_bits']+7)//8)
            assert message==expected and sha(message)==r['message_sha256']
            wire=[]
            if not r['exact_recovery']:
                assert r['status']=='sender_preflight_failed' and r['sender_returncode']==2
                assert worker_response((folder/'sender.stdout.txt').read_text()).get('reason')=='Sender preflight failed: STC did not recover the masked bytes'
                assert not (folder/'frames').exists() and not (folder/'recovered.bin').exists();failures.append(r)
            else:
                assert (folder/'recovered.bin').read_bytes()==message
                answer=json.loads((folder/'receiver.stdout.txt').read_text());assert not answer['original_cover_used'] and not answer['sender_mask_used']
                e=ep.read_frames(folder/'frames');ep.verify(master,e,bytes.fromhex(block['session_hex']),state)
                assert e.epoch==case['epoch'] and e.payload_bits==block['payload_bits']
                pair=bank[case['pair_index']]
                assert sha(e.gamma1)==pair['gamma1_sha256'] and sha(e.gamma2)==pair['gamma2_sha256']
                assert sha(ep.transcript(e))==r['transcript_sha256'] and sha(ep.context(e))==r['context_sha256']
                assert e.stego==Path(r['stego_path']).read_bytes() and sha(e.stego)==r['stego_sha256']
                wire=frame_bytes(e);assert [sha(x) for x in wire]==r['wire_sha256']
                assert all(x==(folder/'frames'/f'C{i+1}.json').read_bytes() for i,x in enumerate(wire))
                with Image.open(r['stego_path']) as im:observed=pixel_summary(originals[cid],np.asarray(im,dtype=np.int16))
                assert all(observed[k]==r[k] for k in observed)
                if observed['pixels_exceeding_one_level']:deviations.append(dict(case_uid=case['case_uid'],cover_id=cid,partition=block['partition'],payload_bpp=block['payload_bpp'],**observed))
                ip=image_public(r['stego_path']);images[caseid]=ip;epochs[caseid]=e
                if case['role']=='history':
                    history.append(dict(epoch=e,query=message,content=content_vector(g[case['pair_index']],ip),vector=challenge_vector(e,g[case['pair_index']],ip)))
                    state.accepted.add(e.sid.hex()+':'+str(e.epoch));state.nonces.update([n.hex() for n in [e.nonce_a,e.nonce_b,e.nonce_c]])
                else:
                    if context is None:context=ep.context(e)
                    assert ep.context(e)==context,'Challenge variants have unmatched public context'
                saved=ep.ReceiverState.read(folder/'receiver_state.json')
                expectedaccepted=set(state.accepted)
                if case['role']=='challenge':expectedaccepted.add(e.sid.hex()+':4')
                assert saved.accepted==expectedaccepted
                expectednonces=set(state.nonces)
                if case['role']=='challenge':expectednonces.update([e.nonce_a.hex(),e.nonce_b.hex(),e.nonce_c.hex()])
                assert saved.nonces==expectednonces
            if case['role']=='history':chain=next_chain(chain,message,wire)
        if all(byid[x]['exact_recovery'] for x in block['case_ids']):
            assert len(history)==4;complete.append(block);histories[block['block_id']]=history
        if (bi+1)%10==0:status(root,phase='views',stage='Auditing native histories and public features',blocks_audited=bi+1,blocks_total=len(blocks))
    if not cfg['development_only']:
        for pay in cfg['payloads']:
            for part,n in cfg['primary_min_complete_blocks'].items():
                count=sum(b['payload_bpp']==pay and b['partition']==part for b in complete);assert count>=n,(pay,part,count)
    assert complete
    first=complete[0];cid=first['case_ids'][4];schema_cols=columns(epochs[cid],g[0],images[cid],histories[first['block_id']]);width=len(schema_cols['full_history'])
    n=len(complete)*156;X=np.lib.format.open_memmap(root/'observer_features.npy',mode='w+',dtype=np.float32,shape=(n,width));y=np.zeros(n,dtype=np.uint8)
    metadata=[];offset=0;started=time.perf_counter();verified_views=0;diagonal=0
    with (root/'public_views.jsonl').open('x') as out:
        for bi,block in enumerate(complete):
            master=Path(block['master_file']).read_bytes();history=histories[block['block_id']];state=ep.ReceiverState()
            for h in history:
                e=h['epoch'];state.accepted.add(e.sid.hex()+':'+str(e.epoch));state.nonces.update([n.hex() for n in [e.nonce_a,e.nonce_b,e.nonce_c]])
            history_hasher=hashlib.sha256()
            for h in history:history_hasher.update(ep.enc([(b'query',h['query']),*[(f'C{i+1}'.encode(),w) for i,w in enumerate(frame_bytes(h['epoch']))]]))
            native=block['case_ids'][4:];tagmap={};metabase=None
            combos=[(1,a,a) for a in range(12)]+[(0,a,b) for a in range(12) for b in range(12)]
            for label,pa,cb in combos:
                carrier=epochs[native[cb]];pair=bank[pa]
                e=ep.authenticate(master,replace(carrier,gamma1=Path(pair['gamma1_path']).read_bytes(),gamma2=Path(pair['gamma2_path']).read_bytes(),tag=b''))
                ep.verify(master,e,bytes.fromhex(block['session_hex']),state);verified_views+=1
                if pa==cb:assert e.tag==epochs[native[pa]].tag
                wire=frame_bytes(e);digests=[sha(w) for w in wire]
                if label:tagmap[pa]=digests
                elif pa==cb:assert digests==tagmap[pa];diagonal+=1
                vector=full_vector(e,g[pa],images[native[cb]],history);assert len(vector)==width and np.isfinite(vector).all()
                mz=vector[schema_cols['metadata_only']]
                if metabase is None:metabase=mz
                assert np.array_equal(mz,metabase),'World-dependent public metadata/lengths'
                X[offset]=vector;y[offset]=label
                record=dict(row=offset,block_id=block['block_id'],cover_id=block['cover_id'],partition=block['partition'],payload_bpp=block['payload_bpp'],
                    label=label,parameter_index=pa,carrier_index=cb,carrier_case_uid=cases[native[cb]]['case_uid'],
                    tag=e.tag.hex(),canonical_transcript_sha256=sha(ep.transcript(e)),wire_sha256=digests)
                hv=history_hasher.copy();hv.update(ep.enc([(f'C{i+1}'.encode(),w) for i,w in enumerate(wire)]));record['observer_sha256']=hv.hexdigest()
                out.write(json.dumps(record)+'\n');metadata.append({k:record[k] for k in ['row','block_id','cover_id','partition','payload_bpp','label','parameter_index','carrier_index']});offset+=1
            assert len(combos)==156
            if (bi+1)%5==0 or bi+1==len(complete):status(root,phase='views',stage='Building matched, authenticated full-public-view worlds',views_completed=offset,views_total=n,
                view_elapsed_sec=time.perf_counter()-started,complete_blocks=len(complete),eta_sec=None)
    X.flush();np.save(root/'observer_labels.npy',y);pd.DataFrame(metadata).to_csv(root/'observer_metadata.csv.gz',index=False,compression='gzip')
    assert offset==n and y.sum()==len(complete)*12 and diagonal==len(complete)*12
    counts=[]
    for pay in cfg['payloads']:
        for part in cfg['cover_counts']:
            planned=[b for b in blocks if b['payload_bpp']==pay and b['partition']==part];used=[b for b in complete if b['payload_bpp']==pay and b['partition']==part]
            counts.append(dict(payload_bpp=pay,partition=part,attempted_blocks=len(planned),complete_blocks=len(used),excluded_blocks=len(planned)-len(used),linked_views=len(used)*12,independent_views=len(used)*144))
    audit=dict(checked_at=now(),all_native_artifacts_checked=True,all_world_tags_verified=True,matched_metadata_exact=True,conditional_marginals_exact=True,
        independent_diagonal_fraction=1/12,independent_diagonal_views=diagonal,views=verified_views,complete_blocks=len(complete),counts=counts,
        attempted_native_cases=len(rows),exact_recoveries=sum(r['exact_recovery'] for r in rows),preflight_failures=len(failures),wrong_accepted_plaintexts=0,
        excluded_blocks=[b['block_id'] for b in blocks if b not in complete],failure_cases=[r['case_uid'] for r in failures],pixel_deviations=deviations,
        valid_native_cases_excluded=sum(r['exact_recovery'] and r['block_id'] in {b['block_id'] for b in blocks if b not in complete} for r in rows),
        fresh_research_keys=len(keys),known_challenge_message='all zero',adaptive_history_queries=4)
    save(root/'world_audit.json',audit);save(root/'pixel_deviations.json',dict(all_recovered_images_checked=audit['exact_recoveries'],deviations=deviations))
    pd.DataFrame(deviations,columns=['case_uid','cover_id','partition','payload_bpp','max_abs_pixel_change',
        'pixels_exceeding_one_level','pixel_change_histogram']).to_csv(root/'pixel_deviations.csv',index=False)
    save(root/'feature_schema.json',dict(columns=schema_cols,feature_count=width,rows=n,feature_sha256=sha((root/'observer_features.npy').read_bytes()),
        label_sha256=sha((root/'observer_labels.npy').read_bytes()),metadata_sha256=sha((root/'observer_metadata.csv.gz').read_bytes()),views_sha256=sha((root/'public_views.jsonl').read_bytes()),
        raw_view_reconstruction='export_view.py; all content bytes/native history frames plus per-view native tag retained',feature_scope='fixed public byte maps; no secret, source-cover, verification result or hidden pairing index as input'))
    status(root,phase='controls',stage='Complete full-public views audited; checking known-leak sensitivity',eta_sec=None)
    print(json.dumps(dict(complete_blocks=len(complete),views=n,features=width,failures=len(failures),pixel_deviations=len(deviations))))

if __name__=='__main__':main()
