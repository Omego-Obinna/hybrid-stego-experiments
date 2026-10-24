import json,math,os,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
from native_common import *
def worker(task):
    cfg=config();case=json.loads(Path(task).read_text());folder=Path(case['folder']);work=folder/'native_work';work.mkdir();shutil.copytree(ROOT/'source/assemblage/pulsar',work/'pulsar');(work/'offline-states').mkdir();(work/'images').mkdir();os.chdir(work)
    os.environ.update(CPM_NATIVE_TEMP=str(work/'estimate.png'),CPM_NATIVE_MODEL=cfg['model']['path'],CPM_NATIVE_KEY_HEX=Path(case['key_file']).read_bytes().hex())
    sys.path.insert(0,str(ROOT/'source/assemblage'));sys.path.insert(0,str(work/'pulsar'))
    import torch
    torch.set_num_threads(1)
    from raptorq import Encoder
    from offline import offline
    from send import send
    from receive import receive
    payload=Path(case['message_path']).read_bytes();mtu=cfg['assemblage']['mtu'];t=time.perf_counter();encoder=Encoder.with_defaults(payload,mtu);packets=encoder.get_encoded_packets(math.ceil(math.ceil(len(payload)/mtu)*.5));encode_sec=time.perf_counter()-t
    for i,b in enumerate(packets):(folder/f'raptor_packet_{i}.bin').write_bytes(b)
    t=time.perf_counter();off=offline('church',len(packets),case['assemblage_seed_start']);offline_sec=time.perf_counter()-t;hashes=json.loads((work/'offline-states/church_hashes.json').read_text());assert len(hashes)==len(packets)
    rows=[];start=time.perf_counter()
    for i,(packet,h) in enumerate(zip(packets,hashes)):
        row=dict(packet_index=i,seed=h['seed'],capacity=h['m_len'],packet_bytes=len(packet),input_packet_path=str(folder/f'raptor_packet_{i}.bin'),input_packet_sha256=sha(folder/f'raptor_packet_{i}.bin'),sent=False,decode_exception=False,decoded_path=None)
        # This is the upstream capacity gate. A send ValueError is retained,
        # including the native MTU versus serialised-packet-length boundary.
        if h['m_len']>=mtu:
            state=work/'offline-states'/f'church_{h["seed"]}.pt';image=work/'images'/f'church_{h["seed"]}.png'
            try:
                ts=time.perf_counter();ret=send('church',str(state),packet,str(image));row.update(sent=True,image_path=str(image),image_sha256=sha(image),carrier_bytes=image.stat().st_size,send_wall_sec=time.perf_counter()-ts)
            except (ValueError,AssertionError) as e:row.update(send_error=str(e));(folder/f'packet_{i}_send_failure.txt').write_text(traceback.format_exc())
        rows.append(row)
    send_sec=time.perf_counter()-start
    for row in rows:
        if not row['sent']:continue
        try:
            ts=time.perf_counter();msg,metrics=receive(row['image_path'],str(work/'offline-states'));p=folder/f'decoded_packet_{row["packet_index"]}.bin';p.write_bytes(msg[:row['packet_bytes']]);row.update(decoded_path=str(p),decoded_sha256=sha(p),decode_wall_sec=time.perf_counter()-ts,native_receiver_metrics=metrics)
        except (ValueError,AssertionError) as e:row.update(decode_exception=True,decode_error=str(e));(folder/f'packet_{row["packet_index"]}_decode_failure.txt').write_text(traceback.format_exc())
    result=dict(case_id=case['case_id'],method='assemblage',message_bytes=len(payload),mtu=mtu,encoded_packets=len(packets),transmitted_packets=sum(r['sent'] for r in rows),wire_bytes=sum(r.get('carrier_bytes',0) for r in rows),raptor_encode_sec=encode_sec,offline_sec=offline_sec,send_sec=send_sec,setup_state_bytes=sum(p.stat().st_size for p in (work/'offline-states').glob('*.pt')),setup_hash_bytes=(work/'offline-states/church_hashes.json').stat().st_size,packets=rows)
    save(folder/'native_metrics.json',result);print(json.dumps({k:v for k,v in result.items() if k!='packets'}),flush=True)
def loss_result(case,method,row,rate,seed):
    from raptorq import Decoder
    rng=np.random.default_rng(seed+case['case_id']*10000+int(rate*1000));expected=Path(case['message_path']).read_bytes()
    if method=='cpm':
        n=3 if row['exact_recovery'] else 0;keep=rng.random(n)>=rate;actual=expected if n==3 and keep.all() else None;received=int(keep.sum());err=False;indices=np.flatnonzero(keep).tolist()
    else:
        sent=[p for p in row['packets'] if p['sent']];keep=rng.random(len(sent))>=rate;selected=[p for p,k in zip(sent,keep) if k];err=any(p['decode_exception'] for p in selected);actual=None
        if not err:
            decoder=Decoder.with_defaults(len(expected),row['mtu'])
            for p in selected:
                actual=decoder.decode(Path(p['decoded_path']).read_bytes())
                if actual is not None:break
        n=len(sent);received=len(selected);indices=[p['packet_index'] for p in selected]
    return dict(case_id=case['case_id'],method=method,message_bytes=len(expected),loss_probability=rate,loss_seed=seed,transmitted_objects=n,received_objects=received,dropped_objects=n-received,selected_indices=indices,native_decode_exception=err,delivered=actual is not None,exact_recovery=actual==expected,incorrect_recovered_message=actual is not None and actual!=expected,result_sha256=__import__('hashlib').sha256(actual).hexdigest() if actual is not None else None)
def main():
    cfg=config();assert not (ROOT/'system_results.jsonl').exists();cases=json.loads((ROOT/'cases.json').read_text());rows=[];start=time.monotonic()
    from native_generate import cpm
    with (ROOT/'system_results.jsonl').open('x') as log,(ROOT/'erasure_traces.jsonl').open('x') as traces:
        for case in cases:
            d=ROOT/'cases'/str(case['case_id']);d.mkdir(parents=True,exist_ok=False);p=d/'cpm';p.mkdir();cr=cpm(case,p);rows.append(cr);append(log,cr)
            p=d/'assemblage';p.mkdir();task=dict(case,folder=str(p));tp=p/'task.json';save(tp,task);ret,wall=execute([sys.executable,str(ROOT/'source/native_distributed.py'),'--worker',str(tp)],p,'native',timeout=3600)
            if ret.returncode:raise RuntimeError('Assemblage native worker stopped: '+(ret.stderr or ret.stdout)[-2000:])
            ar=json.loads((p/'native_metrics.json').read_text());ar.update(folder=str(p),native_wall_sec=wall,**case);rows.append(ar);append(log,ar)
            for rate in cfg['assemblage']['loss_fractions']:
                for seed in cfg['assemblage']['loss_seeds']:
                    for method,r in [('cpm',cr),('assemblage',ar)]:append(traces,loss_result(case,method,r,rate,seed))
            status(state='running',phase='generation',stage='Native Assemblage local carrier encoding, PDQ discovery and finite erasure traces',completed=case['case_id']+1,total=len(cases),elapsed_sec=time.monotonic()-start,detail='Setup costs, sent carriers, capacity skips and decoder failures retained; no online platform posting')
            print('Assemblage group',case['case_id']+1,'/',len(cases),'carriers',ar['transmitted_packets'],flush=True)
    save(ROOT/'generation_completion.json',dict(completed_at=now(),message_groups=len(cases),system_rows=len(rows),erasure_traces=len(cases)*40))
if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2])
    else:main()
