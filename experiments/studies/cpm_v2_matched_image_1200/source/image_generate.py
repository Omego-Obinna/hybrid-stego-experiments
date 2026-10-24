import concurrent.futures,csv,io,json,os,subprocess,sys,threading,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from image_common import *
sys.path.insert(0,str(ROOT/'source'))
import epoch_protocol as ep
def execute(cmd,folder,label):
    t=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    (folder/f'{label}.stdout.txt').write_text(r.stdout);(folder/f'{label}.stderr.txt').write_text(r.stderr)
    return r,time.perf_counter()-t
def main():
    cfg=config();assert not (ROOT/'image_results.jsonl').exists(),'No case reruns'
    start=time.monotonic();rows=[];lock=threading.Lock();stop=threading.Event();bank=json.loads((ROOT/'parameter_bank.json').read_text());cases=json.loads((ROOT/'cases.json').read_text());covers={int(r['cover_id']):r for r in csv.DictReader((ROOT/'covers.csv').open())}
    def update(phase):status(state='running',phase=phase,completed=sum(r['method']=='cpm' for r in rows),total=len(cases),reused_checked=sum(r['method']!='cpm' for r in rows),reused_total=14400,successful=sum(r.get('exact_recovery') is True for r in rows),failed_images=sum(not r['valid_image'] for r in rows),stage='Generating revised CPM images and checking saved baselines',elapsed_sec=time.monotonic()-start,recent=rows[-3:])
    with (ROOT/'image_results.jsonl').open('x') as log:
        # Reuse has no new payload recovery claim. Validate hashes and recorded failures.
        for item in json.loads((ROOT/'historical_inputs.json').read_text()):
            r=item['record'];c=covers[item['cover_id']];good=bool(r['ok']) and bool(r['success_extract']) and r.get('stc_backend_actual')=='pystc'
            row=dict(method=item['method'],cover_id=item['cover_id'],payload_bpp=item['payload_bpp'],payload_bits=int(r['payload_length_L']),partition=c['partition'],cover_path=c['source_path'],cover_sha256=c['cover_sha256'],valid_image=good,exact_recovery=None,source_reported_exact_recovery=bool(r['success_extract']),provenance='historical immutable native-STC output',source_log=item['source_log'],source_run=r['run_id'],error=r.get('error',''))
            if good:
                p=Path(r['stego_path']);assert sha(p)==r['stego_sha256'];assert sha(c['source_path'])==c['cover_sha256']==r['cover_sha256']
                with Image.open(c['source_path']) as im:a=np.asarray(im,dtype=np.uint8).copy()
                with Image.open(p) as im:assert im.mode=='L' and im.size==(512,512);b=np.asarray(im,dtype=np.uint8).copy()
                row.update(stego_path=str(p),stego_sha256=sha(p),carrier_bytes=p.stat().st_size,**image_metrics(a,b))
            rows.append(row);append(log,row)
            if len(rows)%100==0:update('reuse_audit')
        save(ROOT/'reuse_completion.json',dict(checked=14400,valid=sum(r['valid_image'] for r in rows),recovery_claim='Recorded original checks only; original master-secret configuration unavailable for independent expected-byte reconstruction.'))
        def job(case):
            if stop.is_set():return
            folder=ROOT/'cases'/str(case['case_id']);folder.mkdir(parents=True,exist_ok=False);r=dict(case,method='cpm',valid_image=False,exact_recovery=False,provenance='new unchanged accepted CPM protocol and image worker')
            try:
                assert sha(case['cover_path'])==case['cover_sha256'] and sha(case['message_path'])==case['message_sha256']
                pair=bank[case['pair_index']];t=time.perf_counter()
                cmd=[sys.executable,str(ROOT/'source/image_worker.py'),'send','--cover',case['cover_path'],'--gamma1',pair['gamma1_path'],'--gamma2',pair['gamma2_path'],'--message',case['message_path'],'--master-file',case['master_file'],'--session',case['session_hex'],'--epoch',str(case['epoch']),'--payload',str(case['payload_bpp']),'--out',str(folder/'frames'),'--metrics',str(folder/'sender_metrics.json')]
                sent,wall=execute(cmd,folder,'sender');r.update(sender_returncode=sent.returncode,sender_wall_sec=wall)
                if sent.returncode:
                    answer=worker_response(sent.stdout)
                    if sent.returncode==2 and answer.get('reason')=='Sender preflight failed: STC did not recover the masked bytes':r.update(status='sender_preflight_failed',error=answer['reason'])
                    else:raise RuntimeError('Unexpected sender error: '+(sent.stderr or sent.stdout)[-1800:])
                else:
                    got,wall=execute([sys.executable,str(ROOT/'source/image_worker.py'),'receive','--frames',str(folder/'frames'),'--master-file',case['master_file'],'--session',case['session_hex'],'--state',str(folder/'receiver_state.json'),'--out',str(folder/'recovered.bin')],folder,'receiver')
                    r.update(receiver_returncode=got.returncode,receiver_wall_sec=wall)
                    if got.returncode:raise RuntimeError('Unexpected receiver error: '+(got.stderr or got.stdout)[-1800:])
                    expected=ep.truncate(Path(case['message_path']).read_bytes(),case['payload_bits']);actual=(folder/'recovered.bin').read_bytes();assert actual==expected,'Wrong accepted plaintext'
                    e=ep.read_frames(folder/'frames');assert e.payload_bits==case['payload_bits'];p=folder/'stego.pgm';p.write_bytes(e.stego)
                    with Image.open(case['cover_path']) as im:a=np.asarray(im,dtype=np.uint8).copy()
                    with Image.open(p) as im:b=np.asarray(im,dtype=np.uint8).copy()
                    metrics=json.loads((folder/'sender_metrics.json').read_text());wire=sum(x.stat().st_size for x in (folder/'frames').glob('C*.json'))
                    r.update(valid_image=True,exact_recovery=True,status='recovered',plaintext_error_bits=0,stego_path=str(p),stego_sha256=sha(p),carrier_bytes=len(e.stego),wire_bytes=wire,parameter_bytes=len(e.gamma1)+len(e.gamma2),embed_time_sec=metrics['embed_time_sec'],receiver_core_sec=worker_response(got.stdout)['receiver_time_sec'],**image_metrics(a,b))
                r['elapsed_sec']=time.perf_counter()-t
            except BaseException as exc:
                r.update(status='operational_or_correctness_failure',error=str(exc));(folder/'failure.txt').write_text(traceback.format_exc());stop.set()
            with lock:
                rows.append(r);append(log,r);update('native_images')
                if sum(x['method']=='cpm' for x in rows)%20==0 or not r['valid_image']:print(json.dumps(dict(case=case['case_id'],status=r['status'],completed=sum(x['method']=='cpm' for x in rows))),flush=True)
            if stop.is_set():raise RuntimeError(r.get('error'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg['generation_workers']) as pool:
            jobs=[pool.submit(job,c) for c in cases]
            for f in concurrent.futures.as_completed(jobs):f.result()
    assert len(rows)==18000
    blocks=complete_blocks(rows);counts={}
    for pay in cfg['payloads']:
        counts[str(pay)]={part:sum(k[1]==pay and v['cpm']['partition']==part for k,v in blocks.items()) for part in ['train','validation','test']}
        for part,n in counts[str(pay)].items():assert n>=cfg['detector_minimum_complete_counts'][part],('minimum complete count',pay,part,n)
    save(ROOT/'generation_completion.json',dict(completed_at=now(),attempts=18000,new_native=3600,common_valid_counts=counts,pixel_deviation_images=sum(r.get('pixels_over_one',0)>0 for r in rows),wrong_accepted_plaintexts=0))
    status(phase='features',stage='Native image and reuse audit complete; extracting common SRM features',completed=3600,total=3600)
if __name__=='__main__':main()
