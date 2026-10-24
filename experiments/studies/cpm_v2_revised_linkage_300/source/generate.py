#!/usr/bin/env python3
"""Real frozen sender and separate receiver, three concurrent cover groups."""
import argparse,csv,io,json,math,os,subprocess,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from common import sha,save,now,status

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();cfg=json.loads((root/'run_config.json').read_text());source=root/'source'
    sys.path.insert(0,str(source));import epoch_protocol as ep
    assert not (root/'generation_results.jsonl').exists(),'Never overwrite or automatically resume attempts'
    cases=json.loads((root/'case_manifest.json').read_text());bank={r['pair_id']:r for r in json.loads((root/'parameter_bank.json').read_text())}
    grouped={cover:[c for c in cases if c['cover_id']==cover] for cover in cfg['cover_execution_order']}
    rows=[];active={};lock=threading.Lock();stop=threading.Event();started=time.perf_counter()
    log=(root/'generation_results.jsonl').open('x')
    def update(error=None):
        elapsed=time.perf_counter()-started
        status(root,state='running',phase='generation',stage='Generating and independently recovering image epochs',
            completed=len(rows),successful=sum(r['exact_recovery'] for r in rows),failed=sum(not r['exact_recovery'] for r in rows),
            elapsed_sec=elapsed,eta_sec=elapsed/len(rows)*(len(cases)-len(rows)) if rows else None,
            active=list(active.values()),recent=rows[-6:],last_error=error)
    def publish(row):
        with lock:
            rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
            update()
            if len(rows)%12==0 or not row['exact_recovery']:
                print(json.dumps(dict(completed=len(rows),total=len(cases),epoch=row['epoch'],status=row['status'])),flush=True)
    def execute(cmd,folder,label):
        t=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        (folder/f'{label}.stdout.txt').write_text(r.stdout);(folder/f'{label}.stderr.txt').write_text(r.stderr)
        return r,time.perf_counter()-t
    def cover_job(cover):
        items=grouped[cover];path=Path(items[0]['cover_path']);assert sha(path.read_bytes())==items[0]['cover_sha256']
        with Image.open(path) as im:
            assert im.mode=='L' and im.size==(512,512);original=np.asarray(im,dtype=np.int16).copy()
        for case in items:
            if stop.is_set():return
            t=time.perf_counter();folder=root/'epochs'/f'epoch_{case["epoch"]:05d}';folder.mkdir(parents=True,exist_ok=False)
            row={k:case[k] for k in ('epoch','cover_id','partition','cover_sha256','payload_bpp','pair_id','message_sha256','application_payload_bits')}
            row.update(status='not_completed',exact_recovery=False)
            with lock:
                active[cover]=dict(cover_id=cover,payload_bpp=case['payload_bpp'],pair_id=case['pair_id']);update()
            fatal=False
            try:
                assert sha(Path(case['message_path']).read_bytes())==case['message_sha256']
                pair=bank[case['pair_id']]
                command=[sys.executable,str(source/'image_worker.py'),'send','--cover',str(path),
                    '--gamma1',pair['gamma1_path'],'--gamma2',pair['gamma2_path'],'--message',case['message_path'],
                    '--master-file',cfg['master_file'],'--session',cfg['session_hex'],'--epoch',str(case['epoch']),
                    '--payload',str(case['payload_bpp']),'--out',str(folder/'frames'),'--metrics',str(folder/'sender_metrics.json')]
                result,wall=execute(command,folder,'sender');row.update(sender_returncode=result.returncode,sender_wall_sec=wall)
                if result.returncode:
                    answer=json.loads(result.stdout) if result.stdout.strip().startswith('{') else {}
                    if result.returncode==2 and answer.get('reason')=='Sender preflight failed: STC did not recover the masked bytes':
                        row.update(status='sender_preflight_failed',error=answer['reason'])
                    else:raise RuntimeError('Unexpected sender failure: '+(result.stderr or result.stdout)[-1500:])
                else:
                    receive=[sys.executable,str(source/'image_worker.py'),'receive','--frames',str(folder/'frames'),
                        '--master-file',cfg['master_file'],'--session',cfg['session_hex'],'--state',str(folder/'receiver_state.json'),
                        '--out',str(folder/'recovered.bin')]
                    result,wall=execute(receive,folder,'receiver');row.update(receiver_returncode=result.returncode,receiver_wall_sec=wall)
                    if result.returncode:raise RuntimeError('Independent receiver failed: '+(result.stderr or result.stdout)[-1500:])
                    actual=(folder/'recovered.bin').read_bytes();message=Path(case['message_path']).read_bytes()
                    if actual!=message:raise RuntimeError('Wrong accepted plaintext; stop all further work')
                    e=ep.read_frames(folder/'frames');assert e.epoch==case['epoch'] and e.payload_bits==case['application_payload_bits']
                    assert sha(e.gamma1)==pair['gamma1_sha256'] and sha(e.gamma2)==pair['gamma2_sha256']
                    metrics=json.loads((folder/'sender_metrics.json').read_text());assert metrics['profile']['fused_weight']==.75
                    imagepath=folder/'stego.pgm';imagepath.write_bytes(e.stego)
                    with Image.open(io.BytesIO(e.stego)) as im:stego=np.asarray(im,dtype=np.int16).copy()
                    delta=stego-original;assert np.max(np.abs(delta))<=1
                    mse=float(np.mean(delta.astype(np.float64)**2));assert mse>0
                    row.update(status='recovered',exact_recovery=True,plaintext_error_bits=0,recovered_sha256=sha(actual),
                        stego_path=str(imagepath),stego_sha256=sha(e.stego),transcript_sha256=sha(ep.transcript(e)),
                        epoch_context_sha256=sha(ep.context(e)),masked_sha256=metrics['masked_sha256'],
                        psnr_db=10*math.log10(255**2/mse),ssim=float(structural_similarity(original,stego,data_range=255)),
                        modified_pixels=int(np.count_nonzero(delta)),mse=mse,
                        embed_time_sec=metrics['embed_time_sec'],wire_bytes=sum(p.stat().st_size for p in (folder/'frames').glob('C*.json')),
                        canonical_transcript_bytes=len(ep.transcript(e)))
            except BaseException as exc:
                fatal=True;stop.set();row.update(status='operational_or_correctness_failure',error=str(exc))
                (folder/'failure.txt').write_text(traceback.format_exc())
            row.update(elapsed_sec=time.perf_counter()-t,completed_at=now());publish(row)
            if fatal:raise RuntimeError(row['error'])
        with lock:active.pop(cover,None);update()
    try:
        update()
        with ThreadPoolExecutor(max_workers=cfg['generation_workers']) as pool:
            futures=[pool.submit(cover_job,c) for c in cfg['cover_execution_order']]
            for f in as_completed(futures):f.result()
        assert len(rows)==len(cases)
        save(root/'generation_completion.json',dict(completed_at=now(),attempted=len(rows),
            exact_recoveries=sum(r['exact_recovery'] for r in rows),sender_preflight_failures=sum(not r['exact_recovery'] for r in rows),
            wrong_accepted_plaintexts=0,elapsed_sec=time.perf_counter()-started))
        status(root,phase='features',stage='Generation complete; verifying cohorts and extracting content features',active=[],eta_sec=None)
    except BaseException as exc:
        stop.set();status(root,state='failed',phase='generation',stage='Generation stopped; evidence preserved',last_error=str(exc))
        (root/'generation_failure.txt').write_text(traceback.format_exc());raise
    finally:log.close()

if __name__=='__main__':main()
