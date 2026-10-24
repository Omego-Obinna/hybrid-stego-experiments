import argparse,io,json,math,os,shutil,subprocess,sys,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
import epoch_protocol as ep
from common import check_sources,sha,save,status,now,worker_response,frame_bytes,query_message,next_chain,pixel_summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();cfg=check_sources(root)
    cases=json.loads((root/'case_manifest.json').read_text());blocks={x['block_id']:x for x in json.loads((root/'blocks.json').read_text())};bank=json.loads((root/'parameter_bank.json').read_text())
    assert not (root/'generation_results.jsonl').exists(),'Do not repeat a used trial'
    rows=[];lock=threading.Lock();stop=threading.Event();start=time.perf_counter();active={}
    log=(root/'generation_results.jsonl').open('x')
    def update():
        elapsed=time.perf_counter()-start
        status(root,state='running',phase='generation',completed=len(rows),successful=sum(x['exact_recovery'] for x in rows),failed=sum(not x['exact_recovery'] for x in rows),
            pixel_deviation_epochs=sum(x.get('pixels_exceeding_one_level',0)>0 for x in rows),elapsed_sec=elapsed,
            eta_sec=elapsed/len(rows)*(len(cases)-len(rows)) if rows else None,active=list(active.values()),recent=rows[-5:])
    def execute(command,folder,label):
        t=time.perf_counter();result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        (folder/f'{label}.stdout.txt').write_text(result.stdout);(folder/f'{label}.stderr.txt').write_text(result.stderr)
        return result,time.perf_counter()-t
    def block_job(ident):
        block=blocks[ident];assert sha(Path(block['cover_path']).read_bytes())==block['cover_sha256']
        with Image.open(block['cover_path']) as im:original=np.asarray(im,dtype=np.int16).copy()
        base=root/'blocks'/ident;base.mkdir(parents=True,exist_ok=False);statefile=base/'history_state.json';ep.ReceiverState().save(statefile)
        chain=bytes(32)
        for cid in block['case_ids']:
            if stop.is_set():return
            case=cases[cid];folder=root/'cases'/case['case_uid'];folder.mkdir(parents=True,exist_ok=False);row=dict(case,exact_recovery=False,status='not_completed')
            t=time.perf_counter();fatal=False
            with lock:active[ident]=dict(cover_id=block['cover_id'],payload_bpp=block['payload_bpp'],role=case['role']);update()
            try:
                message=query_message(case['role_index'],block['payload_bits'],chain) if case['role']=='history' else bytes((block['payload_bits']+7)//8)
                messagepath=folder/'message.bin';messagepath.write_bytes(message);messagepath.chmod(0o444);row['message_sha256']=sha(message)
                pair=bank[case['pair_index']]
                command=[sys.executable,str(root/'source/image_worker.py'),'send','--cover',block['cover_path'],'--gamma1',pair['gamma1_path'],'--gamma2',pair['gamma2_path'],
                    '--message',str(messagepath),'--master-file',block['master_file'],'--session',block['session_hex'],'--epoch',str(case['epoch']),
                    '--payload',str(block['payload_bpp']),'--out',str(folder/'frames'),'--metrics',str(folder/'sender_metrics.json')]
                result,wall=execute(command,folder,'sender');row.update(sender_returncode=result.returncode,sender_wall_sec=wall)
                if result.returncode:
                    answer=worker_response(result.stdout)
                    if result.returncode==2 and answer.get('reason')=='Sender preflight failed: STC did not recover the masked bytes':row.update(status='sender_preflight_failed',error=answer['reason'])
                    else:raise RuntimeError('Unexpected sender failure: '+(result.stderr or result.stdout)[-1500:])
                else:
                    receiverstate=folder/'receiver_state.json';shutil.copy2(statefile,receiverstate)
                    result,wall=execute([sys.executable,str(root/'source/image_worker.py'),'receive','--frames',str(folder/'frames'),'--master-file',block['master_file'],
                        '--session',block['session_hex'],'--state',str(receiverstate),'--out',str(folder/'recovered.bin')],folder,'receiver')
                    row.update(receiver_returncode=result.returncode,receiver_wall_sec=wall)
                    if result.returncode:raise RuntimeError('Independent receiver failed: '+(result.stderr or result.stdout)[-1500:])
                    actual=(folder/'recovered.bin').read_bytes();assert actual==message,'Wrong accepted plaintext'
                    e=ep.read_frames(folder/'frames');assert e.epoch==case['epoch'] and e.payload_bits==block['payload_bits']
                    assert sha(e.gamma1)==pair['gamma1_sha256'] and sha(e.gamma2)==pair['gamma2_sha256']
                    wire=frame_bytes(e);assert all(w==(folder/'frames'/f'C{i+1}.json').read_bytes() for i,w in enumerate(wire))
                    imagepath=folder/'stego.pgm';imagepath.write_bytes(e.stego)
                    with Image.open(io.BytesIO(e.stego)) as im:stego=np.asarray(im,dtype=np.int16).copy()
                    observed=pixel_summary(original,stego);delta=stego-original;mse=float(np.mean(delta.astype(np.float64)**2));assert mse>0
                    metrics=json.loads((folder/'sender_metrics.json').read_text());assert metrics['profile']['fused_weight']==.75
                    row.update(status='recovered',exact_recovery=True,plaintext_error_bits=0,recovered_sha256=sha(actual),stego_path=str(imagepath),stego_sha256=sha(e.stego),
                        transcript_sha256=sha(ep.transcript(e)),context_sha256=sha(ep.context(e)),wire_sha256=[sha(x) for x in wire],
                        psnr_db=10*math.log10(255**2/mse),ssim=float(structural_similarity(original,stego,data_range=255)),modified_pixels=int(np.count_nonzero(delta)),mse=mse,
                        embed_time_sec=metrics['embed_time_sec'],wire_bytes=sum(map(len,wire)),**observed)
                    if case['role']=='history':shutil.copy2(receiverstate,statefile)
                if case['role']=='history':chain=next_chain(chain,message,wire if row['exact_recovery'] else [])
            except BaseException as exc:
                fatal=True;stop.set();row.update(status='operational_or_correctness_failure',error=str(exc));(folder/'failure.txt').write_text(traceback.format_exc())
            row.update(completed_at=now(),elapsed_sec=time.perf_counter()-t)
            with lock:
                rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno());update()
                if len(rows)%16==0 or not row['exact_recovery']:print(json.dumps(dict(completed=len(rows),total=len(cases),case_uid=case['case_uid'],status=row['status'])),flush=True)
            if fatal:raise RuntimeError(row.get('error'))
        with lock:active.pop(ident,None);update()
    try:
        with ThreadPoolExecutor(max_workers=cfg['generation_workers']) as pool:
            futures=[pool.submit(block_job,x) for x in cfg['block_execution_order']]
            for f in as_completed(futures):f.result()
        assert len(rows)==len(cases) and len({x['case_id'] for x in rows})==len(cases)
        save(root/'generation_completion.json',dict(completed_at=now(),attempted=len(rows),recovered=sum(x['exact_recovery'] for x in rows),
            preflight_failures=sum(not x['exact_recovery'] for x in rows),wrong_accepted_plaintexts=0,elapsed_sec=time.perf_counter()-start))
        status(root,phase='views',stage='Auditing native histories and building authenticated full-view worlds',active=[],eta_sec=None)
    except BaseException as exc:
        (root/'generation_failure.txt').write_text(traceback.format_exc());status(root,state='failed',last_error=str(exc));raise
    finally:log.close()

if __name__=='__main__':main()
