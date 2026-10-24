import json,sys,time
from pathlib import Path
from native_common import *
def cpm(case,folder):
    cfg=config();bank=json.loads((ROOT/'parameter_bank.json').read_text());pair=bank[case['pair_index']];pay=case['message_bytes']*8/(512*512);wp=ROOT/'source/image_worker.py';cmd=[sys.executable,str(wp),'send','--cover',case['cover_path'],'--gamma1',pair['gamma1_path'],'--gamma2',pair['gamma2_path'],'--message',case['message_path'],'--master-file',case['key_file'],'--session',case['session_hex'],'--epoch',str(case['epoch']),'--payload',str(pay),'--out',str(folder/'frames'),'--metrics',str(folder/'sender_metrics.json')]
    sent,wall=execute(cmd,folder,'sender');r=dict(method='cpm',**case,folder=str(folder),exact_recovery=False,sender_wall_sec=wall,sender_returncode=sent.returncode)
    if sent.returncode:
        z=response(sent.stdout);assert sent.returncode==2 and z.get('reason')=='Sender preflight failed: STC did not recover the masked bytes',(sent.stderr,sent.stdout);r['status']='sender_preflight_failed';return r
    sys.path.insert(0,str(ROOT/'source'));import epoch_protocol as ep
    e=ep.read_frames(folder/'frames');p=folder/'stego.pgm';p.write_bytes(e.stego)
    got,wall=execute([sys.executable,str(wp),'receive','--frames',str(folder/'frames'),'--master-file',case['key_file'],'--session',case['session_hex'],'--state',str(folder/'state.json'),'--out',str(folder/'recovered.bin')],folder,'receiver');assert got.returncode==0,(got.stderr,got.stdout);assert (folder/'recovered.bin').read_bytes()==Path(case['message_path']).read_bytes(),'Wrong accepted CPM plaintext'
    met=json.loads((folder/'sender_metrics.json').read_text());r.update(status='recovered',exact_recovery=True,plaintext_error_bits=0,stego_path=str(p),stego_sha256=sha(p),benign_path=case['cover_path'],benign_sha256=case['cover_sha256'],carrier_bytes=len(e.stego),wire_bytes=sum(p.stat().st_size for p in (folder/'frames').glob('C*.json')),parameter_bytes=len(e.gamma1)+len(e.gamma2),core_embedding_sec=met['embed_time_sec'],receiver_wall_sec=wall,receiver_core_sec=response(got.stdout)['receiver_time_sec'])
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity
    a=np.asarray(Image.open(case['cover_path']),dtype=np.int16);b=np.asarray(Image.open(p),dtype=np.int16);delta=b-a;mse=float(np.mean(delta.astype(float)**2))
    r.update(pixels_over_one=int(np.count_nonzero(abs(delta)>1)),pixel_change_max=int(abs(delta).max()),mse=mse,psnr_db=float(10*np.log10(255**2/mse)) if mse else None,ssim=float(structural_similarity(a,b,data_range=255)))
    return r
def main():
    cfg=config();assert not (ROOT/'generation_results.jsonl').exists();cases=json.loads((ROOT/'cases.json').read_text());rows=[];t0=time.monotonic()
    with (ROOT/'generation_results.jsonl').open('x') as log:
        for case in cases:
            assert sha(case['cover_path'])==case['cover_sha256'] and sha(case['message_path'])==case['message_sha256']
            base=ROOT/'cases'/str(case['case_id']);base.mkdir(parents=True,exist_ok=False)
            for method in ['pulsar','cpm']:
                d=base/method;d.mkdir()
                if method=='cpm':row=cpm(case,d)
                else:
                    task=dict(case,folder=str(d));tp=d/'task.json';save(tp,task);row=dict(task,method='pulsar',exact_recovery=False)
                    sent,wall=execute([sys.executable,str(ROOT/'source/native_worker.py'),str(tp),'send'],d,'sender');assert sent.returncode==0,(sent.stderr,sent.stdout);s=json.loads((d/'send_metrics.json').read_text());row.update(sender_wall_sec=wall,sender_native=s,native_capacity=s['native_capacity'],status=s['status'])
                    if s['status']=='sent':
                        got,wall=execute([sys.executable,str(ROOT/'source/native_worker.py'),str(tp),'receive'],d,'receiver');assert got.returncode==0,(got.stderr,got.stdout);r=json.loads((d/'receive_metrics.json').read_text());assert r['native_capacity']==s['native_capacity'],'Independent native region construction disagrees';row.update(receiver_wall_sec=wall,receiver_native=r,status=r['status'],exact_recovery=r.get('exact_recovery',False),plaintext_error_bits=r.get('plaintext_error_bits'),stego_path=str(d/'stego.png'),benign_path=str(d/'benign.png'),stego_sha256=s['stego_sha256'],benign_sha256=s['benign_sha256'],carrier_bytes=s['carrier_bytes'],wire_bytes=s['carrier_bytes'])
                rows.append(row);append(log,row)
            status(state='running',phase='generation',stage='Native Pulsar and revised CPM at matched application byte lengths',completed=case['case_id']+1,total=len(cases),successful=sum(r['exact_recovery'] for r in rows),detail='All capacity and recovery failures retained; separate native benign sources and formats',elapsed_sec=time.monotonic()-t0)
            print(json.dumps(dict(case=case['case_id'],pulsar=rows[-2]['status'],cpm=rows[-1]['status'])),flush=True)
    save(ROOT/'generation_completion.json',dict(completed_at=now(),native_groups=len(cases),attempts=len(rows),exact=sum(r['exact_recovery'] for r in rows),wrong_accepted_cpm_plaintexts=0))
if __name__=='__main__':main()
