import csv,hashlib,io,json,os,secrets,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from image_common import *
sys.path.insert(0,str(ROOT/'source'));import epoch_protocol as ep
from image_generate import execute
def main():
    cfg=config();assert not (ROOT/'fresh_results.jsonl').exists()
    cases=json.loads((ROOT/'fresh_cases.json').read_text());bank=json.loads((ROOT/'parameter_bank.json').read_text());start=time.monotonic();rows=[]
    with (ROOT/'fresh_results.jsonl').open('x') as log:
        for i,task in enumerate(cases):
            folder=Path(task['folder']);folder.mkdir(parents=True,exist_ok=False);r=dict(task,exact_recovery=False,valid_image=False)
            if task['method']=='cpm':
                pair=bank[task['pair_index']];cmd=[sys.executable,str(ROOT/'source/image_worker.py'),'send','--cover',task['cover_path'],'--gamma1',pair['gamma1_path'],'--gamma2',pair['gamma2_path'],'--message',task['message_path'],'--master-file',task['master_file'],'--session',task['session_hex'],'--epoch',str(task['epoch']),'--payload',str(task['payload_bpp']),'--out',str(folder/'frames'),'--metrics',str(folder/'sender_metrics.json')]
                sent,wall=execute(cmd,folder,'sender');r.update(sender_returncode=sent.returncode,sender_wall_sec=wall)
                if sent.returncode:
                    answer=worker_response(sent.stdout);assert sent.returncode==2 and answer.get('reason')=='Sender preflight failed: STC did not recover the masked bytes',answer
                    r.update(status='sender_preflight_failed')
                else:
                    e=ep.read_frames(folder/'frames');(folder/'stego.pgm').write_bytes(e.stego)
                    got,wall=execute([sys.executable,str(ROOT/'source/image_worker.py'),'receive','--frames',str(folder/'frames'),'--master-file',task['master_file'],'--session',task['session_hex'],'--state',str(folder/'state.json'),'--out',str(folder/'recovered.bin')],folder,'receiver')
                    assert got.returncode==0;met=json.loads((folder/'sender_metrics.json').read_text());r.update(core_sec=met['embed_time_sec'],receiver_core_sec=worker_response(got.stdout)['receiver_time_sec'],receiver_wall_sec=wall,carrier_bytes=len(e.stego),wire_bytes=sum(p.stat().st_size for p in (folder/'frames').glob('C*.json')),parameter_bytes=len(e.gamma1)+len(e.gamma2))
            else:
                tp=folder/'task.json';tp.write_text(json.dumps(task));cmd=[sys.executable,str(ROOT/'source/fresh_baseline_worker.py'),str(tp)]
                sent,wall=execute(cmd+['send'],folder,'sender');r.update(sender_returncode=sent.returncode,sender_wall_sec=wall)
                if sent.returncode:r.update(status='native_embedding_failure',error=(sent.stderr or sent.stdout)[-1500:])
                else:
                    got,wall=execute(cmd+['receive'],folder,'receiver');r.update(receiver_returncode=got.returncode,receiver_wall_sec=wall)
                    if got.returncode:r.update(status='native_decoding_failure',error=(got.stderr or got.stdout)[-1500:])
                    else:r.update(**json.loads((folder/'sender_metrics.json').read_text()),**worker_response(got.stdout),carrier_bytes=(folder/'stego.pgm').stat().st_size,wire_bytes=None,protocol_side_information='Native STC seed and application length supplied; no authenticated multichannel wrapper')
            if (folder/'recovered.bin').exists():
                expected=Path(task['message_path']).read_bytes();actual=(folder/'recovered.bin').read_bytes();exact=actual==expected;r.update(exact_recovery=exact,status='recovered' if exact else 'payload_recovery_failure',plaintext_error_bits=sum((a^b).bit_count() for a,b in zip(actual,expected))+8*abs(len(actual)-len(expected)))
                if task['method']=='cpm':assert exact,'Wrong accepted CPM plaintext'
            if (folder/'stego.pgm').exists():
                p=folder/'stego.pgm';r.update(valid_image=True,stego_path=str(p),stego_sha256=sha(p),**image_metrics(np.asarray(Image.open(task['cover_path'])),np.asarray(Image.open(p))))
            rows.append(r);append(log,r);status(phase='fresh_timing',stage='Fresh paired recovery and sequential cost measurements',fresh_completed=i+1,fresh_total=len(cases),phase_elapsed_sec=time.monotonic()-start)
            print('fresh',i+1,'/',len(cases),task['method'],r['status'],flush=True)
    save(ROOT/'fresh_completion.json',dict(completed_at=now(),attempted=len(rows),exact=sum(r['exact_recovery'] for r in rows),scope='Sequential matched 30-development-cover panel; engineering wall times, full protocol versus primitive costs identified separately'))
if __name__=='__main__':main()
