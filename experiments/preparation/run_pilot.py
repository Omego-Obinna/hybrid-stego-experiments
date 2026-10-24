#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import csv,hashlib,json,math,os,secrets,signal,subprocess,sys,time,traceback
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/cpm_v2_pilot_100';CONFIG=json.loads((RUN/'run_config.json').read_text())
sys.path.insert(0,str(RUN/'source'));import epoch_protocol as ep
START=time.time();OUTCOMES=[]
STATE={'state':'running','stage':'Starting the revised sender–receiver pilot','total':CONFIG['attempted_epochs_planned'],'completed':0,'successful':0,'failed':0,'recent':[],'run_id':CONFIG['run_id']}
def atomic(p,obj):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False));q.replace(p)
def update(**kw):
 STATE.update(kw);STATE['elapsed_sec']=round(time.time()-START,2);STATE['updated_at']=datetime.now(timezone.utc).isoformat()
 if STATE['completed']:STATE['eta_sec']=round(STATE['elapsed_sec']/STATE['completed']*(STATE['total']-STATE['completed']))
 atomic(ROOT/'status/pilot.json',STATE)
def execute(cmd,timeout,folder,label):
 t=time.perf_counter();r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
 (folder/f'{label}.stdout.txt').write_text(r.stdout);(folder/f'{label}.stderr.txt').write_text(r.stderr)
 return r,time.perf_counter()-t
def field_summary(rows):
 out=[]
 for pay in CONFIG['payloads_bpp']:
  rs=[x for x in rows if x['payload_bpp']==pay];valid=[x for x in rs if x.get('exact_recovery')]
  d={'payload_bpp':pay,'attempted':len(rs),'successful':len(valid),'failed':len(rs)-len(valid)}
  for k in ['psnr_db','ssim','change_rate','wire_bytes','canonical_transcript_bytes','application_payload_bits','embed_time_sec','receiver_wall_sec','sender_wall_sec','elapsed_sec']:
   v=[float(x[k]) for x in valid if k in x and x[k] is not None and math.isfinite(float(x[k]))]
   d[k+'_mean']=float(np.mean(v)) if v else None;d[k+'_sd']=float(np.std(v,ddof=1)) if len(v)>1 else None
  d['plaintext_error_bits']=sum(x.get('plaintext_error_bits',0) or 0 for x in valid);out.append(d)
 return out
def main():
 assert not (RUN/'pilot_results.jsonl').exists(),'Existing results found; stop instead of overwriting'
 update();covers=list(csv.DictReader((RUN/'cover_manifest.csv').open()));streams=[[x for x in (RUN/f'gamma{i}_stream.txt').read_bytes().splitlines() if x.strip()] for i in [1,2]]
 worker=[sys.executable,str(RUN/'source/image_worker.py')];state=RUN/'receiver_state.json'
 with (RUN/'pilot_results.jsonl').open('x') as log:
  for ci,c in enumerate(covers):
   cover=Path(c['path']);assert hashlib.sha256(cover.read_bytes()).hexdigest()==c['sha256'],'Cover content changed'
   for pay in CONFIG['payloads_bpp']:
    epoch=len(OUTCOMES);folder=RUN/'epochs'/f'epoch_{epoch:03d}';folder.mkdir();t0=time.perf_counter();row={'run_id':CONFIG['run_id'],'epoch':epoch,'cover_id':c['cover_id'],'cover_sha256':c['sha256'],'payload_bpp':pay,'status':'not_completed','exact_recovery':False}
    update(stage=f"Cover {ci+1}/100 · {pay:.2f} bpp · sending",current_cover=c['cover_id'])
    try:
     bits=int(round(pay*512*512));message=ep.truncate(secrets.token_bytes((bits+7)//8),bits);(folder/'message.bin').write_bytes(message)
     for j in [1,2]:(folder/f'gamma{j}.txt').write_bytes(streams[j-1][ci])
     send=worker+['send','--cover',str(cover),'--gamma1',str(folder/'gamma1.txt'),'--gamma2',str(folder/'gamma2.txt'),'--message',str(folder/'message.bin'),'--master-file',CONFIG['master_file'],'--session',CONFIG['session_hex'],'--epoch',str(epoch),'--payload',str(pay),'--out',str(folder/'frames'),'--metrics',str(folder/'sender_metrics.json')]
     sent,sw=execute(send,120,folder,'sender');row.update(sender_returncode=sent.returncode,sender_wall_sec=sw,application_payload_bits=bits)
     if sent.returncode:raise RuntimeError('Sender failed: '+(sent.stderr or sent.stdout)[-1200:])
     update(stage=f"Cover {ci+1}/100 · {pay:.2f} bpp · independent receiver",current_cover=c['cover_id'])
     receive=worker+['receive','--frames',str(folder/'frames'),'--master-file',CONFIG['master_file'],'--session',CONFIG['session_hex'],'--state',str(state),'--out',str(folder/'recovered.bin')]
     received,rw=execute(receive,120,folder,'receiver');row.update(receiver_returncode=received.returncode,receiver_wall_sec=rw)
     if received.returncode:raise RuntimeError('Receiver failed: '+(received.stderr or received.stdout)[-1200:])
     actual=(folder/'recovered.bin').read_bytes();error=sum((a^b).bit_count() for a,b in zip(message,actual))+8*abs(len(message)-len(actual));exact=actual==message
     met=json.loads((folder/'sender_metrics.json').read_text());e=ep.read_frames(folder/'frames');a=np.asarray(Image.open(cover),dtype=np.int16);b=np.asarray(Image.open(__import__('io').BytesIO(e.stego)),dtype=np.int16);diff=b-a;mse=float(np.mean(diff.astype(np.float64)**2));psnr=10*math.log10(255**2/mse) if mse else None
     row.update(status='recovered' if exact else 'wrong_plaintext',exact_recovery=exact,plaintext_error_bits=error,plaintext_ber=error/bits,psnr_db=psnr,ssim=float(structural_similarity(a,b,data_range=255)),mse=mse,modified_pixels=int(np.count_nonzero(diff)),change_rate=float(np.mean(diff!=0)),pixel_change_min=int(diff.min()),pixel_change_max=int(diff.max()),stego_sha256=hashlib.sha256(e.stego).hexdigest(),wire_bytes=sum(p.stat().st_size for p in (folder/'frames').glob('C*.json')),canonical_transcript_bytes=len(ep.transcript(e)),carrier_bytes=len(e.stego),parameter_bytes=len(e.gamma1)+len(e.gamma2),embed_time_sec=met['embed_time_sec'],effective_stc_bpp=met['effective_stc_bpp'],byte_padding_bits=met['byte_padding_bits'],stc_length_header_bits=met['stc_length_header_bits'],receiver_report=json.loads(received.stdout.strip()),sender_original_cover_available_to_receiver=False)
     if not exact:row['error']='Receiver accepted different plaintext'
    except Exception as ex:
     row['status']='failed';row['error']=str(ex)
     STATE['last_error']=f"Epoch {epoch}, cover {c['cover_id']}, {pay:.2f} bpp: {ex}"
    row['elapsed_sec']=round(time.perf_counter()-t0,6);row['completed_at']=datetime.now(timezone.utc).isoformat();OUTCOMES.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
    STATE['completed']=len(OUTCOMES);STATE['successful']=sum(x['exact_recovery'] for x in OUTCOMES);STATE['failed']=STATE['completed']-STATE['successful'];STATE['recent']=OUTCOMES[-10:];update()
    atomic(RUN/'pilot_summary.json',{'state':'running','run_id':CONFIG['run_id'],'scope':CONFIG['purpose'],'summary_by_payload':field_summary(OUTCOMES),'attempted':len(OUTCOMES),'successful':STATE['successful']})
    print(f"{len(OUTCOMES)}/300 cover={c['cover_id']} bpp={pay} exact={row['exact_recovery']}",flush=True)
    # Wrong accepted plaintext is a construction failure; stop rather than propagate invalid evidence.
    if row.get('status')=='wrong_plaintext':raise RuntimeError('Accepted incorrect plaintext; further pilot execution halted')
    if len(OUTCOMES)>=3 and all(x['status']=='failed' for x in OUTCOMES[-3:]):raise RuntimeError('Three consecutive failed attempts; stop to investigate')
 summary={'state':'completed','run_id':CONFIG['run_id'],'scope':CONFIG['purpose'],'attempted':len(OUTCOMES),'successful':STATE['successful'],'failed':STATE['failed'],'completed_at':datetime.now(timezone.utc).isoformat(),'elapsed_sec':time.time()-START,'summary_by_payload':field_summary(OUTCOMES),'population_security_claim':False,'research_fixture_keys_only':True}
 atomic(RUN/'pilot_summary.json',summary)
 with (RUN/'pilot_results.csv').open('w',newline='') as f:
  fields=sorted({k for x in OUTCOMES for k in x if k!='receiver_report'});w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(OUTCOMES)
 update(state='completed',stage=f"Pilot finished: {STATE['successful']}/{len(OUTCOMES)} exact recoveries",eta_sec=0,current_cover=None)
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  update(state='failed',stage='Pilot stopped; attention required',last_error=str(e));traceback.print_exc();raise
