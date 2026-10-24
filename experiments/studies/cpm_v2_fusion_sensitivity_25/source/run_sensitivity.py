#!/usr/bin/env python3
"""Frozen paired validation-only fusion sensitivity, with independent receivers."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import secrets
import sys
import subprocess
import threading
import time
import traceback

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

def sha(data):
    return hashlib.sha256(data).hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)

def execute(args,folder,label):
    start=time.perf_counter()
    result=subprocess.run(args,capture_output=True,text=True,timeout=120)
    (folder/f'{label}.stdout.txt').write_text(result.stdout)
    (folder/f'{label}.stderr.txt').write_text(result.stderr)
    return result,time.perf_counter()-start

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--evidence-root',type=Path,required=True)
    parser.add_argument('--status',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve();evidence=args.evidence_root.resolve();source=root/'source'
    assert not (root/'run_config.json').exists(),'Refusing to overwrite/resume an existing experiment'
    sys.path.insert(0,str(source))
    import epoch_protocol as ep
    manifest=evidence/'future_benchmark_manifest.csv'
    manifest_meta=json.loads((evidence/'future_benchmark_manifest.json').read_text())
    assert sha(manifest.read_bytes())==manifest_meta['sha256']
    covers=list(csv.DictReader(manifest.open()))
    val=[r for r in covers if r['partition']=='val']
    selected=sorted(val,key=lambda r:sha(('cpm-fusion-validation-20261017:'+r['cover_id']).encode()))[:25]
    assert len(selected)==25 and len({r['cover_sha256'] for r in selected})==25
    assert all(r['cover_id'] not in manifest_meta['excluded_duplicate_ids'] for r in selected)
    pilot=evidence/'runs/cpm_v2_pilot_100'
    pilot_ids={r['cover_id'] for r in csv.DictReader((pilot/'cover_manifest.csv').open())}
    assert not ({r['cover_id'] for r in selected}&pilot_ids)
    for c in selected:
        assert sha(Path(c['source_path']).read_bytes())==c['cover_sha256']
    with (root/'cover_manifest.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    weights=[0.,.25,.5,.75,1.];payloads=[.1,.2,.4]
    keydir=root/'secrets';keydir.mkdir(mode=0o700)
    master=keydir/'research_fixture_master.bin';master.write_bytes(secrets.token_bytes(32));master.chmod(0o600)
    sid=secrets.token_hex(16)
    streams=[(pilot/f'gamma{j}_stream.txt').read_bytes().splitlines() for j in (1,2)]
    assert all(len(stream)>=25 for stream in streams)
    base_order=weights.copy();random.Random(4321).shuffle(base_order)
    cases=[]
    for ci,c in enumerate(selected):
        for pi,payload in enumerate(payloads):
            epoch=ci*3+pi
            folder=root/'inputs'/f'case_{epoch:03d}';folder.mkdir(parents=True)
            bits=int(round(payload*512*512))
            message=ep.truncate(secrets.token_bytes((bits+7)//8),bits)
            (folder/'message.bin').write_bytes(message)
            for j in (1,2):
                (folder/f'gamma{j}.txt').write_bytes(streams[j-1][ci])
            for f in folder.iterdir():f.chmod(0o444)
            order=base_order[(ci+pi)%5:]+base_order[:(ci+pi)%5]
            cases.append(dict(epoch=epoch,cover_id=c['cover_id'],cover_sha256=c['cover_sha256'],
                cover_path=c['source_path'],payload_bpp=payload,application_payload_bits=bits,
                input_folder=str(folder),message_sha256=sha(message),
                gamma_sha256=[sha((folder/f'gamma{j}.txt').read_bytes()) for j in (1,2)],
                weight_order=order))
    execution=cases.copy();random.Random(1729).shuffle(execution)
    for payload in payloads:
        for position in range(5):
            assert all(sum(c['weight_order'][position]==weight for c in cases if c['payload_bpp']==payload)==5 for weight in weights)
    config=dict(run_id='cpm_v2_fusion_sensitivity_25_validation_20261017',created_at=now(),
        purpose='Paired validation sensitivity of fidelity, exact recovery, pixel changes and cost; no detector tuning or security-bound estimate',
        cover_count=25,case_count=75,total_attempts=375,partition='val',test_used=False,
        selection_rule='First 25 validation cover IDs by SHA256(cpm-fusion-validation-20261017:ID); fixed before outcomes',
        benchmark_manifest_sha256=sha(manifest.read_bytes()),payloads_bpp=payloads,fusion_weights=weights,
        session_hex=sid,master_file=str(master),fixture_keys_only=True,
        pairing='Within each cover/payload, identical plaintext, parameters, session/epoch/nonces, masked bytes and STC seed across weights; separate counterfactual receiver states',
        source_sha256={p.name:sha(p.read_bytes()) for p in sorted(source.iterdir()) if p.is_file()},
        original_pilot_source_sha256=json.loads((pilot/'run_config.json').read_text())['source_files'],
        weight_order_rule='Seed 4321 initial permutation, rotate by (cover_index+payload_index)%5; each position balanced within every payload',
        case_order_seed=1729,case_execution_order=[c['epoch'] for c in execution],
        runtime=dict(python_executable=sys.executable,python_version=sys.version,
            workers=3,threads_per_process=1,device='CPU',process_startup_in_wall_measurements=True,
            cpu_timing='RUSAGE_SELF user+system CPU for cost construction plus embedding; excludes preflight/import/transport'),
        other_settings='Original QIM/CF/SLIC/STC settings retained; changing fusion weight also propagates through the implementation residual-feature map',
        endpoints='Internal pipeline endpoints, not standalone published MiPOD or S-UNIWARD',
        retry=False,fallback=False,per_process_timeout_sec=120)
    save(root/'run_config.json',config);save(root/'case_manifest.json',cases)
    start=time.perf_counter();rows=[];lock=threading.Lock();stop=threading.Event()
    log=(root/'trial_results.jsonl').open('x')
    active={}
    def status(state='running',error=None):
        elapsed=time.perf_counter()-start
        totals=[]
        for payload in payloads:
            for weight in weights:
                group=[r for r in rows if r['payload_bpp']==payload and r['fusion_weight']==weight]
                valid=[r for r in group if r.get('exact_recovery')]
                totals.append(dict(payload_bpp=payload,fusion_weight=weight,attempted=len(group),successful=len(valid),
                    mean_psnr_db=float(np.mean([r['psnr_db'] for r in valid])) if valid else None,
                    mean_embed_sec=float(np.mean([r['embed_time_sec'] for r in valid])) if valid else None))
        save(args.status,dict(state=state,stage='Fusion sensitivity: paired validation covers' if state=='running' else
            'Fusion sensitivity finished; ready for independent audit' if state=='completed' else 'Stopped; evidence preserved',
            total=375,completed=len(rows),successful=sum(r.get('exact_recovery',False) for r in rows),
            failed=sum(not r.get('exact_recovery',False) for r in rows),elapsed_sec=elapsed,
            eta_sec=elapsed/len(rows)*(375-len(rows)) if rows else None,
            active=list(active.values()),summary=totals,recent=rows[-8:],last_error=error,updated_at=now()))
    status()
    def publish(row):
        with lock:
            rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
            status()
            print(json.dumps({k:row[k] for k in ('epoch','cover_id','payload_bpp','fusion_weight','status','exact_recovery')}),flush=True)
    def one_case(case):
        source_cover=Path(case['cover_path'])
        assert sha(source_cover.read_bytes())==case['cover_sha256']
        inp=Path(case['input_folder']);message=(inp/'message.bin').read_bytes()
        assert sha(message)==case['message_sha256']
        with Image.open(source_cover) as im:
            assert im.mode=='L' and im.size==(512,512)
            original=np.asarray(im,dtype=np.int16).copy()
        prior_mask=None;prior_context=None;fails=0
        for position,weight in enumerate(case['weight_order']):
            if stop.is_set():return
            trial=root/'trials'/f'case_{case["epoch"]:03d}'/f'lambda_{weight:.2f}'
            trial.mkdir(parents=True,exist_ok=False)
            row=dict(epoch=case['epoch'],cover_id=case['cover_id'],cover_sha256=case['cover_sha256'],
                payload_bpp=case['payload_bpp'],fusion_weight=weight,within_case_position=position,
                partition='val',status='not_completed',exact_recovery=False,
                message_sha256=case['message_sha256'],gamma_sha256=case['gamma_sha256'])
            t0=time.perf_counter()
            with lock:
                active[case['epoch']]=dict(cover_id=case['cover_id'],payload_bpp=case['payload_bpp'],fusion_weight=weight)
                status()
            try:
                worker=[sys.executable,str(source/'image_worker_weighted.py'),'send','--fusion-weight',str(weight),
                    '--cover',str(source_cover),'--gamma1',str(inp/'gamma1.txt'),'--gamma2',str(inp/'gamma2.txt'),
                    '--message',str(inp/'message.bin'),'--master-file',str(master),'--session',sid,
                    '--epoch',str(case['epoch']),'--payload',str(case['payload_bpp']),
                    '--out',str(trial/'frames'),'--metrics',str(trial/'sender_metrics.json')]
                sent,sw=execute(worker,trial,'sender')
                row.update(sender_returncode=sent.returncode,sender_wall_sec=sw)
                if sent.returncode:raise RuntimeError('Sender failed: '+(sent.stderr or sent.stdout)[-1200:])
                receive=[sys.executable,str(source/'image_worker.py'),'receive','--frames',str(trial/'frames'),
                    '--master-file',str(master),'--session',sid,'--state',str(trial/'receiver_state.json'),
                    '--out',str(trial/'recovered.bin')]
                received,rw=execute(receive,trial,'receiver')
                row.update(receiver_returncode=received.returncode,receiver_wall_sec=rw)
                if received.returncode:raise RuntimeError('Receiver failed: '+(received.stderr or received.stdout)[-1200:])
                actual=(trial/'recovered.bin').read_bytes()
                metrics=json.loads((trial/'sender_metrics.json').read_text())
                assert metrics['profile']['fused_weight']==weight
                e=ep.read_frames(trial/'frames');context=sha(ep.context(e))
                assert e.epoch==case['epoch'] and e.payload_bits==case['application_payload_bits']
                assert sha(e.gamma1)==case['gamma_sha256'][0] and sha(e.gamma2)==case['gamma_sha256'][1]
                if prior_mask is not None:
                    assert prior_mask==metrics['masked_sha256'] and prior_context==context
                prior_mask,prior_context=metrics['masked_sha256'],context
                with Image.open(io.BytesIO(e.stego)) as im:
                    stego=np.asarray(im,dtype=np.int16).copy()
                delta=stego-original;mse=float(np.mean(delta.astype(np.float64)**2))
                errors=sum((a^b).bit_count() for a,b in zip(message,actual))+8*abs(len(message)-len(actual))
                row.update(status='recovered' if actual==message else 'wrong_plaintext',exact_recovery=actual==message,
                    application_payload_bits=e.payload_bits,plaintext_error_bits=errors,plaintext_ber=errors/e.payload_bits,
                    recovered_sha256=sha(actual),masked_sha256=metrics['masked_sha256'],epoch_context_sha256=context,
                    stego_sha256=sha(e.stego),transcript_sha256=sha(ep.transcript(e)),
                    psnr_db=10*math.log10(255**2/mse) if mse else None,mse=mse,
                    ssim=float(structural_similarity(original,stego,data_range=255)),
                    modified_pixels=int(np.count_nonzero(delta)),change_rate=float(np.mean(delta!=0)),
                    pixel_change_min=int(delta.min()),pixel_change_max=int(delta.max()),
                    embed_time_sec=metrics['embed_time_sec'],embedding_cpu_sec=metrics['embedding_cpu_sec'],
                    preflight_decode_time_sec=metrics['preflight_decode_time_sec'],
                    wire_bytes=sum(p.stat().st_size for p in (trial/'frames').glob('C*.json')),
                    canonical_transcript_bytes=len(ep.transcript(e)),effective_stc_bpp=metrics['effective_stc_bpp'],
                    byte_padding_bits=metrics['byte_padding_bits'],profile=metrics['profile'])
                assert np.max(np.abs(delta))<=1,'Native +/-1 interface changed'
                if actual!=message:stop.set()
            except Exception as exc:
                row.update(status='failed',error=str(exc));fails+=1
                (trial/'failure.txt').write_text(traceback.format_exc())
                if fails>=3:stop.set()
            row['elapsed_sec']=time.perf_counter()-t0;row['completed_at']=now();publish(row)
            if row['status']=='wrong_plaintext':
                raise RuntimeError('Receiver accepted wrong plaintext; all further work stopped')
        with lock:
            active.pop(case['epoch'],None);status()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(one_case,c) for c in execution]
            for future in as_completed(futures):future.result()
        assert len(rows)==375,'Batch stopped before all planned attempts'
        save(root/'completion.json',dict(completed_at=now(),attempted=len(rows),
            exact_recoveries=sum(r.get('exact_recovery',False) for r in rows),
            failed=sum(not r.get('exact_recovery',False) for r in rows),elapsed_sec=time.perf_counter()-start,
            interpretation='Validation-only sensitivity; no detector evaluation or optimal-security weight claim'))
        with lock:status('completed')
    except BaseException as exc:
        stop.set()
        (root/'failure.txt').write_text(traceback.format_exc())
        with lock:status('failed',str(exc))
        raise
    finally:
        log.close()

if __name__=='__main__':
    main()
