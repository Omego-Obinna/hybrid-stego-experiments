from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import csv,hashlib,json,os
R=Path(__file__).resolve().parents[1];RUN=R/'runs/cpm_v2_pilot_100'
summary=json.loads((RUN/'pilot_summary.json').read_text());assert summary['state']=='completed','Pilot has not finished; do not finalise'
rows=[json.loads(x) for x in (RUN/'pilot_results.jsonl').read_text().splitlines() if x.strip()]
assert len(rows)==summary['attempted']==300
assert sum(x['exact_recovery'] for x in rows)==summary['successful']
assert len({x['epoch'] for x in rows})==300 and len({x['cover_id'] for x in rows})==100
assert Counter(x['payload_bpp'] for x in rows)==Counter({.1:100,.2:100,.4:100})
assert all(not x.get('sender_original_cover_available_to_receiver',False) for x in rows)
config=json.loads((RUN/'run_config.json').read_text())
assert all(hashlib.sha256((RUN/'source'/s['name']).read_bytes()).hexdigest()==s['sha256'] for s in config['source_files'])
import sys
sys.path.insert(0,str(RUN/'source'));import epoch_protocol as ep
state=ep.ReceiverState.read(RUN/'receiver_state.json')
assert len(state.accepted)==summary['successful'] and len(state.nonces)==3*summary['successful']
# Verify persisted frames and recovered bytes against the completed per-epoch records.
for row in rows:
 if not row['exact_recovery']:continue
 p=RUN/'epochs'/f"epoch_{row['epoch']:03d}";e=ep.read_frames(p/'frames')
 assert hashlib.sha256(e.stego).hexdigest()==row['stego_sha256']
 assert (p/'message.bin').read_bytes()==(p/'recovered.bin').read_bytes()
 assert sum(x.stat().st_size for x in (p/'frames').glob('C*.json'))==row['wire_bytes']
# Unmodified accepted source version remains the baseline of the experimental snapshot.
base=Path('/home/omego/.codex/.chatgpt-projects/g-p-682bc5782a888191a87d67b2db53dfd6/revision_work/reviewer1_style_preserved');release=base/'release/reviewer5_textual_revision_package'
unchanged={p:hashlib.sha256((base/p).read_bytes()).hexdigest()==hashlib.sha256((release/p).read_bytes()).hexdigest() for p in ['manuscript/sn-article.tex','manuscript/Supplementary.tex','protocol/epoch_protocol.py','protocol/image_worker.py','reviewer1_response.tex','reviewer2_response.tex','reviewer3_response.tex','reviewer4_response.tex','reviewer5_response.tex']};assert all(unchanged.values())
analysis=json.loads((R/'audit/reuse_analysis_summary.json').read_text());extra=json.loads((R/'audit/published_fidelity_reanalysis_summary.json').read_text());boss=json.loads((R/'audit/boss_parent_summary.json').read_text());scan=json.loads((R/'audit/scan_complete.json').read_text())
report={'completed_at':datetime.now(timezone.utc).isoformat(),'audit':{'experiment_tree_entries':scan['files'],'boss_parent_entries':boss['files_inventoried'],'structured_files':scan['structured_files']+boss['structured_logs'],'fully_scanned_structured_files':scan['full_scan_files']+boss['full_scan_files'],'sampled_large_logs':scan['sampled_large_files'],'dataset_images_verified':10000,'saved_historical_stego_files_verified':analysis['historical_stego_files_verified']+extra['verified_earlier_stego_files'],'source_integrity_issues':analysis['stego_integrity_issues']+len(extra['integrity_issues'])},'pilot':summary,'accepted_sources_unchanged':unchanged,'directed_deferral_respected':'No historical CNN rate/worker manuscript changes','new_neural_models_trained':0,'completion_is_not_all_reviewers_closed':True,'remaining':['matched published multichannel/generative-hybrid benchmarks','revised detector and linkage evaluation','timed-window active-adversary simulation','validation-only fusion sensitivity if retained'],'dashboard':'http://127.0.0.1:8765','browser_opening':'user chose manual opening after automated local-address permission was declined'}
(R/'batch_completion.json').write_text(json.dumps(report,indent=2))
lines=['# Completed first experimental batch','',f"Finished: {report['completed_at']}",'',f"Revised pilot: **{summary['successful']}/{summary['attempted']} exact recoveries**, {summary['failed']} failed attempts.",'','These are 100 development covers at three payloads, not a held-out detector benchmark or cryptographic security bound.','', '| Payload | Attempted | Exact recovery | PSNR mean (dB) | SSIM mean | Serialized bytes mean | Sender wall time mean (s) |','|---|---:|---:|---:|---:|---:|---:|']
for x in summary['summary_by_payload']:
 def fmt(k,n=3):v=x.get(k);return 'not available' if v is None else f'{v:.{n}f}'
 lines.append(f"| {x['payload_bpp']:.2f} | {x['attempted']} | {x['successful']} | {fmt('psnr_db_mean',3)} | {fmt('ssim_mean',6)} | {fmt('wire_bytes_mean',1)} | {fmt('sender_wall_sec_mean',3)} |")
lines+=['','Byte counts are the actual three JSON/base64 frame files; IP/TCP and other network headers are not included. The canonical binary transcript size is reported separately in the raw results. Timings were collected on an active workstation during the audit and are engineering observations, not an isolated competitive speed benchmark.','', 'The existing evidence was reused for 72 paired historical fidelity comparisons and nine fixed-model detector uncertainty panels. Original and later embedding cohorts remain separate. No model was retrained.','','Read `audit/reuse_report.md` for what can answer reviewer comments and what still needs new experiments. `DATA_GUIDE.md` preserves source/run locations and evidence boundaries. `future_benchmark_manifest.csv` is prepared for subsequent clean training/validation/test evaluation.','','The local dashboard remains running and displays this batch as completed. No source manuscript, prior output or deferred CNN training-description paragraph was changed.']
(R/'BATCH_RESULTS.md').write_text('\n'.join(lines)+'\n')
# Provide a browser-readable completion summary through the existing report endpoint.
p=R/'audit/reuse_report.md';s=p.read_text();s+='\n\n## Revised pilot completion\n\n'+f"The first revised sender–receiver pilot completed with {summary['successful']}/{summary['attempted']} exact recoveries and {summary['failed']} failures. Full payload-specific results are saved in `../BATCH_RESULTS.md` and `../runs/cpm_v2_pilot_100/pilot_summary.json`. This remains development-only engineering evidence.\n";p.write_text(s)
print(json.dumps({'pilot_attempted':summary['attempted'],'pilot_successful':summary['successful'],'pilot_failed':summary['failed'],'historical_stego_files_verified':report['audit']['saved_historical_stego_files_verified'],'full_structured_scans':report['audit']['fully_scanned_structured_files'],'accepted_sources_preserved':all(unchanged.values())},indent=2))
