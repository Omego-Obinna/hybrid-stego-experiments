from pathlib import Path
import os,json,hashlib,shutil,subprocess,datetime
P=Path(__file__).resolve().parent;R=Path('/home/omego/Documents/Hybrid_Stego_Eva/revision_20261017/runs/cpm_v2_hsmnet_matched_1200');H=R/'resume_20261021_v1';now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
assert not H.exists()
# Do not duplicate a running trainer/driver, even if its recorded PID disappeared.
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,PermissionError):continue
 if len(args)>1 and (str(R/'source/run_hsmnet_pipeline.py').encode()==args[1] or str(R/'source/image_hsm_train.py').encode()==args[1]):raise RuntimeError('An original comparison process is still running')
for n in ['hsm_selection.json','test_scoring_started.json','test_predictions.json','completion.json']:assert not(R/n).exists(),n
H.mkdir();(H/'before').mkdir();(H/'before/models_hsm').mkdir()
for p in P.glob('*'):
 if p.is_file():shutil.copy2(p,H/p.name)
for n in ['process.json','status.json','pipeline.log','image_hsm_train.log','run_config.json']:shutil.copy2(R/n,H/'before'/n)
for p in (R/'models_hsm').iterdir():
 if p.is_file():shutil.copy2(p,H/'before/models_hsm'/p.name)
cfg=json.loads((R/'run_config.json').read_text());hist=[json.loads(l) for l in (R/'models_hsm/0.1_801_history.jsonl').read_text().splitlines()];assert [v['epoch']for v in hist]==list(range(1,27))
for n,d in {**cfg['source_hashes'],**cfg['input_hashes']}.items():assert sha(R/n)==d,n
frozen={n:sha(R/n) for n in ['run_config.json','image_results.jsonl','fresh_results.jsonl','covers.csv','source/image_hsm_train.py']};frozen.update({p:d for p,d in cfg['source_hashes'].items()})
provenance=dict(created_at=now(),author_instruction='Continue HSMNet, preserving frozen settings and without training diagnosis',reason='Recorded driver PID100332 and all matching trainer processes absent in permitted host process checks; saved status stopped during epoch27. No Python failure log.',latest_completed_epoch=26,latest_logged_partial_epoch=27,latest_logged_partial_batch=51,unsaved_partial_epoch_policy='Discard only uncheckpointed partial epoch27 updates and resume that epoch from durable epoch26 model/optimizer/scheduler. Completed epochs1-26 are retained and never retrained. All original logs/status/checkpoints are backed up.',checkpoint_state='Model, Adamax optimizer and scheduler restored together; total completed budget remains100epochs.',rng_recovery='Original last checkpoint omitted RNG state. Reconstruct fixed seed801 CPU/CUDA state after26 completed epochs by replaying only known dropout RNG calls on constant [batch,256] feature tensors and validation DataLoader CPU seed draws. No native forward pass, gradient or completed epoch is repeated. Future last checkpoints additionally save RNG state atomically.',frozen_source_and_inputs=frozen,resume_source_hashes={p.name:sha(p)for p in H.glob('*.py')},before_artifact_hashes={str(p.relative_to(H/'before')):sha(p) for p in (H/'before').rglob('*')if p.is_file()},all_scientific_settings_unchanged=True,training_quality_diagnosis_performed=False,completed_model_retrained=False,test_previously_used=False,native_image_attempts_repeated=False)
save(H/'provenance.json',provenance);save(P/'provenance.json',provenance)
env=os.environ.copy();env.update(CPM_IMAGE_RUN=str(R),PYTHONPATH=str(P.parents[1]/'phase6_matched_comparisons/env/site')+':/home/omego/Documents/Hybrid_Stego_Eva/myenv/lib/python3.12/site-packages',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR=str(R/'matplotlib_cache'))
python='/home/omego/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
with (H/'pipeline.log').open('xb') as log:p=subprocess.Popen([python,str(H/'run_continuation.py')],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,cwd=str(R))
m=dict(pid=p.pid,command=[python,str(H/'run_continuation.py')],started_at=now(),run_root=str(R),continuation='resume_20261021_v1',original_process_receipt='resume_20261021_v1/before/process.json');save(R/'process.json',m);save(P/'process.json',m);print(json.dumps(m))
