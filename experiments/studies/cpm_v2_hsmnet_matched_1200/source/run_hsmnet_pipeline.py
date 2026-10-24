"""Author-directed HSMNet-only execution of the frozen image comparison."""
import fcntl,json,os,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(os.environ['CPM_IMAGE_RUN']);sys.path.insert(0,str(ROOT/'source'))
from image_common import config,records,complete_blocks,save,sha,status,now

def stage(name):
    status(state='running',stage='Running '+name)
    with (ROOT/(name+'.log')).open('x') as log:
        p=subprocess.run([sys.executable,str(ROOT/'source'/(name+'.py'))],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    if p.returncode:raise RuntimeError(f'{name} exited {p.returncode}; preserve all evidence, do not rerun a used selection')

def main():
    cfg=config();assert not (ROOT/'pipeline_started.json').exists()
    for name in ['models_hsm','hsm_selection.json','test_scoring_started.json','predictions','completion.json']:
        assert not (ROOT/name).exists(),name
    parent=Path(json.loads((ROOT/'provenance.json').read_text())['parent_run'])
    for name in ['models_hsm','hsm_selection.json','test_scoring_started.json','test_predictions.json','completion.json']:
        assert not (parent/name).exists(),('Parent has already used HSMNet model/test selection',name)
    assert cfg['hsmnet']==json.loads((parent/'run_config.json').read_text())['hsmnet']
    save(ROOT/'pipeline_started.json',dict(started_at=now(),pid=os.getpid(),config_sha256=sha(ROOT/'run_config.json'),scope='HSMNet only; frozen original settings; no native-generation or SRM stage'))
    status(state='running',phase='audit',stage='Checking preserved image/input hashes before unchanged HSMNet execution',models_completed=0,models_total=6,test_panels_total=15,report_ready=False)
    rows=records();assert len(rows)==18000 and len({(r['method'],r['cover_id'],r['payload_bpp']) for r in rows})==18000
    blocks=complete_blocks(rows);counts={}
    for pay in cfg['payloads']:
        counts[str(pay)]={part:sum(k[1]==pay and v['cpm']['partition']==part for k,v in blocks.items()) for part in cfg['detector_minimum_complete_counts']}
        for part,n in cfg['detector_minimum_complete_counts'].items():assert counts[str(pay)][part]>=n,(pay,part,counts)
    images={}
    for r in rows:
        images[r['cover_path']]=r['cover_sha256']
        if r['valid_image']:images[r['stego_path']]=r['stego_sha256']
    for p,digest in images.items():assert sha(p)==digest,p
    save(ROOT/'input_audit.json',dict(passed=True,checked_at=now(),records=18000,native_attempts=3600,reused_baseline_records=14400,common_valid_counts=counts,image_hashes_checked=len(images),invalid_images_retained=sum(not r['valid_image'] for r in rows),parent_hsm_and_test_unattempted=True,original_hsm_settings_unchanged=True,training_diagnosis_performed=False))
    with (ROOT.parents[1]/'comparison_gpu.lock').open('a') as lock:
        status(phase='gpu_wait',stage='Waiting for shared GPU for the frozen HSMNet comparison')
        fcntl.flock(lock,fcntl.LOCK_EX)
        stage('image_hsm_train');stage('image_score');stage('image_verify')
    stage('image_report')
    status(state='running',phase='report_quality',stage='Versioned reporting QA from saved results only')
    with (ROOT/'report_quality_pass.log').open('x') as log:
        q=subprocess.run([sys.executable,str(ROOT/'source/report_quality_pass.py'),str(ROOT)],stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    if q.returncode:raise RuntimeError('Report quality pass failed; preserve evidence')
    status(state='completed',phase='completed',stage='HSMNet comparison completed; final independent numeric and figure review pending',models_completed=6,report_ready=True)
    completion=json.loads((ROOT/'completion.json').read_text());completion['report_quality_passed']=True;save(ROOT/'completion.json',completion)
    print('HSMNet-only fixed comparison finished; independent numeric/figure review pending',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        (ROOT/'pipeline_failure.txt').write_text(traceback.format_exc());status(state='failed',last_error=str(exc),stage='HSMNet comparison stopped; preserve all evidence');raise
