"""Operational checkpoint continuation of the already authorised fixed HSMNet job."""
import pathlib,os,sys,json,subprocess,traceback,fcntl
ROOT=pathlib.Path(os.environ['CPM_IMAGE_RUN']);HERE=ROOT/'resume_20261021_v1';sys.path.insert(0,str(ROOT/'source'))
from image_common import *
def stage(name,path=None,args=None):
    status(state='running',stage='Continuing '+name,continuation='resume_20261021_v1')
    with (HERE/(name+'.log')).open('x') as log:p=subprocess.run([sys.executable,str(path or ROOT/'source'/(name+'.py'))]+(args or []),stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
    if p.returncode:raise RuntimeError(f'{name} exited {p.returncode}; preserve evidence')
def main():
    config();assert not (HERE/'continuation_started.json').exists()
    for n in ['hsm_selection.json','test_scoring_started.json','test_predictions.json','completion.json']:assert not(ROOT/n).exists(),n
    provenance=json.loads((HERE/'provenance.json').read_text())
    for name,digest in provenance['frozen_source_and_inputs'].items():assert sha(ROOT/name)==digest,name
    for name,digest in provenance['resume_source_hashes'].items():assert sha(HERE/name)==digest,name
    save(HERE/'continuation_started.json',dict(started_at=now(),pid=os.getpid(),config_sha256=sha(ROOT/'run_config.json'),resume_provenance_sha256=sha(HERE/'provenance.json')))
    with (ROOT.parents[1]/'comparison_gpu.lock').open('a') as lock:
        status(phase='gpu_wait',stage='Waiting for GPU to continue from saved epoch26',continuation='resume_20261021_v1');fcntl.flock(lock,fcntl.LOCK_EX)
        stage('image_hsm_train_resume',HERE/'image_hsm_train_resume.py');stage('image_score');stage('image_verify')
    stage('image_report');stage('report_quality_pass',args=[str(ROOT)])
    completion=json.loads((ROOT/'completion.json').read_text());completion['report_quality_passed']=True;completion['operational_continuation']='resume_20261021_v1/provenance.json';save(ROOT/'completion.json',completion)
    status(state='completed',phase='completed',stage='HSMNet comparison completed; final independent numeric and figure review pending',models_completed=6,report_ready=True)
if __name__=='__main__':
    try:main()
    except BaseException as exc:
        (HERE/'pipeline_failure.txt').write_text(traceback.format_exc());status(state='failed',last_error=str(exc),stage='HSMNet continuation stopped; preserve evidence');raise
