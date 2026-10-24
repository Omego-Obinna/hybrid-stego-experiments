import fcntl,json,os,subprocess,sys,time,traceback
from pathlib import Path
from image_common import *
def stage(name):
    status(state='running',stage='Running '+name)
    with (ROOT/(name+'.log')).open('x') as f:r=subprocess.run([sys.executable,str(ROOT/'source'/(name+'.py'))],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
    if r.returncode:raise RuntimeError(f'{name} stopped with return code {r.returncode}; preserve all evidence and investigate')
def main():
    cfg=config();assert not (ROOT/'pipeline_started.json').exists()
    save(ROOT/'pipeline_started.json',dict(started_at=now(),pid=os.getpid(),python=sys.executable,config_sha256=sha(ROOT/'run_config.json')))
    stage('image_generate');stage('image_fresh');stage('image_features');stage('image_srm_train')
    with (REV/'comparison_gpu.lock').open('a') as lock:
        status(phase='gpu_wait',stage='Waiting for the shared GPU after CPU feature/model stages');fcntl.flock(lock,fcntl.LOCK_EX)
        stage('image_hsm_train');stage('image_score');stage('image_verify')
    stage('image_report')
    print('Completed, independently verified native image comparison panel',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        (ROOT/'pipeline_failure.txt').write_text(traceback.format_exc());status(state='failed',last_error=str(e),stage='Image comparison stopped; evidence preserved');raise
