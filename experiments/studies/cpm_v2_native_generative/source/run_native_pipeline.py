import fcntl,json,os,subprocess,sys,traceback
from pathlib import Path
from native_common import *
def stage(name):
    with (ROOT/(name+'.log')).open('x') as f:r=subprocess.run([sys.executable,str(ROOT/'source'/(name+'.py'))],stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
    if r.returncode:raise RuntimeError(name+' stopped with return code '+str(r.returncode))
def main():
    cfg=config();assert not (ROOT/'pipeline_started.json').exists();save(ROOT/'pipeline_started.json',dict(started_at=now(),pid=os.getpid(),config_sha256=sha(ROOT/'run_config.json')))
    with (REV/'comparison_gpu.lock').open('a') as lock:
        status(state='running',phase='gpu_wait',stage='Waiting for shared GPU');fcntl.flock(lock,fcntl.LOCK_EX)
        if cfg['panel']=='generative':stage('native_generate')
        else:stage('native_distributed')
    if cfg['panel']=='generative':stage('native_features');stage('native_evaluate')
    status(phase='verification',stage='Checking saved native images, recovered payloads, model predictions and traces');stage('native_verify_report')
if __name__=='__main__':
    try:main()
    except BaseException as e:(ROOT/'pipeline_failure.txt').write_text(traceback.format_exc());status(state='failed',last_error=str(e),stage='Native comparison stopped; all evidence preserved');raise
