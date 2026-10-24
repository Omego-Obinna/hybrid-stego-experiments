import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from common import check_sources,save,status,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();assert not (root/'pipeline_started.json').exists()
    started=time.perf_counter();save(root/'pipeline_started.json',dict(started_at=now(),pid=__import__('os').getpid()))
    try:
        cfg=check_sources(root)
        for stage in ['generate','build_views','controls','evaluate','verify_report']:
            check_sources(root);print(json.dumps(dict(starting_stage=stage,time=now())),flush=True)
            with (root/f'{stage}.log').open('x') as output:
                result=subprocess.run([sys.executable,str(root/'source'/f'{stage}.py'),'--root',str(root)],stdout=output,stderr=subprocess.STDOUT,timeout=6*3600)
            assert result.returncode==0,f'{stage} failed; inspect {stage}.log'
        save(root/'pipeline_completion.json',dict(completed_at=now(),elapsed_sec=time.perf_counter()-started))
    except BaseException as exc:
        (root/'pipeline_failure.txt').write_text(traceback.format_exc());status(root,state='failed',stage='Stopped; all attempted cases and frozen evidence preserved',last_error=str(exc));raise

if __name__=='__main__':main()
