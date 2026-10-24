#!/usr/bin/env python3
"""Persistent finite pipeline: generate -> audit/features -> fit/test -> verify/report."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from common import save,status,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--evidence-root',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();start=time.perf_counter()
    assert not (root/'pipeline_started.json').exists(),'No automatic retry or rerun of held-out evaluation'
    save(root/'pipeline_started.json',dict(started_at=now(),pid=__import__('os').getpid()))
    try:
        for stage in ('prepare','generate','features','evaluate','verify_and_report'):
            command=[sys.executable,str(root/'source'/f'{stage}.py'),'--root',str(root)]
            if stage=='prepare':command+=['--evidence-root',str(args.evidence_root)]
            print(json.dumps(dict(starting_stage=stage,time=now())),flush=True)
            with (root/f'{stage}.log').open('x') as output:
                result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=6*3600)
            assert result.returncode==0,f'{stage} failed; inspect {root/(stage+".log")}'
        save(root/'pipeline_completion.json',dict(completed_at=now(),elapsed_sec=time.perf_counter()-start))
    except BaseException as exc:
        (root/'pipeline_failure.txt').write_text(traceback.format_exc())
        status(root,state='failed',stage='Pipeline stopped; all evidence preserved',last_error=str(exc))
        raise

if __name__=='__main__':main()
