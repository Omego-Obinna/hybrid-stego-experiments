"""Native-STC measured baseline primitive; no proxy, masking or protocol wrapper."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np,pystc
from PIL import Image
from image_common import ROOT
def module(name):
    s=importlib.util.spec_from_file_location('baseline_'+name,ROOT/'source'/f'{name}_baseline.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def main():
    task=json.loads(Path(sys.argv[1]).read_text());m=module(task['method']);out=Path(task['folder']);mode=sys.argv[2]
    if mode=='send':
        a=np.asarray(Image.open(task['cover_path']),dtype=np.uint8);msg=Path(task['message_path']).read_bytes();t=time.perf_counter();cpu=time.process_time()
        if task['method']=='suniward':rho=m.suniward_cost_map(a,wet_cost=1e10,sigma=1.)
        elif task['method']=='wow':rho=m.wow_cost_map(a,wet_cost=1e10,p=-1.,eps=1e-10)
        elif task['method']=='hill':rho=m.hill_cost_map(a,wet_cost=1e10,sigma=1e-10)
        else:
            rho=m.mipod_cost_map(a,task['payload_bits'],wet_cost=1e10,wiener_size=3,variance_window=9,sigma_floor=.0001)
            if isinstance(rho,tuple):rho=rho[0]
        tc=time.perf_counter()-t;t2=time.perf_counter();b=np.asarray(pystc.hide(msg,a,rho.astype(float),rho.astype(float),task['stc_seed'],mx=255,mn=0),dtype=np.uint8);embedding=time.perf_counter()-t2
        Image.fromarray(b).save(out/'stego.pgm',format='PPM')
        (out/'sender_metrics.json').write_text(json.dumps(dict(core_sec=time.perf_counter()-t,cost_sec=tc,stc_sec=embedding,cpu_sec=time.process_time()-cpu)))
    else:
        a=np.asarray(Image.open(out/'stego.pgm'),dtype=np.uint8);t=time.perf_counter();msg=bytes(pystc.unhide(a,task['stc_seed']));elapsed=time.perf_counter()-t
        (out/'recovered.bin').write_bytes(msg);print(json.dumps(dict(receiver_core_sec=elapsed)))
if __name__=='__main__':main()
