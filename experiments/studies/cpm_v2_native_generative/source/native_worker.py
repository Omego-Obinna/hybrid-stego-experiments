"""Isolated native Pulsar sender or independent receiver. No output substitutions."""
import json,os,sys,time,traceback,shutil
from pathlib import Path
import numpy as np,torch
from native_common import *
def main():
    case=json.loads(Path(sys.argv[1]).read_text());mode=sys.argv[2];d=Path(case['folder']);cfg=config();assert sha(case['key_file'])==case['key_sha256'];os.environ['CPM_NATIVE_TEMP']=str(d/(mode+'_estimate.png'));os.environ['CPM_NATIVE_MODEL']=cfg['model']['path'];sys.path.insert(0,str(ROOT/'source/pulsar'));import pulsar
    sagework=d/(mode+'_sage');shutil.copytree(ROOT/'source/pulsar/sage',sagework);pulsar.SageCode._SCRIPT_DIR=str(sagework);torch.set_num_threads(1);t=time.perf_counter();obj=pulsar.Pulsar(repo=cfg['model']['path'],key=Path(case['key_file']).read_bytes(),seed=bytes.fromhex(case['seed_hex']),benchmarks=False);loaded=time.perf_counter()-t
    t=time.perf_counter();cap=obj.estimate_regions(n_to_gen=1,n_hist_bins=100,end_to_end=True);estimate=time.perf_counter()-t;result=dict(mode=mode,native_capacity=cap,model_load_sec=loaded,estimate_sec=estimate,status='capacity_limited' if cap<case['message_bytes'] else 'ready')
    if cap>=case['message_bytes']:
        t=time.perf_counter()
        try:
            if mode=='send':
                message=Path(case['message_path']).read_bytes();assert len(message)==case['message_bytes'];padded=message+bytes(cap-len(message));(d/'padded_message.bin').write_bytes(padded)
                g=obj.generate_with_regions(padded);obj.save_sample(g['samples'][49]['hidden'],str(d/'stego.png'));obj.save_sample(g['samples'][49]['all0'],str(d/'benign.png'));result.update(status='sent',carrier_bytes=(d/'stego.png').stat().st_size,stego_sha256=sha(d/'stego.png'),benign_sha256=sha(d/'benign.png'))
            else:
                actual=obj.reveal_with_regions(obj.load_sample(str(d/'stego.png')));(d/'recovered_padded.bin').write_bytes(actual);expected=Path(case['message_path']).read_bytes()+bytes(cap-case['message_bytes']);exact=actual==expected;(d/'recovered.bin').write_bytes(actual[:case['message_bytes']]);result.update(status='recovered' if exact else 'native_payload_error',exact_recovery=exact,padding_exact=actual[case['message_bytes']:]==bytes(cap-case['message_bytes']),plaintext_error_bits=sum((a^b).bit_count() for a,b in zip(actual,expected))+8*abs(len(actual)-len(expected)))
        except (ValueError,AssertionError) as e:result.update(status='native_encoding_or_decoding_failure',error=str(e));(d/(mode+'_native_failure.txt')).write_text(traceback.format_exc())
        result['operation_sec']=time.perf_counter()-t
    save(d/(mode+'_metrics.json'),result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
