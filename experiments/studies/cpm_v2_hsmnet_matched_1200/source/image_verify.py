"""Reproduce saved predictions without retraining, selecting or retrying trials."""
import csv,json,sys
from pathlib import Path
import joblib,numpy as np,torch
from torch.utils.data import DataLoader
from image_common import *
from image_hsm_train import Pairs,HSMNet,evaluate
sys.path.insert(0,str(ROOT/'source'));import epoch_protocol as ep
def main():
    cfg=config();rows=records();assert len(rows)==18000 and len({(r['method'],r['cover_id'],r['payload_bpp']) for r in rows})==18000
    for r in rows:
        assert sha(r['cover_path'])==r['cover_sha256']
        if r['valid_image']:assert sha(r['stego_path'])==r['stego_sha256']
        if r['method']=='cpm' and r['exact_recovery']:
            folder=Path(r['stego_path']).parent;expected=ep.truncate(Path(r['message_path']).read_bytes(),r['payload_bits']);assert (folder/'recovered.bin').read_bytes()==expected
            original=ep.read_frames(folder/'frames');tagged=ep.authenticate(Path(r['master_file']).read_bytes(),original);assert ep.transcript(original)==ep.transcript(tagged)
            # authenticate() may return a replaced epoch; compare serialised tags too.
            from dataclasses import asdict
            assert asdict(original)==asdict(tagged),'saved native authentication mismatch'
    blocks=complete_blocks(rows);test=json.loads((ROOT/'test_predictions.json').read_text());checked=0
    for p,h in test['prediction_hashes'].items():assert sha(p)==h
    torch.set_num_threads(1)
    for s in json.loads((ROOT/'hsm_selection.json').read_text())['panels']:
        pay=s['payload_bpp'];loader=DataLoader(Pairs(blocks,pay,'test'),batch_size=cfg['hsmnet']['paired_batch_size'],shuffle=False,num_workers=2,pin_memory=True);scores=[]
        for seed,path in zip(cfg['hsmnet']['seeds'],s['paths']):
            model=HSMNet().cuda();ck=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);ev=evaluate(model,loader);saved=np.load(ROOT/'predictions'/f'hsm_{pay}_{seed}.npz');assert np.array_equal(ev['cover_ids'],saved['cover_ids']) and np.array_equal(ev['labels'],saved['labels']);assert np.allclose(ev['scores'],saved['scores'],rtol=1e-5,atol=1e-5);scores.append(ev['scores']);checked+=1;del model;torch.cuda.empty_cache()
        avg=np.mean(scores,axis=0)*s['orientation']
        for m in cfg['methods']:
            final=np.load(ROOT/'predictions'/f'hsm_{m}_{pay}_ensemble.npz');assert np.allclose(avg[ev['methods']==m],final['scores'],rtol=1e-5,atol=1e-5)
    assert checked==6
    fresh=[json.loads(x) for x in (ROOT/'fresh_results.jsonl').open()];assert len(fresh)==450
    for r in fresh:
        assert sha(r['cover_path'])==r['cover_sha256'] and sha(r['message_path'])==r['message_sha256']
        if r['exact_recovery']:assert (Path(r['folder'])/'recovered.bin').read_bytes()==Path(r['message_path']).read_bytes()
    for n,digest in {**cfg['source_hashes'],**cfg['input_hashes']}.items():assert sha(ROOT/n)==digest,n
    save(ROOT/'verified_summary.json',dict(checked_at=now(),passed=True,attempted_records=18000,new_native_attempts=3600,reused_records=14400,new_native_exact=sum(r['method']=='cpm' and r['exact_recovery'] for r in rows),fresh_attempts=len(fresh),fresh_exact=sum(r['exact_recovery'] for r in fresh),selected_checkpoint_predictions_checked=checked,all_saved_ensemble_scores_checked=True,definition_scope='Finite image-content wardens. No full-view IND-STEGO or CP-UNL bound is inferred.'))
    status(phase='report',stage='All selected checkpoints and saved-artifact checks passed; exporting statistics and figures',checkpoints_checked=checked)
if __name__=='__main__':main()
