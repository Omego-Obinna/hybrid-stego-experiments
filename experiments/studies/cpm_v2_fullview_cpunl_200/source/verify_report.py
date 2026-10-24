import argparse,csv,hashlib,json,math,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
import epoch_protocol as ep
from common import check_sources,read_jsonl,sha,save,status,now,frame_bytes,query_message,next_chain,pixel_summary
from export_view import epoch_for_record
from public_features import gamma_public,image_public,content_vector,challenge_vector,full_vector

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();cfg=check_sources(root);out=root/'evaluation'
    done=json.loads((out/'completion.json').read_text());assert done['models']==90 and done['panels']==30 and done['all_controls_passed']
    selection=json.loads((root/'primary_selection.json').read_text());assert not selection['test_scored_at_selection']
    assert sha((root/'primary_selection.json').read_bytes())==done['primary_selection_sha256']
    assert (root/'primary_selection.json').stat().st_mtime_ns<min(p.stat().st_mtime_ns for p in out.glob('*_test.npz'))
    schema=json.loads((root/'feature_schema.json').read_text());X=np.load(root/'observer_features.npy',mmap_mode='r');y=np.load(root/'observer_labels.npy');meta=pd.read_csv(root/'observer_metadata.csv.gz',dtype={'cover_id':str})
    for name,key in [('observer_features.npy','feature_sha256'),('observer_labels.npy','label_sha256'),('observer_metadata.csv.gz','metadata_sha256'),('public_views.jsonl','views_sha256')]:assert sha((root/name).read_bytes())==schema[key]
    assert np.array_equal(y,meta.label) and len(y)==len(X)==schema['rows']
    covers=list(csv.DictReader((root/'cover_manifest.csv').open()));assert len(covers)==200 and len({c['cover_sha256'] for c in covers})==200
    bycover={c['cover_id']:c for c in covers}
    assert all(bycover[c]['partition']==part for c,part in zip(meta.cover_id,meta.partition))
    for c in covers:assert sha(Path(c['source_path']).read_bytes())==c['cover_sha256']
    blocks={b['block_id']:b for b in json.loads((root/'blocks.json').read_text())};cases=json.loads((root/'case_manifest.json').read_text());native=read_jsonl(root/'generation_results.jsonl');audit=json.loads((root/'world_audit.json').read_text())
    assert len(native)==len({r['case_id'] for r in native})==9600
    assert audit['attempted_native_cases']==9600 and audit['exact_recoveries']+audit['preflight_failures']==9600
    for r in native:
        folder=root/'cases'/r['case_uid']
        if r['exact_recovery']:
            assert (folder/'recovered.bin').read_bytes()==(folder/'message.bin').read_bytes()
            assert sha(Path(r['stego_path']).read_bytes())==r['stego_sha256']
        else:assert r['status']=='sender_preflight_failed' and not (folder/'frames').exists()
    bank=json.loads((root/'parameter_bank.json').read_text());g=[gamma_public(b) for b in bank];cache={};master=None;state=None;histories=None;previous=None;checked_views=0;diagonal=0
    for record in read_jsonl(root/'public_views.jsonl'):
        block=blocks[record['block_id']]
        if previous!=block['block_id']:
            previous=block['block_id'];master=Path(block['master_file']).read_bytes();state=ep.ReceiverState();histories=[];chain=bytes(32);historyhasher=hashlib.sha256()
            indices=meta.loc[meta.block_id==block['block_id']].index.to_numpy();sub=meta.iloc[indices]
            assert len(sub)==156 and sub.label.sum()==12
            linked=sub[sub.label==1];ind=sub[sub.label==0]
            assert len(ind)==144 and len({(int(x.parameter_index),int(x.carrier_index)) for x in ind.itertuples()})==144
            assert all(linked.parameter_index==linked.carrier_index)
            for column in ['parameter_index','carrier_index']:
                assert np.array_equal(np.bincount(linked[column],minlength=12)*12,np.bincount(ind[column],minlength=12))
            mz=np.asarray(X[np.ix_(indices,np.asarray(schema['columns']['metadata_only']))]);assert np.array_equal(mz,np.repeat(mz[:1],156,axis=0))
            for cid in block['case_ids'][:4]:
                case=cases[cid];folder=root/'cases'/case['case_uid'];e=ep.read_frames(folder/'frames');query=(folder/'message.bin').read_bytes()
                assert query==query_message(case['role_index'],block['payload_bits'],chain)
                ep.verify(master,e,bytes.fromhex(block['session_hex']),state);wire=frame_bytes(e);chain=next_chain(chain,query,wire)
                ip=image_public(folder/'stego.pgm');gp=g[case['pair_index']];histories.append(dict(epoch=e,query=query,content=content_vector(gp,ip),vector=challenge_vector(e,gp,ip)))
                state.accepted.add(e.sid.hex()+':'+str(e.epoch));state.nonces.update([e.nonce_a.hex(),e.nonce_b.hex(),e.nonce_c.hex()])
                historyhasher.update(ep.enc([(b'query',query),*[(f'C{i+1}'.encode(),w) for i,w in enumerate(wire)]]))
        e=epoch_for_record(root,record,bank);ep.verify(master,e,bytes.fromhex(block['session_hex']),state)
        assert sha(ep.transcript(e))==record['canonical_transcript_sha256'];wire=frame_bytes(e);assert [sha(w) for w in wire]==record['wire_sha256']
        hv=historyhasher.copy();hv.update(ep.enc([(f'C{i+1}'.encode(),w) for i,w in enumerate(wire)]));assert hv.hexdigest()==record['observer_sha256']
        assert y[record['row']]==record['label']
        if record['label']==0 and record['parameter_index']==record['carrier_index']:diagonal+=1
        # Re-extract one full vector in every block plus every diagonal branch.
        if record['parameter_index']==record['carrier_index']:
            uid=record['carrier_case_uid']
            if uid not in cache:cache[uid]=image_public(root/'cases'/uid/'stego.pgm')
            expected=full_vector(e,g[record['parameter_index']],cache[uid],histories);assert np.array_equal(expected,X[record['row']])
        checked_views+=1
        if checked_views%5000==0:status(root,phase='verification',stage='Verifying full public-view tags, bytes and marginals',views_verified=checked_views)
    assert checked_views==audit['views'] and diagonal==audit['independent_diagonal_views']
    table=pd.read_csv(out/'test_metrics.csv');primary=pd.read_csv(out/'primary_metrics.csv');validation=pd.read_csv(out/'validation_metrics.csv');assert len(table)==30 and len(primary)==3
    checkpoints=0
    for row in table.to_dict('records'):
        tag=row['tag'];saved=np.load(out/f'{tag}_test.npz');idx=saved['indices'];score=saved['ensemble'];assert np.all(meta.iloc[idx].partition=='test')
        assert abs(roc_auc_score(y[idx],score)-row['auc'])<1e-12
        assert abs(balanced_accuracy_score(y[idx],score>=.5)-row['balanced_accuracy'])<1e-12
        pred=score>=.5;gap=float(pred[y[idx]==1].mean()-pred[y[idx]==0].mean());assert abs(gap-row['signed_gap'])<1e-12
        for j,seed in enumerate(cfg['model_seeds']):
            obj=joblib.load(out/'models'/f'{tag}_seed{seed}.joblib');raw=obj['estimator'].predict_proba(np.asarray(X[np.ix_(idx,obj['columns'])]))[:,1]
            actual=raw if row['orientation']==1 else 1-raw;assert np.allclose(actual,saved['seed_scores'][j],atol=1e-12,rtol=0);checkpoints+=1
        assert np.allclose(saved['seed_scores'].mean(axis=0),score,atol=1e-14)
        status(root,phase='verification',stage=f'Full-view checkpoint predictions checked {checkpoints}/90',checkpoints_checked=checkpoints)
    for pay in cfg['payloads']:
        candidates=validation[(validation.payload_bpp==pay)&(validation.view=='full_history')]
        best=candidates.sort_values('validation_auc',ascending=False,kind='stable').iloc[0];chosen=selection['selections'][str(pay)]
        assert best['classifier']==chosen['classifier']==primary[np.isclose(primary.payload_bpp,pay)].iloc[0]['classifier']
    assert checkpoints==90
    controls=json.loads((root/'controls.json').read_text());assert controls['known_leak_detected'] and controls['all_synthetic_tags_valid']
    verified=dict(verified_at=now(),all_independent_checks_passed=True,source_hashes_checked=True,complete_public_views_checked=checked_views,
        all_world_tags_and_wire_hashes_checked=True,matched_metadata_and_conditional_marginals_checked=True,history_query_policy_checked=True,
        cover_separation_checked=True,checkpoints_checked=checkpoints,all_test_predictions_reproduced=True,selection_before_test_checked=True,
        native_attempts=9600,native_exact_recoveries=audit['exact_recoveries'],preflight_failures=audit['preflight_failures'],wrong_accepted_plaintexts=0,
        pixel_deviation_cases=len(audit['pixel_deviations']),pixels_exceeding_one_level=sum(x['pixels_exceeding_one_level'] for x in audit['pixel_deviations']),
        primary=primary.to_dict('records'),controls=controls,scope=cfg['scope'])
    save(root/'verified_summary.json',verified)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pts=primary.sort_values('payload_bpp');fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,lo,hi,reference,label in [(axes[0],'auc','auc_adjusted_ci_low','auc_adjusted_ci_high',.5,'Full-view AUC'),
        (axes[1],'absolute_gap','absolute_gap_ci_low','absolute_gap_ci_high',0.,'Fixed-rule absolute acceptance gap')]:
        xs=np.arange(3);ax.plot(xs,pts[metric],'o-',color='#236d7d');ax.vlines(xs,pts[lo],pts[hi],color='#236d7d')
        ax.hlines(pts[lo],xs-.04,xs+.04,color='#236d7d');ax.hlines(pts[hi],xs-.04,xs+.04,color='#236d7d');ax.axhline(reference,linestyle='--',color='#a26825')
        ax.set_xticks(xs,[f'{x:.2f}' for x in pts.payload_bpp]);ax.set_xlabel('Payload (bpp)');ax.set_ylabel(label);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('CP-UNL: full public challenge and four-query history');fig.tight_layout();fig.savefig(root/'fullview_cpunl.png',dpi=300,bbox_inches='tight');fig.savefig(root/'fullview_cpunl.svg',bbox_inches='tight');plt.close(fig)
    text=['# CP-UNL full-public-view extension','',
        f"All 9,600 native counterfactual/history attempts are accounted for: {audit['exact_recoveries']:,} exact recoveries and {audit['preflight_failures']} retained preflight failures. No incorrect accepted plaintext was observed. No attempted case was retried or replaced.",'',
        'The real-world h=0 comparison uses 200 new BOSSBase covers (100 train / 40 validation / 60 test), twelve fixed equal-length parameter pairs, fresh keys/sessions per cover/payload, and four adaptive chosen-message queries before a known-zero challenge. All challenge contexts and visible lengths are matched between worlds. Both worlds carry valid native HMAC tags. The independent world includes all 144 pair/carrier combinations, including its 1/12 diagonal overlap.','',
        '## Primary held-out full-history results','',
        '| Payload | Family | Test covers | AUC | Adjusted AUC interval | Balanced accuracy | Signed gap | Absolute gap |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for d in pts.to_dict('records'):text.append(f"| {d['payload_bpp']:.2f} | {d['classifier']} | {d['test_covers']} | {d['auc']:.4f} | [{d['auc_adjusted_ci_low']:.4f}, {d['auc_adjusted_ci_high']:.4f}] | {100*d['balanced_accuracy']:.2f}% | {d['signed_gap']:.4f} | {d['absolute_gap']:.4f} |")
    text+=['','AUC chance is 0.5; balanced-accuracy chance is 50%. A fixed-rule gap is its observed TPR minus FPR (or absolute value), not a universal epsilon_unl bound. Selection/orientation were frozen using validation only before any held-out scoring. Primary AUC intervals use 3,000 test-cover cluster bootstrap draws and Bonferroni adjustment over three payloads; other intervals are pointwise 95%. They condition on the fixed models, bank and query policy.', '',f'![Full-view CP-UNL results]({root/"fullview_cpunl.png"})','',
        '## Complete-block accounting','', '| Payload | Partition | Planned blocks | Complete | Excluded | Linked views | Independent views |','|---:|---|---:|---:|---:|---:|---:|']
    for d in audit['counts']:text.append(f"| {d['payload_bpp']:.2f} | {d['partition']} | {d['attempted_blocks']} | {d['complete_blocks']} | {d['excluded_blocks']} | {d['linked_views']} | {d['independent_views']} |")
    text+=['',f"There are {audit['complete_blocks']} complete blocks and {audit['views']:,} full public views. {audit['valid_native_cases_excluded']} otherwise valid native outputs belong to incomplete blocks; all remain in raw accounting. Both worlds exclude incomplete blocks symmetrically. Inference is conditional on complete valid blocks; failure/availability linkage is not estimated.",'',
        '## Every predefined view and family','', '| Payload | Observer features | Family | AUC | Pointwise interval | Balanced accuracy |','|---:|---|---|---:|---:|---:|']
    for d in table.to_dict('records'):text.append(f"| {d['payload_bpp']:.2f} | {d['view']} | {d['classifier']} | {d['auc']:.4f} | [{d['auc_ci_low']:.4f}, {d['auc_ci_high']:.4f}] | {100*d['balanced_accuracy']:.2f}% |")
    text+=['','## What the observer receives','',
        'The full public record contains every challenge frame byte, public session/epoch/header/nonces, lengths/codec/version, actual HMAC tags, and four earlier public query messages and complete responses. The classifier inputs are fixed finite maps of those bytes, including ordered history features and cross-history comparisons. Keys, sender masks/seeds/cost maps, original covers, hidden world/pair labels and file identifiers never enter feature inputs. Full raw bytes can be reconstructed with source/export_view.py using the saved view manifest and native artifacts.', '',
        'This implements the manuscript\'s matched-metadata linked versus independent-content comparison for the declared real-world finite source. Counterfactual challenge branches share context because they are alternative worlds; they are never emitted together as nonce-reusing live traffic. Independent-content views are laboratory-challenger hybrids, retagged consistently, not adversarial forgeries or claimed correct encodings of the known challenge message.', '',
        '## Controls and retained anomalies','',
        'Metadata-only chance scores follow from exact metadata balance and are not an independent secrecy result. A separate deliberately implanted public digest-to-image coupling is processed by the same public feature maps and classifiers. Its valid tags and diagonal overlap are retained; its theoretical maximum AUC is 23/24, not 1. This control checks sensitivity to the implanted signal, not every possible leak.', '',
        *[f"Synthetic known-leak {x['classifier']} AUC: {x['auc']:.4f}." for x in controls['models']], '',
        f"Recovered native cases with pixel changes beyond one level: {len(audit['pixel_deviations'])}; total such pixels: {verified['pixels_exceeding_one_level']}. Exact cases and change histograms remain in pixel_deviations.json. These deviations are retained, not repaired or suppressed.", '',
        '## Interpretation and reproducibility','',
        'Report all observed scores, including above-chance or unfavourable panels. Near-chance results support only the tested full-public-view features, fresh-key source, finite bank and four-query policy; they do not prove security against every PPT warden or estimate the compositional epsilon_unl. No H2 uniform-mask hybrid premise is experimentally certified. Public headers intentionally associate frames. Network scheduling/timing and active decryption queries are outside this CP-UNL experiment. Fixed-length parameter prefixes define a controlled conditional length stratum, not a naturalness or unseen-bank result.', '',
        'The completed earlier content-only study remains unchanged and uses different covers/parameter lengths; do not pool its results. source/DESIGN.md and run_config.json freeze the design/source/model choices; generation_results.jsonl preserves every native attempt; public_views.jsonl binds reconstructible complete views and their hashes; world_audit.json records world marginals/failures; feature_schema.json binds feature arrays; evaluation/ retains all 90 checkpoints, validation selection, predictions, seed metrics and 30 panels. Independent verification reproduced every held-out prediction and verified every world tag, wire/observer hash, conditional marginal and source split. No manuscript, synced source, old experiment or deferred CNN description was edited.','']
    (root/'RESULTS.md').write_text('\n'.join(text))
    files=['RESULTS.md','verified_summary.json','world_audit.json','pixel_deviations.json','fullview_cpunl.png','fullview_cpunl.svg','evaluation/primary_metrics.csv','evaluation/test_metrics.csv','controls.json']
    save(root/'delivery_manifest.json',dict(checked_at=now(),artifacts=[dict(path=f,sha256=sha((root/f).read_bytes())) for f in files]))
    save(root/'completion.json',dict(completed_at=now(),all_independent_checks_passed=True,native_attempts=9600,native_recovered=audit['exact_recoveries'],
        preflight_failures=audit['preflight_failures'],complete_public_views=checked_views,checkpoints_checked=90,primary=primary.to_dict('records')))
    status(root,state='completed',phase='completed',stage='Full-view CP-UNL extension complete; all predictions and public worlds checked',eta_sec=0,last_error=None,primary=primary.to_dict('records'),report=str(root/'RESULTS.md'))
    print(json.dumps(dict(completed=True,views=checked_views,checkpoints=checkpoints)))

if __name__=='__main__':main()
