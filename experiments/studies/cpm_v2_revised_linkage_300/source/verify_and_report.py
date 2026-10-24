#!/usr/bin/env python3
"""Check saved checkpoints/predictions independently, then export exact results."""
import argparse,csv,json,math,sys,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from common import sha,save,now,status,read_jsonl
from metrics import rank_items

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();cfg=json.loads((root/'run_config.json').read_text());out=root/'evaluation'
    for name,digest in cfg['source_sha256'].items():assert sha((root/'source'/name).read_bytes())==digest
    evaluation=json.loads((out/'evaluation_completion.json').read_text())
    assert evaluation['all_controls_passed'] and evaluation['panels']==24 and evaluation['models']==72
    assert sha((root/'primary_selection.json').read_bytes())==evaluation['primary_selection_sha256']
    schema=json.loads((root/'feature_schema.json').read_text());data=np.load(root/'feature_dataset.npz')
    assert sha((root/'feature_dataset.npz').read_bytes())==schema['feature_sha256']
    assert sha((root/'feature_metadata.csv.gz').read_bytes())==schema['metadata_sha256']
    X,y=data['X'],data['y'];meta=pd.read_csv(root/'feature_metadata.csv.gz',dtype={'cover_id':str})
    assert np.array_equal(y,meta.label.to_numpy())
    assert np.array_equal(y,(meta.true_pair_index==meta.candidate_pair_index).to_numpy())
    covers=list(csv.DictReader((root/'cover_manifest.csv').open()))
    cover_part={c['cover_id']:c['partition'] for c in covers}
    assert len(covers)==300 and len({c['cover_sha256'] for c in covers})==300
    assert all(cover_part[str(c)]==p for c,p in zip(meta.cover_id,meta.partition))
    generation=read_jsonl(root/'generation_results.jsonl');cases=json.loads((root/'case_manifest.json').read_text())
    assert len(generation)==len(cases)==10800
    assert len({r['epoch'] for r in generation})==10800
    failures=[r for r in generation if not r['exact_recovery']]
    assert all(r['status']=='sender_preflight_failed' for r in failures)
    source_cases={c['epoch']:c for c in cases}
    # Every source cover and message is hash-checked again; protocol extraction was
    # separately performed by an original worker and checked in features.py.
    for c in covers:assert sha(Path(c['source_path']).read_bytes())==c['cover_sha256']
    for c in cases:assert sha(Path(c['message_path']).read_bytes())==c['message_sha256']
    for r in generation:
        folder=root/'epochs'/f'epoch_{r["epoch"]:05d}'
        if r['exact_recovery']:
            assert (folder/'recovered.bin').read_bytes()==Path(source_cases[r['epoch']]['message_path']).read_bytes()
            assert sha(Path(r['stego_path']).read_bytes())==r['stego_sha256']
        else:assert not (folder/'frames').exists() and not (folder/'recovered.bin').exists()
    tables=pd.read_csv(out/'test_metrics.csv');primary=pd.read_csv(out/'primary_metrics.csv')
    validation=pd.read_csv(out/'validation_models.csv')
    assert len(tables)==24 and len(primary)==3
    selection=json.loads((root/'primary_selection.json').read_text())
    checkpoint_checks=0
    for row in tables.to_dict('records'):
        pay,view,family=(row[k] for k in ('payload_bpp','view','classifier'))
        tag=f'p{int(round(pay*100)):03d}_{view}_{family}'
        saved=np.load(out/f'{tag}_test.npz');idx=saved['row_indices'];scores=saved['ensemble_scores']
        assert np.all(meta.iloc[idx].partition=='test') and np.allclose(meta.iloc[idx].payload_bpp,pay)
        assert np.isclose(roc_auc_score(y[idx],scores),row['auc'],atol=1e-12)
        assert np.isclose(balanced_accuracy_score(y[idx],scores>=.5),row['balanced_accuracy'],atol=1e-12)
        assert np.allclose(saved['seed_scores'].mean(axis=0),scores,atol=1e-14)
        # Reproduce every saved held-out score from all 72 frozen checkpoints.
        check=[]
        for seed in cfg['model_seeds']:
            model=joblib.load(out/'models'/f'{tag}_seed{seed}.joblib')
            raw=model['estimator'].predict_proba(X[np.ix_(idx,model['columns'])])[:,1]
            check.append(raw if row['orientation']>0 else 1-raw);checkpoint_checks+=1
        assert np.allclose(np.stack(check),saved['seed_scores'],atol=1e-12,rtol=0)
        # Independently reconstruct fractional tie rank from each twelve-row item.
        s=scores.reshape(-1,12);truth=y[idx].reshape(-1,12).argmax(axis=1)
        top1=[];mrr=[]
        for values,t in zip(s,truth):
            equal=np.isclose(values,values[t],rtol=0,atol=1e-12)
            above=int(np.sum((values>values[t])&~equal));ties=int(equal.sum())
            top1.append(1/ties if above==0 else 0)
            mrr.append(float(np.mean([1/k for k in range(above+1,above+ties+1)])))
        assert np.isclose(np.mean(top1),row['top1'],atol=1e-12)
        assert np.isclose(np.mean(mrr),row['mrr'],atol=1e-12)
        status(root,phase='verification',stage=f'Checked {checkpoint_checks}/72 checkpoint prediction sets',checkpoints_checked=checkpoint_checks)
    for pay in cfg['payloads']:
        subset=validation[(validation.payload_bpp==pay)&(validation.view=='text_image')]
        best=subset.sort_values('validation_auc',ascending=False,kind='stable').iloc[0]
        chosen=selection['selections'][str(pay)]
        assert best['classifier']==chosen['classifier']
        actual=primary[np.isclose(primary.payload_bpp,pay)].iloc[0]
        assert actual['classifier']==chosen['classifier']
    audit=json.loads((root/'generation_audit.json').read_text())
    assert audit['header_control_correct'] and audit['all_generation_artifact_checks_passed']
    assert audit['attempted']==10800 and audit['preflight_failures']==len(failures)
    verified=dict(verified_at=now(),saved_source_hashes_checked=True,all_checkpoint_test_predictions_reproduced=True,
        checkpoints_checked=checkpoint_checks,all_metric_recomputations_passed=True,heldout_cover_separation_checked=True,
        attempted=10800,exact_recoveries=10800-len(failures),preflight_failures=len(failures),
        wrong_accepted_plaintexts=0,primary_selection_sha256=evaluation['primary_selection_sha256'],
        feature_sha256=schema['feature_sha256'],metadata_sha256=schema['metadata_sha256'],
        primary=primary.to_dict('records'),controls=json.loads((out/'controls.json').read_text()))
    save(root/'verified_summary.json',verified)
    # Stable, interpretable figures with actual intervals, no interval clipping.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.8))
    points=primary.sort_values('payload_bpp')
    for ax,metric,label,baseline,lo,hi in [
        (axes[0],'auc','Held-out content-linkage AUC',.5,'auc_familywise_ci_low','auc_familywise_ci_high'),
        (axes[1],'top1','Correct top-ranked pair',1/12,'top1_ci_low','top1_ci_high')]:
        xs=np.arange(3);means=points[metric].to_numpy()
        ax.plot(xs,means,'o-',color='#246a82');ax.vlines(xs,points[lo],points[hi],color='#246a82')
        ax.hlines(points[lo],xs-.04,xs+.04,color='#246a82');ax.hlines(points[hi],xs-.04,xs+.04,color='#246a82')
        ax.axhline(baseline,color='#9b622c',linestyle='--',label='Chance reference')
        ax.set_xticks(xs,[f'{p:.2f}' for p in points.payload_bpp]);ax.set_xlabel('Payload (bpp)')
        ax.set_ylabel(label);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Revised CPM: bounded held-out content linkage',fontsize=12);fig.tight_layout()
    fig.savefig(root/'primary_linkage.png',dpi=300,bbox_inches='tight');fig.savefig(root/'primary_linkage.svg',bbox_inches='tight');plt.close(fig)
    text=['# Revised CPM content-linkage evaluation','',
        f"**{10800-len(failures):,}/10,800 attempted epochs recovered exactly**; {len(failures)} sender preflight failures "
        'were retained without retries or replacement covers. No incorrect accepted plaintext was observed. '
        'The study uses 300 distinct BOSSBase covers: 150 train, 50 validation and 100 test, '
        'drawn from the prepared clean partitions. Twelve fixed public parameter pairs and '
        'three application payloads are evaluated with the unchanged CPM-v2/fused-0.75 sender.', '',
        '## Held-out primary results','',
        '| Payload | Selected classifier | Test covers | Test images | AUC | Familywise-95% AUC interval | Balanced accuracy | Top-1 / 12 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in points.to_dict('records'):
        text.append(f"| {r['payload_bpp']:.2f} | {r['classifier']} | {r['test_covers']} | {r['test_images']} | "
            f"{r['auc']:.4f} | [{r['auc_familywise_ci_low']:.4f}, {r['auc_familywise_ci_high']:.4f}] | "
            f"{100*r['balanced_accuracy']:.2f}% | {100*r['top1']:.2f}% |")
    text+=['','The primary classifier family and score orientation were chosen using validation scores '
        'and frozen before test scoring. Each score averages three fixed model seeds. Binary AUC '
        'and balanced-accuracy chance references are 0.5 and 50%; random top-1 among twelve '
        'candidates is 8.33%. AUC is a ranking metric, not a cryptographic advantage bound.', '',
        f'![Primary linkage results]({root/"primary_linkage.png"})','',
        'AUC intervals in the primary table use Bonferroni adjustment over the three selected '
        'payload tests. Other diagnostic intervals and the top-1 figure use pointwise 95% '
        'cover-bootstrap intervals. The bootstrap jointly resamples the 100 attempted test '
        'covers across all their payloads and variants; each payload uses its available '
        'complete blocks. It quantifies test-cover sampling conditional on the fixed trained '
        'ensembles and twelve-pair bank, not uncertainty over arbitrary future models/parameter banks.', '',
        '## Complete-block cohort and all-attempt accounting','',
        '| Payload | Partition | Attempts | Recovered | Preflight failures | Complete covers | Images in linkage test |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in audit['counts']:
        text.append(f"| {r['payload_bpp']:.2f} | {r['partition']} | {r['attempted']} | {r['valid']} | {r['failed']} | {r['complete_covers']} | {r['primary_images']} |")
    text+=['',f"In addition to {len(failures)} failed attempts, {audit['valid_images_excluded_in_incomplete_blocks']} "
        'valid images belong to incomplete twelve-pair blocks and are excluded from the primary '
        'content-linkage cohort. This preserves candidate/label balance; it makes the linkage '
        'inference conditional on complete valid blocks. Every failure and excluded epoch remains '
        'in the generation audit/raw logs. No claim is made about leakage through delivery failures.', '',
        '## All predefined feature views and classifier families','',
        '| Payload | View | Classifier | AUC | Pointwise 95% interval | Balanced accuracy | Top-1 |',
        '|---:|---|---|---:|---:|---:|---:|']
    for r in tables.to_dict('records'):
        text.append(f"| {r['payload_bpp']:.2f} | {r['view']} | {r['classifier']} | {r['auc']:.4f} | "
            f"[{r['auc_ci_low']:.4f}, {r['auc_ci_high']:.4f}] | {100*r['balanced_accuracy']:.2f}% | {100*r['top1']:.2f}% |")
    text+=['','## Controls and interpretation','',
        'Image-only and parameter-only views are marginal-balance controls: each complete block '
        'contains every generating pair, and every image is scored with every candidate. Their '
        'chance performance is partly a property of this balanced design. It is not standalone '
        'evidence of covertness or secrecy. Cross-only and combined views test the defined '
        'content association using the fixed handcrafted feature families.', '',
        'A separate synthetic control encodes the generating/candidate pair agreement and must '
        'be detected with AUC above 0.99. This checks functioning labels, splits, classifiers '
        'and scoring. It does not establish sensitivity to every subtle real-world leakage.', '',
        'Matching visible session/epoch/header fields identifies the intended channel association. '
        'The header control verifies that association; protocol IDs and headers are excluded '
        'from content-classifier features. This study does not claim full-view unlinkability '
        'against an observer who can use those intentional association fields.', '',
        'These results supply source-consistent revised content-linkage evidence for the relevant '
        'parts of Reviewer 5 comments 1–2. Near-chance results, if observed, describe only these '
        'finite classifiers and conditional views; a confidence interval containing chance does '
        'not prove absence of linkage or estimate epsilon_unl. Above-chance observations must '
        'be retained with their uncertainty. Modern steganalysis and published multichannel/'
        'generative-hybrid comparisons remain separate unfinished work.', '',
        '## Reproducibility and independent checks','',
        'The original protocol, image worker and legacy embedder were copied unchanged from the '
        'verified pilot. Frozen source hashes, cover/pair/message manifests, source settings, '
        'classifiers, seeds and inference rules are recorded in `run_config.json` and `source/README.md`. '
        '`generation_results.jsonl` and `epochs/` preserve all attempts; `generation_audit.json` '
        'records failures and incomplete blocks. `feature_dataset.npz`, `feature_metadata.csv.gz` '
        'and `feature_schema.json` bind exactly the classifier inputs. Paths, IDs, keys, '
        'headers/nonces, payload labels and generating-pair labels are never input features.', '',
        '`evaluation/validation_models.csv` and `primary_selection.json` document selection '
        'before test scoring. `evaluation/test_metrics.csv`, `primary_metrics.csv`, seed '
        'metrics, prediction arrays and model checkpoints preserve all panels. An independent '
        'stage reproduced every held-out score from all 72 saved checkpoints and checked '
        'AUC/balanced accuracy using sklearn and ranking via a separate calculation. The '
        'cover-weighted AUC bootstrap is also checked against sklearn weighted AUC. '
        '`verified_summary.json` records the completed checks. No manuscript was edited.','']
    (root/'RESULTS.md').write_text('\n'.join(text))
    artifacts=['RESULTS.md','verified_summary.json','primary_linkage.png','primary_linkage.svg',
        'evaluation/primary_metrics.csv','evaluation/test_metrics.csv','evaluation/controls.json']
    save(root/'delivery_manifest.json',dict(verified_at=now(),artifacts=[dict(path=f,sha256=sha((root/f).read_bytes())) for f in artifacts]))
    save(root/'completion.json',dict(completed_at=now(),attempted=10800,recovered=10800-len(failures),
        preflight_failures=len(failures),all_independent_checks_passed=True,primary=primary.to_dict('records')))
    status(root,state='completed',phase='completed',stage='Revised linkage study complete; saved predictions independently checked',
        eta_sec=0,last_error=None,report=str(root/'RESULTS.md'),primary=primary.to_dict('records'))
    print(json.dumps(dict(completed=True,attempted=10800,recovered=10800-len(failures),checkpoints_checked=72)))

if __name__=='__main__':main()
