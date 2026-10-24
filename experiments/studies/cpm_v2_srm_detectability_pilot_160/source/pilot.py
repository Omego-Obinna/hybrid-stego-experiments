"""Fixed short SRM pilot; saved images only, no new native embeddings."""
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

import joblib
import numpy as np
import sealwatch as sw
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(os.environ['CPM_PILOT_RUN'])


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def status(**kw):
    path = ROOT / 'status.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(kw, updated_at=now())
    save(path, data)
    save(ROOT.parents[1] / 'status/detectpilot.json', data)


def extract(item):
    key, path, digest = item
    assert sha(path) == digest, ('image changed', key)
    cfg = json.loads((ROOT / 'run_config.json').read_text())
    cached = Path(cfg['full_study_root']) / 'features' / (digest + '.npy')
    dest = ROOT / 'features' / (digest + '.npy')
    if cached.exists():
        a = np.load(cached, allow_pickle=False)
        assert a.shape == (34671,) and a.dtype == np.float32 and np.isfinite(a).all()
        return key, str(cached), True
    if dest.exists():
        a = np.load(dest, allow_pickle=False)
        assert a.shape == (34671,) and a.dtype == np.float32 and np.isfinite(a).all()
        return key, str(dest), False
    a = sw.tools.flatten(sw.srm.extract_from_file(path)).astype(np.float32)
    assert a.shape == (34671,) and np.isfinite(a).all()
    np.save(dest, a, allow_pickle=False)
    return key, str(dest), False


def blocks(rows, methods):
    result = {}
    for row in rows:
        result.setdefault((row['cover_id'], row['payload_bpp']), {})[row['method']] = row
    return {key: group for key, group in result.items()
            if set(group) == set(methods) and all(r['valid_image'] for r in group.values())}


def choose_threshold(y, scores):
    fpr, tpr, thresholds = roc_curve(y, scores)
    allowed = np.flatnonzero(np.isfinite(thresholds))
    return float(thresholds[allowed[np.argmax((tpr + 1 - fpr)[allowed])]])


def write_table(name, rows):
    with (ROOT / name).open('x', newline='') as file:
        writer = csv.DictWriter(file, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert not (ROOT / 'pipeline_started.json').exists(), 'Pilot already attempted'
    cfg = json.loads((ROOT / 'run_config.json').read_text())
    assert sha(ROOT / 'source/pilot.py') == cfg['source_sha256']
    for name, digest in cfg['input_hashes'].items():
        assert sha(ROOT / name) == digest
    save(ROOT / 'pipeline_started.json', dict(started_at=now(), pid=os.getpid(), config_sha256=sha(ROOT / 'run_config.json')))
    rows = json.loads((ROOT / 'saved_image_records.json').read_text())
    cohort = json.loads((ROOT / 'cohort.json').read_text())
    part = {c['cover_id']: c['pilot_partition'] for c in cohort}
    assert len(cohort) == 160 and all(c['full_partition'] == 'train' for c in cohort)
    assert len(rows) == 960
    complete = blocks(rows, cfg['methods'])
    counts = {}
    for pay in cfg['payloads']:
        counts[str(pay)] = {p: sum(k[1] == pay and part[k[0]] == p for k in complete)
                            for p in ['train', 'validation', 'test']}
        for p, minimum in cfg['minimum_complete_counts'].items():
            assert counts[str(pay)][p] >= minimum, ('fixed minimum not met', pay, p, counts)
    save(ROOT / 'cohort_accounting.json', dict(selected_source_groups=160, saved_method_attempts=960,
         valid_saved_images=sum(r['valid_image'] for r in rows), common_valid_counts=counts,
         full_study_test_groups_used=0, native_images_generated=0, failed_cases_replaced=0))
    items = {}
    for (cover, pay), group in complete.items():
        c = group['cpm']
        items[f'cover_{cover}'] = (c['cover_path'], c['cover_sha256'])
        for method, row in group.items():
            items[f'{method}_{cover}_{pay}'] = (row['stego_path'], row['stego_sha256'])
    (ROOT / 'features').mkdir()
    fm = {}; reused = 0; start = time.monotonic()
    status(state='running', phase='features', stage='Extracting pilot SRM features; existing cache reused',
           features_completed=0, features_total=len(items), models_completed=0, models_total=12)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(extract, (key, path, digest)) for key, (path, digest) in items.items()]
        for future in concurrent.futures.as_completed(futures):
            key, path, was_reused = future.result(); fm[key] = path; reused += was_reused
            if len(fm) % 10 == 0 or len(fm) == len(items):
                status(features_completed=len(fm), features_reused=reused,
                       stage='Pilot full-SRM extraction', phase_elapsed_sec=time.monotonic()-start)
                print('features', len(fm), '/', len(items), 'reused', reused, flush=True)
    save(ROOT / 'feature_manifest.json', fm)
    hashes = {path: sha(path) for path in sorted(set(fm.values()))}
    save(ROOT / 'feature_hashes.json', hashes)
    save(ROOT / 'feature_completion.json', dict(completed_at=now(), feature_items=len(fm), reused=reused,
         dimensions=34671, feature_manifest_sha256=sha(ROOT / 'feature_manifest.json')))

    # Planted-feature control, separate from all native pilot observations.
    rng = np.random.default_rng(cfg['control_seed'])
    x0 = rng.standard_normal((120, 34671), dtype=np.float32)
    x1 = rng.standard_normal((120, 34671), dtype=np.float32) + 4
    control_aucs = []
    for seed in cfg['seeds']:
        trainer = sw.ensemble_classifier.FldEnsembleTrainer(x0[:80], x1[:80], seed=seed,
                   L=cfg['learners'], d_sub=cfg['subspace'], verbose=0)
        model, _ = trainer.train()
        score = model.predict_confidence(np.r_[x0[80:], x1[80:]])
        auc = float(roc_auc_score(np.r_[np.zeros(40), np.ones(40)], score))
        assert auc >= .95, ('Planted-feature control failed before native test', auc)
        control_aucs.append(auc)
    save(ROOT / 'known_leak_control.json', dict(aucs=control_aucs,
         scope='Synthetic full-dimensional shifted-feature control; not native steganalysis or a test of detector completeness.'))
    del x0, x1
    (ROOT / 'models').mkdir(); selections = []; done = 0
    for pay in cfg['payloads']:
        ids = {p: sorted(k[0] for k in complete if k[1] == pay and part[k[0]] == p)
               for p in ['train', 'validation']}
        xc = {p: np.stack([np.load(fm[f'cover_{c}'], allow_pickle=False) for c in ids[p]]) for p in ids}
        for method in cfg['methods']:
            xs = {p: np.stack([np.load(fm[f'{method}_{c}_{pay}'], allow_pickle=False) for c in ids[p]]) for p in ids}
            scores = []; paths = []
            for seed in cfg['seeds']:
                status(phase='training', stage=f'Pilot SRM: {method} at {pay:.2f} bpp', models_completed=done)
                trainer = sw.ensemble_classifier.FldEnsembleTrainer(xc['train'], xs['train'],
                           seed=seed, L=cfg['learners'], d_sub=cfg['subspace'], verbose=0)
                model, history = trainer.train(); path = ROOT / 'models' / f'{method}_{pay}_{seed}.joblib'
                joblib.dump(model, path); save(path.with_suffix('.history.json'), history)
                score = model.predict_confidence(np.r_[xc['validation'], xs['validation']])
                np.save(path.with_suffix('.validation.npy'), score, allow_pickle=False)
                paths.append(str(path)); scores.append(score); done += 1
                status(models_completed=done); print('models', done, '/12', flush=True)
            y = np.r_[np.zeros(len(ids['validation'])), np.ones(len(ids['validation']))]
            avg = np.mean(scores, axis=0); orientation = 1 if roc_auc_score(y, avg) >= .5 else -1
            selections.append(dict(method=method, payload_bpp=pay, paths=paths,
                 orientation=orientation, threshold=choose_threshold(y, avg*orientation),
                 validation_auc=float(roc_auc_score(y, avg*orientation))))
    save(ROOT / 'selection.json', dict(selected_at=now(), panels=selections,
         rule='Fixed model budget; orientation and threshold use validation only; no hyperparameter search.'))
    assert not (ROOT / 'heldout_started.json').exists()
    save(ROOT / 'heldout_started.json', dict(started_at=now(), selection_sha256=sha(ROOT / 'selection.json')))
    (ROOT / 'predictions').mkdir(); metrics = []; predictions = {}; checked = 0
    test_ids = sorted(c['cover_id'] for c in cohort if c['pilot_partition'] == 'test')
    weights = np.random.default_rng(cfg['bootstrap_seed']).multinomial(len(test_ids),
                   np.full(len(test_ids), 1/len(test_ids)), size=cfg['bootstrap_draws'])
    index = {c:i for i,c in enumerate(test_ids)}; boot = {}
    status(phase='heldout_scoring', stage='One pilot held-out pass; all model settings frozen')
    for sel in selections:
        pay, method = sel['payload_bpp'], sel['method']
        ids = sorted(k[0] for k in complete if k[1] == pay and part[k[0]] == 'test')
        xc = np.stack([np.load(fm[f'cover_{c}'], allow_pickle=False) for c in ids])
        xs = np.stack([np.load(fm[f'{method}_{c}_{pay}'], allow_pickle=False) for c in ids])
        x = np.r_[xc, xs]; y = np.r_[np.zeros(len(ids)), np.ones(len(ids))]
        seed_scores = np.stack([joblib.load(path).predict_confidence(x) for path in sel['paths']])
        scores = seed_scores.mean(axis=0) * sel['orientation']; ids2 = np.tile(ids, 2)
        path = ROOT / 'predictions' / f'{method}_{pay}.npz'
        np.savez_compressed(path, scores=scores, labels=y, cover_ids=ids2, seed_scores=seed_scores)
        predictions[(method, pay)] = dict(path=path, feature_matrix=x, selection=sel)
        delta = scores[y==1, None] - scores[None, y==0]
        concordance = (delta>0).astype(float) + .5*(delta==0)
        w = weights[:, [index[c] for c in ids]]
        draws = np.einsum('bi,ij,bj->b', w, concordance, w, optimize=True) / w.sum(1)**2
        auc = float(roc_auc_score(y, scores)); assert abs(auc-float(concordance.mean())) < 1e-12
        lo, hi = np.quantile(draws, [.025, .975]); accepted = scores >= sel['threshold']
        gap = float(accepted[y==1].mean() - accepted[y==0].mean())
        metrics.append(dict(method=method, payload_bpp=pay, test_groups=len(ids), auc=auc,
             ci_low=float(lo), ci_high=float(hi), balanced_accuracy=float((1+gap)/2),
             acceptance_gap=gap, threshold=sel['threshold']))
        boot[(method, pay)] = draws
    # Reproduction checks are planned fixed-checkpoint verification, not reselection.
    status(phase='verification', stage='Verifying all saved pilot checkpoint predictions')
    for item in predictions.values():
        z = np.load(item['path'], allow_pickle=False)
        for i, path in enumerate(item['selection']['paths']):
            actual = joblib.load(path).predict_confidence(item['feature_matrix'])
            assert np.array_equal(actual, z['seed_scores'][i]); checked += 1
    assert checked == 12
    contrasts = []
    for pay in cfg['payloads']:
        a = next(r for r in metrics if r['method']=='cpm' and r['payload_bpp']==pay)
        b = next(r for r in metrics if r['method']=='suniward' and r['payload_bpp']==pay)
        lo, hi = np.quantile(boot[('cpm',pay)]-boot[('suniward',pay)], [.05/6, 1-.05/6])
        contrasts.append(dict(payload_bpp=pay, cpm_auc_minus_suniward=a['auc']-b['auc'],
                         adjusted_ci_low=float(lo), adjusted_ci_high=float(hi)))
    for path, digest in hashes.items(): assert sha(path)==digest
    for name, digest in cfg['input_hashes'].items(): assert sha(ROOT/name)==digest
    assert sha(ROOT/'source/pilot.py') == cfg['source_sha256']
    write_table('detectability.csv', metrics); write_table('paired_auc_contrasts.csv', contrasts)
    save(ROOT/'verified_summary.json', dict(passed=True, checked_at=now(),
         selected_checkpoint_predictions_checked=checked, image_records_retained=960,
         source_input_and_feature_hashes_checked=True, full_study_test_groups_used=0,
         native_trials_regenerated=0, cohorts=counts, figure_data='detectability.csv; paired_auc_contrasts.csv'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'font.size':10,
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(5.7,3.2), layout='constrained')
    for method, color, marker in [('cpm','#176B87','o'),('suniward','#B76A22','s')]:
        rr = [r for r in metrics if r['method']==method]; yy=np.array([r['auc'] for r in rr])
        ax.errorbar([r['payload_bpp'] for r in rr],yy,
          yerr=[yy-np.array([r['ci_low'] for r in rr]), np.array([r['ci_high'] for r in rr])-yy],
          color=color,marker=marker,capsize=3,label='CPM' if method=='cpm' else 'S-UNIWARD')
    ax.axhline(.5,color='.5',ls='--',lw=.8); ax.set(xlabel='Application payload (bpp)',ylabel='Pilot held-out detection AUC',ylim=(0,1.025))
    ax.set_xticks(cfg['payloads']);ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.savefig(ROOT/'pilot_detectability.png',dpi=300);fig.savefig(ROOT/'pilot_detectability.pdf');plt.close(fig)
    lines=['# Short revised-CPM detectability pilot','',
      'Separate SRM-only development pilot. The full SRM/HSMNet experiment remains paused. No new image embedding or HSMNet training occurred. All 160 covers come from the full study training pool; its original validation/test covers are untouched. The fixed-seed cohort is drawn from training covers with cached, valid 0.10-bpp CPM/S-UNIWARD features for speed. This cache-availability cohort is not a fresh random sample of the complete BOSSBase population.', '',
      'Fixed 100/20/40 pilot training/validation/test split by cover identity. CPM and the recorded S-UNIWARD implementation use the same common-valid groups at 0.10/0.20/0.40 bpp. Full 34,671-dimensional SRM, 20 FLD learners, fixed subspace 128 and seeds 801/802. Only score orientation and threshold use validation; pilot test is used once. All failures remain in cohort accounting.', '',
      '| Method | bpp | Test groups | AUC [pointwise 95% CI] | Balanced accuracy |',
      '|---|---:|---:|---|---:|']
    for row in metrics: lines.append(f"| {row['method']} | {row['payload_bpp']:.2f} | {row['test_groups']} | {row['auc']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | {row['balanced_accuracy']:.4f} |")
    lines+=['','| bpp | CPM minus S-UNIWARD AUC | Three-payload adjusted interval |','|---:|---:|---|']
    for row in contrasts:lines.append(f"| {row['payload_bpp']:.2f} | {row['cpm_auc_minus_suniward']:.4f} | [{row['adjusted_ci_low']:.4f}, {row['adjusted_ci_high']:.4f}] |")
    lines+=['',f"All {checked} saved checkpoint predictions reproduced. Source-group uncertainty uses {cfg['bootstrap_draws']} shared multinomial draws over the forty pilot test covers. Counts after symmetric exclusions are in cohort_accounting.json. Validity of historical S-UNIWARD images is source-recorded; this pilot does not independently reconstruct historical payload bytes.", '',
      'The synthetic planted-feature control tests pipeline sensitivity to a strong artificial shift; it does not establish native detector completeness. The pilot is small, development-only evidence with a reduced fixed training budget. Near-chance AUC is not proof of security or equivalence, and a method difference needs its paired adjusted interval. Pilot results must not change the frozen full-study sample, features, training budget or selection rules.', '',
      '![Pilot detectability](pilot_detectability.png)']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save(ROOT/'completion.json',dict(completed_at=now(),verified=True,report_sha256=sha(ROOT/'RESULTS.md')))
    status(state='completed',phase='completed',stage='Pilot completed and checkpoint predictions verified; final review pending',
           models_completed=12,models_total=12,report_ready=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        (ROOT/'pipeline_failure.txt').write_text(traceback.format_exc())
        status(state='failed',stage='Pilot stopped; preserve evidence and investigate',last_error=str(error))
        raise
