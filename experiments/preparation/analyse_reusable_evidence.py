#!/usr/bin/env python3
"""Reanalyse historical evidence without changing source data or manuscript."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv,hashlib,json,math,os,sys,time,traceback
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score,accuracy_score,roc_curve
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent;OUT=ROOT/'audit';START=time.time()
def dump(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False,default=str));t.replace(p)
def state(stage,**kw):dump(ROOT/'status/analysis.json',{'state':'running','stage':stage,'elapsed_sec':round(time.time()-START,2),'updated_at':datetime.now(timezone.utc).isoformat(),**kw})
def writecsv(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def num(v):
 try:x=float(v);return x if math.isfinite(x) else None
 except (ValueError,TypeError):return None
def yes(v):return str(v).lower() in ('1','1.0','true','yes')
def valid(r):return yes(r.get('ok')) and yes(r.get('success_extract')) and yes(r.get('stc_used')) and r.get('stc_backend_actual')=='pystc' and num(r.get('ber'))==0
with (OUT/'bossbase_manifest.csv').open() as f:BOSS={r['cover_id']:r for r in csv.DictReader(f)}
SOURCES={'Ours-QIM historical v3.2':'ours_qim_v3_2_fused075_payloads_FULL/logs/embed_log_ours_qim.jsonl','S-UNIWARD recorded implementation':'suniward_results_pystc/logs/embed_log_suniward.jsonl','MiPOD recorded implementation':'mipod_results_pystc/logs/embed_log_mipod.jsonl','WOW recorded implementation':'wow_results_pystc/logs/embed_log_wow.jsonl','HILL recorded implementation':'hill_results_pystc/logs/embed_log_hill.jsonl'}
def main():
 state('Verifying recorded image cohorts and saved stego-file hashes')
 cohorts={};integrity=[];cohort_summary=[]
 for mi,(method,rel) in enumerate(SOURCES.items()):
  p=DATA/rel;rows={};counts=Counter();duplicate=0;conflict=set();profiles=Counter()
  with p.open() as f:
   for line in f:
    r=json.loads(line);pay=num(r.get('payload_bpp'))
    if pay not in (.1,.2,.4):continue
    if method.startswith('Ours') and r.get('run_id')!='ours-qim-v3-2-20260606T204306-d220f9':continue
    ident=str(r.get('image_id'));key=(pay,ident);counts[pay]+=1
    if key in rows:
     duplicate+=1
     if any(str(rows[key].get(k))!=str(r.get(k)) for k in ['stego_sha256','psnr_db','ok','success_extract']):conflict.add(key)
     continue
    rows[key]=r;profiles[str(r.get('run_id'))]+=1
  cohort={};checked=0;mismatch=0;missing=0;cover_mismatch=0;failures=0;integrity_by_payload=defaultdict(Counter)
  for key,r in rows.items():
   pay,ident=key
   if not valid(r) or key in conflict:failures+=1;continue
   if ident not in BOSS or r.get('cover_sha256')!=BOSS[ident]['sha256']:cover_mismatch+=1;continue
   path=Path(r.get('stego_path',''));status='ok'
   if not path.is_file():missing+=1;status='missing'
   else:
    actual=hashlib.sha256(path.read_bytes()).hexdigest();checked+=1
    if actual!=r.get('stego_sha256'):mismatch+=1;status='hash_mismatch'
   integrity_by_payload[pay][status]+=1
   if status!='ok':integrity.append({'method':method,'payload_bpp':pay,'image_id':ident,'status':status,'path':str(path)})
   else:cohort[key]=r
   if checked%1000==0:state(f'{method}: {checked:,} stego files checked',checked_files=checked,method_index=mi+1,total_methods=len(SOURCES))
  cohorts[method]=cohort
  for pay in [.1,.2,.4]:
   candidates=[r for (a,b),r in rows.items() if a==pay];verified=[r for (a,b),r in cohort.items() if a==pay]
   cohort_summary.append({'method':method,'payload_bpp':pay,'source':rel,'attempted_log_rows':counts[pay],'unique_attempted_image_ids':len(candidates),'logged_valid':sum(valid(r) for r in candidates),'verified_stego_files':len(verified),'missing_stego_files':integrity_by_payload[pay]['missing'],'mismatched_stego_files':integrity_by_payload[pay]['hash_mismatch'],'psnr_db_verified_mean':float(np.mean([float(r['psnr_db']) for r in verified])) if verified else None,'run_ids':';'.join(profiles),'duplicate_rows':duplicate,'conflicting_keys':len(conflict),'scope':'historical recorded implementation, not cpm-epoch-v2'})
 writecsv('historical_cohort_integrity.csv',cohort_summary);writecsv('stego_integrity_issues.csv',integrity)
 state('Computing paired historical fidelity differences and uncertainty')
 pairs=[];rng=np.random.default_rng(20261017);ours=cohorts['Ours-QIM historical v3.2']
 for method,other in cohorts.items():
  if method.startswith('Ours'):continue
  for pay in [.1,.2,.4]:
   keys=sorted(k for k in ours if k[0]==pay and k in other)
   for metric in ['psnr_db','ssim','change_rate']:
    by_hash=defaultdict(list)
    for k in keys:
     a=num(ours[k].get(metric));b=num(other[k].get(metric))
     if a is not None and b is not None:by_hash[BOSS[k[1]]['sha256']].append(a-b)
    diff=np.array([np.mean(v) for v in by_hash.values()]);n=len(diff)
    if not n:continue
    boot=np.empty(2000)
    for j in range(0,2000,50):boot[j:j+50]=diff[rng.integers(0,n,size=(50,n))].mean(axis=1)
    nonzero=diff[diff!=0];wp=float(wilcoxon(nonzero,alternative='two-sided',method='approx').pvalue) if len(nonzero) else 1.
    pairs.append({'method_a':'Ours-QIM historical v3.2','method_b':method,'payload_bpp':pay,'metric':metric,'paired_image_ids':len(keys),'unique_cover_content_groups':n,'mean_difference_a_minus_b':float(diff.mean()),'median_difference':float(np.median(diff)),'bootstrap95_low':float(np.quantile(boot,.025)),'bootstrap95_high':float(np.quantile(boot,.975)),'wilcoxon_two_sided_p':wp,'bootstrap_replicates':2000,'scope':'conditional on common valid verified historical outputs; full failures reported separately'})
 # Holm adjustment within each metric across all comparator/payload tests.
 for metric in ['psnr_db','ssim','change_rate']:
  rs=sorted([x for x in pairs if x['metric']==metric],key=lambda x:x['wilcoxon_two_sided_p']);prev=0.
  for i,r in enumerate(rs):prev=max(prev,min(1.,(len(rs)-i)*r['wilcoxon_two_sided_p']));r['holm_adjusted_p_within_metric']=prev
 writecsv('historical_paired_fidelity.csv',pairs)
 state('Recomputing uncertainty from saved detector predictions; no retraining')
 preds=[];detector_sources=[]
 for model,dirname,stem in [('Ye-Net','yenet_main_results_v3_2_fused075_lr0003_gamma08_noearly_seed20260605','yenet'),('Yedroudj-Net','yedroudj_main_results_v3_2_fused075_payload_020_lr0001_gamma03_seed20260605','yedroudj')]:
  for code in ['010','020','040']:detector_sources.append((model,float(code)/100,DATA/dirname/f'payload_{code}/logs/{stem}_test_predictions.csv','prob_stego'))
 for code in ['010','020','040']:detector_sources.append(('SRM+EC selected fast30 run',float(code)/100,DATA/f'SRM_EC/runs/payload_{code}_seed20260605_srm_ec_fast30/predictions.csv','score_stego'))
 for model,pay,p,scorecol in detector_sources:
  with p.open() as f:records=[x for x in csv.DictReader(f) if x.get('split','test')=='test']
  grouped=defaultdict(list)
  for r in records:
   ident=Path(r.get('filename',r.get('path',r.get('image_path','')))).stem
   if ident not in BOSS:raise ValueError(f'Unknown detector image id {ident}')
   grouped[BOSS[ident]['sha256']].append((int(r['label']),float(r[scorecol]),int(r['pred']),ident))
  vals=list(grouped.values());flat=[v for g in vals for v in g];y=np.array([x[0] for x in flat]);s=np.array([x[1] for x in flat]);pr=np.array([x[2] for x in flat]);auc=float(roc_auc_score(y,s));acc=float(accuracy_score(y,pr));fpr,tpr,_=roc_curve(y,s);pe=float(np.min((fpr+1-tpr)/2))
  # Group each cover and its stego counterpart, including duplicate content, during bootstrap.
  aucs=[];ng=len(vals)
  for _ in range(1000):
   ix=rng.integers(0,ng,size=ng);v=[x for j in ix for x in vals[j]];yy=np.fromiter((x[0] for x in v),dtype=int);ss=np.fromiter((x[1] for x in v),dtype=float);aucs.append(float(roc_auc_score(yy,ss)))
  row={'model':model,'payload_bpp':pay,'source':str(p.relative_to(DATA)),'observations':len(flat),'unique_cover_content_groups':ng,'auc_recomputed':auc,'auc_bootstrap95_low':float(np.quantile(aucs,.025)),'auc_bootstrap95_high':float(np.quantile(aucs,.975)),'accuracy_recomputed':acc,'error_at_recorded_decision_rule':1-acc,'minimum_empirical_balanced_ROC_error':pe,'bootstrap_replicates':1000,'uncertainty_scope':'fixed trained model; content-grouped test-cohort sampling, not independent training seeds; historical carriers'}
  # Identify observed train/test duplicate-content overlap for this detector split.
  if model.startswith('SRM'):
   pair=DATA/f'SRM_EC/pairs/pairs_payload_{int(pay*100):03d}.csv'
   with pair.open() as f:training={BOSS[str(x['pair_id'])]['sha256'] for x in csv.DictReader(f) if x['split']=='train'}
  else:
   pair=DATA/f'yenet_main_split_v3_2_fused075/payload_{int(pay*100):03d}/split_manifest.csv'
   with pair.open() as f:training={BOSS[Path(x['filename']).stem]['sha256'] for x in csv.DictReader(f) if x['split']=='train'}
  overlap=set(grouped)&training;clean=[x for h,g in grouped.items() if h not in overlap for x in g];row['train_test_duplicate_content_groups']=len(overlap);row['clean_subset_observations']=len(clean);row['clean_subset_auc_sensitivity']=float(roc_auc_score([x[0] for x in clean],[x[1] for x in clean])) if clean else None
  preds.append(row);state(f'Historical detector predictions checked: {model}, {pay:.2f} bpp')
 writecsv('historical_detector_uncertainty.csv',preds)
 state('Auditing historical simulator and attack scenario coverage')
 scenario=[]
 for rel in ['qim_resync_results/run_summary.csv','qim_resync_attack_results/run_summary.csv']:
  with (DATA/rel).open() as f:rs=list(csv.DictReader(f))
  cols=['seed','num_epochs','loss_rate','jitter_ms','skew_ms','window_W','timeout_tau_ms','heartbeat_h','ecc_enabled','burst_loss_prob','burst_len','replay_rate','tamper_rate','base_delay_ms','epoch_interval_ms'];seen=set();unique=[];conflicts=[]
  for r in rs:
   key=tuple(r.get(k) for k in cols)
   if key not in seen:seen.add(key);unique.append(r)
  scenario.append({'source':rel,'raw_rows':len(rs),'unique_parameter_seed_scenarios':len(unique),'duplicate_scenarios_removed_for_this_summary':len(rs)-len(unique),'replay_rates':sorted({x['replay_rate'] for x in unique}),'tamper_rates':sorted({x['tamper_rate'] for x in unique}),'injected_replay_frames':int(sum(float(x['replay_injected_frames']) for x in unique)),'tampered_frames':int(sum(float(x['tampered_injected_frames']) for x in unique)),'observed_false_accepts':int(sum(float(x['false_accepts']) for x in unique)),'seeds':sorted({int(x['seed']) for x in unique}),'scope':'historical abstract digest/erasure simulator; not revised full-transcript image receiver'})
 dump(OUT/'historical_simulator_coverage.json',scenario)
 result={'historical_methods_audited':len(cohorts),'historical_stego_files_verified':sum(x['verified_stego_files'] for x in cohort_summary),'stego_integrity_issues':len(integrity),'paired_metric_comparisons':len(pairs),'saved_detector_panels':len(preds),'new_detector_models_trained':0,'new_embedding_experiments_in_this_analysis':0,'elapsed_sec':time.time()-START,'interpretation':'Reanalysis supports only the recorded historical variants. It is not a revised-version benchmark or matched modern-method comparison.'}
 dump(OUT/'reuse_analysis_summary.json',result)
 dump(ROOT/'status/analysis.json',{'state':'completed','stage':'Historical reanalysis finished; source-specific reuse report available','updated_at':datetime.now(timezone.utc).isoformat(),**result})
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(ROOT/'status/analysis.json',{'state':'failed','stage':'Reused-data analysis needs attention','last_error':str(e),'updated_at':datetime.now(timezone.utc).isoformat()});traceback.print_exc();raise
