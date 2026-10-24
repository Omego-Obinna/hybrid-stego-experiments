from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv,hashlib,json,math,os,time,sys,traceback
import numpy as np
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent;OUT=ROOT/'audit';START=time.time()
def dump(p,v):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def state(s,**kw):dump(ROOT/'status/analysis.json',{'state':'running','stage':s,'updated_at':datetime.now(timezone.utc).isoformat(),**kw})
def yes(v):return str(v).lower() in ('1','1.0','true')
def valid(r):
 try:return yes(r.get('ok')) and yes(r.get('success_extract')) and yes(r.get('stc_used')) and r.get('stc_backend_actual')=='pystc' and float(r.get('ber',-1))==0
 except (TypeError,ValueError):return False
def load(rel):
 d={};counts=Counter()
 with (DATA/rel).open() as f:
  for line in f:
   r=json.loads(line);pay=float(r['payload_bpp'])
   if pay not in [.1,.2,.4]:continue
   k=(pay,str(r['image_id']));counts[pay]+=1
   if k in d:raise ValueError(f'Ambiguous duplicate row {rel}: {k}')
   d[k]=r
 return d,counts
with (OUT/'bossbase_manifest.csv').open() as f:BOSS={r['cover_id']:r for r in csv.DictReader(f)}
def main():
 state('Matching the manuscript fidelity table to its original historical cohort')
 ours,attempts=load('ours_qim_results_pystc/logs/embed_log_ours_qim.jsonl');good={};issues=[]
 for i,(k,r) in enumerate(ours.items()):
  if not valid(r):continue
  p=Path(r['stego_path']);ok=p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==r['stego_sha256'] and r['cover_sha256']==BOSS[k[1]]['sha256']
  if ok:good[k]=r
  else:issues.append({'payload_bpp':k[0],'cover_id':k[1],'path':str(p)})
  if i%1500==0:state(f'Original fidelity cohort: {i:,} / {len(ours):,} records verified')
 source=[]
 with (DATA/'embedding_fidelity_analysis/csv/embedding_fidelity_table.csv').open() as f:published={float(r['payload_bpp']):r for r in csv.DictReader(f) if r['method']=='Ours-QIM'}
 for pay in [.1,.2,.4]:
  rs=[r for (p,i),r in good.items() if p==pay];mean=float(np.mean([float(x['psnr_db']) for x in rs]));row={'payload_bpp':pay,'source':'ours_qim_results_pystc/logs/embed_log_ours_qim.jsonl','attempted':attempts[pay],'logged_valid':sum(valid(r) for (p,i),r in ours.items() if p==pay),'verified_stego_files':len(rs),'psnr_db_verified_mean':mean,'published_valid_count':int(published[pay]['valid_pairs']),'published_psnr_mean':float(published[pay]['psnr_db_mean']),'published_cohort_matches':len(rs)==int(published[pay]['valid_pairs']) and abs(mean-float(published[pay]['psnr_db_mean']))<1e-6};source.append(row)
 assert all(x['published_cohort_matches'] for x in source),'Publication fidelity source mismatch remains unresolved'
 rng=np.random.default_rng(20261018);results=[]
 for method,rel in [('S-UNIWARD','suniward_results_pystc/logs/embed_log_suniward.jsonl'),('MiPOD','mipod_results_pystc/logs/embed_log_mipod.jsonl'),('WOW','wow_results_pystc/logs/embed_log_wow.jsonl'),('HILL','hill_results_pystc/logs/embed_log_hill.jsonl')]:
  baseline,_=load(rel);baseline={k:r for k,r in baseline.items() if valid(r)}
  for pay in [.1,.2,.4]:
   keys=[k for k in good if k[0]==pay and k in baseline]
   for metric in ['psnr_db','ssim','change_rate']:
    groups=defaultdict(list)
    for k in keys:groups[BOSS[k[1]]['sha256']].append(float(good[k][metric])-float(baseline[k][metric]))
    d=np.array([np.mean(x) for x in groups.values()]);n=len(d);boot=np.empty(2000)
    for j in range(0,2000,50):boot[j:j+50]=d[rng.integers(0,n,(50,n))].mean(axis=1)
    nz=d[d!=0];pv=float(wilcoxon(nz,method='approx').pvalue) if len(nz) else 1.
    results.append({'method_a':'Ours-QIM published fidelity cohort (historical earlier embedder)','method_b':method,'payload_bpp':pay,'metric':metric,'paired_image_ids':len(keys),'unique_cover_content_groups':n,'mean_difference_a_minus_b':float(d.mean()),'median_difference':float(np.median(d)),'bootstrap95_low':float(np.quantile(boot,.025)),'bootstrap95_high':float(np.quantile(boot,.975)),'wilcoxon_two_sided_p':pv,'bootstrap_replicates':2000,'scope':'actual manuscript fidelity cohort; historical recorded implementation, not revised CPM'})
 for metric in ['psnr_db','ssim','change_rate']:
  rows=sorted([r for r in results if r['metric']==metric],key=lambda x:x['wilcoxon_two_sided_p']);prev=0.
  for i,r in enumerate(rows):prev=max(prev,min(1.,(len(rows)-i)*r['wilcoxon_two_sided_p']));r['holm_adjusted_p_within_metric']=prev
 for name,rs in [('published_fidelity_cohort_check.csv',source),('published_fidelity_paired_statistics.csv',results)]:
  with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,list(rs[0]));w.writeheader();w.writerows(rs)
 summary={'cohort_matches_manuscript_fidelity_table':True,'verified_earlier_stego_files':len(good),'integrity_issues':issues,'paired_comparisons':len(results),'current_manuscript_edited':False,'source_versions_pooled':False,'elapsed_sec':time.time()-START,'completed_at':datetime.now(timezone.utc).isoformat()}
 dump(OUT/'published_fidelity_reanalysis_summary.json',summary)
 prior=json.loads((OUT/'reuse_analysis_summary.json').read_text());prior.update(published_fidelity_cohort_verified=True,published_fidelity_paired_comparisons=len(results),additional_earlier_stego_files_verified=len(good));dump(OUT/'reuse_analysis_summary.json',prior)
 dump(ROOT/'status/analysis.json',{'state':'completed','stage':'Historical reanalysis finished, including the manuscript’s actual fidelity cohort','updated_at':datetime.now(timezone.utc).isoformat(),**prior})
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:dump(ROOT/'status/analysis.json',{'state':'failed','stage':'Publication-cohort verification needs attention','last_error':str(e),'updated_at':datetime.now(timezone.utc).isoformat()});traceback.print_exc();raise
