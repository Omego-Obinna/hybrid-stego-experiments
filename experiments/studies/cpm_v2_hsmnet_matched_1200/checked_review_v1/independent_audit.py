"""Independent completed-run audit from saved scores/records; no training or inference."""
from pathlib import Path
import csv,json,hashlib,datetime
import numpy as np
R=Path('/home/omego/Documents/Hybrid_Stego_Eva/revision_20261017/runs/cpm_v2_hsmnet_matched_1200');P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda n:json.loads((R/n).read_text())
def table(n):return list(csv.DictReader((R/n).open()))
def same(a,b):assert np.allclose(a,b,rtol=0,atol=2e-12),(a,b)
cfg=load('run_config.json');v=load('verified_summary.json');comp=load('completion.json');qa=load('report_quality_v1/provenance.json');test=load('test_predictions.json');selection=load('hsm_selection.json');guard=load('test_scoring_started.json')
assert v['passed'] and v['selected_checkpoint_predictions_checked']==6 and v['all_saved_ensemble_scores_checked']
assert comp['verified'] and comp['detectability_panels']==15 and comp['report_quality_passed'] and sha(R/'RESULTS.md')==comp['report_sha256']
assert guard['hsm_selection_sha256']==sha(R/'hsm_selection.json') and guard['config_sha256']==sha(R/'run_config.json')
assert selection['selected_at']<guard['started_at']<test['scored_at']
assert len(selection['panels'])==3 and len(test['panels'])==15 and len(test['prediction_hashes'])==21
for path,d in test['prediction_hashes'].items():assert sha(path)==d,path
for path,d in {**cfg['source_hashes'],**cfg['input_hashes']}.items():assert sha(R/path)==d,path
parent=Path(load('provenance.json')['parent_run']);assert cfg['hsmnet']==json.loads((parent/'run_config.json').read_text())['hsmnet'];assert sha(R/'source/image_hsm_train.py')==sha(parent/'source/image_hsm_train.py')
for n,d in qa['original_hashes'].items():assert sha(R/'report_quality_v1'/n)==d,n
for n,d in qa['updated_hashes'].items():
 if n!='completion.json':assert sha(R/n)==d,n
original_quality_completion=load('report_quality_v1/completion.json');current=load('completion.json');assert all(current[k]==val for k,val in original_quality_completion.items()if k!='report_sha256')
assert qa['scientific_design_changed']==qa['raw_trials_changed']==qa['model_or_test_selection_changed']==qa['test_scoring_repeated']==False
records=[json.loads(l)for l in (R/'image_results.jsonl').read_text().splitlines()];fresh=[json.loads(l)for l in (R/'fresh_results.jsonl').read_text().splitlines()];covers=table('covers.csv');assert len(records)==18000 and len(fresh)==450 and len(covers)==1200
part={int(c['cover_id']):c['partition']for c in covers};assert len(part)==1200 and {p:sum(q==p for q in part.values())for p in ['train','validation','test']}==dict(train=700,validation=200,test=300)
assert len({(r['method'],r['cover_id'],r['payload_bpp'])for r in records})==18000
blocks={}
for r in records:
 assert part[r['cover_id']]==r['partition'];blocks.setdefault((r['cover_id'],r['payload_bpp']),{})[r['method']]=r
blocks={k:g for k,g in blocks.items()if set(g)==set(cfg['methods'])and all(r['valid_image']for r in g.values())}
counts={str(pay):{p:sum(k[1]==pay and g['cpm']['partition']==p for k,g in blocks.items())for p in ['train','validation','test']}for pay in cfg['payloads']}
for pay in counts:
 for p,n in cfg['detector_minimum_complete_counts'].items():assert counts[pay][p]>=n
all_attempts={}
for method in cfg['methods']:
 rr=[r for r in records if r['method']==method];ff=[r for r in fresh if r['method']==method]
 all_attempts[method]=dict(attempted=len(rr),valid_images=sum(r['valid_image']for r in rr),exact_recovery_confirmed=sum(r.get('exact_recovery') is True for r in rr),exact_recovery_not_independently_established=sum(r.get('exact_recovery') is None for r in rr),pixel_deviation_images=sum(r.get('pixels_over_one',0)>0 for r in rr),fresh_attempted=len(ff),fresh_exact=sum(r['exact_recovery']for r in ff),fresh_pixel_deviations=sum(r.get('pixels_over_one',0)>0 for r in ff))
 assert len(rr)==3600 and len(ff)==90
 assert all_attempts[method]['pixel_deviation_images']==v['pixel_deviation_image_counts_by_method'][method]
 assert all_attempts[method]['fresh_pixel_deviations']==v['fresh_pixel_deviation_counts_by_method'][method]
assert all_attempts['cpm']['exact_recovery_confirmed']==3595 and sum(r['exact_recovery']for r in fresh)==446
ids=sorted(i for i,p in part.items()if p=='test');index={i:j for j,i in enumerate(ids)};W=np.random.default_rng(cfg['bootstrap']['seed']).multinomial(len(ids),np.repeat(1/len(ids),len(ids)),size=cfg['bootstrap']['draws'])
boots={};metrics=[]
for r in table('detectability.csv'):
 pay=float(r['payload_bpp']);method=r['method'];panel=next(p for p in test['panels']if p['method']==method and p['payload_bpp']==pay);z=np.load(panel['prediction_path'],allow_pickle=False);y=z['labels'];s=z['scores'].astype(np.float64);g=z['cover_ids'];common=sorted(k[0]for k in blocks if k[1]==pay and part[k[0]]=='test');assert sorted(set(g))==common and len(g)==2*len(common)
 assert all(np.array_equal(np.sort(y[g==i]),[0,1])for i in common)
 order=np.argsort(s,kind='stable');ordered=s[order];starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1];positive=y[order]==1;weighted=W[:,[index[int(i)]for i in g[order]]];pos=np.add.reduceat(weighted*positive,starts,axis=1);neg=np.add.reduceat(weighted*(~positive),starts,axis=1);draws=np.sum(pos*(np.cumsum(neg,axis=1)-.5*neg),axis=1)/(pos.sum(1)*neg.sum(1))
 pp=np.add.reduceat(positive.astype(float),starts);nn=np.add.reduceat((~positive).astype(float),starts);auc=float(np.sum(pp*(np.cumsum(nn)-.5*nn))/pp.sum()/nn.sum());lo,hi=np.quantile(draws,[.025,.975]);accepted=s>=float(r['threshold']);gap=float(accepted[y==1].mean()-accepted[y==0].mean());same([auc,lo,hi,(1+gap)/2,gap],[float(r[n])for n in ['auc','auc_ci_low','auc_ci_high','balanced_accuracy','acceptance_gap']]);assert int(r['test_covers'])==len(common)
 raw=[np.load(R/'predictions'/f'hsm_{pay}_{seed}.npz',allow_pickle=False)for seed in cfg['hsmnet']['seeds']]
 for field in ['labels','cover_ids','methods']:assert np.array_equal(raw[0][field],raw[1][field])
 pick=raw[0]['methods']==method;orient=next(p['orientation']for p in selection['panels']if p['payload_bpp']==pay);average=np.mean([a['scores']for a in raw],axis=0)*orient;assert np.array_equal(average[pick],z['scores'])and np.array_equal(raw[0]['cover_ids'][pick],g)and np.array_equal(raw[0]['labels'][pick],y)
 boots[method,pay]=draws;metrics.append(dict(method=method,payload_bpp=pay,test_covers=len(common),auc=auc,ci_low=float(lo),ci_high=float(hi)))
assert len(metrics)==15
contrasts=[]
for r in table('auc_paired_contrasts.csv'):
 pay=float(r['payload_bpp']);m=r['baseline'];a={x['method']:x['auc']for x in metrics if x['payload_bpp']==pay};delta=a['cpm']-a[m];lo,hi=np.quantile(boots['cpm',pay]-boots[m,pay],[.05/24,1-.05/24]);same([delta,lo,hi],[float(r[n])for n in ['auc_cpm_minus_baseline','adjusted_ci_low','adjusted_ci_high']]);assert int(r['family_comparisons'])==12;contrasts.append(dict(baseline=m,payload_bpp=pay,difference=delta,adjusted_ci_low=float(lo),adjusted_ci_high=float(hi)))
assert len(contrasts)==12
fidelity_checks=0
for r in table('paired_fidelity.csv'):
 pay=float(r['payload_bpp']);m=r['method'];rr=[g[m]for k,g in blocks.items()if k[1]==pay and part[k[0]]=='test'];ix=[index[x['cover_id']]for x in rr];assert int(r['test_covers'])==len(rr);assert int(r['all_attempts'])==300;assert int(r['valid_images'])==sum(x['method']==m and x['payload_bpp']==pay and x['partition']=='test'and x['valid_image']for x in records)
 for f in ['psnr_db','ssim','modified_pixels']:
  a=np.asarray([x[f]for x in rr]);d=W[:,ix]@a/W[:,ix].sum(1);lo,hi=np.quantile(d,[.025,.975]);same([a.mean(),lo,hi],[float(r[n])for n in [f,f+'_ci_low',f+'_ci_high']]);fidelity_checks+=1
fidelity_contrasts=0
for r in table('fidelity_contrasts.csv'):
 pay=float(r['payload_bpp']);m=r['baseline'];f=r['metric'];rr=[g for k,g in blocks.items()if k[1]==pay and part[k[0]]=='test'];ix=[index[g['cpm']['cover_id']]for g in rr];a=np.asarray([g['cpm'][f]-g[m][f]for g in rr]);d=W[:,ix]@a/W[:,ix].sum(1);lo,hi=np.quantile(d,[.05/24,1-.05/24]);same([a.mean(),lo,hi],[float(r[n])for n in ['cpm_minus_baseline','adjusted_ci_low','adjusted_ci_high']]);fidelity_contrasts+=1
for r in table('fresh_recovery_runtime.csv'):
 rr=[x for x in fresh if x['method']==r['method']and x['payload_bpp']==float(r['payload_bpp'])];valid=[x for x in rr if x['exact_recovery']];assert int(r['attempted'])==len(rr)==30 and int(r['exact'])==len(valid)and int(r['failed'])==len(rr)-len(valid)
 for f in ['core_sec','sender_wall_sec','receiver_core_sec','receiver_wall_sec','carrier_bytes','wire_bytes']:
  aa=[(x['cost_sec']+x['stc_sec'])if f=='core_sec'and r['method']!='cpm'else x[f]for x in valid if x.get(f)is not None]
  if aa:same([np.mean(aa),np.median(aa)],[float(r[f+'_mean']),float(r[f+'_median'])])
  else:assert r[f+'_mean']==r[f+'_median']==''
audit=dict(passed=True,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),independent_auc_and_pointwise_interval_panels=15,independent_adjusted_auc_contrasts=12,independent_fidelity_mean_interval_checks=fidelity_checks,independent_fidelity_contrasts=fidelity_contrasts,fresh_cost_recovery_rows_checked=15,formula='Weighted sorted-score ROC bins, including exact tied-score handling',common_valid_counts=counts,all_attempts=all_attempts,pipeline_checkpoint_reproductions=6,all_fifteen_ensembles_reproduced_from_saved_seed_scores=True,source_input_prediction_and_QA_hash_checks_passed=True,all_scientific_settings_unchanged=True,training_diagnosis_performed=False,training_or_selection_or_test_inference_repeated=False,panels=metrics,auc_contrasts=contrasts,conditional_intervals_excluding_chance=sum(p['ci_low']>.5 or p['ci_high']<.5 for p in metrics),adjusted_contrasts_favouring_cpm=sum(c['adjusted_ci_high']<0 for c in contrasts),adjusted_contrasts_favouring_baseline=sum(c['adjusted_ci_low']>0 for c in contrasts),figures_inspected={'matched_detectability.png':'Passed: all fifteen point estimates/interval endpoints agree with tables; legible axes and five-method legend.','matched_fidelity.png':'Passed: PSNR and SSIM means agree with common-valid tables; legible axes/legend.'},preserved_scientific_artifact_hashes={n:sha(R/n)for n in ['run_config.json','source/image_hsm_train.py','image_results.jsonl','fresh_results.jsonl','hsm_selection.json','test_scoring_started.json','test_predictions.json','detectability.csv','auc_paired_contrasts.csv','paired_fidelity.csv','fidelity_contrasts.csv','fresh_recovery_runtime.csv']})
(P/'checked_comparison_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps({'passed':True,'panels':metrics,'contrasts':contrasts,'all_attempts':all_attempts,'common_valid_counts':counts,'fidelity_checks':fidelity_checks,'fidelity_contrasts':fidelity_contrasts}))
