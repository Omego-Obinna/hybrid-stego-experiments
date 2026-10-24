"""Uniform cover-content delete-one jackknife intervals, retaining bootstrap diagnostics."""
from pathlib import Path
from collections import defaultdict
import csv,json,math
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT.parent;OUT=ROOT/'audit'
with (OUT/'bossbase_manifest.csv').open() as f:BOSS={x['cover_id']:x for x in csv.DictReader(f)}
def jackknife(groups):
 flat=[x for g in groups for x in g];pos=np.array([x[1] for x in flat if x[0]==1]);neg=np.array([x[1] for x in flat if x[0]==0]);np_,nn=len(pos),len(neg);sp=np.sort(pos);sn=np.sort(neg)
 def pc(p):return np.searchsorted(sn,p,'left')+.5*(np.searchsorted(sn,p,'right')-np.searchsorted(sn,p,'left'))
 def nc(n):return np_-np.searchsorted(sp,n,'right')+.5*(np.searchsorted(sp,n,'right')-np.searchsorted(sp,n,'left'))
 U=float(pc(pos).sum());point=U/(np_*nn);js=[]
 for g in groups:
  pp=np.array([x[1] for x in g if x[0]==1]);gg=np.array([x[1] for x in g if x[0]==0]);within=float(((pp[:,None]>gg[None,:])+.5*(pp[:,None]==gg[None,:])).sum());rem=U-float(pc(pp).sum())-float(nc(gg).sum())+within
  js.append(rem/((np_-len(pp))*(nn-len(gg))))
 js=np.array(js);m=len(js);se=float(np.sqrt((m-1)/m*np.sum((js-js.mean())**2)))
 return point,se,js
# Exact rank-count identity checked against direct AUC deletion on toy and real groups.
for groups in [[[(0,.2),(1,.4)],[(0,.6),(1,.7)],[(0,.3),(1,.1)]],[[(0,.5),(1,.5)],[(0,.5),(1,.5)],[(0,.5),(1,.5)]]]:
 point,se,js=jackknife(groups)
 for i,v in enumerate(js):
  x=[x for j,g in enumerate(groups) if j!=i for x in g];assert abs(v-roc_auc_score([x[0] for x in x],[x[1] for x in x]))<1e-12
p=OUT/'historical_detector_uncertainty.csv'
with p.open() as f:rows=list(csv.DictReader(f))
checks=0
for row in rows:
 path=DATA/row['source'];score='score_stego' if row['model'].startswith('SRM') else 'prob_stego'
 with path.open() as f:records=[x for x in csv.DictReader(f) if x.get('split','test')=='test']
 grouped=defaultdict(list)
 for r in records:
  i=Path(r.get('filename',r.get('path',r.get('image_path','')))).stem;grouped[BOSS[i]['sha256']].append((int(r['label']),float(r[score])))
 groups=list(grouped.values());point,se,js=jackknife(groups);assert abs(point-float(row['auc_recomputed']))<1e-12
 for j in np.linspace(0,len(groups)-1,10,dtype=int):
  x=[x for i,g in enumerate(groups) if i!=j for x in g];direct=roc_auc_score([x[0] for x in x],[x[1] for x in x]);assert abs(js[j]-direct)<1e-12;checks+=1
 row.update(auc_group_jackknife_standard_error=se,auc_primary95_low=max(0.,point-1.959963984540054*se),auc_primary95_high=min(1.,point+1.959963984540054*se),primary_interval_method='cover-content group delete-one jackknife normal approximation, applied uniformly to all panels',bootstrap_interval_role='secondary diagnostic; percentile bias flag retained where present')
with p.open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'detector_uncertainty_method_check.json').write_text(json.dumps({'primary_method':'cover-content group delete-one jackknife normal approximation for all nine fixed-model panels','secondary_method':'original paired percentile bootstrap retained as diagnostic','real_group_deletions_verified_against_direct_auc':checks,'toy_cases_verified':2,'max_identity_tolerance':1e-12,'interpretation':'Approximate fixed-model test-cohort uncertainty. Not population warden bounds or training-seed uncertainty. Near-chance score ties and earlier split leakage remain separately reported.'},indent=2))
print('Checked uniform group-jackknife intervals for nine panels against 90 direct deletions.')
