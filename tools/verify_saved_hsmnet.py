#!/usr/bin/env python3
"""Reproduce saved HSMNet statistics only: no torch, selection, training or inference."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'experiments/studies/cpm_v2_hsmnet_matched_1200'
REC=ROOT/'records'

def table(n):
    with (REC/n).open(newline='') as f:return list(csv.DictReader(f))

def load(n):return json.loads((REC/n).read_text())

def same(a,b):
    assert np.allclose(a,b,rtol=0,atol=2e-12),(a,b)

def main():
    cfg=load('run_config.json'); selection=load('hsm_selection.json')
    records=[json.loads(s) for s in (REC/'image_results.jsonl').read_text().splitlines()]
    fresh=[json.loads(s) for s in (REC/'fresh_results.jsonl').read_text().splitlines()]
    covers=table('covers.csv')
    part={int(c['cover_id']):c['partition'] for c in covers}
    assert len(records)==18000 and len(fresh)==450 and len(part)==1200
    assert {p:sum(v==p for v in part.values()) for p in ['train','validation','test']}==dict(train=700,validation=200,test=300)
    assert len({(r['method'],r['cover_id'],r['payload_bpp']) for r in records})==18000
    blocks={}
    for r in records:
        assert part[r['cover_id']]==r['partition']
        blocks.setdefault((r['cover_id'],r['payload_bpp']),{})[r['method']]=r
    blocks={k:g for k,g in blocks.items() if set(g)==set(cfg['methods']) and all(r['valid_image'] for r in g.values())}
    counts={str(pay):{p:sum(k[1]==pay and g['cpm']['partition']==p for k,g in blocks.items())
        for p in ['train','validation','test']} for pay in cfg['payloads']}
    for dd in counts.values():
        for p,n in cfg['detector_minimum_complete_counts'].items():assert dd[p]>=n
    attempts={}
    verified=load('verified_summary.json')
    for method in cfg['methods']:
        rr=[r for r in records if r['method']==method]; ff=[r for r in fresh if r['method']==method]
        assert len(rr)==3600 and len(ff)==90
        deviations=sum(r.get('pixels_over_one',0)>0 for r in rr)
        assert deviations==verified['pixel_deviation_image_counts_by_method'][method]
        assert sum(r.get('pixels_over_one',0)>0 for r in ff)==verified['fresh_pixel_deviation_counts_by_method'][method]
        attempts[method]=dict(attempted=len(rr),valid_images=sum(r['valid_image'] for r in rr),
            exact_confirmed=sum(r.get('exact_recovery') is True for r in rr),
            exact_not_independently_established=sum(r.get('exact_recovery') is None for r in rr),
            pixel_deviation_images=deviations,fresh_attempted=len(ff),fresh_exact=sum(r['exact_recovery'] for r in ff))
    assert attempts['cpm']['exact_confirmed']==3595 and sum(r['exact_recovery'] for r in fresh)==446
    ids=sorted(i for i,p in part.items() if p=='test'); index={i:j for j,i in enumerate(ids)}
    W=np.random.default_rng(cfg['bootstrap']['seed']).multinomial(len(ids),np.repeat(1/len(ids),len(ids)),size=cfg['bootstrap']['draws'])
    boots={}; aucs={}; ensembles=0
    for r in table('detectability.csv'):
        pay=float(r['payload_bpp']);m=r['method']
        with np.load(ROOT/'predictions'/f'hsm_{m}_{pay}_ensemble.npz',allow_pickle=False) as z:
            y=z['labels'];s=z['scores'].astype(np.float64);g=z['cover_ids']
            saved_scores=z['scores'].copy()
        common=sorted(k[0] for k in blocks if k[1]==pay and part[k[0]]=='test')
        assert sorted(set(g))==common and len(g)==2*len(common)
        assert all(np.array_equal(np.sort(y[g==i]),[0,1]) for i in common)
        order=np.argsort(s,kind='stable'); ordered=s[order]
        starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]; positive=y[order]==1
        weighted=W[:,[index[int(i)] for i in g[order]]]
        pos=np.add.reduceat(weighted*positive,starts,axis=1)
        neg=np.add.reduceat(weighted*(~positive),starts,axis=1)
        draws=np.sum(pos*(np.cumsum(neg,axis=1)-.5*neg),axis=1)/(pos.sum(1)*neg.sum(1))
        pp=np.add.reduceat(positive.astype(float),starts);nn=np.add.reduceat((~positive).astype(float),starts)
        auc=float(np.sum(pp*(np.cumsum(nn)-.5*nn))/pp.sum()/nn.sum())
        lo,hi=np.quantile(draws,[.025,.975]); accepted=s>=float(r['threshold'])
        gap=float(accepted[y==1].mean()-accepted[y==0].mean())
        same([auc,lo,hi,(1+gap)/2,gap],[float(r[n]) for n in ['auc','auc_ci_low','auc_ci_high','balanced_accuracy','acceptance_gap']])
        assert int(r['test_covers'])==len(common)
        raw=[]
        for seed in cfg['hsmnet']['seeds']:
            with np.load(ROOT/'predictions'/f'hsm_{pay}_{seed}.npz',allow_pickle=False) as z:
                raw.append({k:z[k] for k in z.files})
        for k in ['labels','cover_ids','methods']:assert np.array_equal(raw[0][k],raw[1][k])
        pick=raw[0]['methods']==m
        orientation=next(p['orientation'] for p in selection['panels'] if p['payload_bpp']==pay)
        avg=np.mean([z['scores'] for z in raw],axis=0)*orientation
        assert np.array_equal(avg[pick],saved_scores)
        assert np.array_equal(raw[0]['labels'][pick],y) and np.array_equal(raw[0]['cover_ids'][pick],g)
        boots[m,pay]=draws; aucs[m,pay]=auc; ensembles+=1
    assert ensembles==15
    contrasts=table('auc_paired_contrasts.csv'); assert len(contrasts)==12
    for r in contrasts:
        pay=float(r['payload_bpp']);m=r['baseline'];delta=aucs['cpm',pay]-aucs[m,pay]
        lo,hi=np.quantile(boots['cpm',pay]-boots[m,pay],[.05/24,1-.05/24])
        same([delta,lo,hi],[float(r[k]) for k in ['auc_cpm_minus_baseline','adjusted_ci_low','adjusted_ci_high']])
        assert int(r['family_comparisons'])==12
    fidelity=0
    for r in table('paired_fidelity.csv'):
        pay=float(r['payload_bpp']);m=r['method']
        rr=[g[m] for k,g in blocks.items() if k[1]==pay and part[k[0]]=='test']
        ix=[index[x['cover_id']] for x in rr]
        assert int(r['test_covers'])==len(rr) and int(r['all_attempts'])==300
        assert int(r['valid_images'])==sum(x['method']==m and x['payload_bpp']==pay and x['partition']=='test' and x['valid_image'] for x in records)
        for f in ['psnr_db','ssim','modified_pixels']:
            a=np.asarray([x[f] for x in rr]);d=W[:,ix]@a/W[:,ix].sum(1)
            lo,hi=np.quantile(d,[.025,.975])
            same([a.mean(),lo,hi],[float(r[k]) for k in [f,f+'_ci_low',f+'_ci_high']]);fidelity+=1
    fcontrasts=0
    for r in table('fidelity_contrasts.csv'):
        pay=float(r['payload_bpp']);m=r['baseline'];f=r['metric']
        rr=[g for k,g in blocks.items() if k[1]==pay and part[k[0]]=='test']
        ix=[index[g['cpm']['cover_id']] for g in rr];a=np.asarray([g['cpm'][f]-g[m][f] for g in rr])
        d=W[:,ix]@a/W[:,ix].sum(1);lo,hi=np.quantile(d,[.05/24,1-.05/24])
        same([a.mean(),lo,hi],[float(r[k]) for k in ['cpm_minus_baseline','adjusted_ci_low','adjusted_ci_high']]);fcontrasts+=1
    timing=table('fresh_recovery_runtime.csv')
    for r in timing:
        rr=[x for x in fresh if x['method']==r['method'] and x['payload_bpp']==float(r['payload_bpp'])]
        valid=[x for x in rr if x['exact_recovery']]
        assert int(r['attempted'])==len(rr)==30 and int(r['exact'])==len(valid) and int(r['failed'])==len(rr)-len(valid)
        for f in ['core_sec','sender_wall_sec','receiver_core_sec','receiver_wall_sec','carrier_bytes','wire_bytes']:
            a=[x['cost_sec']+x['stc_sec'] if f=='core_sec' and r['method']!='cpm' else x[f] for x in valid if x.get(f) is not None]
            if a:same([np.mean(a),np.median(a)],[float(r[f+'_mean']),float(r[f+'_median'])])
            else:assert r[f+'_mean']==r[f+'_median']==''
    print(json.dumps(dict(passed=True,auc_and_interval_panels=15,adjusted_auc_contrasts=12,
        saved_ensembles_reproduced=15,raw_seed_prediction_files=6,fidelity_mean_interval_checks=fidelity,
        adjusted_fidelity_contrasts=fcontrasts,recovery_timing_rows=len(timing),
        common_valid_counts=counts,all_attempts=attempts,checkpoint_forward_passes_executed=0,
        training_or_selection_executed=False),indent=2))

if __name__=='__main__':main()
