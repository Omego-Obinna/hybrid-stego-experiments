import csv,json,math
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from image_common import *
def table(p,rows):
    keys=sorted({k for r in rows for k in r})
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def main():
    cfg=config();verified=json.loads((ROOT/'verified_summary.json').read_text());assert verified['passed']
    assert not (ROOT/'completion.json').exists()
    rows=records();blocks=complete_blocks(rows);panels=json.loads((ROOT/'test_predictions.json').read_text())['panels'];covers=list(csv.DictReader((ROOT/'covers.csv').open()));testids=sorted(int(c['cover_id']) for c in covers if c['partition']=='test');index={v:i for i,v in enumerate(testids)}
    rng=np.random.default_rng(cfg['bootstrap']['seed']);weights=rng.multinomial(len(testids),np.repeat(1/len(testids),len(testids)),size=cfg['bootstrap']['draws']);auc_boot={};metrics=[]
    for p in panels:
        z=np.load(p['prediction_path'],allow_pickle=False);ids=z['cover_ids'];y=z['labels'];s=z['scores'];pred=s>=p['threshold'];col=np.array([index[int(i)] for i in ids]);boot=np.array([roc_auc_score(y,s,sample_weight=w[col]) for w in weights]);auc_boot[(p['detector'],p['method'],p['payload_bpp'])]=boot
        low,hi=np.quantile(boot,[.025,.975]);r=dict(p,auc=float(roc_auc_score(y,s)),auc_ci_low=float(low),auc_ci_high=float(hi),balanced_accuracy=float(balanced_accuracy_score(y,pred)),acceptance_gap=float(np.mean(pred[y==1])-np.mean(pred[y==0])));metrics.append(r)
    contrasts=[]
    for p in metrics:
        if p['method']=='cpm':continue
        c=next(r for r in metrics if r['method']=='cpm' and r['payload_bpp']==p['payload_bpp'] and r['detector']==p['detector']);d=auc_boot[(p['detector'],'cpm',p['payload_bpp'])]-auc_boot[(p['detector'],p['method'],p['payload_bpp'])];lo,hi=np.quantile(d,[.05/24,1-.05/24]);contrasts.append(dict(detector=p['detector'],baseline=p['method'],payload_bpp=p['payload_bpp'],auc_cpm_minus_baseline=c['auc']-p['auc'],adjusted_ci_low=float(lo),adjusted_ci_high=float(hi),family_comparisons=12))
    fidelity=[];fidelity_diffs=[]
    for pay in cfg['payloads']:
        for m in cfg['methods']:
            rr=[v[m] for k,v in blocks.items() if k[1]==pay and v[m]['partition']=='test'];ids=[index[r['cover_id']] for r in rr]
            out=dict(method=m,payload_bpp=pay,test_covers=len(rr),all_attempts=sum(r['method']==m and r['payload_bpp']==pay and r['partition']=='test' for r in rows),valid_images=sum(r['method']==m and r['payload_bpp']==pay and r['partition']=='test' and r['valid_image'] for r in rows))
            for field in ['psnr_db','ssim','modified_pixels']:
                a=np.array([r[field] for r in rr]);v=weights[:,ids]@a/weights[:,ids].sum(axis=1);lo,hi=np.quantile(v,[.025,.975]);out.update({field:float(a.mean()),field+'_ci_low':float(lo),field+'_ci_high':float(hi)})
                if m!='cpm':
                    b=np.array([blocks[(r['cover_id'],pay)]['cpm'][field]-r[field] for r in rr]);dist=weights[:,ids]@b/weights[:,ids].sum(axis=1);lo,hi=np.quantile(dist,[.05/24,1-.05/24]);fidelity_diffs.append(dict(baseline=m,payload_bpp=pay,metric=field,cpm_minus_baseline=float(b.mean()),adjusted_ci_low=float(lo),adjusted_ci_high=float(hi)))
            fidelity.append(out)
    fresh=[json.loads(x) for x in (ROOT/'fresh_results.jsonl').open()];fresh_summary=[]
    for pay in cfg['payloads']:
        for m in cfg['methods']:
            rr=[r for r in fresh if r['method']==m and r['payload_bpp']==pay];valid=[r for r in rr if r['exact_recovery']];out=dict(method=m,payload_bpp=pay,attempted=len(rr),exact=len(valid),failed=len(rr)-len(valid),wrong_recovered_payloads=sum(r.get('plaintext_error_bits',0)>0 for r in rr))
            for field in ['core_sec','sender_wall_sec','receiver_core_sec','receiver_wall_sec','carrier_bytes','wire_bytes']:
                a=[r[field] for r in valid if r.get(field) is not None];out[field+'_mean']=float(np.mean(a)) if a else None;out[field+'_median']=float(np.median(a)) if a else None
            fresh_summary.append(out)
    table(ROOT/'detectability.csv',metrics);table(ROOT/'auc_paired_contrasts.csv',contrasts);table(ROOT/'paired_fidelity.csv',fidelity);table(ROOT/'fidelity_contrasts.csv',fidelity_diffs);table(ROOT/'fresh_recovery_runtime.csv',fresh_summary)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'legend.fontsize':8,'xtick.labelsize':9,'ytick.labelsize':9,'svg.fonttype':'none','pdf.fonttype':42})
    colors=dict(cpm='#007C83',suniward='#E69F00',mipod='#6B5B95',wow='#0072B2',hill='#A15A5A');fig,one_ax=plt.subplots(figsize=(6.8,3.7),layout='constrained');axes=[one_ax]
    for ax,det in zip(axes,dict.fromkeys(p['detector'] for p in metrics)):
        for mi,m in enumerate(cfg['methods']):
            pp=sorted([p for p in metrics if p['method']==m and p['detector']==det],key=lambda p:p['payload_bpp']);x=np.array([p['payload_bpp'] for p in pp]);y=np.array([p['auc'] for p in pp]);lo=np.array([p['auc_ci_low'] for p in pp]);hi=np.array([p['auc_ci_high'] for p in pp]);xx=x+(mi-2)*.003;ax.vlines(xx,lo,hi,color=colors[m],lw=1.2);ax.hlines(lo,xx-.0015,xx+.0015,color=colors[m]);ax.hlines(hi,xx-.0015,xx+.0015,color=colors[m]);ax.plot(xx,y,label=m.upper() if m=='cpm' else m,marker='o',markersize=4,lw=1.4,color=colors[m])
        ax.axhline(.5,color='.5',ls='--',lw=.8);ax.set(xlabel='Application payload (bits/pixel)',ylabel='Held-out detection AUC',title=det,xticks=[.1,.2,.4],ylim=(0,1.025));ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[-1].legend(loc='best');fig.savefig(ROOT/'matched_detectability.png',dpi=300);fig.savefig(ROOT/'matched_detectability.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for m in cfg['methods']:
        pp=sorted([p for p in fidelity if p['method']==m],key=lambda p:p['payload_bpp'])
        for ax,field,label in zip(axes,['psnr_db','ssim'],['PSNR (dB)','SSIM']):ax.plot([r['payload_bpp'] for r in pp],[r[field] for r in pp],marker='o',label=m,color=colors[m]);ax.set(xlabel='Application payload (bits/pixel)',ylabel=label,xticks=[.1,.2,.4]);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[-1].legend();fig.savefig(ROOT/'matched_fidelity.png',dpi=300);fig.savefig(ROOT/'matched_fidelity.svg');plt.close(fig)
    lines=['# HSMNet matched image comparison','',f'Independent saved-artifact verification passed for {verified["selected_checkpoint_predictions_checked"]} selected checkpoints.','',
      'Fixed BOSSBase panel: 700 training / 200 validation / 300 test covers; native 512 × 512 grayscale. Three application payloads: 0.10, 0.20 and 0.40 bpp. All source versions, failed attempts, symmetric common-valid exclusions and model choices are retained.','',
      'Only HSMNet is evaluated in this continuation, using the same equal-method mixture observer for all methods at each payload; the original SRM job remains paused. HSMNet is the published architecture with a declared bounded training budget, not a reproduction of its original paper scores. Scores describe these image-content wardens and do not estimate IND-STEGO-MC or CP-UNL security bounds.','',
      'The classical images are reused from the original study and labelled by their recorded implementations. In particular, the local MiPOD variance estimator is an approximation; these rows must not be described as certified reference-code results. Historical recovery is originally reported, whereas the separate 450-attempt fresh panel has saved known messages and independent native extraction.','',
      '## Held-out detection','', '| Detector | Method | bpp | Covers | AUC (95% interval) | Balanced accuracy |','|---|---|---:|---:|---|---:|']
    for p in metrics:lines.append(f'| {p["detector"]} | {p["method"]} | {p["payload_bpp"]:.2f} | {p["test_covers"]} | {p["auc"]:.4f} [{p["auc_ci_low"]:.4f}, {p["auc_ci_high"]:.4f}] | {p["balanced_accuracy"]:.4f} |')
    lines+=['','Paired CPM-minus-baseline AUC differences and family-adjusted intervals are in `auc_paired_contrasts.csv`; positive differences indicate easier detection of CPM by the tested observer. Lower AUC is favourable only relative to the fixed, validation-oriented observer. No universal superiority claim is inferred.','',
      '## Fidelity, recovery and cost','', 'Fidelity uses identical test covers and common-valid cover/payload blocks for every method. `paired_fidelity.csv` and `fidelity_contrasts.csv` include PSNR, SSIM and modified pixels with cover-cluster uncertainty. All attempts remain in the separate accounting.','',
      f'Fresh sequential development panel: {len(fresh)} attempts, {sum(r["exact_recovery"] for r in fresh)} exact recoveries. `fresh_recovery_runtime.csv` separates core embedding, process wall time, extraction, carrier bytes and actual CPM serialized frame bytes. It compares 30 fixed development covers across all five methods and payloads.','',
      'The single-carrier classical primitives do not implement CPM authentication, replay handling or public parameter channels. Missing protocol-wire and cross-channel linkage quantities are not applicable, never zero. Timing includes local desktop contention and is an engineering observation. Existing completed full-view CP-UNL results stay separate.','',
      'Generative and distributed-carrier comparisons are a separate native-format panel; they are not mixed into this BOSSBase AUC table.','',
      '![Matched detectability](matched_detectability.png)','', '![Matched fidelity](matched_fidelity.png)']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save(ROOT/'completion.json',dict(completed_at=now(),verified=True,report_sha256=sha(ROOT/'RESULTS.md'),selected_checkpoints_checked=6,detectability_panels=15,fresh_attempts=450))
    status(state='completed',phase='completed',stage='HSMNet comparison completed; final independent report review pending; SRM remains paused',report_ready=True,detectability_panels=15,checkpoints_checked=6)
if __name__=='__main__':main()
