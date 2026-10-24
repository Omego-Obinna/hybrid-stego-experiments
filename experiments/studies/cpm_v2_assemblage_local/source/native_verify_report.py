import csv,json
from pathlib import Path
import joblib,numpy as np
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from native_common import *
def table(path,rows):
    keys=sorted({k for r in rows for k in r})
    with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def generative(cfg):
    rows=[json.loads(l) for l in (ROOT/'generation_results.jsonl').open()];cases=json.loads((ROOT/'cases.json').read_text());assert len(rows)==2*len(cases)
    for r in rows:
        assert sha(r['cover_path'])==r['cover_sha256'] and sha(r['message_path'])==r['message_sha256'] and sha(r['key_file'])==r['key_sha256']
        for which in ['stego','benign']:
            if r.get(which+'_path'):assert sha(r[which+'_path'])==r[which+'_sha256']
        if r['exact_recovery']:
            assert (Path(r['folder'])/'recovered.bin').read_bytes()==Path(r['message_path']).read_bytes()
            if r['method']=='pulsar':assert (Path(r['folder'])/'recovered_padded.bin').read_bytes()==Path(r['message_path']).read_bytes()+bytes(r['native_capacity']-r['message_bytes'])
    test=json.loads((ROOT/'test_predictions.json').read_text());fc=json.loads((ROOT/'feature_completion.json').read_text());assert sha(ROOT/'feature_dataset.npz')==fc['feature_sha256'];x=np.load(ROOT/'feature_dataset.npz')['features'];checked=0
    for p,h in test['prediction_hashes'].items():assert sha(p)==h
    for item in test['checkpoint_checks']:
        saved=np.load(item['prediction_path'],allow_pickle=False);actual=joblib.load(item['model_path']).predict_proba(x[saved['feature_indices']])[:,1];assert np.allclose(actual,saved['scores'],rtol=0,atol=1e-12);checked+=1
    summary=[]
    for size in cfg['payload_bytes']:
        for method in ['cpm','pulsar']:
            rr=[r for r in rows if r['method']==method and r['message_bytes']==size];valid=[r for r in rr if r['exact_recovery']];r=dict(method=method,message_bytes=size,attempted=len(rr),exact=len(valid),failed=len(rr)-len(valid),capacity_limited=sum(a['status']=='capacity_limited' for a in rr),incorrect_decoded_payloads=sum((a.get('plaintext_error_bits') or 0)>0 for a in rr),pixel_deviation_images=sum(a.get('pixels_over_one',0)>0 for a in rr))
            for field in ['carrier_bytes','wire_bytes','sender_wall_sec','receiver_wall_sec','native_capacity','core_embedding_sec','psnr_db','ssim']:
                v=[a[field] for a in valid if a.get(field) is not None];r[field+'_mean']=float(np.mean(v)) if v else None
            summary.append(r)
    metrics=[];rng=np.random.default_rng(cfg['bootstrap']['seed']);testgroups=sorted({c['group_id'] for c in cases if c['partition']=='test'});idx={v:i for i,v in enumerate(testgroups)};w=rng.multinomial(len(testgroups),np.repeat(1/len(testgroups),len(testgroups)),size=cfg['bootstrap']['draws']);boots={}
    for panel in test['panels']:
        z=np.load(panel['prediction_path']);y=z['labels'];s=z['scores'];cols=np.array([idx[int(i)] for i in z['group_ids']]);b=np.array([roc_auc_score(y,s,sample_weight=a[cols]) for a in w]);lo,hi=np.quantile(b,[.025,.975]);r=dict(panel,auc=float(roc_auc_score(y,s)),auc_ci_low=float(lo),auc_ci_high=float(hi),balanced_accuracy=float(balanced_accuracy_score(y,s>=panel['threshold'])),acceptance_gap=float(np.mean(s[y==1]>=panel['threshold'])-np.mean(s[y==0]>=panel['threshold'])));metrics.append(r);boots[(r['method'],r['message_bytes'],r['family'])]=b
    contrasts=[]
    for n in cfg['payload_bytes']:
        rr=[r for r in metrics if r['message_bytes']==n and r['primary']]
        if len(rr)!=2:continue
        c=next(r for r in rr if r['method']=='cpm');p=next(r for r in rr if r['method']=='pulsar');diff=boots[('cpm',n,c['family'])]-boots[('pulsar',n,p['family'])];lo,hi=np.quantile(diff,[.05/6,1-.05/6]);contrasts.append(dict(message_bytes=n,family=c['family'],cpm_auc_minus_pulsar=c['auc']-p['auc'],adjusted_ci_low=float(lo),adjusted_ci_high=float(hi)))
    table(ROOT/'native_recovery_cost.csv',summary);table(ROOT/'native_detectability.csv',metrics);table(ROOT/'native_auc_contrasts.csv',contrasts)
    save(ROOT/'verified_summary.json',dict(passed=True,checked_at=now(),attempts=len(rows),exact=sum(r['exact_recovery'] for r in rows),selected_checkpoint_predictions_checked=checked,all_native_file_hashes_checked=True,all_exact_recovered_bytes_checked=True,source_scope='native format and benign distribution, finite within-source content classifiers, matched application bytes',evaluable=fc['evaluable']))
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.4),layout='constrained')
    for method,color in [('cpm','#007C83'),('pulsar','#C87229')]:
        rr=sorted([r for r in metrics if r['primary'] and r['method']==method],key=lambda r:r['message_bytes'])
        if rr:
            xx=np.array([r['message_bytes'] for r in rr]);yy=np.array([r['auc'] for r in rr]);lo=np.array([r['auc_ci_low'] for r in rr]);hi=np.array([r['auc_ci_high'] for r in rr]);axes[0].errorbar(xx,yy,yerr=[np.maximum(yy-lo,0),np.maximum(hi-yy,0)],color=color,marker='o',capsize=3,label=method.upper())
        rr=[r for r in summary if r['method']==method];axes[1].plot([r['message_bytes'] for r in rr],[r['wire_bytes_mean'] for r in rr],marker='o',color=color,label=method.upper())
    axes[0].axhline(.5,color='.5',ls='--',lw=.8);axes[0].set(ylabel='Native-source detection AUC',ylim=(0,1.025));axes[1].set(ylabel='Transmitted native bytes',yscale='log')
    for ax in axes:ax.set_xlabel('Application payload (bytes)');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False);ax.legend()
    fig.savefig(ROOT/'native_comparison.png',dpi=300);fig.savefig(ROOT/'native_comparison.svg');plt.close(fig)
    lines=['# Native generative comparison: Pulsar and CPM','',f'All {checked} selected checkpoint prediction sets passed independent reproduction; native image hashes and exact recovered byte strings were checked.','',
      'Fixed panel: 200 new source groups (100 training / 40 validation / 60 test) at 128, 256 and 384 application bytes. Each group pairs a BOSSBase cover for CPM with a native Google church DDPM seed for Pulsar. The common quantity is application bytes, not bpp or original-cover PSNR. CPM uses 512×512 grayscale PGM with three authenticated frames. Pulsar keeps 256×256 RGB 16-bit PNG, its native 50-step sampler and original region ECC.','',
      'Each detector is trained against the appropriate benign source: original natural covers for CPM and the same unmodified sampler output for Pulsar, serialized identically to the stego images. Fixed native-precision content features are shared; no secret key, mask, seed, source-cover pair or verification result is supplied as a feature. Family and tuning choices use validation only. A common family is selected across the two sources at each payload. This is a bounded content test, not a full-view proof or a reproduction of published detector results.','',
      '| Method | Bytes | Attempts | Exact | Capacity failures | Other failures | Mean native transmitted bytes |','|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["method"]} | {r["message_bytes"]} | {r["attempted"]} | {r["exact"]} | {r["capacity_limited"]} | {r["failed"]-r["capacity_limited"]} | {r["wire_bytes_mean"]} |')
    lines+=['','| Method | Bytes | Primary family | Test groups | AUC (95% interval) |','|---|---:|---|---:|---|']
    for r in metrics:
        if r['primary']:lines.append(f'| {r["method"]} | {r["message_bytes"]} | {r["family"]} | {r["test_groups"]} | {r["auc"]:.4f} [{r["auc_ci_low"]:.4f}, {r["auc_ci_high"]:.4f}] |')
    lines+=['','All capacity and native decoder failures remain in all-attempt counts. Detector panels condition symmetrically on both native methods recovering; insufficient fixed minimum counts make a panel not evaluable. No failed group is replaced. Primary AUC contrasts have three-payload adjusted intervals; secondary families are retained in native_detectability.csv.','',
      'Raw byte cost compares native serialized services, which differ: CPM includes two public objects, JSON/base64 frames, authentication and freshness context; Pulsar supplies a native image primitive and shared model/key/seed context. Runtime separates sampler/region setup, encoding, decoding and process/model-loading overhead in raw per-case metrics. These do not establish equivalent protocol protection or universal efficiency superiority.','',
      'Original-cover PSNR/SSIM apply only to CPM in native_recovery_cost.csv; they are not assigned to synthetic Pulsar carriers. Cross-channel CP-UNL is not defined for this single-carrier interface and is not reported as zero. The earlier CPM full-public-view study remains separate.','',
      '![Native comparison](native_comparison.png)']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
def distributed(cfg):
    from native_distributed import loss_result
    cases=json.loads((ROOT/'cases.json').read_text());rows=[json.loads(l) for l in (ROOT/'system_results.jsonl').open()];lookup={(r['case_id'],r['method']):r for r in rows};traces=[json.loads(l) for l in (ROOT/'erasure_traces.jsonl').open()];assert len(traces)==len(cases)*40
    for r in rows:
        assert sha(r['message_path'])==r['message_sha256']
        if r['method']=='cpm':
            if r['exact_recovery']:assert (Path(r['folder'])/'recovered.bin').read_bytes()==Path(r['message_path']).read_bytes();assert sha(r['stego_path'])==r['stego_sha256']
        else:
            for p in r['packets']:
                assert sha(p['input_packet_path'])==p['input_packet_sha256']
                if p['sent']:assert sha(p['image_path'])==p['image_sha256']
                if p['decoded_path']:assert sha(p['decoded_path'])==p['decoded_sha256']
    for t in traces:assert loss_result(cases[t['case_id']],t['method'],lookup[(t['case_id'],t['method'])],t['loss_probability'],t['loss_seed'])==t
    summary=[];cost=[];rng=np.random.default_rng(61020);weights=rng.multinomial(10,np.repeat(.1,10),size=3000)
    for n in cfg['payload_bytes']:
        for m in ['cpm','assemblage']:
            rr=[r for r in rows if r['method']==m and r['message_bytes']==n];out=dict(method=m,message_bytes=n,message_groups=len(rr))
            for k in ['wire_bytes','sender_wall_sec','receiver_wall_sec','native_wall_sec','offline_sec','send_sec','setup_state_bytes','setup_hash_bytes','transmitted_packets','encoded_packets']:
                a=[r[k] for r in rr if r.get(k) is not None];out[k+'_mean']=float(np.mean(a)) if a else None
            cost.append(out)
            for rate in cfg['assemblage']['loss_fractions']:
                tt=[t for t in traces if t['method']==m and t['message_bytes']==n and t['loss_probability']==rate];v=np.array([np.mean([t['exact_recovery'] for t in tt if cases[t['case_id']]['group_id']==g]) for g in range(10)]);b=weights@v/10;lo,hi=np.quantile(b,[.025,.975]);summary.append(dict(method=m,message_bytes=n,loss_probability=rate,unique_message_groups=10,recorded_traces=len(tt),exact_recovery_rate=float(v.mean()),ci_low=float(lo),ci_high=float(hi),incorrect_messages=sum(t['incorrect_recovered_message'] for t in tt),decode_exception_traces=sum(t['native_decode_exception'] for t in tt),transmitted_objects=sum(t['transmitted_objects'] for t in tt),dropped_objects=sum(t['dropped_objects'] for t in tt)))
    table(ROOT/'delivery_summary.csv',summary);table(ROOT/'native_system_cost.csv',cost);save(ROOT/'verified_summary.json',dict(passed=True,checked_at=now(),message_groups=len(cases),erasure_traces_checked=len(traces),all_saved_packet_and_image_hashes_checked=True,all_saved_decoded_packets_used_without_oracle_filtering=True,source_scope='Finite local native-carrier and per-object iid-erasure comparison; repeated loss seeds and zero-loss repetitions are not independent trials'))
    fig,axes=plt.subplots(1,3,figsize=(10,3.1),layout='constrained')
    for ax,n in zip(axes,cfg['payload_bytes']):
        for m,color in [('cpm','#007C83'),('assemblage','#C87229')]:
            rr=[r for r in summary if r['method']==m and r['message_bytes']==n];x=np.array([r['loss_probability'] for r in rr]);y=np.array([r['exact_recovery_rate'] for r in rr]);lo=np.array([r['ci_low'] for r in rr]);hi=np.array([r['ci_high'] for r in rr]);ax.errorbar(x,y,yerr=[np.maximum(y-lo,0),np.maximum(hi-y,0)],marker='o',capsize=2,label=m,color=color)
        ax.set(title=f'{n:,} application bytes',xlabel='Per-object erasure probability',ylim=(-.025,1.025));ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Exact message recovery');axes[-1].legend();fig.savefig(ROOT/'distributed_delivery.png',dpi=300);fig.savefig(ROOT/'distributed_delivery.svg');plt.close(fig)
    lines=['# Native local distributed-message comparison','',f'Independent saved-packet replay verified all {len(traces)} finite erasure traces. The experiment has ten unique source-cover/message groups per payload, not 1,200 independent messages.','',
      'Published Assemblage (FOCI 2026) uses its exact official Pulsar submodule, native PDQ discovery, Sage region ECC, church generator and RaptorQ packetization with MTU 408 and 50% additional repair packets. CPM uses its unchanged authenticated three-channel protocol. Both carry identical saved application messages of 1,024, 2,048 and 4,096 bytes. No online platform is accessed.','',
      'All native offline candidates, capacity skips, packet encoding failures, sent images, decoder exceptions and incorrect decoded packets are retained. Failed native cases are not replaced. The receiver sees actual decoded packet bytes; incorrect packets are not removed by consulting ground truth. As in upstream receive_raptor, a native decode exception among received carriers aborts the message.','',
      'Loss traces apply the same fixed iid per-object erasure law and seeds to each native system. A CPM object is one authenticated channel frame; an Assemblage object is one native image. Object sizes and counts differ and are reported; these traces are not byte-level network-loss experiments. The native images/frames are generated and decoded once, then their saved outcomes are replayed under fixed erasure patterns. CPM requires all three frames; Assemblage uses its native rateless decoder.','',
      '| Method | Bytes | Erasure probability | Unique groups | Exact recovery (95% interval) | Incorrect messages |','|---|---:|---:|---:|---|---:|']
    for r in summary:lines.append(f'| {r["method"]} | {r["message_bytes"]} | {r["loss_probability"]:.2f} | 10 | {r["exact_recovery_rate"]:.3f} [{r["ci_low"]:.3f}, {r["ci_high"]:.3f}] | {r["incorrect_messages"]} |')
    lines+=['','Intervals cluster the five loss seeds and repeated payload observations by the ten unique source groups. At zero loss all five seeds repeat the same intact native transmission and do not multiply the independent sample size.','',
      'native_system_cost.csv separates transmitted image/frame bytes from pre-shared offline state and hash-bank bytes. Setup, model loading, offline sampling, native send and receive costs are preserved per message. Authentication, replay resistance, rendezvous discovery and rateless loss tolerance are different services; no absent service is assigned zero cost or zero leakage.','',
      'This is a local multi-carrier comparator. It does not reproduce a deployed network, censorship, scheduling anonymity, a multi-receiver multichannel protocol, or a CPM-style CP-UNL interface. The original CPM theorem and full-public-view evidence retain their declared conditions.','',
      '![Distributed delivery](distributed_delivery.png)']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
def main():
    cfg=config();assert not (ROOT/'completion.json').exists();plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'legend.fontsize':9,'svg.fonttype':'none','pdf.fonttype':42})
    if cfg['panel']=='generative':generative(cfg)
    else:distributed(cfg)
    save(ROOT/'completion.json',dict(completed_at=now(),verified=True,report_sha256=sha(ROOT/'RESULTS.md')));status(state='completed',phase='completed',stage='Native comparison completed and independently checked',report_ready=True)
if __name__=='__main__':main()
