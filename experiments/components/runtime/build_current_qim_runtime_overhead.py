#!/usr/bin/env python3
"""Create current QIM runtime/overhead tables from existing results only.

This script never runs the embedder or decoder and never modifies stego images.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path

IMG_EXTS={'.png','.pgm','.bmp','.tif','.tiff','.jpg','.jpeg'}

class InputError(RuntimeError):
    pass

@dataclass(frozen=True)
class Layout:
    epoch_index_bytes:int=8
    epoch_auth_bytes:int=32
    nonce_bytes:int=16
    mac_bytes:int=32
    channel_id_bytes:int=1
    version_flags_bytes:int=1
    length_field_bytes:int=4
    heartbeat_cadence_epochs:int=10
    network_overhead_per_frame:int=0

    @property
    def frame_fixed_bytes(self):
        return (self.epoch_index_bytes+self.epoch_auth_bytes+self.nonce_bytes+
                self.channel_id_bytes+self.version_flags_bytes+self.length_field_bytes)
    @property
    def fixed_control_bytes(self):
        return 3*self.frame_fixed_bytes+self.mac_bytes
    @property
    def heartbeat_amortized_bytes(self):
        if self.heartbeat_cadence_epochs<=0:
            return 0.0
        return 3*(self.epoch_index_bytes+self.epoch_auth_bytes)/self.heartbeat_cadence_epochs

def parse_pm(v):
    t=str(v).strip().replace('$','').replace('\\pm','±').replace('+/-','±').replace(' ','')
    m=re.fullmatch(r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?:±([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?))?',t)
    if not m:
        raise InputError(f'Cannot parse mean±std value: {v!r}')
    return float(m.group(1)), float(m.group(2) or 0.0)

def parse_payload_dir(v):
    if '=' not in v:
        raise InputError(f'Expected PAYLOAD=DIR, got {v}')
    p,d=v.split('=',1)
    d=Path(d).expanduser().resolve()
    if not d.is_dir():
        raise InputError(f'Stego directory does not exist: {d}')
    return float(p),d

def read_lines(path):
    path=Path(path).expanduser().resolve()
    if not path.is_file():
        raise InputError(f'File not found: {path}')
    vals=[x.strip() for x in path.read_text(encoding='utf-8',errors='replace').splitlines() if x.strip()]
    if not vals:
        raise InputError(f'No non-empty lines in {path}')
    return vals

def images_in(path,max_files):
    vals=sorted(x for x in path.rglob('*') if x.is_file() and x.suffix.lower() in IMG_EXTS)
    return vals[:max_files] if max_files>0 else vals

def load_summary(path):
    import pandas as pd
    path=Path(path).expanduser().resolve()
    df=pd.read_csv(path)
    req=['Payload (bpp)','Images','Embed time (s)','Extract time (s)','RSS memory (MB)','Peak RSS (MB)','GPU allocated (MB)','GPU reserved (MB)']
    missing=[c for c in req if c not in df.columns]
    if missing:
        raise InputError(f'Missing summary columns: {missing}')
    rows=[]
    for _,r in df.iterrows():
        em,es=parse_pm(r['Embed time (s)']); xm,xs=parse_pm(r['Extract time (s)'])
        rm,rs=parse_pm(r['RSS memory (MB)']); pm,ps=parse_pm(r['Peak RSS (MB)'])
        am,as_=parse_pm(r['GPU allocated (MB)']); gm,gs=parse_pm(r['GPU reserved (MB)'])
        rows.append(dict(payload_bpp=float(r['Payload (bpp)']),images=int(r['Images']),
            embed_mean_s=em,embed_std_s=es,extract_mean_s=xm,extract_std_s=xs,
            rss_mean_mb=rm,rss_std_mb=rs,peak_rss_mean_mb=pm,peak_rss_std_mb=ps,
            gpu_alloc_mean_mb=am,gpu_alloc_std_mb=as_,gpu_reserved_mean_mb=gm,gpu_reserved_std_mb=gs))
    return pd.DataFrame(rows).sort_values('payload_bpp')

def discover_logs(roots):
    out=[]
    for item in roots:
        p=Path(item).expanduser().resolve()
        if p.is_file() and p.suffix.lower()=='.csv': out.append(p)
        elif p.is_dir(): out.extend(p.rglob('embed_log_ours_qim.csv'))
    seen=set(); uniq=[]
    for p in out:
        if p not in seen: uniq.append(p); seen.add(p)
    return uniq

def load_raw(paths):
    import pandas as pd
    frames=[]
    wanted=['run_id','image_id','payload_bpp','embed_time_sec','extract_time_sec','process_rss_mb','peak_rss_mb','gpu_mem_allocated_mb','gpu_mem_reserved_mb','success_extract','ber','ok']
    for p in paths:
        try:
            cols=pd.read_csv(p,nrows=0).columns.tolist()
            use=[c for c in wanted if c in cols]
            if not {'payload_bpp','embed_time_sec','extract_time_sec'}.issubset(use):
                print(f'[WARN] skipping {p}: required columns absent'); continue
            d=pd.read_csv(p,usecols=use,low_memory=False)
            d['source_log']=str(p); frames.append(d)
        except Exception as e:
            print(f'[WARN] cannot read {p}: {e}')
    if not frames: return None
    raw=pd.concat(frames,ignore_index=True)
    for c in ['payload_bpp','embed_time_sec','extract_time_sec','process_rss_mb','peak_rss_mb','gpu_mem_allocated_mb','gpu_mem_reserved_mb','ber']:
        if c in raw: raw[c]=pd.to_numeric(raw[c],errors='coerce')
    if 'ok' in raw:
        ok=raw['ok'].astype(str).str.lower(); raw=raw[ok.isin(['true','1','yes'])|raw['ok'].isna()]
    keys=[c for c in ['run_id','image_id','payload_bpp'] if c in raw]
    if keys: raw=raw.drop_duplicates(keys,keep='last')
    return raw

def summarize_raw(raw):
    import pandas as pd
    rows=[]
    for payload,g in raw.groupby('payload_bpp'):
        row={'payload_bpp':float(payload),'raw_images':len(g)}
        for col,prefix in [('embed_time_sec','embed'),('extract_time_sec','extract'),('process_rss_mb','rss'),('peak_rss_mb','peak_rss')]:
            if col not in g: continue
            x=g[col].dropna().astype(float)
            if x.empty: continue
            row.update({f'{prefix}_median':x.median(),f'{prefix}_p95':x.quantile(.95),f'{prefix}_min':x.min(),f'{prefix}_max':x.max()})
        if 'ber' in g:
            x=g['ber'].dropna().astype(float)
            if not x.empty:
                row['ber_mean']=x.mean(); row['exact_recovery_rate']=(x==0).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values('payload_bpp')

def build_overhead(specs,g1,g2,layout,w,h,max_files):
    import pandas as pd
    rows=[]
    for payload,d in specs:
        imgs=images_in(d,max_files)
        if not imgs: raise InputError(f'No images in {d}')
        secret=math.ceil(payload*w*h/8)
        for i,img in enumerate(imgs):
            b1=len(g1[i%len(g1)].encode()); b2=len(g2[i%len(g2)].encode()); bs=img.stat().st_size
            net=3*layout.network_overhead_per_frame
            app=b1+b2+bs+layout.fixed_control_bytes+layout.heartbeat_amortized_bytes
            total=app+net
            rows.append(dict(payload_bpp=payload,epoch=i,secret_bytes=secret,cover_message_bytes=b1+b2,
                stego_object_bytes=bs,fixed_control_bytes=layout.fixed_control_bytes,
                heartbeat_amortized_bytes=layout.heartbeat_amortized_bytes,network_overhead_bytes=net,
                application_total_bytes=app,total_bytes=total,control_ratio=(layout.fixed_control_bytes+layout.heartbeat_amortized_bytes)/app,
                payload_efficiency=secret/total,expansion_ratio=total/secret))
    raw=pd.DataFrame(rows)
    out=[]
    metrics=[c for c in raw.columns if c not in ['payload_bpp','epoch']]
    for p,g in raw.groupby('payload_bpp'):
        r={'payload_bpp':p,'epochs':len(g)}
        for c in metrics:
            r[c+'_mean']=g[c].mean(); r[c+'_std']=g[c].std(ddof=1) if len(g)>1 else 0; r[c+'_median']=g[c].median()
        out.append(r)
    return raw,pd.DataFrame(out).sort_values('payload_bpp')

def write_runtime_tex(summary,raws,path):
    lines=[r'\begin{table*}[!t]',r'\centering',r'\caption{Runtime and memory consumption of Ours-QIM-v3.2-Fused-CF over 10,000 images per payload. Values are mean $\pm$ standard deviation.}',r'\label{tab:qim_v32_runtime}',r'\begin{tabular}{rrrrrr}',r'\toprule',r'Payload & Images & $T_{\mathrm{enc}}$ (s) & $T_{\mathrm{dec}}$ (s) & RSS (MB) & Peak RSS (MB) \\',r'\midrule']
    for _,r in summary.iterrows():
        lines.append(f"{r.payload_bpp:.2f} & {int(r.images)} & {r.embed_mean_s:.3f}$\\pm${r.embed_std_s:.3f} & {r.extract_mean_s:.4f}$\\pm${r.extract_std_s:.4f} & {r.rss_mean_mb:.1f}$\\pm${r.rss_std_mb:.1f} & {r.peak_rss_mean_mb:.1f}$\\pm${r.peak_rss_std_mb:.1f} \\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
    if raws is not None and not raws.empty:
        lines += [r'\begin{table*}[!t]',r'\centering',r'\caption{Distributional runtime statistics recomputed from available per-image logs.}',r'\label{tab:qim_v32_runtime_distribution}',r'\begin{tabular}{rrrrrr}',r'\toprule',r'Payload & Raw images & Embed median & Embed P95 & Extract median & Extract P95 \\',r'\midrule']
        for _,r in raws.iterrows():
            lines.append(f"{r.payload_bpp:.2f} & {int(r.raw_images)} & {getattr(r,'embed_median',float('nan')):.3f} & {getattr(r,'embed_p95',float('nan')):.3f} & {getattr(r,'extract_median',float('nan')):.4f} & {getattr(r,'extract_p95',float('nan')):.4f} \\\\")
        lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
    path.write_text('\n'.join(lines),encoding='utf-8')

def write_overhead_tex(s,path):
    lines=[r'\begin{table*}[!t]',r'\centering',r'\caption{Communication accounting from the existing Ours-QIM-v3.2-Fused-CF stego files. The stego-object is protocol traffic rather than control overhead.}',r'\label{tab:qim_v32_overhead}',r'\begin{tabular}{rrrrrrrrr}',r'\toprule',r'Payload & Epochs & Secret (B) & Covers (B) & Stego (B) & Control (B) & Heartbeat (B) & Total (B) & Expansion \\',r'\midrule']
    for _,r in s.iterrows():
        lines.append(f"{r.payload_bpp:.2f} & {int(r.epochs)} & {r.secret_bytes_mean:.0f} & {r.cover_message_bytes_mean:.1f} & {r.stego_object_bytes_mean:.1f} & {r.fixed_control_bytes_mean:.0f} & {r.heartbeat_amortized_bytes_mean:.1f} & {r.total_bytes_mean:.1f} & {r.expansion_ratio_mean:.2f}$\\times$ \\\\")
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}','']
    path.write_text('\n'.join(lines),encoding='utf-8')

def figures(summary,raws,over,out,formats,dpi,width,height):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    x=np.arange(len(summary)); labs=[f'{p:.2f}' for p in summary.payload_bpp]
    fig,ax=plt.subplots(figsize=(width,height)); bw=.36
    ax.bar(x-bw/2,summary.embed_mean_s,bw,yerr=summary.embed_std_s,capsize=4,label='Embedding')
    ax.bar(x+bw/2,summary.extract_mean_s,bw,yerr=summary.extract_std_s,capsize=4,label='Extraction')
    ax.set_xticks(x,labs); ax.set_xlabel('Payload (bpp)'); ax.set_ylabel('Time (s)'); ax.set_title('Ours-QIM-v3.2-Fused-CF runtime'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.3)
    for e in formats: fig.savefig(out/f'fig_qim_v32_runtime.{e}',dpi=dpi,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(width,height)); ax.errorbar(summary.payload_bpp,summary.rss_mean_mb,yerr=summary.rss_std_mb,marker='o',capsize=4,label='RSS'); ax.errorbar(summary.payload_bpp,summary.peak_rss_mean_mb,yerr=summary.peak_rss_std_mb,marker='s',capsize=4,label='Peak RSS'); ax.set_xlabel('Payload (bpp)'); ax.set_ylabel('Memory (MB)'); ax.set_title('Ours-QIM-v3.2-Fused-CF memory consumption'); ax.legend(frameon=False); ax.grid(alpha=.3)
    for e in formats: fig.savefig(out/f'fig_qim_v32_memory.{e}',dpi=dpi,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(width,height)); bottom=np.zeros(len(over)); x=np.arange(len(over))
    for c,l in [('cover_message_bytes_mean','Cover messages'),('stego_object_bytes_mean','Stego-object'),('fixed_control_bytes_mean','Fixed control'),('heartbeat_amortized_bytes_mean','Heartbeat'),('network_overhead_bytes_mean','Network headers')]:
        vals=over[c].to_numpy(float); ax.bar(x,vals,bottom=bottom,label=l); bottom+=vals
    ax.set_xticks(x,[f'{p:.2f}' for p in over.payload_bpp]); ax.set_xlabel('Payload (bpp)'); ax.set_ylabel('Bytes per protocol epoch'); ax.set_title('Protocol-epoch communication composition'); ax.legend(frameon=False); ax.grid(axis='y',alpha=.3)
    for e in formats: fig.savefig(out/f'fig_qim_v32_epoch_bytes.{e}',dpi=dpi,bbox_inches='tight')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runtime-summary-csv',required=True)
    ap.add_argument('--raw-log-root',action='append',default=[])
    ap.add_argument('--stego',action='append',required=True)
    ap.add_argument('--gamma1',required=True); ap.add_argument('--gamma2',required=True)
    ap.add_argument('--out-dir',required=True); ap.add_argument('--image-width',type=int,default=512); ap.add_argument('--image-height',type=int,default=512); ap.add_argument('--max-files',type=int,default=10000)
    ap.add_argument('--heartbeat-cadence',type=int,default=10); ap.add_argument('--network-overhead-per-frame',type=int,default=0)
    ap.add_argument('--formats',nargs='+',default=['pdf','png','svg']); ap.add_argument('--dpi',type=int,default=300); ap.add_argument('--width',type=float,default=8); ap.add_argument('--height',type=float,default=4.2)
    ap.add_argument('--hardware-note',default='Insert CPU, RAM, Ubuntu, Python, and storage details used for the experiment')
    a=ap.parse_args()
    import pandas as pd
    out=Path(a.out_dir).expanduser().resolve(); tables=out/'tables'; figs=out/'figures'; tables.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True)
    summary=load_summary(a.runtime_summary_csv)
    logpaths=discover_logs(a.raw_log_root); raw=load_raw(logpaths) if logpaths else None; raws=summarize_raw(raw) if raw is not None and not raw.empty else None
    if raw is not None: raw.to_csv(out/'qim_v32_runtime_raw_combined.csv',index=False)
    if raws is not None: raws.to_csv(tables/'qim_v32_runtime_distribution.csv',index=False)
    specs=[parse_payload_dir(v) for v in a.stego]; g1=read_lines(a.gamma1); g2=read_lines(a.gamma2)
    layout=Layout(heartbeat_cadence_epochs=a.heartbeat_cadence,network_overhead_per_frame=a.network_overhead_per_frame)
    oraw,osum=build_overhead(specs,g1,g2,layout,a.image_width,a.image_height,a.max_files)
    summary.to_csv(tables/'qim_v32_runtime_summary_clean.csv',index=False); oraw.to_csv(out/'qim_v32_overhead_per_epoch.csv',index=False); osum.to_csv(tables/'qim_v32_overhead_summary.csv',index=False)
    write_runtime_tex(summary,raws,tables/'table_qim_v32_runtime.tex'); write_overhead_tex(osum,tables/'table_qim_v32_overhead.tex'); figures(summary,raws,osum,figs,a.formats,a.dpi,a.width,a.height)
    first,last=summary.iloc[0],summary.iloc[-1]
    draft=f'''\\subsection{{Communication Overhead and Runtime Efficiency}}\n\\label{{subsec:runtime_overhead}}\n\nThe runtime analysis uses the existing Ours-QIM-v3.2-Fused-CF measurements collected over 10,000 images at each payload; no additional embedding run was required. Mean embedding time changed only slightly from ${first.embed_mean_s:.3f}\\pm{first.embed_std_s:.3f}$~s at ${first.payload_bpp:.2f}$~bpp to ${last.embed_mean_s:.3f}\\pm{last.embed_std_s:.3f}$~s at ${last.payload_bpp:.2f}$~bpp. Extraction increased from ${first.extract_mean_s:.4f}\\pm{first.extract_std_s:.4f}$~s to ${last.extract_mean_s:.4f}\\pm{last.extract_std_s:.4f}$~s. Mean RSS increased from ${first.rss_mean_mb:.1f}\\pm{first.rss_std_mb:.1f}$~MB to ${last.rss_mean_mb:.1f}\\pm{last.rss_std_mb:.1f}$~MB, while no GPU memory was allocated.\n\n\\input{{tables/table_qim_v32_runtime}}\n\\input{{tables/table_qim_v32_overhead}}\n\nUnder the stated binary serialisation, the fixed control cost is {layout.fixed_control_bytes} bytes per protocol epoch and heartbeat messages add an amortised {layout.heartbeat_amortized_bytes:.1f} bytes per epoch. The two cover messages and stego-object are reported separately because they are transmitted protocol data rather than control overhead.\n\n\\paragraph{{Hardware and software environment.}} {a.hardware_note}.\n'''
    (out/'communication_overhead_runtime_draft.tex').write_text(draft,encoding='utf-8')
    (out/'analysis_metadata.json').write_text(json.dumps({'no_embedding_performed':True,'runtime_summary_csv':str(Path(a.runtime_summary_csv).resolve()),'raw_logs_discovered':[str(x) for x in logpaths],'raw_payloads':sorted(raw.payload_bpp.dropna().unique().tolist()) if raw is not None else [],'layout':asdict(layout)},indent=2)+'\n')
    print('[OK] No embedding or extraction executable was run.')
    print(f'[OK] Raw logs discovered: {len(logpaths)}')
    if raw is not None: print(f'[OK] Raw payloads found: {sorted(raw.payload_bpp.dropna().unique().tolist())}')
    print(f'[OK] Output directory: {out}')

if __name__=='__main__':
    try: main()
    except InputError as e: raise SystemExit(f'[ERROR] {e}')
