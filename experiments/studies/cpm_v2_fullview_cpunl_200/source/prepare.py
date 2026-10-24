import argparse,csv,importlib.metadata,json,os,random,secrets,sys
from pathlib import Path
from common import sha,save,status,now

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--evidence-root',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    root=a.root.resolve();evidence=a.evidence_root.resolve();assert not (root/'run_config.json').exists()
    root.mkdir(parents=True,exist_ok=True);manifest=evidence/'future_benchmark_manifest.csv'
    assert sha(manifest.read_bytes())==json.loads((evidence/'future_benchmark_manifest.json').read_text())['sha256']
    allcovers=list(csv.DictReader(manifest.open()));excluded=set()
    for name in ['cpm_v2_pilot_100','cpm_v2_fusion_sensitivity_25','cpm_v2_revised_linkage_300']:
        excluded|={r['cover_id'] for r in csv.DictReader((evidence/'runs'/name/'cover_manifest.csv').open())}
    counts={'train':100,'val':40,'test':60} if not a.smoke else {'train':1,'val':0,'test':0}
    selected=[]
    for part,n in counts.items():
        pool=[r for r in allcovers if r['partition']==part and r['cover_id'] not in excluded]
        selected+=sorted(pool,key=lambda r:sha(f'cpunl-fullview-20261017:{part}:{r["cover_id"]}'.encode()))[:n]
    if a.smoke:
        old=next(csv.DictReader((evidence/'runs/cpm_v2_pilot_100/cover_manifest.csv').open()))
        selected=[next(x for x in allcovers if x['cover_id']==old['cover_id'])];assert selected[0]['partition']=='train'
    assert len(selected)==sum(counts.values()) and len({r['cover_sha256'] for r in selected})==len(selected)
    for c in selected:assert sha(Path(c['source_path']).read_bytes())==c['cover_sha256']
    with (root/'cover_manifest.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    oldbank=json.loads((evidence/'runs/cpm_v2_revised_linkage_300/parameter_bank.json').read_text());bank=[]
    for old in oldbank:
        item={'pair_id':old['pair_id']};folder=root/'parameter_bank'/old['pair_id'];folder.mkdir(parents=True)
        for j in [1,2]:
            data=Path(old[f'gamma{j}_path']).read_bytes();assert sha(data)==old[f'gamma{j}_sha256'];assert len(data)>=65536
            target=folder/f'gamma{j}.bin';target.write_bytes(data[:65536]);target.chmod(0o444)
            item.update({f'gamma{j}_path':str(target),f'gamma{j}_sha256':sha(data[:65536]),f'gamma{j}_bytes':65536,
                         f'gamma{j}_original_sha256':sha(data)})
        bank.append(item)
    assert len(bank)==len({(b['gamma1_sha256'],b['gamma2_sha256']) for b in bank})==12;save(root/'parameter_bank.json',bank)
    (root/'secrets').mkdir(mode=0o700)
    blocks=[];cases=[];payloads=[.1,.2,.4] if not a.smoke else [.2]
    for c in selected:
        for pay in payloads:
            ident=f'c{c["cover_id"]}_p{round(pay*100):03d}';keyfile=root/'secrets'/f'{ident}.bin';keyfile.write_bytes(secrets.token_bytes(32));keyfile.chmod(0o600)
            block=dict(block_id=ident,cover_id=c['cover_id'],partition=c['partition'],cover_path=c['source_path'],cover_sha256=c['cover_sha256'],
                payload_bpp=pay,payload_bits=round(pay*512*512),master_file=str(keyfile),session_hex=secrets.token_hex(16))
            block['history_pair_indices']=[secrets.randbelow(12) for _ in range(4)]
            block['case_ids']=[]
            for h,q in [('history',i) for i in range(4)]+[('challenge',i) for i in range(12)]:
                pair=block['history_pair_indices'][q] if h=='history' else q
                case=dict(case_id=len(cases),case_uid=f'{ident}_{h}{q:02d}',block_id=ident,cover_id=c['cover_id'],partition=c['partition'],
                    role=h,role_index=q,pair_index=pair,epoch=q if h=='history' else 4,payload_bpp=pay)
                cases.append(case);block['case_ids'].append(case['case_id'])
            blocks.append(block)
    save(root/'blocks.json',blocks);save(root/'case_manifest.json',cases)
    private=root/'secrets/private_manifest.json'
    save(private,[dict(path=b['master_file'],sha256=sha(Path(b['master_file']).read_bytes())) for b in blocks]);private.chmod(0o600)
    order=list(range(len(blocks)));random.Random(261017).shuffle(order)
    cfg=dict(run_id='cpm_v2_fullview_cpunl_200' if not a.smoke else 'development_fullview_smoke',created_at=now(),development_only=a.smoke,
        cover_counts=counts,payloads=payloads,covers=len(selected),history_queries=4,parameter_pairs=12,total_attempts=len(cases),
        block_execution_order=[blocks[i]['block_id'] for i in order],generation_workers=3,fusion_weight=.75,automatic_retry=False,
        public_length_stratum=65536,metadata_conditioning='same session/epoch/header/three nonces/payload/codec/version/channel fields/content and wire lengths within a block',
        source_sha256={p.name:sha(p.read_bytes()) for p in sorted((root/'source').iterdir()) if p.is_file()},
        input_sha256={n:sha((root/n).read_bytes()) for n in ['cover_manifest.csv','parameter_bank.json','blocks.json','case_manifest.json']},
        private_manifest_sha256=sha(private.read_bytes()),
        benchmark_manifest_sha256=sha(manifest.read_bytes()),independent_world_includes_diagonal=True,world_priors='equal; balanced class weights for 12 versus 144 rows',
        primary_view='full_history',views=['metadata_only','content_only','authentication_only','full_challenge','full_history'],
        classifiers=['logistic','extra_trees'],model_seeds=[13,41,97],models_total=90,
        hyperparameters=dict(logistic_C=1.,max_iter=3000,trees=256,min_samples_leaf=4,max_features='sqrt'),
        primary_min_complete_blocks={'train':80,'val':30,'test':45},uncertainty=dict(seed=20261019,replicates=3000,unit='attempted test cover jointly across worlds/histories/payloads',primary_alpha=.05/3),
        selection_rule='Validation-only orientation; full_history validation AUC chooses family, exact tie logistic; freeze before first test scoring',
        scope='h=0 finite-bank full-public-view comparison with four adaptive public chosen-message queries; complete-block conditional inference; no universal PPT/epsilon or network-scheduling claim',
        query_policy='h0 zero; h>0 truncated SHAKE256(domain||rolling SHA256 of public requests/responses); challenge zero',
        environment=dict(python=sys.executable,packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','Pillow','scikit-image','scikit-learn','pandas','pystcstego']}))
    save(root/'run_config.json',cfg)
    for n in ['run_config.json','cover_manifest.csv','parameter_bank.json','blocks.json','case_manifest.json']:(root/n).chmod(0o444)
    status(root,state='running',phase='generation',development_only=a.smoke,total=len(cases),completed=0,successful=0,failed=0,models_completed=0,models_total=90,
        stage='Native history and matched-context challenge generation',cover_counts=counts,last_error=None)
    print(json.dumps(dict(covers=len(selected),blocks=len(blocks),attempts=len(cases),development_only=a.smoke)))

if __name__=='__main__':main()
