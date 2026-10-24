#!/usr/bin/env python3
"""Freeze bounded content-linkage study before any new stego outcomes."""
import argparse,csv,importlib.metadata,json,os,random,secrets,shutil,sys
from pathlib import Path
from common import sha,save,now,status

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--evidence-root',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();root,evidence=args.root.resolve(),args.evidence_root.resolve()
    assert not (root/'run_config.json').exists(),'Existing run is never overwritten'
    root.mkdir(parents=True,exist_ok=True)
    source=root/'source';assert source.exists()
    pilot=evidence/'runs/cpm_v2_pilot_100'
    oldcfg=json.loads((pilot/'run_config.json').read_text())
    for item in oldcfg['source_files']:
        assert sha((source/item['name']).read_bytes())==item['sha256']
    manifest=evidence/'future_benchmark_manifest.csv'
    meta=json.loads((evidence/'future_benchmark_manifest.json').read_text())
    assert sha(manifest.read_bytes())==meta['sha256']
    allcovers=list(csv.DictReader(manifest.open()))
    exclude={r['cover_id'] for r in csv.DictReader((pilot/'cover_manifest.csv').open())}
    sens=evidence/'runs/cpm_v2_fusion_sensitivity_25/cover_manifest.csv'
    if sens.exists():exclude|={r['cover_id'] for r in csv.DictReader(sens.open())}
    counts={'train':150,'val':50,'test':100} if not args.smoke else {'train':1,'val':0,'test':0}
    if args.smoke:
        # Development fixture only: never consume a final test cover in checks.
        cover=next(csv.DictReader((pilot/'cover_manifest.csv').open()))
        selected=[next(r for r in allcovers if r['cover_id']==cover['cover_id'])]
    else:
        selected=[]
        for partition,count in counts.items():
            candidates=[r for r in allcovers if r['partition']==partition and r['cover_id'] not in exclude]
            selected+=sorted(candidates,key=lambda r:sha(('cpm-linkage-20261017:'+partition+':'+r['cover_id']).encode()))[:count]
    assert len(selected)==sum(counts.values())
    assert len({r['cover_sha256'] for r in selected})==len(selected)
    for c in selected:
        assert sha(Path(c['source_path']).read_bytes())==c['cover_sha256']
    with (root/'cover_manifest.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    pairmeta=Path('/home/omego/Documents/Hybrid_Stego_Eva/linkability_probes/matched_outputs/prepared/gamma_pair_features.csv')
    bank=[]
    for old in csv.DictReader(pairmeta.open()):
        ident=old['candidate_gamma_pair_id'];folder=root/'parameter_bank'/ident;folder.mkdir(parents=True)
        row=dict(pair_id=ident)
        for j in (1,2):
            original=Path(old[f'gamma{j}_path']);data=original.read_bytes()
            assert sha(original.read_text().encode())==old[f'gamma{j}_sha256']
            target=folder/f'gamma{j}.txt';target.write_bytes(data);target.chmod(0o444)
            row.update({f'gamma{j}_path':str(target),f'gamma{j}_sha256':sha(data),f'gamma{j}_bytes':len(data)})
        bank.append(row)
    bank.sort(key=lambda r:r['pair_id']);assert len(bank)==12
    assert len({(r['gamma1_sha256'],r['gamma2_sha256']) for r in bank})==12
    save(root/'parameter_bank.json',bank)
    secret=root/'secrets';secret.mkdir(mode=0o700)
    master=secret/'research_fixture_master.bin';master.write_bytes(secrets.token_bytes(32));master.chmod(0o600)
    payloads=[.1,.2,.4] if not args.smoke else [.2]
    cases=[]
    for ci,c in enumerate(selected):
        for pi,payload in enumerate(payloads):
            folder=root/'inputs'/f'cover_{c["cover_id"]}'/f'payload_{int(round(payload*100)):03d}';folder.mkdir(parents=True)
            bits=int(round(payload*512*512));data=bytearray(secrets.token_bytes((bits+7)//8))
            if bits%8:data[-1]&=(255<<(8-bits%8))&255
            message=folder/'message.bin';message.write_bytes(data);message.chmod(0o444)
            for qi,pair in enumerate(bank):
                cases.append(dict(epoch=len(cases),cover_id=c['cover_id'],partition=c['partition'],
                    cover_path=c['source_path'],cover_sha256=c['cover_sha256'],payload_bpp=payload,
                    application_payload_bits=bits,pair_id=pair['pair_id'],message_path=str(message),message_sha256=sha(data)))
    save(root/'case_manifest.json',cases)
    order=list(range(len(selected)));random.Random(1907).shuffle(order)
    config=dict(run_id='cpm_v2_revised_linkage_300' if not args.smoke else 'development_linkage_smoke',
        created_at=now(),purpose='Bounded revised content-linkage diagnostic, not full-view unlinkability or cryptographic-bound estimation',
        covers=len(selected),cover_counts=counts,total_attempts=len(cases),parameter_pairs=12,payloads=payloads,
        selection='SHA256(cpm-linkage-20261017:partition:cover_id); exclude prior pilot and fusion-panel covers',
        benchmark_manifest_sha256=sha(manifest.read_bytes()),master_file=str(master),session_hex=secrets.token_hex(16),
        research_fixture_key_only=True,source_sha256={f.name:sha(f.read_bytes()) for f in sorted(source.iterdir()) if f.is_file()},
        frozen_pilot_sources=oldcfg['source_files'],cover_execution_order=[selected[i]['cover_id'] for i in order],
        generation_workers=3,threads_per_process=1,fusion_weight=.75,automatic_retry=False,fallback=False,
        common_message='Same random plaintext within a cover/payload; unique authenticated epoch/context and STC seed per pair',
        primary_cohort='Complete 12-pair blocks per cover/payload; report every failed attempt and excluded block',
        binary_task='True pair versus all 11 wrong candidates for the same image; class-balanced fitting and balanced accuracy at threshold 0.5',
        ranking='All 12 candidates; fractional tie credit at absolute score tolerance 1e-12',
        views=['image_only','text_only','cross_only','text_image'],classifiers=['logistic','extra_trees'],
        model_seeds=[13,41,97],ensemble='Mean probability across three fixed seeds',
        hyperparameters=dict(logistic_C=1.,logistic_max_iter=3000,extra_trees_n_estimators=400,min_samples_leaf=2,max_features='sqrt'),
        primary_selection='Choose logistic versus extra_trees on validation AUC for combined text_image view per payload; save before test scoring',
        score_orientation='Flip ensemble probabilities only if validation AUC is below 0.5; freeze orientation before test scoring',
        controls=['image-only and text-only marginal balance','synthetic pair-agreement leakage','intentional visible epoch/header association'],
        feature_scope='Unchanged historical handcrafted text/image/cross features, hash dimension 32; no IDs, paths, session/header/nonces, ground truth or secret inputs in classifier features',
        uncertainty=dict(bootstrap_replicates=3000,seed=20261018,unit='Test cover, jointly across payloads and variants; fixed trained models',
            diagnostic_CI='pointwise 95%',primary_CI='Bonferroni across three selected payload AUCs: familywise 95%'),
        primary_min_complete_cover_counts={'train':120,'val':40,'test':80},
        scope_limits=['300 source covers and a fixed bank of twelve parameter pairs','conditional on complete valid blocks',
            'handcrafted finite classifier family','content-only view; headers intentionally link traffic','no new recent steganalysis/baseline result'],
        environment=dict(python=sys.executable,python_version=sys.version,
            packages={x:importlib.metadata.version(x) for x in ['numpy','scipy','Pillow','scikit-image','scikit-learn','pandas','pystcstego']}))
    save(root/'run_config.json',config)
    status(root,state='running',phase='generation',stage='Generating source-consistent image epochs',
           total=len(cases),completed=0,successful=0,failed=0,models_completed=0,models_total=72,cover_counts=counts)
    print(json.dumps(dict(root=str(root),covers=len(selected),attempts=len(cases),counts=counts)))

if __name__=='__main__':main()
