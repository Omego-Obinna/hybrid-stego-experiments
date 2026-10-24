#!/usr/bin/env python3
"""Read-only release checks. Does not import or execute experiment modules."""
from pathlib import Path
import ast
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def rows(p):
    with p.open(newline='') as f:
        return list(csv.DictReader(f))

def main():
    sums = ROOT/'SHA256SUMS'
    checked = 0
    for line in sums.read_text().splitlines():
        digest, rel = line.split('  ', 1)
        p=ROOT/rel
        assert p.is_file(), f'Missing release file: {rel}'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest, f'Changed release file: {rel}'
        checked += 1
    manifest=json.loads((ROOT/'provenance/source_manifest.json').read_text())
    for r in manifest:
        p=ROOT/r['path']
        assert p.stat().st_size==r['bytes'], r['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['packaged_sha256'], r['path']
        if r['transformation']=='byte-identical':
            assert r['source_sha256']==r['packaged_sha256'], r['path']
    syntax=0
    references=json.loads((ROOT/'provenance/non_executable_reference_files.json').read_text())
    for p in ROOT.rglob('*.py'):
        if '.venv' in p.parts: continue
        rel=str(p.relative_to(ROOT))
        if rel in references:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==references[rel]['sha256'],rel
            continue
        ast.parse(p.read_bytes(),filename=str(p.relative_to(ROOT)))
        syntax+=1
    studies=json.loads((ROOT/'provenance/studies.json').read_text())
    assert len(studies)==11
    for s in studies:
        p=ROOT/'experiments/studies'/s['run_id']
        assert (p/'source').is_dir() and (p/'records/run_config.json').is_file(), s['run_id']
        if s['status']=='complete' and s['run_id']!='cpm_v2_pilot_100':
            assert (p/'records/completion.json').is_file(),s['run_id']
    h=ROOT/'experiments/studies/cpm_v2_hsmnet_matched_1200'
    d=rows(h/'records/detectability.csv'); c=rows(h/'records/auc_paired_contrasts.csv')
    assert len(d)==15 and len(c)==12
    assert len(list((h/'predictions').glob('*.npz')))==21
    assert all(float(x['auc_ci_low'])>.5 for x in d)
    assert sum(float(x['adjusted_ci_high'])<0 for x in c)==7
    assert sum(float(x['adjusted_ci_low'])<=0<=float(x['adjusted_ci_high']) for x in c)==5
    v=json.loads((h/'records/verified_summary.json').read_text())
    assert v['passed'] and v['selected_checkpoint_predictions_checked']==6
    transport=ROOT/'results/components/transport/run_summary_raw.csv'
    unique=ROOT/'results/audit/transport_unique_runs.csv'
    assert len(rows(transport))==26191 and len(rows(unique))==25920
    bad=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or '.venv' in p.parts or '.git' in p.parts:continue
        if p.name in {'fixture_master.bin','master.bin','.env'} or 'secrets' in p.relative_to(ROOT).parts:
            bad.append(str(p.relative_to(ROOT)))
        assert p.stat().st_size<100_000_000, f'Ordinary Git file exceeds release limit: {p}'
    assert not bad,bad
    print(json.dumps(dict(passed=True,release_files_checked=checked,copied_files_checked=len(manifest),
        python_sources_parsed=syntax,unused_upstream_reference_files=len(references),study_status_records=11,hsm_auc_rows=15,hsm_contrasts=12,
        raw_transport_rows=26191,unique_transport_scenarios=25920,
        scientific_driver_execution=False),indent=2))

if __name__=='__main__':main()
