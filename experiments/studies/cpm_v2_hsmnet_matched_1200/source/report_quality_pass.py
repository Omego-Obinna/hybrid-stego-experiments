"""Versioned reporting-only corrections; never edits source, trials or selections.

Run once per completed, independently verified panel, before reporting it to the
author. Every replaced report/table/summary/completion file is backed up first.
"""
import csv,hashlib,json,statistics,sys
from datetime import datetime,timezone
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False))
def main():
    root=Path(sys.argv[1]).resolve();assert (root/'completion.json').exists();assert json.loads((root/'verified_summary.json').read_text())['passed']
    dest=root/'report_quality_v1';assert not dest.exists(),'Already applied; inspect provenance instead of repeating';dest.mkdir()
    files=['RESULTS.md','verified_summary.json','completion.json']
    if (root/'fresh_recovery_runtime.csv').exists():files.append('fresh_recovery_runtime.csv')
    if (root/'native_system_cost.csv').exists():files.append('native_system_cost.csv')
    original={}
    for name in files:
        p=root/name;original[name]=sha(p);(dest/name).write_bytes(p.read_bytes())
    report=(root/'RESULTS.md').read_text();summary=json.loads((root/'verified_summary.json').read_text());notes=[]
    if root.name in ['cpm_v2_matched_image_1200','cpm_v2_hsmnet_matched_1200']:
        rows=[json.loads(x) for x in (root/'image_results.jsonl').open()];fresh=[json.loads(x) for x in (root/'fresh_results.jsonl').open()];counts={m:sum(r['method']==m and r.get('pixels_over_one',0)>0 for r in rows) for m in ['cpm','suniward','mipod','wow','hill']};fc={m:sum(r['method']==m and r.get('pixels_over_one',0)>0 for r in fresh) for m in counts}
        table=list(csv.DictReader((root/'fresh_recovery_runtime.csv').open()))
        for r in table:
            valid=[x for x in fresh if x['method']==r['method'] and x['payload_bpp']==float(r['payload_bpp']) and x['exact_recovery']]
            values=[(x['cost_sec']+x['stc_sec']) if x['method']!='cpm' else x['core_sec'] for x in valid]
            r['core_sec_mean']=statistics.mean(values) if values else '';r['core_sec_median']=statistics.median(values) if values else '';r['core_timing_definition']='cost-map computation plus native STC embedding; image serialization excluded'
        with (root/'fresh_recovery_runtime.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,table[0].keys());w.writeheader();w.writerows(table)
        summary['pixel_deviation_image_counts_by_method']=counts;summary['fresh_pixel_deviation_counts_by_method']=fc
        report+='\n## Reporting audit\n\nImages with an absolute pixel change exceeding one level remain retained. Counts in the main image panel, by method: '+', '.join(f'{m}: {n}' for m,n in counts.items())+'. Fresh development panel: '+', '.join(f'{m}: {n}' for m,n in fc.items())+'. These observations do not imply incorrect recovered plaintext.\n\nFresh core-time comparisons consistently use cost-map computation plus native STC embedding and exclude image serialization. For the baseline workers this is derived from their separately saved `cost_sec + stc_sec`; the original raw `core_sec` also included the image save. Raw measurements, scientific settings, outcomes, model choices and detector results are unchanged. `report_quality_v1/provenance.json` preserves the original table and the correction.\n'
        notes.append('Correct the baseline table core-time denominator using already saved cost_sec + stc_sec; the planned pure embedding-core metric excludes serialization. No trial or scientific result was rerun.')
    else:
        log='generation_results.jsonl' if 'generative' in root.name else 'system_results.jsonl';rows=[json.loads(x) for x in (root/log).open()];n=sum(r['method']=='cpm' and r.get('pixels_over_one',0)>0 for r in rows);summary['cpm_pixel_deviation_images']=n
        report+=f'\n## Reporting audit\n\nThe CPM rows retain {n} exactly recovered images with absolute pixel changes exceeding one level. Native synthetic carriers have no original-cover one-level distortion comparison. All observed deviations remain in the raw records and checked summary.\n'
        if 'generative' in root.name:
            report+='\nThe mean transmitted-byte and timing summaries for the native generative panel condition on exact recovery; their denominators are the explicitly reported exact-recovery counts. All capacity and recovery failures remain in the all-attempt table. These means must not be described as all-attempt communication cost.\n'
        else:
            table=list(csv.DictReader((root/'native_system_cost.csv').open()))
            for r in table:
                rr=[x for x in rows if x['method']==r['method'] and x['message_bytes']==int(r['message_bytes'])]
                wire=[]
                for x in rr:
                    if x.get('wire_bytes') is None:
                        assert x['method']=='cpm' and not x['exact_recovery'];assert not list((Path(x['folder'])/'frames').glob('C*.json'));wire.append(0)
                    else:wire.append(x['wire_bytes'])
                r['wire_bytes_all_attempt_total']=sum(wire);r['wire_bytes_all_attempt_mean']=statistics.mean(wire);r['wire_bytes_mean_denominator']='cases with a recorded native wire value; all-attempt columns count verified unsent CPM preflight failures as zero'
            with (root/'native_system_cost.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,table[0].keys());w.writeheader();w.writerows(table)
            report+='\nThe native system-cost table also gives actual sent-wire totals and means over all attempts. A recognized CPM sender preflight failure contributes zero sent bytes only after checking that no authenticated frame files exist. The earlier `wire_bytes_mean` column retains its original recorded-value denominator and is labelled accordingly; the new all-attempt columns support consistent accounting without dropping failures.\n'
            notes.append('Add explicitly denominated all-attempt sent-wire columns and verify unsent CPM preflight cases have no frame files. Original cost columns are retained.')
    notes.append('Expose retained pixel-limit deviations in RESULTS.md and verified_summary.json; no image is removed or changed.')
    (root/'RESULTS.md').write_text(report);summary['report_quality_provenance']='report_quality_v1/provenance.json';write(root/'verified_summary.json',summary)
    completion=json.loads((root/'completion.json').read_text());completion['report_sha256']=sha(root/'RESULTS.md');completion['report_quality_pass']='report_quality_v1/provenance.json';write(root/'completion.json',completion)
    write(dest/'provenance.json',dict(applied_at=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),original_hashes=original,updated_hashes={n:sha(root/n) for n in files},changes=notes,scientific_design_changed=False,raw_trials_changed=False,model_or_test_selection_changed=False,test_scoring_repeated=False))
    print('Reporting quality pass saved with original artifacts and provenance:',dest)
if __name__=='__main__':main()
