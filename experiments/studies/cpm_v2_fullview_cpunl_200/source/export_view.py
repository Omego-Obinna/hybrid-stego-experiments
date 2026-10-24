"""Reconstruct complete observable bytes without keys or hidden world labels."""
import argparse,json,hashlib,shutil
from dataclasses import replace
from pathlib import Path
import epoch_protocol as ep
from common import frame_bytes,sha

def epoch_for_record(root,record,bank):
    e=ep.read_frames(root/'cases'/record['carrier_case_uid']/'frames');pair=bank[record['parameter_index']]
    return replace(e,gamma1=Path(pair['gamma1_path']).read_bytes(),gamma2=Path(pair['gamma2_path']).read_bytes(),tag=bytes.fromhex(record['tag']))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--row',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();bank=json.loads((root/'parameter_bank.json').read_text());blocks={b['block_id']:b for b in json.loads((root/'blocks.json').read_text())};cases=json.loads((root/'case_manifest.json').read_text())
    record=None
    for line in (root/'public_views.jsonl').open():
        candidate=json.loads(line)
        if candidate['row']==a.row:record=candidate;break
    assert record is not None and not a.out.exists();a.out.mkdir(parents=True)
    h=hashlib.sha256()
    for i,cid in enumerate(blocks[record['block_id']]['case_ids'][:4]):
        source=root/'cases'/cases[cid]['case_uid'];target=a.out/f'query_{i}';target.mkdir()
        query=(source/'message.bin').read_bytes();(target/'request.bin').write_bytes(query)
        e=ep.read_frames(source/'frames');ep.write_frames(e,target/'response')
        h.update(ep.enc([(b'query',query),*[(f'C{j+1}'.encode(),b) for j,b in enumerate(frame_bytes(e))]]))
    e=epoch_for_record(root,record,bank);ep.write_frames(e,a.out/'challenge');wire=frame_bytes(e)
    assert [sha(x) for x in wire]==record['wire_sha256'];h.update(ep.enc([(f'C{i+1}'.encode(),b) for i,b in enumerate(wire)]))
    assert h.hexdigest()==record['observer_sha256']
    (a.out/'OBSERVATION.md').write_text('Four public chosen-message request/response pairs followed by one complete challenge.\nNo private key, world label, generating-pair index or source cover is exported.\n')
    print(json.dumps(dict(public_observer_sha256=h.hexdigest(),output=str(a.out))))

if __name__=='__main__':main()
