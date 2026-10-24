#!/usr/bin/env python3
import copy
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
import epoch_protocol as ep

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get("CPM_RESULTS_DIR", str(ROOT / "results")))


def flip(value):
    out = bytearray(value)
    out[len(out)//2] ^= 1
    return bytes(out)


def main():
    master = (OUT / "fixture_master.bin").read_bytes()
    validation = json.loads((OUT / "receiver_validation.json").read_text())
    packets = [ep.read_frames(OUT / f"epoch_{r['epoch']:03d}/received_frames")
               for r in validation["outcomes"] if r["exact_recovery"]]
    assert len(packets) == 36
    rows = []
    checks = []
    for field, label in [("gamma1", "C1 parameter"), ("gamma2", "C2 parameter"),
                          ("stego", "C3 image"), ("nonce_a", "C1 nonce"),
                          ("nonce_b", "C2 nonce"), ("nonce_c", "C3 nonce"),
                          ("hdr", "Header"), ("tag", "Epoch tag")]:
        rows.append((label, lambda e,j,field=field: replace(e, **{field:flip(getattr(e,field))})))
    rows.extend([
        ("Payload length", lambda e,j:replace(e,payload_bits=e.payload_bits+1)),
        ("Fresh-epoch rebinding", lambda e,j:replace(e,epoch=e.epoch+1000,
             hdr=ep.header(master,e.sid,e.epoch+1000))),
        ("Channel swapping", lambda e,j:replace(e,nonce_a=e.nonce_b,nonce_b=e.nonce_a,
                                                gamma1=e.gamma2,gamma2=e.gamma1)),
        ("Cross-epoch C1 splice", lambda e,j:replace(e,nonce_a=packets[(j+1)%36].nonce_a,
                                                     gamma1=packets[(j+1)%36].gamma1)),
        ("Cross-epoch C2 splice", lambda e,j:replace(e,nonce_b=packets[(j+1)%36].nonce_b,
                                                     gamma2=packets[(j+1)%36].gamma2)),
        ("Cross-session transplant", lambda e,j:replace(e,sid=flip(e.sid))),
    ])
    results = []
    for label, mutate in rows:
        rejects = 0
        before_decode = 0
        for j, e in enumerate(packets):
            altered = mutate(e,j)
            called = []
            try:
                ep.recover(master,altered,e.sid,ep.ReceiverState(),
                           lambda image,seed:called.append(True) or b"")
            except ep.ProtocolError:
                rejects += 1
            before_decode += not called
        results.append(dict(family=label, attempts=36, rejected=rejects,
                            unauthorised_accepts=36-rejects, rejected_before_decode=before_decode))
    # Authentic repeated epochs must also be rejected after state has been committed.
    persisted = ep.ReceiverState.read(OUT / "receiver_state.json")
    rejected = 0
    for e in packets:
        try:ep.verify(master,e,e.sid,persisted)
        except ep.ProtocolError:rejected += 1
    results.append(dict(family="Accepted-epoch replay",attempts=36,rejected=rejected,
                        unauthorised_accepts=36-rejected,rejected_before_decode=rejected))

    first = packets[0]
    records = [json.loads((OUT / "epoch_000/received_frames" / f"C{i}.json").read_text())
               for i in [1,2,3]]
    for label, change in [
        ("Wrong codec", lambda rs:rs[2].update(codec="wrong")),
        ("Wrong version", lambda rs:rs[0].update(version="wrong")),
        ("Wrong channel label", lambda rs:rs[0].update(channel="C2")),
        ("Unexpected JSON field", lambda rs:rs[1].update(unbound="value")),
        ("Missing C2 frame", None),
    ]:
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name);rs=copy.deepcopy(records)
            if change:change(rs)
            for i,r in enumerate(rs,1):
                if label == "Missing C2 frame" and i == 2:continue
                (folder/f"C{i}.json").write_text(json.dumps(r))
            try:ep.read_frames(folder)
            except ep.ProtocolError:passed=True
            else:passed=False
            checks.append(dict(case=label,passed=passed))
    for bits in [1,7,8,9,255,256,257,26214,52429,104858]:
        e=replace(first,payload_bits=bits,stego=b"",tag=b"")
        message=ep.truncate(bytes([0xa5])*((bits+7)//8),bits)
        encoded=ep.mask_message(master,e,message)
        checks.append(dict(case=f"mask_inverse_{bits}_bits",
                           passed=ep.mask_message(master,e,encoded)==message))
    checks.append(dict(case="distinct_purpose_keys",passed=len({ep.key(master,first.sid,p)
        for p in [b"header",b"nonce",b"payload-mask",b"cost-modulation",b"embedding",b"epoch-mac"]})==6))
    checks.append(dict(case="distinct_contexts_change_mask",passed=ep.mask(master,first)!=
        ep.mask(master,replace(first,nonce_c=flip(first.nonce_c)))))
    # This finite exact example falsifies the previous use of mask loss to pay
    # for sampler mismatch; it is not an empirical estimate of a PRF bound.
    sampler_example=dict(ideal_sampler="always A",benign_law="always B",
                         distinguisher="accept exactly A",sampler_gap=1.0,
                         independent_uniform_mask_gap=0.0,
                         old_sampler_le_mask_claim_holds=False,
                         acceptance_gap=1.0,equal_prior_guessing_advantage=0.5)
    result=dict(protocol=ep.VERSION.decode(),authenticated_field_tests=results,
                attack_attempts=sum(r["attempts"] for r in results),
                unauthorised_accepts=sum(r["unauthorised_accepts"] for r in results),
                additional_checks=checks,additional_passed=sum(c["passed"] for c in checks),
                sampler_counterexample=sampler_example,
                scope="Deterministic engineering checks over 36 validated real-image epochs; not independent population attack trials")
    (OUT/"protocol_checks.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ["authenticated_field_tests","additional_checks"]},indent=2))
    assert result["unauthorised_accepts"]==0 and all(c["passed"] for c in checks)
    assert all(r["rejected_before_decode"]==r["attempts"] for r in results)


if __name__=="__main__":main()
