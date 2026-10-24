#!/usr/bin/env python3
"""Deterministic policy tests; native real-image validation is run separately."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

from window_receiver import WindowReceiver


class Boundaries(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.master, self.sid = b"k"*32, b"s"*16
        self.cfg = dict(window=3, timeout_ms=200, epoch_interval_ms=100, epoch_count=30,
            skew_ms=100, candidates_per_channel=2, buffer_bytes=8*1024*1024,
            failed_cache_per_epoch=128)
        self.r = WindowReceiver(ep, lambda data, seed:b"\0", self.master,
                                self.sid, self.cfg, self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def frames(self, epoch=0):
        nonces = [ep.mac(self.master, ep.uint(epoch), bytes([i]))[:16] for i in (1,2,3)]
        e = ep.authenticate(self.master, ep.Epoch(self.sid, epoch,
            ep.header(self.master,self.sid,epoch),nonces[0],b"a",nonces[1],b"b",nonces[2],8,b"s"))
        folder = Path(self.tmp.name)/f"fixture_{epoch}"
        ep.write_frames(e, folder)
        return [(folder/f"C{c}.json").read_text() for c in (1,2,3)]

    def test_skew_equality_is_ready(self):
        frames = self.frames()
        for raw,t in zip(frames,(20,70,120)):
            result = self.r.frame(raw,t)
        self.assertEqual(result["public"],"accepted")

    def test_skew_exceeded_does_not_decode_or_commit(self):
        for raw,t in zip(self.frames(),(20,70,121)):
            result = self.r.frame(raw,t)
        self.assertEqual(result["public"],"buffered")
        self.assertFalse(self.r.state.accepted)
        self.assertEqual(self.r.count["decode_calls"],0)

    def test_deadline_is_exclusive(self):
        for raw,t in zip(self.frames(),(101,150,200)):
            result = self.r.frame(raw,t)
        self.assertEqual(result["reason"],"window_rejects")
        self.assertEqual(self.r.count["expired_missing_epochs"],1)
        self.assertFalse(self.r.state.accepted)

    def test_future_frame_cannot_advance_window(self):
        result = self.r.frame(self.frames(3)[0],20)
        self.assertEqual(result["reason"],"window_rejects")
        self.assertEqual(self.r.a,0)

    def test_mac_failure_preserves_state_then_honest_candidate_recovers(self):
        frames = self.frames()
        bad = json.loads(frames[2]);bad["tag"]="00"*32
        self.r.frame(frames[0],20);self.r.frame(frames[1],21)
        result = self.r.frame(json.dumps(bad),22)
        self.assertEqual(result["public"],"rejected")
        self.assertFalse(self.r.state.nonces)
        self.assertEqual(self.r.count["decode_calls"],0)
        self.assertEqual(self.r.frame(frames[2],23)["public"],"accepted")
        self.r.restart(24)
        self.assertEqual(self.r.frame(frames[0],25)["reason"],"freshness_rejects")
        self.assertEqual(len(self.r.state.nonces),3)

    def test_byte_budget_and_duplicate_keys(self):
        raw = self.frames()[0]
        self.cfg["buffer_bytes"]=1
        self.assertEqual(self.r.frame(raw,20)["reason"],"byte_budget_rejects")
        self.assertEqual(self.r.usage(),(0,0))
        self.assertEqual(self.r.frame(raw[:-1]+',"epoch":0}',21)["reason"],"front_end_parse_rejects")


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--protocol-source",required=True)
    args,remaining=parser.parse_known_args()
    sys.path.insert(0,args.protocol_source)
    import epoch_protocol as ep
    unittest.main(argv=[sys.argv[0]]+remaining)
