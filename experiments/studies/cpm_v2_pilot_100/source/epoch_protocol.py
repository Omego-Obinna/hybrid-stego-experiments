"""Versioned receiver-reproducible masking and complete-epoch authentication.

Only byte framing, masking and authentication are implemented here. Image
embedding/extraction lives in image_worker.py; no image feature is needed here.
"""
from __future__ import annotations
import base64
import hashlib
import hmac
import json
from dataclasses import dataclass, field
from pathlib import Path

VERSION = b"cpm-epoch-v2"
CODEC = b"pystc-pgm8-v1"
MAX_FRAME_BYTES = 4 * 1024 * 1024


class ProtocolError(ValueError):
    pass


def enc(fields):
    out = bytearray()
    for label, value in fields:
        if not isinstance(label, bytes) or not isinstance(value, bytes):
            raise ProtocolError("Canonical fields must be bytes")
        out += len(label).to_bytes(2, "big") + label
        out += len(value).to_bytes(8, "big") + value
    return bytes(out)


def uint(value):
    if type(value) is not int or not 0 <= value < 2**64:
        raise ProtocolError("Invalid unsigned 64-bit integer")
    return value.to_bytes(8, "big")


def mac(key, *parts):
    return hmac.new(key, b"".join(parts), hashlib.sha256).digest()


def key(master, sid, purpose):
    if len(master) < 32 or len(sid) != 16:
        raise ProtocolError("Use a master secret of at least 32 bytes and a 16-byte session ID")
    return mac(master, b"derive", enc(((b"version", VERSION), (b"session", sid),
                                     (b"purpose", purpose))))


def header(master, sid, epoch):
    return mac(key(master, sid, b"header"), b"hdr", uint(epoch))[:16]


def expand(secret, label, context, length):
    output = bytearray()
    for counter in range((length + 31) // 32):
        output += mac(secret, label, context, uint(counter))
    return bytes(output[:length])


def truncate(data, bits):
    out = bytearray(data[:(bits + 7) // 8])
    if len(out) != (bits + 7) // 8:
        raise ProtocolError("Insufficient payload bytes")
    if bits % 8:
        out[-1] &= (255 << (8 - bits % 8)) & 255
    return bytes(out)


def xor(*streams):
    if not streams or len({len(s) for s in streams}) != 1:
        raise ProtocolError("Unequal bitstream lengths")
    return bytes(a ^ b for a, b in zip(streams[0], xor(*streams[1:]))) if len(streams) > 2 else (
        bytes(a ^ b for a, b in zip(*streams)) if len(streams) == 2 else streams[0])


@dataclass(frozen=True)
class Epoch:
    sid: bytes
    epoch: int
    hdr: bytes
    nonce_a: bytes
    gamma1: bytes
    nonce_b: bytes
    gamma2: bytes
    nonce_c: bytes
    payload_bits: int
    stego: bytes = b""
    tag: bytes = b""

    def validate(self, complete=True):
        uint(self.epoch)
        uint(self.payload_bits)
        if not 1 <= self.payload_bits <= 8 * MAX_FRAME_BYTES:
            raise ProtocolError("Payload length out of range")
        if len(self.sid) != 16 or len(self.hdr) != 16:
            raise ProtocolError("Invalid session/header length")
        nonces = (self.nonce_a, self.nonce_b, self.nonce_c)
        if any(len(n) != 16 for n in nonces) or len(set(nonces)) != 3:
            raise ProtocolError("Invalid or repeated channel nonces")
        if any(not isinstance(v, bytes) or len(v) > MAX_FRAME_BYTES
               for v in (self.gamma1, self.gamma2, self.stego)):
            raise ProtocolError("Invalid content")
        if not self.gamma1 or not self.gamma2:
            raise ProtocolError("Empty cover parameter")
        if complete and (not self.stego or len(self.tag) != 32):
            raise ProtocolError("Incomplete authenticated epoch")


def context(e):
    return enc(((b"version", VERSION), (b"session", e.sid), (b"epoch", uint(e.epoch)),
                (b"header", e.hdr), (b"nonce-a", e.nonce_a), (b"nonce-b", e.nonce_b),
                (b"nonce-c", e.nonce_c), (b"payload-bits", uint(e.payload_bits)),
                (b"codec", CODEC)))


def parameter_bits(e):
    size = (e.payload_bits + 7) // 8
    a = hashlib.shake_256(enc(((b"domain", b"cp1"), (b"object", e.gamma1)))).digest(size)
    b = hashlib.shake_256(enc(((b"domain", b"cp2"), (b"object", e.gamma2)))).digest(size)
    return truncate(xor(a, b), e.payload_bits)


def mask(master, e):
    e.validate(complete=False)
    rho = expand(key(master, e.sid, b"payload-mask"), b"payload-mask", context(e),
                 (e.payload_bits + 7) // 8)
    return truncate(xor(parameter_bits(e), rho), e.payload_bits)


def mask_message(master, e, message):
    return truncate(xor(truncate(message, e.payload_bits), mask(master, e)), e.payload_bits)


def embedding_seed(master, e):
    # pySTC uses this 31-bit permutation seed. Confidentiality rests on K_mask,
    # never on the seed's limited space.
    return int.from_bytes(mac(key(master, e.sid, b"embedding"), b"stc", context(e))[:4],
                          "big") & 0x7fffffff


def transcript(e):
    return enc(((b"context", context(e)),
                (b"channel", b"C1"), (b"nonce", e.nonce_a), (b"content", e.gamma1),
                (b"channel", b"C2"), (b"nonce", e.nonce_b), (b"content", e.gamma2),
                (b"channel", b"C3"), (b"nonce", e.nonce_c), (b"content", e.stego)))


def authenticate(master, e):
    from dataclasses import replace
    return replace(e, tag=mac(key(master, e.sid, b"epoch-mac"), b"epoch", transcript(e)))


@dataclass
class ReceiverState:
    accepted: set = field(default_factory=set)
    nonces: set = field(default_factory=set)

    @classmethod
    def read(cls, path):
        if not Path(path).exists():
            return cls()
        data = json.loads(Path(path).read_text())
        return cls(set(data["accepted"]), set(data["nonces"]))

    def save(self, path):
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(dict(accepted=sorted(self.accepted), nonces=sorted(self.nonces))))
        tmp.replace(path)


def verify(master, e, expected_sid, state):
    e.validate()
    if e.sid != expected_sid:
        raise ProtocolError("Wrong session")
    ident = e.sid.hex() + ":" + str(e.epoch)
    if ident in state.accepted or any(n.hex() in state.nonces for n in
                                      (e.nonce_a, e.nonce_b, e.nonce_c)):
        raise ProtocolError("Replay/freshness failure")
    if not hmac.compare_digest(e.hdr, header(master, e.sid, e.epoch)):
        raise ProtocolError("Header authentication failure")
    expected = mac(key(master, e.sid, b"epoch-mac"), b"epoch", transcript(e))
    if not hmac.compare_digest(e.tag, expected):
        raise ProtocolError("Complete-epoch authentication failure")


def recover(master, e, expected_sid, state, decode):
    verify(master, e, expected_sid, state)
    extracted = decode(e.stego, embedding_seed(master, e))
    if len(extracted) != (e.payload_bits + 7) // 8:
        raise ProtocolError("Decoded length differs from authenticated length")
    message = mask_message(master, e, extracted)
    # Called only after authentication and successful decoding; this state must
    # be persisted by the host before another acceptance is processed.
    state.accepted.add(e.sid.hex() + ":" + str(e.epoch))
    state.nonces.update(n.hex() for n in (e.nonce_a, e.nonce_b, e.nonce_c))
    return message


def write_frames(e, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    common = dict(version=VERSION.decode(), session=e.sid.hex(), epoch=e.epoch, header=e.hdr.hex())
    for number, nonce, content in [(1, e.nonce_a, e.gamma1), (2, e.nonce_b, e.gamma2),
                                    (3, e.nonce_c, e.stego)]:
        record = dict(common, channel=f"C{number}", nonce=nonce.hex(),
                      content=base64.b64encode(content).decode())
        if number == 3:
            record.update(codec=CODEC.decode(), payload_bits=e.payload_bits, tag=e.tag.hex())
        (folder / f"C{number}.json").write_text(json.dumps(record, sort_keys=True))


def read_frames(folder):
    records = []
    try:
        for number in [1, 2, 3]:
            path = Path(folder) / f"C{number}.json"
            if path.stat().st_size > 2 * MAX_FRAME_BYTES:
                raise ProtocolError("Oversized encoded frame")
            def unique(items):
                d = {}
                for k, v in items:
                    if k in d:
                        raise ProtocolError("Duplicate JSON key")
                    d[k] = v
                return d
            r = json.loads(path.read_text(), object_pairs_hook=unique)
            allowed = {"version", "session", "epoch", "header", "channel", "nonce", "content"}
            if number == 3:
                allowed |= {"codec", "payload_bits", "tag"}
            if set(r) != allowed or r["version"] != VERSION.decode() or r["channel"] != f"C{number}":
                raise ProtocolError("Invalid fields/version/channel")
            r["raw"] = base64.b64decode(r["content"], validate=True)
            records.append(r)
        first, second, third = records
        for r in records[1:]:
            if any(r[k] != first[k] for k in ["session", "epoch", "header"]):
                raise ProtocolError("Cross-channel metadata mismatch")
        if third["codec"] != CODEC.decode():
            raise ProtocolError("Unsupported codec")
        e = Epoch(bytes.fromhex(first["session"]), first["epoch"], bytes.fromhex(first["header"]),
                  bytes.fromhex(first["nonce"]), first["raw"], bytes.fromhex(second["nonce"]),
                  second["raw"], bytes.fromhex(third["nonce"]), third["payload_bits"],
                  third["raw"], bytes.fromhex(third["tag"]))
        e.validate()
        return e
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(str(exc)) from exc
