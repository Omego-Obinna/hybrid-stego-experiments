# Phase 2: real-image timed-window active attacks

This is a new experimental adapter around the frozen `cpm-epoch-v2` pilot protocol,
not a modification of that protocol or a reproduction of the older abstract
transport simulator. The manuscript has not been edited by this experiment.

## Frozen design

- Corpus: pilot epochs 0–29, ten development covers at 0.10/0.20/0.40 bpp.
  The complete JSON frames, sender transcript hashes and original messages are
  reused. This phase does not create new stego images or use held-out covers.
- Eleven scenarios, two profiles, seeds 11/23/37/53/71: 110 traces and 3,300
  offered epoch opportunities. These repeat the same 30 authenticated fixtures.
  They are correlated simulation traces, not 3,300 independent security trials.
- Profiles: W=3 / timeout=200 ms and W=5 / timeout=400 ms. Both use 100 ms
  epoch spacing, maximum selected-triple skew 100 ms, two candidates per
  channel/epoch, 8 MiB retained serialized-frame budget, and 128 cached failed
  triples per epoch. The paired profiles do not isolate the effect of W from
  the effect of timeout.
- Normal arrivals occur 20–35 ms after send. Bounded reordering chooses a
  shared 20 or 145 ms epoch delay plus 0–15 ms channel jitter, allowing later
  epochs to complete first. Burst disruption affects epochs 8–11; the clean
  tail is epochs 12–29. Excess delay adds 350 ms to C2. Persistent suppression
  removes every C2 frame. No ECC or heartbeat recovery is assumed.
- Injection flood: 16 altered frames on each channel for each of four target
  epochs, before the legitimate arrivals. The target header is copied, not
  forged. Changes include recorded public content rebound to new context,
  altered parameter/image bytes, tag changes and authenticated payload length
  changes. A separate adaptive trace makes two further C1 injections per
  target epoch in response to a public rejection. It sees no receiver state,
  secret, ground truth or success oracle beyond this explicit packet response.
- Malformed/wrong-context probes cover duplicate JSON keys, wrong session,
  wrong header and invalid base64. Replay occurs after an epoch's first
  acceptance. Restart replay reloads the saved state and clears transient
  buffers; it is a state-reload test within the A5 assumption, not an OS/power
  crash or endpoint-compromise experiment.

## Exact timeout and window policy

The experiment explicitly assumes a shared initial epoch clock, with epoch e
sent at 100e ms. At time t, the window starts at

`a(t) = max(0, floor((t - timeout_ms) / 100) + 1)`.

Epoch e therefore expires at `100e + timeout_ms`; equality expires. Header or
packet receipt never advances this window. A candidate triple must also have
maximum minus minimum arrival time at most 100 ms and remain within its
candidate lifetime. Failed triples are cached; individual components can be
recombined until FIFO eviction or timeout. Candidate/cache counts and retained
serialized bytes are bounded. This is a declared instantiation of the
manuscript's agreed skip policy. Clock uncertainty, heartbeat alignment and
real scheduler traces are not evaluated here.

## Receiver/evaluator separation

The receiver runs in a separate process and receives only frame bytes,
virtual arrival times, shared configuration/session/secret and replay state.
It uses the frozen canonical three-frame parser, full-transcript HMAC,
masking and native pySTC image extractor. It gets no original cover, sender
mask, plaintext, expected outcome or attack label. Public front-end checks
precede bounded buffering. Invalid complete MACs must not call the decoder or
change accepted-epoch/nonce state. State is saved, file and directory flushed,
and read back before plaintext is emitted. The private evaluator then compares
actual plaintext bytes and the complete transcript hash with the authorised
fixture. Freshness checks prevent duplicate release.

Recovered epochs are labelled by epoch for ordered reassembly. The evaluator
reports whether the whole 30-fragment set is complete; missing fragments are
erasures, never fabricated payload or assumed code recovery. The adapter does
not release a reconstructed complete message when that set is incomplete.

## Measures and interpretation

Each trace preserves event times, frame hashes, public decisions, exact
accepted plaintext comparisons, transcript identities, rejection counters,
full verification attempts, decoder calls, buffer maxima and persistent state
size. Rejections preserve freshness state. The full-MAC attempt count is the
verification history for this finite trace, not a MAC-forgery bound.

Report accepted/erased epochs, unauthorised transcript acceptance, wrong
plaintext, repeated epoch acceptance and clean-tail recovery separately.
Virtual arrival latency excludes CPU service/queueing time; actual receiver
work is measured separately as wall time. Retained serialized-frame bytes
are not process RSS or total memory. Freshness state grows with accepted
epochs and is reported separately from bounded transient buffers.

Zero observed incorrect acceptances is finite functional evidence. Do not
derive a cryptographic epsilon, general denial-of-service resistance,
deployed-network throughput, packet-distortion recovery, covertness or
cross-channel unlinkability from it. Injection budgets, shared-clock
assumptions and authentic frame availability condition delivery outcomes.

## Validation before the recorded batch

`check_boundaries.py` checks skew equality, excessive skew, exclusive epoch
deadline, future-window rejection, failed MAC before decoding/state mutation,
successful later candidate, replay after state reload, byte-budget rejection,
and duplicate JSON keys. Its synthetic decoder checks policy only. A separate
real-image smoke run covers all eleven scenarios; a later reordering check
covers both profiles after adding actual cross-epoch arrival reversal. Those
development outputs remain separate from the frozen 110-trace recorded batch.

No automatic retries or fallback are permitted. Unexpected delivery,
authentication, decoding or invariant outcomes stop the batch and preserve
the trace for investigation.
