import json
import time

# Message layer operations
from offline import offline
from send import send
from receive import receive

REPS = 30
model_names = ["church", "celebahq", "bedroom", "cat"]

all_results = []
# This message will be padded to the maximum available length by send()
sample_msg = b"Attack@Dawn"

for model_name in model_names:
    for rep in range(REPS):
        print("=== MODEL {}, INDEX {} ===".format(model_name, rep))
        rep_result = {}
        rep_result["model"] = model_name
        rep_result["index"] = rep

        try:
            print("--- offline() ---")
            rep_result["offline"] = offline(model_name, 1, rep)

            print("--- send() ---")
            offline_state = "offline-states/{}_{}.pt".format(model_name, rep)
            img_vector = "images/{}_{}.png".format(model_name, rep)
            rep_result["send"] = send(model_name, offline_state, sample_msg, img_vector)

            print("--- receive() ---")
            hashes_dir = "offline-states/"

            recv_msg, rep_result["receive"] = receive(img_vector, hashes_dir)
            if sample_msg not in recv_msg:
                # Make sure that the sent message is actually received
                rep_result["error"] = "message in send() not in receive()"
        except ValueError as e:
            rep_result["error"] = str(e)

        if "error" in rep_result:
            print("!!! {}".format(rep_result["error"]))

        # Compile the results together and continue
        all_results.append(rep_result)

# Dump everything to json for analysis
with open("benchmarks/vector_{}.json".format(int(time.time())), "w") as f:
    json.dump(all_results, f)
