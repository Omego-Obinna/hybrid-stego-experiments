import sys
import torch
import pdqhash
import cv2
import os
import json
import time

# adding pulsar submodule to search path
sys.path.insert(1, "pulsar/")
import pulsar


def offline(model_name, num_to_gen, seed):
    model_repo = "google/ddpm-{}-256".format(model_name)
    if model_repo not in pulsar.REPOS:
        raise ValueError("invalid model selection {}".format(model_name))

    hash_vectors = []
    experiment_result = {}

    def pickleFunc(pulsarMod, fName, m_len):
        del pulsarMod.model
        torch.save(pulsarMod, fName)

    for i in range(seed, seed + num_to_gen):
        seed_bytes = i.to_bytes(2, "big")
        load_start = time.time()
        alice_stego = pulsar.Pulsar(seed=seed_bytes, repo=model_repo)
        load_end = time.time()
        experiment_result["model_load"] = (
            load_end - load_start
            if "model_load" not in experiment_result
            else experiment_result["model_load"]
        )

        model_start = time.time()
        m_len = alice_stego.estimate_regions(
            n_hist_bins=100, n_to_gen=1, end_to_end=True
        )
        model_end = time.time()
        experiment_result["model_run"] = (
            model_end - model_start
            if "model_run" not in experiment_result
            else experiment_result["model_run"]
        )

        alice_stego.m_len = m_len
        experiment_result["m_len"] = (
            m_len if "m_len" not in experiment_result else experiment_result["m_len"]
        )

        pickle_start = time.time()
        tempName = "offline-states/{}_{}.pt".format(model_name, i)
        pickleFunc(alice_stego, tempName, m_len)
        pickle_end = time.time()
        experiment_result["serialization"] = (
            pickle_end - pickle_start
            if "serialization" not in experiment_result
            else experiment_result["serialization"]
        )
        experiment_result["size_serial"] = os.path.getsize(tempName)

        # Pulsar stores a PNG of the estimated offline state
        # We use this PNG to create a PDQ hash for the state
        hash_start = time.time()
        image = cv2.imread(os.environ["CPM_NATIVE_TEMP"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hash_vector, quality = pdqhash.compute(image)
        hash_end = time.time()
        experiment_result["hash"] = (
            hash_end - hash_start
            if "hash" not in experiment_result
            else experiment_result["hash"]
        )

        hash_vector_list = hash_vector.tolist()
        hash_vectors.append(
            {
                "seed": i,
                "m_len": m_len,
                "hash_vector": hash_vector_list,
                "quality": quality,
            }
        )

    json_filename = "offline-states/{}_hashes.json".format(model_name)
    with open(json_filename, "w") as json_file:
        json.dump(hash_vectors, json_file, indent=4)
    experiment_result["size_hash"] = os.path.getsize(json_filename)

    return experiment_result


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 offline.py <model_name> <num_to_gen> <seed>")
        sys.exit(1)

    model_name = sys.argv[1]
    num_to_gen = int(sys.argv[2])
    seed = int(sys.argv[3])

    experiment_result = offline(model_name, num_to_gen, seed)
    # print(json.dumps(experiment_result))
