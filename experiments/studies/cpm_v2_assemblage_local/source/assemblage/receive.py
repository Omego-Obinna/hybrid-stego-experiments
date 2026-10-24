#!/usr/bin/env python

import torch
import pdqhash
import time
import json
import cv2
import os
import sys

# adding pulsar submodule to search path
sys.path.insert(1, "pulsar/")


def receive(img_vector, hashes_dir):
    model_name, seed_val, experiment_result = getSeed(hashes_dir, img_vector)
    fName = "offline-states/{}_{}.pt".format(model_name, seed_val)

    unpickle_start = time.time()
    bob_stego = torch.load(fName, weights_only=False)
    unpickle_end = time.time()
    experiment_result["deserialization"] = unpickle_end - unpickle_start

    image_start = time.time()
    received_hidden_sample = bob_stego.load_sample(img_vector)
    image_end = time.time()
    experiment_result["image_load"] = image_end - image_start

    # Use Pulsar to decode the vector
    # We have to change into the Pulsar directory since it uses hardcoded paths into sage/
    os.chdir("pulsar/")
    run_start = time.time()
    try:
        message = bob_stego.reveal_with_regions(received_hidden_sample)
    except Exception as e:
        # Clean up the chdir before continuing up the error chain
        os.chdir("../")
        raise e

    run_end = time.time()
    experiment_result["decode"] = run_end - run_start
    # All done with Pulsar, so change back
    os.chdir("../")

    return message, experiment_result


def generate_single_json(imgName):
    image = cv2.imread(imgName)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hash_vector, quality = pdqhash.compute(image)

    # Quality Threshold where we recommend discarding hashes
    # https://github.com/facebook/ThreatExchange/tree/main/pdq#matching
    if quality <= 49:
        raise ValueError(
            "provided image {} has low quality when hashed ({})".format(
                imgName, quality
            )
        )

    # Convert hash_vector from ndarray to list
    return hash_vector.tolist()


def load_hashes_from_dir(directory):
    json_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            model_name = filename.split("_")[0]
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r") as file:
                    data = json.load(file)
                    json_data[model_name] = data
            except json.JSONDecodeError as e:
                print(f"Error parsing {filename}: {e}")
    return json_data


def getSeed(hashes_dir, pngName):
    experiment_result = {}

    # Load all offline images
    load_hash_start = time.time()
    hashes = load_hashes_from_dir(hashes_dir)
    load_hash_end = time.time()
    experiment_result["load_hash"] = load_hash_end - load_hash_start

    hash_start = time.time()
    alice_data = generate_single_json(pngName)
    hash_end = time.time()
    experiment_result["hash"] = hash_end - hash_start

    find_start = time.time()
    for model_name in hashes:
        for h in hashes[model_name]:
            # Hamming distance between the image's hash and the offline state's hash
            dist = sum(el1 != el2 for el1, el2 in zip(alice_data, h["hash_vector"]))

            # Distance Threshold to consider two hashes to be similar/matching: <=31
            # https://github.com/facebook/ThreatExchange/tree/main/pdq#matching
            if dist <= 31:
                find_end = time.time()
                experiment_result["find_match"] = find_end - find_start
                return model_name, h["seed"], experiment_result

    raise ValueError("could not find matching hash vector for {}".format(pngName))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python3 receive.py <path to candidate image vector> <path to hash directory>"
        )
        sys.exit(1)

    file_name = sys.argv[1]
    hashes_dir = sys.argv[2]

    # Using just the image file and the directory of hashes, we should be able to receive the data
    hiddenMsg, experiment_result = receive(file_name, hashes_dir)

    # Test Case
    # sample_msg = b"obfs4 193.11.166.194:27020 86AC7B8D430DAC4117E9F42C9EAED18133863AAF cert=0LDeJH4JzMDtkJJrFphJCiPqKx7loozKN7VNfuukMGfHO0Z8OGdzHVkhVAOfo1mUdv9cMg iat-mode=0"
    # assert sample_msg in hiddenMsg

    print(hiddenMsg)
    # print(json.dumps(experiment_result))
