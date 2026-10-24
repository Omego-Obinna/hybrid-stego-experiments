#!/usr/bin/env python


import sys

# adding pulsar submodule to search path
sys.path.insert(1, "pulsar/")

import pulsar
import torch
from diffusers import UNet2DModel
import png
import numpy as np
import os
import time


def prepare_message(ourMsg, m_len):
    if len(ourMsg) > m_len:
        # Throw a ValueError with a custom message
        raise ValueError(
            "the message you want to send (length {}) exceeds the available message length {}".format(
                len(ourMsg), m_len
            )
        )
    # Pad the rest of the size with zeroes
    if len(ourMsg) < m_len:
        # Pad the byte array with zeroes on the right
        ourMsg = ourMsg.ljust(m_len, b"\x00")
    return ourMsg


def send(model_name, offline_state, message, img_vector):
    experiment_result = {}
    model_repo = "google/ddpm-{}-256".format(model_name)
    if model_repo not in pulsar.REPOS:
        raise ValueError("invalid model selection {}".format(model_name))

    unpickle_start = time.time()
    unpickled_instance = torch.load(offline_state, weights_only=False)
    unpickle_end = time.time()
    experiment_result["deserialization"] = unpickle_end - unpickle_start

    m_len = unpickled_instance.m_len
    message = prepare_message(message, m_len)

    # Add the model back
    # We removed it in the offline step to save storage space
    load_start = time.time()
    unpickled_instance.model = UNet2DModel.from_pretrained(os.environ["CPM_NATIVE_MODEL"])
    unpickled_instance.model.to(unpickled_instance.device)
    load_end = time.time()
    experiment_result["model_load"] = load_end - load_start

    # Use Pulsar to generate the vector
    # We have to change into the Pulsar directory since it uses hardcoded paths into sage/
    os.chdir("pulsar/")
    run_start = time.time()
    try:
        alice_generate_results = unpickled_instance.generate_with_regions(message)
    except Exception as e:
        # Clean up the chdir before continuing up the error chain
        os.chdir("../")
        raise e
    last = unpickled_instance.scheduler.num_inference_steps - 1
    alice_hidden_sample = alice_generate_results["samples"][last]["hidden"]
    run_end = time.time()
    experiment_result["encode"] = run_end - run_start
    # All done with Pulsar, so change back
    os.chdir("..")

    image_start = time.time()
    unpickled_instance.save_sample(alice_hidden_sample, img_vector)
    image_end = time.time()
    experiment_result["image_save"] = image_end - image_start
    experiment_result["size_image"] = os.path.getsize(img_vector)

    return experiment_result


def save_sample(sample, fname, dtype=np.uint16):
    image_processed = sample.cpu().permute(0, 2, 3, 1).numpy()
    dtype_max = np.iinfo(dtype).max

    y = image_processed[0]
    z = (y + 1.0) * (dtype_max / 2)
    image_processed = z.astype(dtype)

    # https://stackoverflow.com/a/25814423
    with open(fname, "wb") as f:
        writer = png.Writer(
            width=image_processed.shape[1],
            height=image_processed.shape[0],
            bitdepth=int(np.log2(dtype_max)) + 1,
            greyscale=False,
        )
        image2list = image_processed.reshape(
            -1, image_processed.shape[1] * image_processed.shape[2]
        ).tolist()
        writer.write(f, image2list)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 send.py <path to offline state file> <optional message>")
        sys.exit(1)

    offline_state = sys.argv[1]

    filename = offline_state.split("/")[-1].split(".")[0]
    model_name, seedInt = filename.rsplit("_", 1)

    img_vector = "images/{}_{}.png".format(model_name, seedInt)

    # A sample message -- a Tor bridge string
    # This bridge string is used as an example in the Tor documentation
    sample_msg = b"obfs4 193.11.166.194:27020 86AC7B8D430DAC4117E9F42C9EAED18133863AAF cert=0LDeJH4JzMDtkJJrFphJCiPqKx7loozKN7VNfuukMGfHO0Z8OGdzHVkhVAOfo1mUdv9cMg iat-mode=0"
    message = sys.argv[2] if len(sys.argv) > 2 else sample_msg

    experiment_result = send(
        model_name,
        offline_state,
        sample_msg,
        img_vector,
    )

    # print(json.dumps(experiment_result))
