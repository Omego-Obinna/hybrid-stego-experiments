import os
import random
import zlib
import math
import time
import csv
import sys
import json
from collections import deque
from typing import *

from raptorq import Encoder, Decoder
from wonderwords import RandomWord

# Message layer operations
from offline import offline
from send import send
from receive import receive

erasure_exp = {}

# MTU = maximum transmission unit for the Raptor code
# $\bar{x} - s$ from Assemblage paper for vector capacity
model_mtu = {
    "church": 618 - 210,
    "celebahq": 418 - 89,
    "bedroom": 376 - 142,
    "cat": 369 - 262,
}


def send_raptor(message: str, model_name: str) -> tuple[list[str], int, dict]:
    # Assuming we weren't going to use Raptor codes, this would be
    # how many MTUs (i.e., images) we need to send the whole message
    num_mtu = math.ceil(len(message) / model_mtu[model_name])

    # Create the Encoder, with an MTU defined above
    start_time = time.time()
    encoder = Encoder.with_defaults(message, model_mtu[model_name])
    end_time = time.time()
    erasure_exp["time to create encoder"] = round(end_time - start_time, 6)

    # Perform the encoding, and serialize to bytes for transmission
    # The amount = 50% more than the number of packets we would need, rounded up
    start_time = time.time()
    packets = encoder.get_encoded_packets(math.ceil(num_mtu * 0.5))
    end_time = time.time()
    erasure_exp["time to encode and serialize"] = round(end_time - start_time, 6)

    erasure_exp["total # of packets generated"] = len(packets)

    min_reconstruct = num_mtu
    erasure_exp["minimum number of packets needed to reconstruct"] = min_reconstruct

    # Encode each packet into a Pulsar image
    # First, run the offline step to generate enough offline states
    # The number of offline states we generate is related to the *average capacity* of the model.
    num_vectors = len(packets)
    start_time = time.time()
    offline(model_name, num_vectors, 1)
    end_time = time.time()
    erasure_exp["time to perform offline step"] = end_time - start_time

    erasure_exp["number of Pulsar offline states"] = num_vectors

    # Next, extract the maximum lengths for each offline state
    max_len = []
    with open("offline-states/{}_hashes.json".format(model_name)) as f:
        hashes = json.load(f)
        max_len.extend([x["m_len"] for x in hashes])

    # Finally, embed the image data using Pulsar
    images = []
    for i in range(0, len(packets)):
        if max_len[i] >= model_mtu[model_name]:
            offline_state = "offline-states/{}_{}.pt".format(model_name, i + 1)
            message = packets[i]
            img_vector = "images/{}_{}.png".format(model_name, i + 1)
            send(model_name, offline_state, message, img_vector)
            images.append(img_vector)

    return images, len(packets[0])


def receive_raptor(
    images: list[str], model_name: str, message_len: int, packet_len: int
):
    # Decode each packet from a Pulsar image
    packets_rcvd = []
    for file_name in images:
        recv_msg, _ = receive(file_name, "offline-states/")
        packets_rcvd.append(recv_msg[:packet_len])

    start_time = time.time()
    decoder = Decoder.with_defaults(message_len, model_mtu[model_name])
    end_time = time.time()
    erasure_exp["time to create decoder"] = round(end_time - start_time, 6)

    # Perform the decoding
    start_time = time.time()
    result = None
    for packet in packets_rcvd:
        result = decoder.decode(packet)
        if result is not None:
            break
    end_time = time.time()
    erasure_exp["time to perform decode"] = round(end_time - start_time, 6)

    return result


if __name__ == "__main__":
    model_name = sys.argv[1]
    # record iteration count from runner script
    erasure_exp["run"] = int(sys.argv[2])

    # Generate random words as the data to send
    data = RandomWord().random_words(int(sys.argv[3]))
    data = " ".join(data).encode("utf-8")
    erasure_exp["data length"] = len(data)

    # Compress the data
    data = zlib.compress(data)
    erasure_exp["compressed length"] = len(data)

    try:
        start_time = time.time()
        images, packet_len = send_raptor(data, model_name)
        end_time = time.time()
        erasure_exp["time to send with raptor"] = end_time - start_time
        erasure_exp["time to send only without offline"] = (
            erasure_exp["time to send with raptor"]
            - erasure_exp["time to perform offline step"]
        )

        erasure_exp["number of Raptor packets in images"] = len(images)

        # Here we simulate losing some of the packets randomly.
        # Reconstruction probability after receiving K + h packets = 1 - 1/256^(h + 1).
        # Where K is the number of packets in the original message,
        # and h is the number of additional packets received.
        random.shuffle(images)
        min_reconstruct = erasure_exp["minimum number of packets needed to reconstruct"]
        if len(images) >= min_reconstruct:
            images = images[:min_reconstruct]

            start_time = time.time()
            result = receive_raptor(images, model_name, len(data), packet_len)
            end_time = time.time()
            erasure_exp["time to receive with raptor"] = end_time - start_time

            # Check that even though some of the data was lost we are able to reconstruct the original message
            assert result == data
        else:
            # Not enough images to reconstruct = failure, and don't try to
            pass
    except Exception as e:
        print(e, file=sys.stderr)

    print(json.dumps(erasure_exp))
