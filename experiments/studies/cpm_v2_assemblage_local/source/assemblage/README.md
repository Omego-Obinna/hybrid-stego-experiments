# Assemblage: Chipping Away at Censorship with Generative Steganography (FOCI '26)

A paper by [Tushar Jois](https://tjo.is), [Cora Rowena Ruiz](https://crowenaruiz.com/), and [Gabe Kaptchuk](https://www.cs.umd.edu/~kaptchuk/).

[Our paper](https://www.petsymposium.org/foci/2026/foci-2026-0005.php) appears in [Free and Open Communications on the Internet (FOCI)](https://foci.community) 2026(1), and was presented at the FOCI 2026 winter event. 

## Installation

This project was developed using:

- Python 3.10 (if you use the [`asdf` version manager](https://asdf-vm.com/guide/getting-started.html), a `.tool-versions` file is included)
- macOS Sequoia 15.3.2 with the MPS backend (but CUDA on other platforms should also work)

 To install:
 
 1. Clone this repository recursively to get the Pulsar submodule:

 ```sh
 git clone --recursive https://github.com/spacelab-ccny/assemblage.git
 ```

 2. Follow the installation instructions for Pulsar which are located in the submodule: `pulsar/README.md`.

 3. After completing the Pulsar installation, install additional packages for Assemblage: 

 ```sh
 # In your SageMath Python environment as set up during the Pulsar installation
 pip3 install opencv-python pdqhash raptorq wonderwords
 ```

## Usage

The code is primarily divided into three scripts, each corresponding to an algorithm defined in our paper:

```sh
offline.py # offline() -- Algorithm 1 in the paper
send.py # send() -- Algorithm 2 in the paper
receive.py # receive() -- Algorithm 3 in the paper
```

Each of the above files can be run stand-alone. In addition to these scripts, another script supports the use of erasure codes in Assemblage:

```sh
raptor.py # adds erasure codes to send() and receive()
```

To generate the experiments in the paper, run:

```sh
 # In your SageMath Python environment as set up during the Pulsar installation
 python3 benchmark_vector.py # for the single-vector efficiency experiments
 python3 benchmark_raptor.py # for the erasure-code efficiency experiments
```

## Citation

If you found our code useful in your work, please use the following citation:

```bibtex
@article{assemblage2026,
  title   = {Assemblage: Chipping Away at Censorship with Generative Steganography},
  author  = {Tushar M. Jois and Cora Rowena Ruiz and Gabriel Kaptchuk},
  volume  = {2026},
  number  = {1},
  journal = {Free and Open Communications on the Internet},
  pages   = {33--42},
  month   = Feb,
  year    = {2026}
}
```
