# Installation

Prerequisites:
* Python > 3.10
* [Poetry](https://python-poetry.org)
* patchelf

Suggested build environment:
```bash
# general build dependencies
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
# mujoco dependencies
apt-get -y install wget unzip software-properties-common \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev patchelf
# mujoco installation
curl -OL https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
tar -zxf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco
rm mujoco210-linux-x86_64.tar.gz
```

To install, run

```bash
git checkout ntrlize_one_mass_neuron_in_one_time

pip install -e .

pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install chex==0.1.84 flax==0.8.5 tensorflow-probability==0.24.0

pip uninstall nvidia-cusolver-cu12 
```

For further instructions on running this code on GPU, please follow instructions from [the official repository](https://github.com/google/jax).

For MuJoCo inslattion, you may need to add the following lines in the `.bashrc`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/costa/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

# Development 

If you want to modify the code, install following the instructions above.


# [Examples](examples/)

# Troubleshooting

If you experience out-of-memory errors, especially with enabled video saving, please consider reading [docs](https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html#gpu-memory-allocation) on JAX GPU memory allocation. Also, you can try running with the following environment variable:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 python ...
```

If you run your code on a remote machine and want to save videos for DeepMind Control Suite, please use EGL for rendering:
```bash
MUJOCO_GL=egl python train.py --env_name=cheetah-run --save_dir=./tmp/ --save_video
```

# Tensorboard

Launch tensorboard to see training and evaluation logs

```bash
tensorboard --logdir=./tmp/
```

# Docker

## Build

Copy your MuJoCo key to ./vendor

```bash
cd remote
docker build -t ikostrikov/jaxrl . -f Dockerfile 
```

## Test
```bash
 sudo docker run -v <examples-dir>:/jaxrl/ ikostrikov/jaxrl:latest python /jaxrl/train.py --env_name=HalfCheetah-v2 --save_dir=/jaxrl/tmp/

# On GPU
 sudo docker run --rm --gpus all -v <examples-dir>:/jaxrl/ --gpus=all ikostrikov/jaxrl:latest python /jaxrl/train.py --env_name=HalfCheetah-v2 --save_dir=/jaxrl/tmp/
```

# Acknowledgements 

This code is based on [jaxrl]([https://github.com/evgenii-nikishin](https://github.com/ikostrikov/jaxrl/tree/main)) implemented by Ilya Kostrikov.
