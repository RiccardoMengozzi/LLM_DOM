INSTALL CUDA TOOLKIT AND CUDA DRIVER AND CUDNN AND libnccl2( MY CASE: TOOLKIT = 11.8, DRIVER = 570.124.06, CUDNN 9)
CUDA Toolkit:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8
```
Cuda driver:
```
sudo apt install nvidia-driver-570 -y
```

CUDNN:
```
sudo apt-get -y install cudnn9-cuda-11
```

libncc2:

```
sudo apt install libnccl2 libnccl-dev

```


then export variables
```
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=./venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export LD_PRELOAD=./venv/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9
```


```
source ~/.bashrc
```

Install pytorch compatible with 11.8:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install genesis-world:
```
pip install genesis-world
```


--- FOR CONDA ---
```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```
 (use system libraries to solve this error):
libGL error: MESA-LOADER: failed to open iris:

NOTES:

Currently envs parallelization is only available for environments with only rigid entities: not possible to simulate
multiple environments using deformable objects....



1.
real	2m19,153s
user	2m56,533s
sys	0m51,706s

2.
