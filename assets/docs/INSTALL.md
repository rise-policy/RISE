# 🛠️ Installation

## 💻 Environments

Please follow the instructions to install the conda environments and the dependencies of the codebase. We recommend using CUDA 11.4 or 11.7 during installations to avoid compatibility issues. For CUDA 12.1, we provide [the modified MinkowskiEngine for CUDA 12.1](https://github.com/chenxi-wang/MinkowskiEngine/tree/cuda-12-1).

1. Create a new conda environment and activate the environment.
    ```bash
    conda create -n rise python=3.8
    conda activate rise
    ```

2. Manually install cudatoolkit, then install necessary dependencies.
    ```bash
    pip install -r requirements.txt
    ```

3. Install MinkowskiEngine. We have modified MinkowskiEngine for better adpatation.
    ```bash
    mkdir dependencies && cd dependencies
    conda install openblas-devel -c anaconda
    export CUDA_HOME=/path/to/cuda
    git clone git@github.com:chenxi-wang/MinkowskiEngine.git
    cd MinkowskiEngine
    python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas_library_dirs=${CONDA_PREFIX}/lib --blas=openblas
    cd ../..
    ```
    For CUDA 12.1, switch to the `cuda-12-1` branch after cloning the modified repository.
    ```bash
    git checkout -b cuda-12-1 origin/cuda-12-1
    ```

4. Install [Pytorch3D](https://github.com/facebookresearch/pytorch3d) manually.
    ```bash
    cd dependencies
    git clone git@github.com:facebookresearch/pytorch3d.git
    cd pytorch3d
    pip install -e .
    cd ../..
    ```


## 🦾 Real Robot

**Hardwares**.
- Flexiv Rizon 4 Robotic Arm (or other robotic arms)
- Dahuan AG-95 Gripper (or other grippers)
- Intel RealSense RGB-D Camera (D415/D435/L515)

**Softwares**.
- Ubuntu 20.04 (tested) with previous environment installed.
- If you are using Flexiv Rizon robotic arm, install the [Flexiv RDK](https://rdk.flexiv.com/manual/getting_started.html) to allow the remote control of the arm. Specifically, download [FlexivRDK v0.9](https://github.com/flexivrobotics/flexiv_rdk/releases/tag/v0.9) and copy `lib_py/flexivrdk.cpython-38-[arch].so` to the `device/robot/` directory. Please specify `[arch]` according to your settings. For our platform, `[arch]` is `x86_64-linux-gnu`.
- If you are using Dahuan AG-95 gripper, install the following python packages for communications.
  ```
  pip install pyserial==3.5 modbus_tk==1.1.3 
  ```
- If you are using Intel RealSense RGB-D camera, install the python wrapper `pyrealsense2` of `librealsense` according to [the official installation instructions](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python#installation).
