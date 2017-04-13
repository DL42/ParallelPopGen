# Parallel PopGen Package v0.3 README

The Parallel PopGen Package is a library of CUDA C/C++ APIs consisting of the GO_Fish simulation API (with an associated optional Sim_Model API) and the Spectrum API to simulate allele trajectories and generate SFS respectively. Programs using the APIs can compiled on Mac OSX, Linux, and Windows operating systems. An offline link to the API manual can be found in documents/API.html (Mac/Linux softlink for convenience) and docs/index.html (Mac/Linux/Windows). Online access for the manual of the latest release is at: [http://dl42.github.io/ParallelPopGen/](http://dl42.github.io/ParallelPopGen/). The documentation/ folder also contains a copy of the paper: "Accelerating Wright-Fisher Forward Simulations on the Graphics Processing Unit", which should be cited if the Parallel PopGen Package is used (currently on [bioRxiv](http://biorxiv.org/content/early/2017/04/11/042622)).

Parallel PopGen Package requires an NVIDIA GPU with a minimum of compute capability 3.0. [List of NVIDIA GPUs.](https://developer.nvidia.com/cuda-gpus)  

To use the API package, [install CUDA Toolkit 6.5. or greater.](https://developer.nvidia.com/cuda-toolkit) For those with [Install CUDA Toolkit 6.5.](https://developer.nvidia.com/cuda-toolkit-65) and Maxwell 2 GPUs (GeForce GTX9xx GPUs) first install the previous toolkit to get the drivers, then [download CUDA Toolkit 6.5 here to get the updated toolkit.](https://developer.nvidia.com/cuda-downloads-geforce-gtx9xx)

Once installed, then for Unix systems (Mac + Linux), add nvcc (the CUDA compiler) to your Terminal's Path by adding the following to your .bash_profile or .bashrc:

export PATH=$PATH:/usr/local/cuda/bin  
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib

Or alter the path in the makefile from nvcc to the location of nvcc on your hard drive. (e.g. for Macs the path would be /Developer/NVIDIA/CUDA-6.5/bin/nvcc - or /usr/local/cuda/bin/nvcc should work for the most recent CUDA install for general Unix systems).

Code can be developed and compiled in an Integrated Development Environment - [Nsight](http://www.nvidia.com/object/nsight.html), which comes with the CUDA toolkit, is recommended which on Linux/Mac is an Eclipse IDE and on Windows is a Visual Studio IDE, each modified by Nvidia for CUDA development. Also, in folder examples/ are several example programs using the API with custom makefiles. Standard makefile generator programs should also work.

Change-log:

GO_Fish v0.9:

- minor changes and major bug fixes to simulation algorithm
- API creation and documentation


Sim_Model v0.9:

- API creation and documentation


Spectrum v0.1:

- prototype API for SFS generation