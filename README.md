# Parallel PopGen Package v0.3 README

The Parallel PopGen Package (3P) is a library of CUDA C/C++ APIs consisting of the GO_Fish simulation API (with an associated optional Sim_Model API) and the Spectrum API to simulate allele trajectories and generate SFS respectively. Programs using the APIs can compiled on Mac OSX, Linux, and Windows operating systems. An offline link to the API manual can be found in documents/API.html (Mac/Linux softlink for convenience) and docs/index.html (Mac/Linux/Windows). Online access for the manual of the latest release is at: [http://dl42.github.io/ParallelPopGen/](http://dl42.github.io/ParallelPopGen/). The documentation/ folder also contains a copy of the paper: "Accelerating Wright-Fisher Forward Simulations on the Graphics Processing Unit", which should be cited if the Parallel PopGen Package is used (currently on [bioRxiv](http://biorxiv.org/content/early/2017/04/11/042622)).

Parallel PopGen Package requires an NVIDIA GPU with a minimum of compute capability 3.0. [List of NVIDIA GPUs.](https://developer.nvidia.com/cuda-gpus)  

To use the API package, [install CUDA Toolkit 6.5. or greater.](https://developer.nvidia.com/cuda-toolkit) For those with [CUDA Toolkit 6.5.](https://developer.nvidia.com/cuda-toolkit-65) and Maxwell 2 GPUs (e.g. GeForce GTX9xx GPUs) first install the previous toolkit to get the drivers, then [download CUDA Toolkit 6.5 here to get the updated toolkit.](https://developer.nvidia.com/cuda-downloads-geforce-gtx9xx) For MacOS, Linux. Pascal GPUs (e.g. GeForce GTX10xx GPUs) require CUDA 8 or above. 

Occasionally for newest operating system versions with the newest compilers, Nvidia can take awhile to issue an update to sync the CUDA compiler, nvcc, with clang, gcc, msvc tools. An example error for MacOS Sierra with the newest Xcode command line tools (CLT): 

>	nvcc fatal   : The version ('80100') of the host compiler ('Apple clang') is not supported*. 

An example of this error for Ubuntu 16.04:

>	/usr/include/string.h: In function ‘void* __mempcpy_inline(void*, const void*, size_t)’:
>
>	/usr/include/string.h:652:42: error: ‘memcpy’ was not declared in this scope
>
>   return (char *) memcpy (__dest, __src, __n) + __n;"

Solutions: 1. wait for CUDA update (e.g. CUDA 8 fixes above error for Ubuntu 16.04). 2. For the Mac example above downgrading from command line tools [(CLT) 8.3 to 8.2 works](https://github.com/arrayfire/arrayfire/issues/1384) while waiting for the CUDA update.

Once installed, then for Unix systems (Mac + Linux), add nvcc (the CUDA compiler) to your Terminal's Path by adding the following to your .bash_profile or .bashrc:

export PATH=$PATH:/usr/local/cuda/bin  
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib

Or alter the path in the makefile from nvcc to the location of nvcc on your hard drive. (e.g. for Macs the path would be /Developer/NVIDIA/CUDA-6.5/bin/nvcc - or /usr/local/cuda/bin/nvcc should work for the most recent CUDA install for general Unix systems).

Code can be developed and compiled in an Integrated Development Environment - [Nsight](http://www.nvidia.com/object/nsight.html), which comes with the CUDA toolkit, is recommended which on Linux/Mac is an Eclipse IDE and on Windows is a Visual Studio IDE, each modified by Nvidia for CUDA development. Also, in folder examples/ are several example programs using the API with custom makefiles written for Linux/macOS. Standard makefile generator programs should also work.

All of the API files are in folder 3P/. The files in folders _internal/ and _outside_libraries/ within 3P/ contain implementations of the methods used in the package. All example makefiles and example source programs are in the folder examples/. Documentation for the examples and their respective makefiles can be found here: [http://dl42.github.io/ParallelPopGen/examples.html](http://dl42.github.io/ParallelPopGen/examples.html).   

Change-log 3P v0.3.2:

- updated file and folder structure
- updated README and API Documentation
- updated example makefiles which change build location to be within source code folder (no extra folder needs to be created by user)

GO_Fish v0.9.1:

- updated include statements to reflect new file structure

Sim_Model v0.9.1:

- updated include statements to reflect new file structure

Spectrum v0.1.1:

- updated include statements to reflect new file structure