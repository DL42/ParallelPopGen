# GO Fish v0.1

Through CUDA, GO Fish should run on Mac OSX, Linux, and Windows operating systems. 

GO Fish requires an NVIDIA GPU with a minimum of compute capability 3.0. [List of NVIDIA GPUs.](https://developer.nvidia.com/cuda-gpus)  

[Install CUDA Toolkit 6.5.](https://developer.nvidia.com/cuda-toolkit-65) I have not yet tested GO Fish on Toolkit 7.0 or higher. For those with Maxwell 2 GPUs (GeForce GTX9xx GPUs) first install the previous toolkit to get the drivers, then [download CUDA Toolkit 6.5 here to get the updated toolkit.](https://developer.nvidia.com/cuda-downloads-geforce-gtx9xx)

Once installed, to add nvcc (the CUDA compiler) to your Terminal's Path, add the following to your .bash_profile or .bashrc:
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib

Or alter the path in the makefile from nvcc to the location of nvcc on your hard drive. (e.g. for Macs the path would be /Developer/NVIDIA/CUDA-6.5/bin/nvcc)

Currently the code is set to run the speed test described [in the GO Fish paper.](http://dx.doi.org/10.1101/042622) More scenario functions for demography, mutation, inbreeding, etc ...  are available for testing. At the moment, only a simple parallel histogram function is available to create a site frequency spectrum of a single population. 

To compile the code into an executable (GOFish), navigate your terminal to the GOFish folder, type make. To delete the executable and object files, type make clean. 

The make file creates an executable which, for the OS it is compiled on, will run on any NVIDIA GPU compute architecture from 3.0 to 5.2 and will JIT compile for any architecture above 5.2. To decrease compile time and executable size, you can delete any —-generate-code arch=compute_xx,code=sm_xx flag which does not correspond to your GPU's architecture.