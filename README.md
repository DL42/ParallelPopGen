# GO Fish v0.1

Through CUDA, GO Fish should run on Mac OSX, Linux, and Windows operating systems. 

GO Fish requires an NVIDIA GPU with a minimum of compute capability 3.0. [List of NVIDIA GPUs.](https://developer.nvidia.com/cuda-gpus) If you have a GPU with higher compute capability than 3.0 you can replace the -arch=compute_30 with -arch=compute_35 or -arch=compute_50 or -arch=compute_52 depending on your GPU. GO Fish will run with -arch=compute_30 on higher compute capability GPUs, but *may* run faster with the compute flag specific to your GPU. 

[Install CUDA Toolkit 6.5.](https://developer.nvidia.com/cuda-toolkit-65) I have not yet tested GO Fish on Toolkit 7.0 or higher. 

Once installed, to add nvcc (the CUDA compiler) to your Terminal's Path, add the following to your .bash_profile or .bashrc:
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib

Or alter the path in the makefile from nvcc to the location of nvcc on your hard drive. (e.g. for Macs the path would be /Developer/NVIDIA/CUDA-6.5/bin/nvcc)

Currently the code is set to run the speed test described [in the GO Fish paper.](http://dx.doi.org/10.1101/042622) More scenario functions for demography, mutation, inbreeding, etc ...  are available for testing. At the moment, only a simple parallel histogram function is available to create a site frequency spectrum of a single population. 
