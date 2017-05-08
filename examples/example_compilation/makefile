# Description of Mac/Linux/Unix Makefile for example_compilation.
# makefile_nvcc is useful when combining GPU-accelerated code into an existing C++ project and using NVCC to compile all files and link objects into executable.
# makefile_gpp is useful when combining GPU-accelerated code into an existing C++ project and user wants to keep non-CUDA code compiled by a different compiler (e.g. g++). 
#
#############################
# build_path := Where to build program and put executable (note: folder must already exist)
# EXEC_FILE := Name of executable 
#
# cpp_object_names := CPP objects required for executable
# cpp_objects := Prepends build path to object names
#
# cuda_object_names := CUDA objects required for executable
# cuda_objects := Prepends build path to object names
#
# nvcc := Command 'make' or 'make nvcc' calls the makefile makefile_nvcc
#
# gpp := Command 'make gpp' calls the makefile makefile_gpp
#
# .PHONY := Defines 'nvvc', 'gpp', and 'clean' as not true targets (i.e. don't remake executable if can't find files called 'all' or 'clean')
#
# clean := Action to perform for command 'make clean'
# 	Remove all objects and EXEC_FILE from build_path
#############################

build_path = ../example_compilation
EXEC_FILE = GOFish

cpp_object_names = main.o
cpp_objects = $(addprefix $(build_path)/,$(cpp_object_names))

cuda_object_names = run.o shared.o go_fish_impl.o
cuda_objects = $(addprefix $(build_path)/,$(cuda_object_names))

nvcc: 
	make -f makefile_nvcc
gpp:
	make -f makefile_gpp

.PHONY: nvcc gpp clean

clean:
	rm -f $(cuda_objects) $(cpp_objects) $(build_path)/gpuCode.o $(build_path)/$(EXEC_FILE)