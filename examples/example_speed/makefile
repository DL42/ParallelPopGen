# Description of Mac/Linux/Unix Makefile for example_speed. 
#
#############################
# build_path := Where to build program and put executable (note: folder must already exist)
# api_path_source := Location of API source folder
# api_path_include := Location of API include folder
# EXEC_FILE := Name of executable 
#
# NVCC := Compiler path, in this case nvcc is in $PATH
# CFLAGS := Compiler Flags: optimize most, fast math, add API include folder to include search path, equivalent to --relocatable-device-code=true --compile
# CODE := GPU types for which to build explicitly (I have a NVIDIA GTX 780M and 980) https://developer.nvidia.com/cuda-gpus
#
# object_names := Objects required for executable
# objects := Prepends build path to object names
#
# all := Command 'make' or 'make all' builds executable file $(EXEC_FILE) in location $(build_path)/
#
# ##### Object Dependencies Lists #####
# If one of the files on the right hand side of the : changes, the corresponding object must be recompiled.
# This is a users makefile - it assumes there will be no changes to the GOFish API files, so does not include
# all the non-*.cu (non source file) dependencies. If changes to the API .h or .cuh files are expected see
# Object Dependencies Lists in example_dadi or run 'make clean' before each 'make all'.
# ##### End Object Dependencies Lists #####
#
# $(objects) := Make target all objects
# 	Compile source code into objects, $< := dependencies from Object Dependencies Lists, $@ := object in $(objects)
#
# $(build_path)/$(EXEC_FILE) := Make target executable EXEC_FILE which depends on all objects
# 	Link objects into executable EXEC_FILE, $@ := $(build_path)/$(EXEC_FILE)
#
# .PHONY := Defines 'all' and 'clean' as not true targets (i.e. don't remake executable if can't find files called 'all' or 'clean')
#
# clean := Action to perform for command 'make clean'
# 	Remove all objects and EXEC_FILE from build_path
#############################

build_path = ../../../build/example_speed
api_path_source = ../../source
api_path_include = ../../include
EXEC_FILE = GOFish

NVCC = nvcc
CFLAGS = -O3 --use_fast_math -I $(api_path_include)/ -dc
CODE = --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_52,code=sm_52 

object_names = run.o spectrum.o shared.o go_fish_impl.o
objects = $(addprefix $(build_path)/,$(object_names))

all:$(build_path)/$(EXEC_FILE)

##### OBJECT DEPENDENCIES #####
$(build_path)/run.o: run.cu
$(build_path)/spectrum.o: $(api_path_source)/spectrum.cu
$(build_path)/shared.o: $(api_path_source)/shared.cu
$(build_path)/go_fish_impl.o: $(api_path_source)/go_fish_impl.cu
##### END OBJECT DEPENDENCIES #####

$(objects):
	$(NVCC) $(CODE) $(CFLAGS) $< -o $@

$(build_path)/$(EXEC_FILE): $(objects)

	$(NVCC) $(CODE) $(objects) -o $@

.PHONY: all clean

clean:
	rm -f $(objects) $(build_path)/$(EXEC_FILE)