objects = main.o run.o sfs.o shared.o

all: $(objects)

	nvcc --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_32,code=sm_32 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_50,code=sm_50, --generate-code arch=compute_52,code=sm_52 --generate-code arch=compute_52,code=compute_52 $(objects) -o GOFish

%.o: src/%.cpp
	nvcc -O3 --use_fast_math -c $< -o $@

%.o: src/%.cu
	nvcc --generate-code arch=compute_30,code=sm_30 --generate-code arch=compute_32,code=sm_32 --generate-code arch=compute_35,code=sm_35 --generate-code arch=compute_50,code=sm_50 --generate-code arch=compute_52,code=sm_52 --generate-code arch=compute_52,code=compute_52 -O3 --use_fast_math -I include_cub/ -I include_samples/  -I include_random123/ -dc $< -o $@


clean:
	rm -f *.o GOFish