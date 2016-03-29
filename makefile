objects = main.o run.o sfs.o shared.o

all: $(objects)

	nvcc -arch=compute_30 $(objects) -o GOFish

%.o: src/%.cpp
	nvcc -O3 --use_fast_math -c $< -o $@

%.o: src/%.cu
	nvcc -arch=compute_30 -O3 --use_fast_math -I include_cub/ -I include_samples/  -I include_random123/ -dc $< -o $@


clean:
	rm -f *.o GOFish