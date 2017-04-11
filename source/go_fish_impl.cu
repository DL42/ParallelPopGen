/*
 * go_fish_impl.cu
 *
 *      Author: David Lawrie
 *      implementation of non-template, non-inline functions for GO Fish simulation
 */

#include "go_fish_impl.cuh"

//!\cond
namespace go_fish_details{

/*
 *  CUB scan (sum) phenomenon: float errors in the mse_integral can accumulate differently each run
 *  e.g.
 *  GO_Fish::const_parameter mutation(pow(10.f,-9)); //per-site mutation rate
	GO_Fish::const_parameter inbreeding(1.f); //constant inbreeding
	GO_Fish::const_demography demography(pow(10.f,5)*(1+inbreeding(0,0))); //number of individuals in population, set to maintain consistent effective number of chromosomes
	GO_Fish::const_equal_migration migration(0.f,a.sim_input_constants.num_populations); //constant migration rate
	float gamma = -5; //effective selection
	GO_Fish::const_selection selection(gamma/(2*demography(0,0))); //constant selection coefficient
	GO_Fish::const_parameter dominance(0.f); //constant allele dominance
 *  a.sim_input_constants.compact_interval = 20;
    a.sim_input_constants.num_generations = pow(10.f,3);
    a.sim_input_constants.num_sites = 20*2*pow(10.f,7);
    a.sim_input_constants.seed1 = 0xbeeff00d + 2*14; //random number seeds
    a.sim_input_constants.seed2 = 0xdecafbad - 2*14;

    one solution is to round the results in reverse array
    another is to ignore (currently implemented)
    another is to switch back to using doubles ... at least when summing up the mse_integral (difference in speed was slight)
 */

__global__ void reverse_array(float * array, const int N){
	int myID = blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < N/2; id += blockDim.x*gridDim.x){
		float temp = array[N - id - 1];
		array[N - id - 1] = array[id];
		//float temp = roundf(10000*array[N - id - 1])/10000.f;
		//array[N - id - 1] = roundf(10000*array[id])/10000.f;
		array[id] = temp;
	}
}

__global__ void initialize_mse_mutation_array(float * mutations_freq, const int * freq_index, const int * scan_index, const int offset, const int Nchrom, const int population, const int num_populations, const int array_Length){
	int myIDy = blockIdx.y*blockDim.y + threadIdx.y;
	for(int idy = myIDy; idy < (Nchrom-1); idy+= blockDim.y*gridDim.y){
		int myIDx = blockIdx.x*blockDim.x + threadIdx.x;
		int start = scan_index[offset+idy];
		int num_mutations = freq_index[offset+idy];
		float freq = (idy+1.f)/Nchrom;
		for(int idx = myIDx; idx < num_mutations; idx+= blockDim.x*gridDim.x){
			for(int pop = 0; pop < num_populations; pop++){ mutations_freq[pop*array_Length + start + idx] = 0; }
			mutations_freq[population*array_Length + start + idx] = freq;
		}
	}
}

__global__ void mse_set_mutID(int4 * mutations_ID, const float * const mutations_freq, const int mutations_Index, const int num_populations, const int array_Length, const bool preserve_mutations){
	int myID = blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < mutations_Index; id+= blockDim.x*gridDim.x){
		for(int pop = 0; pop < num_populations; pop++){
			if(mutations_freq[pop*array_Length+id] > 0){
				if(!preserve_mutations){ mutations_ID[id] = make_int4(0,pop,(id+1),0); }
				else{ mutations_ID[id] = make_int4(0,pop,-1*(id+1),0); } //age: eventually will replace where mutations have age <= 0 (age before sim start)//threadID//population//to ensure that ID is non-zero, so that preservation flag can be a -ID
				break; //assumes mutations are only in one population at start
			}
		}
	}
}

/*__global__ void print_Device_array_uint(unsigned int * array, int num){

	for(int i = 0; i < num; i++){
		//if(i%1000 == 0){ printf("\n"); }
		printf("%d: %d\t",i,array[i]);
	}
}

__global__ void sum_Device_array_bit(unsigned int * array, int num){
//	int sum = 0;
	for(int i = 0; i < num; i++){
		//if(i%1000 == 0){ printf("\n"); }
		unsigned int n = array[i];
		while (n) {
		    if (n & 1)
		    	sum+=1;
		    n >>= 1;
		}
		printf("%d\t",__popc(array[i]));
	}
}

__global__ void sum_Device_array_uint(unsigned int * array, int num){
	int j = 0;
	for(int i = 0; i < num; i++){
		j += array[i];
	}
	printf("%d",j);
}

__global__ void sum_Device_array_float(float * array, int start, int end){
	double j = 0;
	for(int i = start; i < end; i++){
		j += array[i];
	}
	printf("%lf\n",j);
}

__global__ void compareDevicearray(int * array1, int * array2, int array_length){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < array_length; id+= blockDim.x*gridDim.x){
		if(array1[id] != array2[id]){ printf("%d,%d,%d\t",id,array1[id],array2[id]); }
	}
}

__global__ void copyDevicearray(int * array1, int * array2, int array_length){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < array_length; id+= blockDim.x*gridDim.x){ array1[id] = array2[id]; }
}

__global__ void compareDevicearray(float * array1, float * array2, int array_length){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < array_length; id+= blockDim.x*gridDim.x){
		if(array1[id] != array2[id]){ printf("%d,%f,%f\t",id,array1[id],array2[id]); return; }
	}
}

__global__ void copyDevicearray(float * array1, float * array2, int array_length){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < array_length; id+= blockDim.x*gridDim.x){ array1[id] = array2[id]; }
}

__global__ void print_Device_array_float(float * array, int num){
	printf("%5.10e\n",array[num]);
}*/

__global__ void add_new_mutations(float * mutations_freq, int4 * mutations_ID, const int prev_mutations_Index, const int new_mutations_Index, const int array_Length, float freq, const int population, const int num_populations, const int generation){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; (id < (new_mutations_Index-prev_mutations_Index)) && ((id + prev_mutations_Index) < array_Length); id+= blockDim.x*gridDim.x){
		for(int pop = 0; pop < num_populations; pop++){ mutations_freq[(pop*array_Length+prev_mutations_Index+id)] = 0; }
		mutations_freq[(population*array_Length+prev_mutations_Index+id)] = freq;
		mutations_ID[(prev_mutations_Index+id)] = make_int4(generation,population,(id+1),0); //to ensure that ID is non-zero, so that preservation flag can be a -ID
	}
}

__global__ void scatter_arrays(float * new_mutations_freq, int4 * new_mutations_ID, const float * const mutations_freq, const int4 * const mutations_ID, const unsigned int * const flag, const unsigned int * const scan_Index, const int padded_mut_Index, const int new_array_Length, const int old_array_Length, const bool preserve_mutations, const int warp_size){
//adapted from https://www.csuohio.edu/engineering/sites/csuohio.edu.engineering/files/Research_Day_2015_EECS_Poster_14.pdf
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	int population = blockIdx.y;

	for(int id = myID; id < (padded_mut_Index >> 5); id+= blockDim.x*gridDim.x){
		int lnID = threadIdx.x % warp_size;
		int warpID = id >> 5;

		unsigned int predmask;
		unsigned int cnt;

		predmask = flag[(warpID<<5)+lnID];
		cnt = __popc(predmask);

		//parallel prefix sum
#pragma unroll
		for(int offset = 1; offset < 32; offset<<=1){
			unsigned int n = __shfl_up(cnt, offset);
			if(lnID >= offset) cnt += n;
		}

		unsigned int global_index = 0;
		if(warpID > 0) global_index = scan_Index[warpID - 1];

		for(int i = 0; i < 32; i++){
			unsigned int mask = __shfl(predmask, i); //broadcast from thread i
			unsigned int sub_group_index = 0;
			if(i > 0) sub_group_index = __shfl(cnt, i-1);
			if(mask & (1 << lnID)){
				int write = global_index + sub_group_index + __popc(mask & ((1 << lnID) - 1));
				int read = (warpID<<10)+(i<<5)+lnID;
				new_mutations_freq[population*new_array_Length + write] = mutations_freq[population*old_array_Length+read];
				if(population == 0){
					if(preserve_mutations){
						int4 ID = mutations_ID[read];
						new_mutations_ID[write] = make_int4(ID.x,ID.y,-1*abs(ID.z),ID.w);
					}else{ new_mutations_ID[write] = mutations_ID[read]; }
				}
			}
		}
	}
}

__global__ void preserve_prev_run_mutations(int4 * mutations_ID, const int mutations_Index){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < mutations_Index; id+= blockDim.x*gridDim.x){ mutations_ID[id].z = -1*abs(mutations_ID[id].z); } //preservation flag is a -ID, use of absolute value is to ensure that if ID is already
}

} /* ----- end namespace go_fish_details ----- */
//!\endcond
