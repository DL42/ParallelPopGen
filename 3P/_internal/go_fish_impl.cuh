/*
 * go_fish_impl.cuh
 *
 *      Author: David Lawrie
 *      implementation of template and inline functions for GO Fish simulation
 */

#ifndef GO_FISH_IMPL_CUH_
#define GO_FISH_IMPL_CUH_

#include "../_outside_libraries/cub/device/device_scan.cuh"
#include "../_internal/rng.cuh"

/**
 *  GO_Fish is a single-locus Wright-Fisher forward simulation where individual sites are assumed to be independent from each other and mutations are irreversible (Poisson Random Field model).
 *  Mutations are the “unit” of simulation for the single-locus Wright-Fisher algorithm. Thus a generation of organisms is represented by an array of mutations and their frequency in the (each) population (if there are multiple in the simulation).
 *  There are several options for how to initialize the mutation array to start a simulation: a blank mutation array, the output of a previous simulation run, or mutation-selection equilibrium. Simulating each discrete generation consists of
 *  calculating the new allele frequency of each mutation after a round of migration, selection, and drift. Concurrently, new mutations are added to the array. Those mutations that become lost or fixed are discarded in a compact step. The resulting offspring array
 *  of mutation frequencies becomes the parent array of the next generation and the cycle is repeated until the end of the simulation when the final mutation array is output. Further, the user can sample individual generations in the simulation.
 *  \n\n
 *  The function `run_sim` runs a GO_Fish simulation (see documentation for `run_sim` below). The sampled and final generations are stored in allele_trajectories `all_results`. `all_results` stores the frequency(ies) in the population(s) and mutID of every mutation in the sample in RAM,
 *  from which population genetics statistics can be calculated or which can be manipulated and output as the user sees fit.
 *  \n\n
 *  To use all GO_Fish functions and objects, include header file: go_fish.cuh.
 *  \n Optionally, to use only the GO_Fish data structures, include header file: go_fish_data_struct.h.
 */
namespace GO_Fish{
//!\cond
namespace details{

template<typename Functor_demography>
struct round_demography{
	Functor_demography dem;
	round_demography() { }
	round_demography(Functor_demography dem): dem(dem) { }
	__host__ __device__ __forceinline__ float operator()(const unsigned int generation, const unsigned int population) const{ return round(dem(generation,population)); }
};

template<typename Functor>
struct trapezoidal_upper{
	Functor fun;
	trapezoidal_upper() { }
	trapezoidal_upper(Functor xfun): fun(xfun) { }
	__device__ __forceinline__ double operator()(double a, double step_size) const{ return step_size*(fun(a)+fun(a-step_size))/2; } //upper integral
};

//generates an array of areas from 1 to 0 of frequencies at every step size
template <typename Functor_Integrator>
__global__ static void calculate_area(double * d_freq, const uint num_freq, const double step_size, Functor_Integrator trapezoidal){
	int myID = blockIdx.x*blockDim.x + threadIdx.x;

	for(uint id = myID; id < num_freq; id += blockDim.x*gridDim.x){ d_freq[id] = trapezoidal((1.0 - id*step_size), step_size); }
}

__global__ static void reverse_array(double * array, const uint N){
 	int myID = blockIdx.x*blockDim.x + threadIdx.x;
 	for(uint id = myID; id < N/2; id += blockDim.x*gridDim.x){
		double temp = array[N - id - 1];
 		array[N - id - 1] = array[id];
 		array[id] = temp;
 	}
}

struct offset_array{ uint array[33]; };

//determines number of mutations at each frequency in the initial population, sets it equal to mutation-selection balance
template <typename Functor_mse_integrand>
__global__ void initialize_mse_frequency_array(uint * freq_index, double * mse_integral, const uint offset, const float mu, const uint Nchrom_e, const float L, Functor_mse_integrand mse, const uint2 seed, const uint population){
	int myID = blockIdx.x*blockDim.x + threadIdx.x;
	double mse_total = mse_integral[0]; //integral from frequency 0 to 1
	for(uint id = myID; id < (Nchrom_e-1); id += blockDim.x*gridDim.x){ //exclusive, number of freq in pop is chromosome population size N-1
		double i = (id+1.0)/Nchrom_e;
		double j = 1-i;
		float lambda;
		if(mse.neutral()){ lambda = 2*mu*L/i; }
		else{ lambda = (2*mu*L*mse_integral[id])/(mse(i)*mse_total*i*j); }
		freq_index[offset+id] = RNG::rand1_approx_pois(lambda, lambda, L*Nchrom_e, seed, 0, id, population);//mutations are poisson distributed in each frequency class //for round(lambda);//rounding can significantly under count for large N:  //
	}
}

//fills in mutation array using the freq and scan indices
//y threads correspond to freq_index/scan_index indices, use grid-stride loops
//x threads correspond to mutation array indices, use grid-stride loops
//using scan number to define start of array, freq_index to define num_new_mutations_index (if 0 simply ignore) and myIDx used to calculate allele_count
__global__ static void initialize_mse_mutation_arrays(uint * mutations_freq, uint4 * mutations_ID, const uint * freq_index, const uint * scan_index, const offset_array o_array, const unsigned int num_populations, const unsigned int array_length){
	auto myIDy = blockIdx.y*blockDim.y + threadIdx.y;
	auto population = blockIdx.z;
	auto offset = o_array.array[population];
	auto end = o_array.array[population+1]-offset;
	for(auto idy = myIDy; idy < end; idy += blockDim.y*gridDim.y){
		auto myIDx = blockIdx.x*blockDim.x + threadIdx.x;
		auto start = scan_index[offset+idy];
		auto num_mutations = freq_index[offset+idy];
		auto freq = (idy+1U);
		for(auto idx = myIDx; idx < num_mutations; idx += blockDim.x*gridDim.x){
			for(uint pop = 0; pop < num_populations; pop++){ mutations_freq[pop*array_length + start + idx] = 0; }
			mutations_freq[population*array_length + start + idx] = freq;
			mutations_ID[start + idx] = make_uint4(0,population,(start + idx),0);  //age: eventually will replace where mutations have age <= 0 (age before sim start)//population//threadID
		}
	}
}

template <typename Functor_DFE>
__global__ static void add_new_mutation_IDs(uint4 * mutations_ID, const unsigned int prev_mutations_index, const unsigned int new_mutations_index, const unsigned int population, const unsigned int generation, Functor_DFE dfe){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < (new_mutations_index-prev_mutations_index); id+= blockDim.x*gridDim.x){ mutations_ID[(prev_mutations_index+id)] = make_uint4(generation,population,id,dfe(generation,population,id)); }
}

__host__ __device__ __forceinline__ float eff_chrom_f(float Nind, float F){ return 2.f*Nind/(1.f+F); }

__host__ __device__ __forceinline__ unsigned int eff_chrom_u(float Nind, float F){ return ppp::fround_u(eff_chrom_f(Nind, F)); }

static constexpr uint tile_size = 32; //a good compiler should turn /32 into >> 5 anywhere it is needed

//calculates new frequencies for every mutation in the population, initializes new mutations to count of 1
//uses seed for random number generator philox's key space, mutID, population, and generation for its counter space in the pseudorandom sequence
template <typename Functor_demography, typename Functor_selection, typename Functor_migration, typename Functor_inbreeding, typename Functor_dominance>
__global__ void migration_selection_drift(uint * mutations_freq, const uint * const prev_freq, const uint4 * const mutations_ID, const uint mutations_index, const uint array_length, const Functor_demography dem, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding f_inbred, const Functor_dominance dominance, const uint2 seed, const unsigned int num_populations, const unsigned int generation, const unsigned int next_event_generation){

	//----- initialize thread group -----
	auto tile = tiled_partition<tile_size>(this_thread_block()); //tried it using num_populations as a template parameter and tiling based on num_populations, no performance difference, code for non-power-of 2 populations would be harder
	auto tile_rank = tile.thread_rank();
	auto population = tile_rank%num_populations;
	auto myID = blockIdx.x*blockDim.x + threadIdx.x;

	if(tile_rank+num_populations-population <= tile_size){ //tile populations across the tile, ignoring lanes which would not fully cover all the populations (i.e. num_populations is not a divisor of tile_size)
		auto total_mutations_block = coalesced_threads().size()/num_populations*blockDim.x/tile_size; //total_mutations is the actual number of mutations being dealt with by a block
		auto my_start_mut =  myID - population; //multiple threads must deal with the same mutation but in different populations
		if(myID >= tile_size && coalesced_threads().size() < tile_size){ my_start_mut -= tile_size%num_populations*(myID/tile_size); } 	//after the first warp, account for unused threads in all previous warps (this block and prev block included in my_start_mut) if num_populations is not a divisor of tile_size
		my_start_mut /= num_populations;
	//----- end -----

		for(auto mut_index = my_start_mut; mut_index < mutations_index; mut_index += (total_mutations_block)*gridDim.x){

			//----- initialize mutation processing -----
			uint i_next; //input current mutation frequency in population
			auto mutID = mutations_ID[mut_index]; //not worth trying to store in shared memory, decreases register pressure, but increases latency
			auto start_gen = generation; //always start processing a mutation 1 generation after the mutation is born
			if(mutID.x < generation){ i_next = prev_freq[population*array_length+mut_index]; } //input current mutation frequency in population
			else{ start_gen = mutID.x+1U; i_next = ((mutID.x <= next_event_generation) && (mutID.y == population)) ? 1U : 0U; } //start at allele count 1 or 0 if mutation is born during/after this round of processing in this/another population
			auto rng_index = (start_gen-(mutID.x+1U))%4U; //make sure to restart RNG when it was stopped at last event
			union { uint4 rng; uint rng_arr[4];} my_rng; //because of dynamic indexing will be in CUDA's "local" memory which can be thread-private global memory but will likely be L1/2 cache since it's small enough (hence no performance improvement moving it to shared memory)
			//----- end -----
			auto active = coalesced_threads();
			my_rng.rng = RNG::Philox(seed, (mutID.z + 2U), mutID.x, population+num_populations*mutID.y, start_gen-(mutID.x+1U)-rng_index); //min mutID.z is 0, minimum of 2 to avoid conflict with initial_mse and mutation RNG
			for(auto gen = start_gen; gen <= next_event_generation; ++gen){ //process mutations until next event generation
				if(rng_index == 0 && gen > start_gen){ my_rng.rng = RNG::Philox(seed, (mutID.z + 2U), mutID.x, population+num_populations*mutID.y, gen-(mutID.x+1U)); } //only call RNG every 4 generations

				//----- migration -----
				float i_mig = 0;
				auto F = f_inbred(gen,population);
				auto N = dem(gen,population);
				float mig_total = 0;
				float i_holder = 0;
				for(uint pop = 0; pop < num_populations; pop++){
					//all threads in group must participate in shuffle or it is UB (https://stackoverflow.com/questions/76067411/ensuring-thread-warp-synchronicity-post-volta-independent-thread-scheduling)
					auto shuffled_freq = active.shfl(i_next,tile_rank-(population-pop)); //active can only be threads 0 to x where x is <= 31 and the most lanes occupied by a valid mutation given the number of populations and also mutations_index
					if(N == 0) { continue; }
					float mig_p = pop==population ? 1: mig_prop(gen,pop,population); //proportion of migrants in population from pop in generation gen
					if(mig_p == 0){ continue; }
					auto N_pop_prev = dem(gen-1,pop);
					if(N_pop_prev == 0){ continue; }
					auto F_pop_prev = f_inbred(gen-1,pop);
					float i_temp = mig_p*((N/N_pop_prev)*(1.f+F_pop_prev)/(1.f+F))*shuffled_freq;//ensures mutation at frequency x in gen-1,pop is at frequency x in the migrants in population, gen
					if(population != pop){ mig_total += mig_p; i_mig += i_temp; }
					else{ i_holder = i_temp; }
				}
				if(1.f-mig_total > 0){ i_mig += (1.f-mig_total)*i_holder; }
				//----- end -----

				//----- selection -----
				auto Nchrom_e = eff_chrom_f(N,F);
				if(i_mig == 0 || i_mig == Nchrom_e){ i_next = i_mig; continue; }
				auto freq = i_mig/Nchrom_e;
				auto s = fmaxf(sel_coeff(gen,population,freq),-1.f); //ensures selection is never lower than -1
				auto h = dominance(gen,population,freq);
				auto i_mig_sel = (Nchrom_e*(s*i_mig*i_mig+Nchrom_e*i_mig+(F+h-h*F)*s*i_mig*(Nchrom_e-i_mig)))/(s*i_mig*i_mig+(F+2*h-2*h*F)*s*i_mig*(Nchrom_e-i_mig)+Nchrom_e*Nchrom_e); //expected allele count after selection and migration
				//----- end -----

				//----- drift -----
				i_next = RNG::approx_binom(my_rng.rng_arr[rng_index],i_mig_sel,((Nchrom_e-i_mig_sel)/Nchrom_e)*i_mig_sel,Nchrom_e);
				//----- end -----

				rng_index = (rng_index == 3) ? 0 : rng_index+1; //update rng_index
			}
			mutations_freq[population*array_length+mut_index] = i_next; //output final allele freq in next_event_generation generation
		}
	}
}

__device__ __forceinline__ bool boundary_0(uint count){ return count == 0; }

__device__ __forceinline__ bool boundary_1(uint count, uint max){ return count == max; }

//tests indicate accumulating mutations in non-migrating populations is not much of a problem
template <typename Functor_demography, typename Functor_inbreeding>
__global__ void flag_segregating_mutations(uint * flag, uint * counter, const Functor_demography demography, const Functor_inbreeding f_inbred, const uint * const mutations_freq, const bool compact_fixations, const bool compact_losses, const unsigned int num_populations, const unsigned int start_mut_index, const unsigned int padded_mut_index, const unsigned int mutations_index, const unsigned int array_length, const unsigned int generation){
//adapted from https://www.csuohio.edu/engineering/sites/csuohio.edu.engineering/files/Research_Day_2015_EECS_Poster_14.pdf
	auto myID =  blockIdx.x*blockDim.x + threadIdx.x;
	auto tile = tiled_partition<32>(this_thread_block());
	for(int id = myID; id < (padded_mut_index >> 5); id+= blockDim.x*gridDim.x){
		auto lnID = tile.thread_rank();
		auto cgID = id >> 5;

		uint predmask; //32-bit predicate mask: each of 32 threads in a cg declares 1 if their allele is segregating in any population or across populations (i.e. fixed in some and lost in others) or preserved, 0 otherwise
		uint cnt = 0; //count of segregating alleles

		for(int j = 0; j < 32; j++){
			//----- check if allele is segregating -----
			auto one = compact_fixations;
			auto zero = compact_losses;
			auto index = (cgID<<10)+(j<<5)+lnID; //mutation index being processed
			if(index < mutations_index){
				for(uint pop = 0; pop < num_populations; pop++){
					auto Nchrom_e = eff_chrom_u(demography(generation,pop),f_inbred(generation,pop));
					if(Nchrom_e > 0){ //not protected if population goes extinct but demography function becomes non-zero again (shouldn't happen anyway, error msg will be spit out in check_sim_paramters)
						auto i = mutations_freq[pop*array_length + index + start_mut_index];
						zero = zero && boundary_0(i);
						one = one && boundary_1(i,Nchrom_e);
						//must be lost in all or gained in all populations to be lost or fixed - e.g. above ensure that if lost in population 0, zero will be true and 1 will be false and if fixed in population 1, both zero and one will be false
					}
				}
			}//-----end -----

			predmask = tile.ballot(!(zero||one)); //vote of every thread in cg stored as flag in predmask

			if(lnID == 0) {
				flag[(cgID<<5)+j] = predmask; //store predmask in flag
				cnt += __popc(predmask); //count votes in cg
			}
		}

		if(lnID == 0) counter[cgID] = cnt; //store sum of 32 counts of 32 votes (i.e. sum of 32x32 flags)
	}
}

__global__ static void scatter_arrays(uint * new_mutations_freq, uint4 * new_mutations_ID, const uint * const mutations_freq, const uint4 * const mutations_ID, const uint * const flag, const uint * const scan_index, const unsigned int start_mut_index, const unsigned int padded_mut_index, const unsigned int new_array_Length, const unsigned int old_array_Length){
//adapted from https://www.csuohio.edu/engineering/sites/csuohio.edu.engineering/files/Research_Day_2015_EECS_Poster_14.pdf
	auto myID =  blockIdx.x*blockDim.x + threadIdx.x;
	auto population = blockIdx.y;
	auto tile = tiled_partition<32>(this_thread_block());

	for(int id = myID; id < (padded_mut_index >> 5); id+= blockDim.x*gridDim.x){
		auto lnID = tile.thread_rank();
		auto cgID = id >> 5;

		uint predmask;
		uint cnt;

		predmask = flag[(cgID<<5)+lnID]; //predmask for a particular lnID in a cg
		cnt = __popc(predmask); //count of segregating alleles in that predmask

		//parallel prefix sum for the number of segregating alleles
#pragma unroll
		for(int offset = 1; offset < 32; offset<<=1){ //offset goes by multiples of 2
			auto n = tile.shfl_up(cnt, offset);
			if(lnID >= offset) cnt += n;
		}

		uint global_index = 0;
		if(cgID > 0) global_index = scan_index[cgID - 1]; //total number of segregating alleles stored by previous cg's

		for(int i = 0; i < 32; i++){
			auto mask = tile.shfl(predmask,i); //broadcast mask from lane i
			uint sub_group_index = 0;
			if(i > 0) sub_group_index = tile.shfl(cnt,i-1); //total number of segregating alleles stored by previous lane or 0 if i == 0
			if(mask & (1 << lnID)){ //if allele in lnID from predmask from lane i is segregating
				auto write = global_index + sub_group_index + __popc(mask & ((1 << lnID) - 1)); //location in new array to write to
				auto read = (cgID<<10)+(i<<5)+lnID; //location in old array to read from
				new_mutations_freq[population*new_array_Length + write + start_mut_index] = mutations_freq[population*old_array_Length + read + start_mut_index]; //scatter mutation frequency in pop population from old array to new array
				if(population == 0){ new_mutations_ID[write + start_mut_index] = mutations_ID[read + start_mut_index]; } //if in population 0 also scatter mutID (tested and not faster if I make this its own block)
			}
		}
	}
}

__global__ static void transfer_arrays(uint * new_mutations_freq, uint4 * new_mutations_ID, const uint * const mutations_freq, const uint4 * const mutations_ID, const unsigned int new_array_Length, const unsigned int old_array_Length, const unsigned int new_start_mutation, const unsigned int old_start_mutation, const unsigned int num_mutations){
	auto myID =  blockIdx.x*blockDim.x + threadIdx.x;
	auto population = blockIdx.y;

	if(population > 0){
		population--;
		for(int mut_index = myID; mut_index < num_mutations; mut_index += blockDim.x*gridDim.x){ new_mutations_freq[population*new_array_Length + mut_index + new_start_mutation] = mutations_freq[population*old_array_Length + mut_index + old_start_mutation]; }
	}else{
		for(int mut_index = myID; mut_index < num_mutations; mut_index += blockDim.x*gridDim.x){ new_mutations_ID[mut_index + new_start_mutation] =  mutations_ID[mut_index + old_start_mutation]; }
	}
}

///////////////////////////////////////////////////////////////

__host__ inline void check_sim_input_parameters(sim_constants & sim_input_constants, const allele_trajectories & prev_sim){
	if(sim_input_constants.num_populations == 0){ fprintf(stderr,"check_sim_input_parameters: minimum number of populations is 1"); exit(1); }
	if(sim_input_constants.num_populations > 32){ fprintf(stderr,"check_sim_input_parameters: maximum number of populations is 32"); exit(1); }
	if(sim_input_constants.num_sites < 1000){ fprintf(stderr,"check_sim_input_parameters: minimum num_sites is 1000"); exit(1); } //make sure poisson assumptions in rng.cuh hold
	if(sim_input_constants.compact_type != compact_scheme::compact_off && sim_input_constants.compact_interval == 0){ fprintf(stderr,"check_sim_input_parameters: compact_type is not off, sim_input_constants.compact_interval must be > 0"); exit(1); }
	if(sim_input_constants.num_generations > std::numeric_limits<int>::max()){ fprintf(stderr,"check_sim_input_parameters: number of generations must be <= %d",std::numeric_limits<int>::max()); exit(1); } //origin generation must be stored as int
	int sample_index = sim_input_constants.prev_sim_sample;

	if(sim_input_constants.init_mse && prev_sim.num_time_samples() > 0){ fprintf(stderr,"check_sim_input_parameters error: conflicting input: both init_mse is true and non-empty prev_sim input provided"); exit(1); }

	if(!sim_input_constants.init_mse){
		if(sample_index < prev_sim.num_time_samples()){
			auto prev_sim_num_sites = prev_sim.num_sites();
			auto prev_sim_num_populations = prev_sim.num_populations();
			if(sim_input_constants.num_sites != prev_sim_num_sites || sim_input_constants.num_populations != prev_sim_num_populations){
				fprintf(stderr,"check_sim_input_parameters error: prev_sim parameters do not match current simulation parameters: prev_sim num_sites %f\tcurrent_sim num_sites %f,\tprev_sim num_populations %d\tcurrent_sim num_populations %d\n",prev_sim_num_sites,sim_input_constants.num_sites,prev_sim_num_populations,sim_input_constants.num_populations); exit(1);
			}
		}
		else if(prev_sim.num_time_samples() > 0 && sample_index >= prev_sim.num_time_samples()){ fprintf(stderr,"check_sim_input_parameters error: requested sample index out of bounds for prev_sim: sample %d\t[0\t %d)\n",sample_index, prev_sim.num_time_samples()); exit(1); }
	}
}

//for internal simulation function passing
struct data_struct{
	//from sim_input_constants
	uint2 seed; //random number seeds for this simulation
	unsigned int compact_interval; //interval between compact generations
	compact_scheme compact_type; //determines compact type
	unsigned int num_populations; //number of populations in the simulation (# rows for freq)
	float num_sites; //number of sites in the simulation
	unsigned int prev_sim_sample_index; //sample index to take of prev_sim (if any)

	//simulation variables
	unsigned int start_generation; //first generation
	unsigned int generation; //current generation
	unsigned int next_event_generation; //when the next event (sample, compact, or final) generation will occur
	unsigned int final_generation; //final generation
	unsigned int sample_index; //number of currently stored samples
	unsigned int preservation_index; //all mutations below the index have been preserved
	unsigned int array_length; //full length of the mutation array, total number of mutations across all populations (# columns for freq)
	unsigned int mutations_index; //number of mutations in the population (last mutation is at mutations_index-1)

	//host arrays
	std::vector<std::vector<bool>> h_extinct; //boolean if population has gone extinct
	std::vector<std::vector<unsigned int>> h_new_mutation_indices; //indices of new mutations of every population in every generation
	std::vector<unsigned long> h_total_mutations; //total number of mutations generated by the simulation (including initialization)
	std::vector<unsigned int> sample_generation; //when to take a sample

	//device arrays
	ppp::unique_device_ptr<uint> d_mutations_freq; //allele frequency of current mutations
	ppp::unique_device_ptr<uint> d_prev_freq; // meant for storing frequency values so changes in previous populations' frequencies don't affect later populations' migration
	ppp::unique_device_ptr<uint4> d_mutations_ID;  //generation in which mutation appeared, population in which mutation first arose, ID that generated mutation (negative value indicates preserved state), reserved for later use

	__host__ inline data_struct(const sim_constants & sim_input_constants, const cudaDeviceProp & devProp);
};

__host__ inline data_struct::data_struct(const sim_constants & sim_input_constants, const cudaDeviceProp & devProp): h_new_mutation_indices(0), h_total_mutations(sim_input_constants.num_generations+1), h_extinct(0), sample_generation(0){
	seed.x = sim_input_constants.seed1;
	seed.y = sim_input_constants.seed2;
	compact_interval = sim_input_constants.compact_interval;
	compact_type = sim_input_constants.compact_type;
	num_populations = sim_input_constants.num_populations;
	num_sites = sim_input_constants.num_sites;
	prev_sim_sample_index = sim_input_constants.prev_sim_sample;

	start_generation = 0;
	generation = 0;
	final_generation = sim_input_constants.num_generations;
	next_event_generation = final_generation;
	sample_index = 0;
	preservation_index = 0;
	array_length = 0;
	mutations_index = 0;

	h_new_mutation_indices.emplace_back(num_populations+1);
	h_extinct.emplace_back(num_populations);
	for(unsigned int pop = 0; pop < num_populations; pop++){
		h_extinct[0][pop] = false;
		h_new_mutation_indices[0][pop] = 0;
	}


	d_mutations_freq = nullptr;
	d_prev_freq = nullptr;
	d_mutations_ID = nullptr;
}

struct stream_struct{
	ppp::raii_stream_array streams;
	ppp::raii_event_array events;

	__host__ inline stream_struct(const int num_streams);
};

__host__ inline stream_struct::stream_struct(const int num_streams): streams(num_streams), events(num_streams) { }

//checks mutation rate, demography, migration, and inbreeding functors, sets extinct boolean if population has gone extinct to catch functor logic errors in later generations
template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_inbreeding>
__host__ inline void check_evolutionary_functions(const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_inbreeding f_inbred, data_struct & data, unsigned int generation){
	auto num_pop = data.num_populations;
	auto gen_index = generation - data.start_generation;
	for(unsigned int pop = 0; pop < num_pop; pop++){
		float migration = 0;
		if(mu_rate(generation,pop) < 0){ fprintf(stderr,"check_evolutionary_functions: mutation error: mu_rate < 0\tgeneration %d\t population %d\n",generation,pop); exit(1); }
		auto Nind = demography(generation,pop);
		auto F = f_inbred(generation,pop);
		if(F < 0) { fprintf(stderr,"check_evolutionary_functions: inbreeding error: inbreeding coefficient < 0\tgeneration %d\t population %d\n",generation,pop); exit(1); }
		if(F > 1) { fprintf(stderr,"check_evolutionary_functions: inbreeding error: inbreeding coefficient > 1\tgeneration %d\t population %d\n",generation,pop); exit(1); }
		float Nchrom_e_f = eff_chrom_f(Nind,F)+0.5f; //using float +0.5f to check maximum Nchrom_e
		auto Nchrom_e = eff_chrom_u(Nind,F);
		uint UMax = std::numeric_limits<uint>::max();
		if(Nchrom_e_f > UMax){ fprintf(stderr,"check_evolutionary_functions: demography & inbreeding error: effective number of chromosomes too big, 2*Nind/(1.f+F) %f > Max UInt %u\tgeneration %d\t population %d\n",Nchrom_e_f,(UMax-1U),generation,pop); exit(1); }
		if(Nchrom_e_f < 0){ fprintf(stderr,"check_evolutionary_functions: demography & inbreeding error: effective number of chromosomes to be rounded down < 0, 2*Nind/(1.f+F)+0.5f %f Nchrom_e %u \tgeneration %d\t population %d\n",Nchrom_e_f,Nchrom_e,generation,pop); exit(1); }
		if(Nchrom_e > 0 && data.h_extinct[gen_index][pop]){ fprintf(stderr,"check_evolutionary_functions: demography error: extinct population with population size > 0\tgeneration %d\t population %d\n",generation,pop); exit(1); }
		else if(Nchrom_e == 0 && generation > 0){ //only mse can check at 0 or data.start_generation for that matter, all others must check starting at data.start_generation+1
			//previous generation, the population was alive, now the population is considered extinct
			auto Nchrom_e_prev = eff_chrom_u(demography(generation-1,pop),f_inbred(generation-1,pop));
			if(Nchrom_e_prev > 0){ data.h_extinct[gen_index][pop] = true; }
		}
		for(unsigned int pop2 = 0; pop2 < num_pop; pop2++){
			auto m_from = mig_prop(generation,pop,pop2);
			if(pop != pop2 && m_from < 0){ fprintf(stderr,"check_evolutionary_functions: migration error: migration rate < 0\tgeneration %d\t population_from %d\t population_to %d\n",generation,pop,pop2); exit(1); }
			auto Nchrom_e_pop2 = eff_chrom_u(demography(generation,pop2),f_inbred(generation,pop2));
			if(m_from > 0 && Nchrom_e == 0 && Nchrom_e_pop2 > 0){ fprintf(stderr,"migration error, migration from non-existant population\tgeneration %d\t population_from %d\t population_to %d\n",generation,pop,pop2); exit(1); } //error only if Nchrom_e == 0 && Nchrom_e_pop2 > 0 because if neither population exists, it doesn't matter
			auto m_to = mig_prop(generation,pop2,pop);
			if(pop != pop2){ migration += m_to; }
		}
		if(migration > 1.f && Nchrom_e > 0){ fprintf(stderr,"check_evolutionary_functions: migration error: migration rate cannot sum to 1\tgeneration %d\t population_TO %d\t total_migration_TO %f\n",generation,pop,migration); exit(1); }
	}
}

template <typename Functor_mse_integrand>
__host__ inline void integrate_mse(double * d_mse_integral, const uint Nchrom_e, Functor_mse_integrand mse_fun, const unsigned int pop, const stream_struct & streams){
	auto d_freq = ppp::make_unique_device_ptr<double>(Nchrom_e,0,pop);

	trapezoidal_upper<Functor_mse_integrand> trap(mse_fun);

	calculate_area<<<10,1024,0,streams.streams[0]>>>(d_freq.get(), Nchrom_e, (double)(1.0/(Nchrom_e)), trap); //setup array frequency values to integrate over (upper integral from 1 to 0)
	cudaCheckErrorsAsync(cudaPeekAtLastError(),0,pop);

	auto d_temp_storage = ppp::make_unique_device_ptr<void>(0,0,pop);
	size_t temp_storage_bytes = 0;
	cudaCheckErrorsAsync(cub::DeviceScan::InclusiveSum(d_temp_storage.get(), temp_storage_bytes, d_freq.get(), d_mse_integral, Nchrom_e, streams.streams[0]),0,pop);
	ppp::reset_device_ptr(d_temp_storage,temp_storage_bytes,0,pop);
	cudaCheckErrorsAsync(cub::DeviceScan::InclusiveSum(d_temp_storage.get(), temp_storage_bytes, d_freq.get(), d_mse_integral, Nchrom_e, streams.streams[0]),0,pop);

	reverse_array<<<10,1024,0,streams.streams[0]>>>(d_mse_integral, Nchrom_e);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),0,pop);
}

//calculates based on the num_initial_mutations how many new mutations will be added in the subsequent generations, sets how many samples to take and when to take them, checks the simulation functors to make sure they are working and sets extinction boolean if any populations go extinct
template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_inbreeding, typename Functor_timesample>
__host__ inline void preprocess_simulation(data_struct & data, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_inbreeding f_inbred, const Functor_timesample take_sample, const unsigned int num_initial_mutations){

	data.mutations_index = num_initial_mutations;
	data.h_total_mutations[0] = data.mutations_index;
	if(take_sample(data.generation) && data.generation != data.final_generation){ data.sample_generation.push_back(data.generation); }

	for(auto gen = data.generation+1; gen <= data.final_generation; gen++){
		auto gen_index = gen - data.start_generation;
		data.h_extinct.emplace_back(data.num_populations);
		for(unsigned int pop = 0; pop < data.num_populations; pop++){ data.h_extinct[gen_index][pop] = data.h_extinct[gen_index-1][pop]; }
		check_evolutionary_functions(mu_rate, demography, mig_prop, f_inbred, data, gen);
		data.h_new_mutation_indices.emplace_back(data.num_populations+1);
		data.h_new_mutation_indices[gen_index][0] = 0;
		unsigned int num_new_mutations_gen = 0;
		data.h_total_mutations[gen_index] = data.h_total_mutations[gen_index-1];
		for(unsigned int pop = 0; pop < data.num_populations; pop++){
			uint Nchrom_e = eff_chrom_u(demography(gen,pop),f_inbred(gen,pop));
			float mu = mu_rate(gen,pop);
			float lambda = mu*Nchrom_e*data.num_sites;
			if(lambda > 0){ num_new_mutations_gen += RNG::rand1_approx_pois(lambda, lambda, Nchrom_e*data.num_sites, data.seed, 1, gen, pop); }
			data.h_new_mutation_indices[gen_index][pop+1] = num_new_mutations_gen;
		}
		data.h_total_mutations[gen_index] += num_new_mutations_gen;
		if(take_sample(gen) && data.generation != data.final_generation){ data.sample_generation.push_back(gen); }
	}

	data.sample_generation.push_back(data.final_generation);
}

//uses compact_interval, final generation, compact_type, and sample_generation to determine how many subsequent generations will be processed at once (i.e. the next event generation)
__host__ inline void set_next_event_generation(data_struct & data){
	auto next_compact_generation = data.final_generation;
	if(data.compact_type != compact_scheme::compact_off){
		auto temp = min(data.generation+data.compact_interval,data.final_generation);
		if(temp > data.generation){ next_compact_generation = temp; } //temp should be greater than data.generation (or equal to in the case of data.generation == data.final_generation), but just in case of unsigned int rollover from data.generation+data.compact_interval ... next_compact_generation will be a coherent value => data.final_generation
	}

	auto next_sample_generation = data.final_generation;
	if(data.sample_generation.size() > 1 && data.generation < data.final_generation){ next_sample_generation = data.sample_generation[data.sample_index+1]; }
	data.next_event_generation = min(next_compact_generation,next_sample_generation);
}

//calls set_next_event_generation, calculates the new array length based on compact_type and next_event_generation
__host__ inline void setup_next_generations(data_struct & data){
	set_next_event_generation(data);
	if(data.compact_type == compact_scheme::compact_off){ data.array_length = data.h_total_mutations[data.h_total_mutations.size()-1]; }
	else{
		data.array_length = data.mutations_index;
		data.array_length += data.h_total_mutations[data.next_event_generation - data.start_generation] - data.h_total_mutations[data.generation - data.start_generation];
	}
}

template <typename Functor_mse, typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_timesample>
__host__ inline void initialize_mse(data_struct & data, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding f_inbred, const Functor_dominance dominance, const Functor_timesample take_sample, const stream_struct & streams){
	check_evolutionary_functions(mu_rate, demography, mig_prop, f_inbred, data, 0);

	offset_array o_array;
	uint num_freq = 0; //number of frequencies
	o_array.array[0] = num_freq;
	for(unsigned int pop = 0; pop < data.num_populations; pop++){
		float Nind = demography(0,pop);
		uint Nchrom_e = eff_chrom_u(Nind,f_inbred(0,pop));
		if(Nind > 0 && Nchrom_e > 1){ num_freq += (Nchrom_e - 1); }
		o_array.array[pop+1] = num_freq;
	}

	auto d_freq_index = ppp::make_unique_device_ptr<unsigned int>(num_freq,0,-1);
	auto d_scan_index = ppp::make_unique_device_ptr<unsigned int>(num_freq,0,-1);

	for(unsigned int pop = 0; pop < data.num_populations; pop++){
		float mu = mu_rate(0,pop);
		uint Nchrom_e = 1U + o_array.array[pop+1] - o_array.array[pop];
		auto mse_integral = ppp::make_unique_device_ptr<double>(Nchrom_e,0,pop);
		Functor_mse mse_integrand(demography,sel_coeff,f_inbred,dominance,0,pop);
		if(Nchrom_e <= 1){ continue; }
		if(!mse_integrand.neutral()){ integrate_mse(mse_integral.get(), Nchrom_e, mse_integrand, pop, streams); }
		initialize_mse_frequency_array<<<6,400,0,streams.streams[0]>>>(d_freq_index.get(), mse_integral.get(), o_array.array[pop], mu,  Nchrom_e, data.num_sites, mse_integrand, data.seed, pop);
		cudaCheckErrorsAsync(cudaPeekAtLastError(),0,pop);
	}

	auto d_temp_storage = ppp::make_unique_device_ptr<void>(0,0,-1);
	size_t temp_storage_bytes = 0;
	cudaCheckErrorsAsync(cub::DeviceScan::ExclusiveSum(d_temp_storage.get(), temp_storage_bytes, d_freq_index.get(), d_scan_index.get(), num_freq, streams.streams[0]),0,-1);
	ppp::reset_device_ptr(d_temp_storage,temp_storage_bytes,0,-1);
	cudaCheckErrorsAsync(cub::DeviceScan::ExclusiveSum(d_temp_storage.get(), temp_storage_bytes, d_freq_index.get(), d_scan_index.get(), num_freq, streams.streams[0]),0,-1);

	unsigned int prefix_sum_result;
	unsigned int final_freq_count;
	//final index is numfreq-1
	cudaCheckErrorsAsync(cudaMemcpyAsync(&prefix_sum_result, &d_scan_index.get()[(num_freq-1)], sizeof(unsigned int), cudaMemcpyDeviceToHost, streams.streams[0]),0,-1); //has to be in sync with host as result is used straight afterwards
	cudaCheckErrorsAsync(cudaMemcpyAsync(&final_freq_count, &d_freq_index.get()[(num_freq-1)], sizeof(unsigned int), cudaMemcpyDeviceToHost, streams.streams[0]),0,-1); //has to be in sync with host as result is used straight afterwards
	if(cudaStreamQuery(streams.streams[0]) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(streams.streams[0]),0,-1); } //has to be in sync with the host since prefix_sum_result+final_freq_count is manipulated on CPU right after
	unsigned int num_mutations = prefix_sum_result+final_freq_count;

	preprocess_simulation(data, mu_rate, demography, mig_prop, f_inbred, take_sample, num_mutations);
	setup_next_generations(data);

	ppp::reset_device_ptr(data.d_mutations_freq,data.num_populations*data.array_length,0,-1);
	ppp::reset_device_ptr(data.d_prev_freq,data.num_populations*data.array_length,0,-1);
	ppp::reset_device_ptr(data.d_mutations_ID,data.array_length,0,-1);

	const dim3 blocksize(4,256,1);
	const dim3 gridsize(32,32,data.num_populations);
	initialize_mse_mutation_arrays<<<gridsize,blocksize,0,streams.streams[0]>>>(data.d_prev_freq.get(), data.d_mutations_ID.get(), d_freq_index.get(), d_scan_index.get(), o_array, data.num_populations, data.array_length);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),0,-1);

	cudaCheckErrorsAsync(cudaEventRecord(streams.events[0],streams.streams[0]),0,-1);
}

template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_inbreeding, typename Functor_timesample>
__host__ inline void init_blank_prev_run(data_struct & data, const allele_trajectories & prev_sim, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_inbreeding f_inbred, const Functor_timesample take_sample, const stream_struct & streams){
	int num_mutations = 0;

	if(prev_sim.num_time_samples() > 0){
		data.start_generation = prev_sim.sampled_generation(data.prev_sim_sample_index);
		data.generation = data.start_generation;
		data.final_generation += data.generation;
		data.next_event_generation += data.generation;
		num_mutations = prev_sim.num_mutations_time_sample(data.prev_sim_sample_index);
		auto prev_sim_extinct = prev_sim.extinct_span(data.prev_sim_sample_index);
		for(unsigned int pop = 0; pop < data.num_populations; pop++){ data.h_extinct[0][pop] = prev_sim_extinct[pop]; }
	}

	preprocess_simulation(data, mu_rate, demography, mig_prop, f_inbred, take_sample, num_mutations);
	setup_next_generations(data);

	ppp::reset_device_ptr(data.d_mutations_freq,data.num_populations*data.array_length,0,-1);
	ppp::reset_device_ptr(data.d_prev_freq,data.num_populations*data.array_length,0,-1);
	ppp::reset_device_ptr(data.d_mutations_ID,data.array_length,0,-1);

	//if prev_sim's num_mutations == 0, don't try to copy data that isn't there (still copies extinct data above)
	if(num_mutations > 0){
		auto h_prev_sim_mutID = prev_sim.mutID_span();
		cudaCheckErrors(cudaMemcpy(data.d_mutations_ID.get(), h_prev_sim_mutID.data(), h_prev_sim_mutID.size_bytes(), cudaMemcpyHostToDevice),0,-1);

		auto h_prev_sim_allele = prev_sim.allele_count_span(data.prev_sim_sample_index,0,data.num_populations);
		cudaCheckErrors(cudaMemcpy2D(data.d_prev_freq.get(), data.array_length*sizeof(uint), h_prev_sim_allele.data(), num_mutations*sizeof(uint), num_mutations*sizeof(uint), data.num_populations, cudaMemcpyHostToDevice),0,-1);

		cudaCheckErrorsAsync(cudaEventRecord(streams.events[0],streams.streams[0]),0,-1);
	}
}

template <typename Functor_demography, typename Functor_inbreeding>
__host__ inline void compact(data_struct & data, const Functor_demography demography, const Functor_inbreeding f_inbred, const stream_struct & streams){
	auto gen_index = data.generation - data.start_generation;
	auto num_new_mutations = data.h_new_mutation_indices[gen_index][data.num_populations]; //mutations added in current generation, obviously can't be lost/fixed
	auto num_processed_mutations = data.mutations_index - num_new_mutations - data.preservation_index; //number of mutations undergone migration_selection_drift since last compact that are not preserved
	unsigned int padded_mut_index = (((num_processed_mutations>>10)+1*(num_processed_mutations%1024!=0))<<10);
	auto generation = data.generation;

	//tell simulation to pause here if not yet done streaming recoded data to host (store_time_sample)
	if(cudaStreamQuery(streams.streams[3]) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(streams.streams[3]),generation,-1); }

	ppp::reset_device_ptr(data.d_prev_freq,0,generation,-1);

	{
		unsigned int h_num_seg_mutations = 0;
		auto d_flag = ppp::make_unique_device_ptr<uint>(padded_mut_index>>5,generation,-1);
		auto d_count = ppp::make_unique_device_ptr<uint>(padded_mut_index>>10,generation,-1);
		auto d_scan_index = ppp::make_unique_device_ptr<uint>(padded_mut_index>>10,generation,-1);

		if(num_processed_mutations > 0){
			bool compact_fixations = (data.compact_type == compact_scheme::compact_all || data.compact_type == compact_scheme::compact_fixations);
			bool compact_losses = (data.compact_type == compact_scheme::compact_all || data.compact_type == compact_scheme::compact_losses);
			flag_segregating_mutations<<<800,128,0,streams.streams[0]>>>(d_flag.get(), d_count.get(), demography, f_inbred, data.d_mutations_freq.get(), compact_fixations, compact_losses,  data.num_populations, data.preservation_index, padded_mut_index, num_processed_mutations, data.array_length, generation);
			cudaCheckErrorsAsync(cudaPeekAtLastError(),generation,-1);

			auto d_temp_storage = ppp::make_unique_device_ptr<void>(0,generation,-1);
			size_t temp_storage_bytes = 0;
			cudaCheckErrorsAsync(cub::DeviceScan::InclusiveSum(d_temp_storage.get(), temp_storage_bytes, d_count.get(), d_scan_index.get(), (padded_mut_index>>10), streams.streams[0]),generation,-1);
			ppp::reset_device_ptr(d_temp_storage,temp_storage_bytes,generation,-1);
			cudaCheckErrorsAsync(cub::DeviceScan::InclusiveSum(d_temp_storage.get(), temp_storage_bytes, d_count.get(), d_scan_index.get(), (padded_mut_index>>10), streams.streams[0]),generation,-1);

			cudaCheckErrorsAsync(cudaPeekAtLastError(),generation,-1);

			cudaCheckErrorsAsync(cudaMemcpyAsync(&h_num_seg_mutations, &d_scan_index.get()[(padded_mut_index>>10)-1], sizeof(unsigned int), cudaMemcpyDeviceToHost, streams.streams[0]),generation,-1);
			if(cudaStreamQuery(streams.streams[0]) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(streams.streams[0]),generation,-1); } //has to be in sync with the host since h_num_seq_mutations is manipulated on CPU right after
		}

		auto old_array_Length = data.array_length;
		data.mutations_index = h_num_seg_mutations + num_new_mutations + data.preservation_index;
		setup_next_generations(data);

		ppp::reset_device_ptr(data.d_prev_freq,data.num_populations*data.array_length,generation,-1);
		auto d_IDtemp = ppp::make_unique_device_ptr<uint4>(data.array_length,generation,-1);

		if(num_processed_mutations > 0){
			const dim3 gridsize(800,data.num_populations,1);
			scatter_arrays<<<gridsize,128,0,streams.streams[0]>>>(data.d_prev_freq.get(), d_IDtemp.get(), data.d_mutations_freq.get(), data.d_mutations_ID.get(), d_flag.get(), d_scan_index.get(), data.preservation_index, padded_mut_index, data.array_length, old_array_Length);
			cudaCheckErrorsAsync(cudaPeekAtLastError(),generation,-1);
		}
		if(num_new_mutations > 0){
			if(num_processed_mutations == 0){ cudaCheckErrorsAsync(cudaStreamWaitEvent(streams.streams[1],streams.events[0],0),data.generation,-1); } //if num_processed_mutations > 0 CPU has to sync with stream 0 anyway
			const dim3 gridsize2(10,data.num_populations+1,1);
			transfer_arrays<<<gridsize2,1024,0,streams.streams[1]>>>(data.d_prev_freq.get(), d_IDtemp.get(), data.d_mutations_freq.get(), data.d_mutations_ID.get(), data.array_length, old_array_Length, (h_num_seg_mutations+data.preservation_index), (num_processed_mutations+data.preservation_index), num_new_mutations);
			cudaCheckErrorsAsync(cudaPeekAtLastError(),generation,-1);
		}
		if(data.preservation_index > 0){
			if(num_processed_mutations == 0){ cudaCheckErrorsAsync(cudaStreamWaitEvent(streams.streams[2],streams.events[0],0),data.generation,-1); }
			const dim3 gridsize3(80,data.num_populations+1,1);
			transfer_arrays<<<gridsize3,1024,0,streams.streams[2]>>>(data.d_prev_freq.get(), d_IDtemp.get(), data.d_mutations_freq.get(), data.d_mutations_ID.get(), data.array_length, old_array_Length, 0, 0, data.preservation_index);
			cudaCheckErrorsAsync(cudaPeekAtLastError(),generation,-1);
		}

		data.d_mutations_ID = std::move(d_IDtemp);
	}  //uses local scope to free device memory so uses less peak memory
	for(auto i = 0; i < 3; i++){
		cudaCheckErrorsAsync(cudaEventRecord(streams.events[i],streams.streams[i]),generation,-1);
		cudaCheckErrorsAsync(cudaStreamWaitEvent(streams.streams[0],streams.events[i],0),data.generation,-1); //don't start mig_sel_drift until all data copied
	}
	ppp::reset_device_ptr(data.d_mutations_freq,data.num_populations*data.array_length,generation,-1);
}

template <typename Functor_demography, typename Functor_inbreeding>
__host__ inline void store_time_sample(allele_trajectories & all_results, data_struct & data, Functor_demography demography, Functor_inbreeding f_inbred, const stream_struct & streams){
	for(auto i = 0; i < 3; i++){ cudaCheckErrorsAsync(cudaStreamWaitEvent(streams.streams[3],streams.events[i],0),data.generation,-1); }
	data.preservation_index = data.mutations_index;
	auto num_mutations = data.mutations_index;
	auto sampled_generation = data.generation;
	auto gen_index = sampled_generation - data.start_generation;
	all_results.initialize_time_sample(data.sample_index,sampled_generation,num_mutations,data.h_total_mutations[gen_index]);

	if(num_mutations > 0){
		auto out_mutations_freq = all_results.allele_count_span(data.sample_index,0,data.num_populations);
		auto pinned_out_freq = ppp::register_host_ptr<uint>(out_mutations_freq.data(),out_mutations_freq.size(),sampled_generation,-1); //pinned memory allows for asynchronous transfer to host
		cudaCheckErrorsAsync(cudaMemcpy2DAsync(pinned_out_freq.get(), num_mutations*sizeof(uint), data.d_prev_freq.get(), data.array_length*sizeof(uint), num_mutations*sizeof(uint), data.num_populations, cudaMemcpyDeviceToHost, streams.streams[1]),sampled_generation,-1); //removes padding
	}
	if(sampled_generation == data.final_generation){
		if(num_mutations > 0){
			auto out_mutations_ID = all_results.mutID_span();
			auto pinned_out_ID = ppp::register_host_ptr<mutID>(out_mutations_ID.data(),out_mutations_ID.size(),sampled_generation,-1); //pinned memory allows for asynchronous transfer to host
			cudaCheckErrorsAsync(cudaMemcpyAsync(pinned_out_ID.get(), data.d_mutations_ID.get(), out_mutations_ID.size_bytes(), cudaMemcpyDeviceToHost, streams.streams[1]),sampled_generation,-1); //mutations array is 1D
		}
	}

	auto out_extinct = all_results.extinct_span(data.sample_index);
	auto out_Nchrom_e = all_results.popsize_span(data.sample_index);
	for(unsigned int pop = 0; pop < data.num_populations; pop++){
		out_extinct[pop] = data.h_extinct[gen_index][pop];
		out_Nchrom_e[pop] = eff_chrom_u(demography(sampled_generation,pop),f_inbred(sampled_generation,pop));
	}

	cudaCheckErrorsAsync(cudaEventRecord(streams.events[3],streams.streams[3]),sampled_generation,-1);
	data.sample_index++;
	//1 round of migration_selection_drift and add_new_mutation_IDs can be done simultaneously with above as they change d_mutations_freq array, not d_prev_freq
}

//generates new mutation IDs for all mutation generated from generation through the next event generation
template <typename Functor_DFE>
__host__ inline void generate_new_mutation_IDs(data_struct & data, Functor_DFE dfe, stream_struct & streams){
	for(auto gen = data.generation; gen <= data.next_event_generation; gen++){
		auto gen_index = gen-data.start_generation;
		for(unsigned int pop = 0; pop < data.num_populations; pop++){
			auto start = data.h_new_mutation_indices[gen_index][pop] + data.mutations_index;
			auto end = data.h_new_mutation_indices[gen_index][pop+1] + data.mutations_index;
			if(end > start){
				add_new_mutation_IDs<<<20,512,0,streams.streams[0]>>>(data.d_mutations_ID.get(), start, end, pop, gen, dfe);
				cudaCheckErrorsAsync(cudaPeekAtLastError(),gen,pop); //keeping the kernels like this single population, single generation, actually seems to be the fastest! multistream, multidimensional kernel calls are no faster and sometimes slower
			}
		}
		data.mutations_index += data.h_new_mutation_indices[gen_index][data.num_populations];
	}
}

//applies the forces of migration, selection, and drift to the frequencies of all mutations from generation through the next event generation
template < typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance>
__host__ inline void migration_selection_drift(data_struct & data, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding f_inbred, const Functor_dominance dominance, stream_struct & streams){
	const unsigned int num_tiles = 10;
	migration_selection_drift<<<600,tile_size*num_tiles,0,streams.streams[0]>>>(data.d_mutations_freq.get(), data.d_prev_freq.get(), data.d_mutations_ID.get(), data.mutations_index, data.array_length, demography, mig_prop, sel_coeff, f_inbred, dominance, data.seed, data.num_populations, data.generation, data.next_event_generation);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),data.generation,0);
}

template <typename Functor_mse, typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_timesample, typename Functor_dfe>
__host__ inline allele_trajectories run_sim_impl(sim_constants & sim_input_constants, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding f_inbred, const Functor_dominance dominance, const Functor_timesample take_sample, const allele_trajectories & prev_sim, Functor_dfe dfe){
	check_sim_input_parameters(sim_input_constants, prev_sim);

	//----- initialize simulation structures -----
	cudaDeviceProp devProp = ppp::set_cuda_device(sim_input_constants.device);
	data_struct data(sim_input_constants, devProp);
	stream_struct streams(4);
	//----- end -----

	//----- initialize simulation -----
	if(sim_input_constants.init_mse){ initialize_mse<Functor_mse>(data, mu_rate, demography, mig_prop, sel_coeff, f_inbred, dominance, take_sample, streams); } //initialize mutation-selection equilibrium
	else{ init_blank_prev_run(data, prev_sim, mu_rate, demography, mig_prop, f_inbred, take_sample, streams); } //initialize from results of previous simulation run or to blank
	allele_trajectories all_results(sim_input_constants,data.sample_generation.size()); //initialize simulation output
	//----- end -----
	std::cout<<data.mutations_index<<"\t"<<data.generation<<std::endl;
	//----- take initial time sample of allele trajectories? -----
	if(data.sample_generation[data.sample_index] == data.generation){ store_time_sample(all_results, data, demography, f_inbred, streams); }
	//----- end -----

	//----- simulation steps: generate new mutation IDs; apply migration, selection, drift; advance simulation to the next event generation; compact?; sample? -----
	while((data.generation+1) <= data.final_generation){ //end of simulation
		data.generation++;
		generate_new_mutation_IDs(data, dfe, streams);
		migration_selection_drift(data, demography, mig_prop, sel_coeff, f_inbred, dominance, streams);
		data.generation = data.next_event_generation; //if advancing only 1 generation, these two may already be equal to each other
		std::cout<<data.mutations_index<<"\t"<<data.generation<<std::endl;
		if(sim_input_constants.compact_type != compact_scheme::compact_off){ compact(data, demography, f_inbred, streams); }
		else{ std::swap(data.d_prev_freq,data.d_mutations_freq); set_next_event_generation(data); }
		if(data.sample_generation[data.sample_index] == data.generation){ store_time_sample(all_results, data, demography, f_inbred, streams); }
		std::cout<<data.mutations_index<<"\t"<<data.generation<<std::endl;
	}
	//----- end -----

	if(cudaStreamQuery(streams.streams[3]) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(streams.streams[3]), data.generation, -1); } //ensures writes to host are finished before host can manipulate the data
	return all_results;
}

} /* ----- end namespace GO_Fish::details ----- */
//!\endcond


/** A simulation run is controlled by the template functions and sim_input_constants (which are then accessible from allele_trajectories.last_run_constants()). The user can write their own
 * simulation functions to input into `run_sim` or use those provided in namespace Sim_Model. For details on how to write your own simulation functions, go to the <a href="modules.html">Modules</a> page,
 * click on the simulation function group which describes the function you wish to write, and read its detailed description. They can be standard functions, functors,
 * or (coming with C+11 support) lambdas.
 *
 * <B>Pro Tip:</B> For extra speed, it is desirable that the simulation functions input into run_sim are known at compile-time (i.e. avoid function pointers and non-inline functions unless necessary).
 * The parameters input into the constructors of functors (as used by Sim_Model) may be set at runtime, but the the function itself (the structure/operator in the case of a functor) should be known at compile-time.
 * The functions are input into `run_sim` via templates, so that, at compile-time, known functions can be compiled directly into run_sim's code (fast) as opposed to called from the function stack (slow).
 * This is especially important for Selection, Migration, and, to a lesser extent, Demographic functions, which are run on the GPU many times over for every mutation, every generation (on the GPU every mutation, every compact generation for Demography).
 *
 * \param all_results `sim_input_constants` help control the simulation run whose results are stored in `all_results`
 * \param mu_rate Function specifying the mutation rate per site for a given `population`, `generation`
 * \param demography Function specifying then population size (individuals) for a given `population`, `generation`
 * \param mig_prop Function specifying the migration rate, which is the proportion of chromosomes in population `pop_TO` from population `pop_FROM` for a given `generation`
 * \param sel_coeff Function specifying the selection coefficient for a given `population`, `generation`, `frequency`
 * \param f_inbred Function specifying the inbreeding coefficient for a given `population`, `generation`
 * \param dominance Function specifying the dominance coefficient for a given `population`, `generation`
 * \param take_sample Function specifying if a time sample should be taken in a `generation` - note this will preserve the mutations present for the rest of the simulation run
 * \param prev_sim then `run_sim` will use the time sample corresponding to `prev_sim_sample` in `sim_input_constants` in prev_sim to initialize the new simulation provided that the number of populations
 * and number of sites in prev_sim are equivalent to those in `sim_input_constants` or an error will be thrown. Default input is a blank simulation (so when starting from mutation-selection-equilbirium or blank simulation, can simply leave the parameter off).
 * If a non-empty prev_sim is provided and init_mse is set to true, an error will be thrown for conflicting input.
*/
template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_timesample, typename Functor_mse, typename Functor_dfe>
__host__ inline allele_trajectories run_sim(sim_constants sim_input_constants, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding f_inbred, const Functor_dominance dominance, const Functor_timesample take_sample, const allele_trajectories & prev_sim, Functor_mse, Functor_dfe DFE){
	details::round_demography<Functor_demography> rounded_demography(demography); //ensures that the demography function returns a round number of individuals, otherwise frequency and other calculations are a bit off
	return details::run_sim_impl<Functor_mse>(sim_input_constants, mu_rate, rounded_demography, mig_prop, sel_coeff, f_inbred, dominance, take_sample, prev_sim, DFE);
}

} /* ----- end namespace GO_Fish ----- */

#endif /* GO_FISH_IMPL_CUH */
