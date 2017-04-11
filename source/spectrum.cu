/*
 * spectrum.cu
 *
 *      Author: David Lawrie
 */

#include "../include/spectrum.h"
#include "../source/shared.cuh"
#include "../outside_libraries/cub/device/device_scan.cuh"
#include "../outside_libraries/cub/block/block_reduce.cuh"
#include <math_constants.h>

//!\cond
namespace Spectrum_details{

class transfer_allele_trajectories{

	struct time_sample{
		float * mutations_freq; //allele frequency of mutations in final generation
		bool * extinct; //extinct[pop] == true, flag if population is extinct by end of simulation
		int * Nchrom_e; //effective number of chromosomes in each population
		int num_mutations; //number of mutations in frequency array (columns array length for freq)
		int sampled_generation; //number of generations in the simulation at point of sampling

		time_sample(): num_mutations(0), sampled_generation(0) { mutations_freq = NULL; extinct = NULL; Nchrom_e = NULL; }
		time_sample(const GO_Fish::allele_trajectories & in, int sample_index): num_mutations(in.time_samples[sample_index]->num_mutations), sampled_generation(in.time_samples[sample_index]->sampled_generation){
			//can replace with weak pointers when moving to CUDA 7+ and C++11 (or maybe not, see notes)
			mutations_freq = in.time_samples[sample_index]->mutations_freq;
			extinct = in.time_samples[sample_index]->extinct;
			Nchrom_e = in.time_samples[sample_index]->Nchrom_e;
		}
		~time_sample(){ mutations_freq = NULL; extinct = NULL; Nchrom_e = NULL; } //don't actually delete information, just null pointers as this just points to the real data held
	};

	time_sample ** time_samples;
	GO_Fish::mutID * mutations_ID; //unique ID consisting of generation, population, threadID, and device
	unsigned int num_samples;
	int all_mutations; //number of mutations in mutation ID array - maximal set of mutations in the simulation

	//----- initialization parameters -----
	struct sim_constants{
		int num_generations;
		float num_sites;
		int num_populations;

		sim_constants();
		sim_constants(const GO_Fish::allele_trajectories & in){
			num_generations = in.sim_run_constants.num_generations;
			num_sites = in.sim_run_constants.num_sites;
			num_populations = in.sim_run_constants.num_populations;
		}
	}sim_run_constants;
	//----- end -----

	transfer_allele_trajectories(): num_samples(0), all_mutations(0) { time_samples = 0; mutations_ID = 0; }

	transfer_allele_trajectories(const GO_Fish::allele_trajectories & in): sim_run_constants(in), all_mutations(in.all_mutations) {
		//can replace with weak pointers when moving to CUDA 7+ and C++11 (or maybe not, see notes)
		if(!in.time_samples || in.num_samples == 0){ fprintf(stderr,"error transferring allele_trajectories to spectrum: empty allele_trajectories\n"); exit(1); }
		num_samples = in.num_samples;
		time_samples = new time_sample *[num_samples];

		for(int i = 0; i < num_samples; i++){ time_samples[i] = new time_sample(in,i); }
		mutations_ID = in.mutations_ID;
	}

	friend void Spectrum::population_frequency_histogram(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, int cuda_device);

	friend void Spectrum::site_frequency_spectrum(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, const int sample_size, int cuda_device);

	~transfer_allele_trajectories(){ delete [] time_samples; time_samples = NULL; mutations_ID = NULL; num_samples = 0; } //don't actually delete anything, this is just a pointer class, actual data held by GO_Fish::trajectory, delete [] time_samples won't call individual destructors and even if it did, the spectrum time sample destructors don't delete anything
};

__global__ void population_hist(unsigned int * out_histogram, float * in_mutation_freq, int Nchrome_e, int num_mutations, int num_sites){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < num_mutations; id+= blockDim.x*gridDim.x){
		int index = round(Nchrome_e*in_mutation_freq[id]);
		if(index == Nchrome_e){ index = 0; }
		atomicAdd(&out_histogram[index],1);
	}
	if(myID == 0){  atomicAdd(&out_histogram[0], (num_sites - num_mutations));  }
}

__global__ void uint_to_float(float * out_array, unsigned int * in_array, int N){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < N; id+= blockDim.x*gridDim.x){ out_array[id] = in_array[id]; }
}

__global__ void binom_coeff(float * binom_coeff, int half_n, int n){
	int myIDx =  blockIdx.x*blockDim.x + threadIdx.x;

	for(int idx = (myIDx+1); idx < half_n; idx+= blockDim.x*gridDim.x){ binom_coeff[idx] =  logf((n+1.f-idx)/((float)idx)); }
	if(myIDx == 0){ binom_coeff[0] = 0.0; }
}

__global__ void print_Device_array_float(float * array, int num){

		//if(i%1000 == 0){ printf("\n"); }
	for(int j = 0; j < num; j++){ printf("%d: %f\t",j,array[j]); }
	printf("\n");
}

__global__ void print_Device_array_double(double * array, int start, int end){

		//if(i%1000 == 0){ printf("\n"); }
	for(int j = start; j < end; j++){ printf("%d: %f\t",j,array[j]); }
	printf("\n");
}

template <int BLOCK_THREADS>
__global__ void binom(float * d_histogram, const float * const d_mutations_freq, const float * const d_binom_coeff, const int half_n, const int num_levels, float num_sites, int num_mutations){
	int myIDx =  blockIdx.x*blockDim.x + threadIdx.x;
	int myIDy = blockIdx.y;
	typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduceT;
	__shared__ typename BlockReduceT::TempStorage temp_storage;
	float thread_data[1];

	for(int idy = myIDy; idy <= num_levels; idy+= blockDim.y*gridDim.y){
		thread_data[0] = 0;
		for(int idx = myIDx; idx < num_mutations; idx+= blockDim.x*gridDim.x){
			float pf = d_mutations_freq[idx];
			if(pf > 0 && pf < 1){ //segregating in this population
				float qf = 1-pf;
				float powp = idy*logf(pf);
				float powq = (num_levels-idy)*logf(qf);
				float coeff;
				if(idy < half_n){ coeff = d_binom_coeff[idy]; } else{ coeff = d_binom_coeff[num_levels-idy]; }
				thread_data[0] += expf(coeff+powp+powq);
			}else if(idy == 0){ thread_data[0] += 1.f; } //segregating in other populations: this is the marginal SFS in one population, so they count as monomorphic only
		}
		float aggregate = BlockReduceT(temp_storage).Sum(thread_data);
		if(threadIdx.x == 0){
			if(idy == num_levels){ atomicAdd(&d_histogram[0],aggregate); }
			else{ atomicAdd(&d_histogram[idy],aggregate);  }
		}
	}
	if(myIDx == 0 && myIDy == 0){  atomicAdd(&d_histogram[0],(float)(num_sites-num_mutations));  }
}

} /*----- end namespace Spectrum_details ----- */
//!\endcond

/** To use Spectrum functions and objects, include header file: spectrum.h
 *
 */
namespace Spectrum{

SFS::SFS(): num_populations(0), num_sites(0), num_mutations(0), sampled_generation(0) {frequency_spectrum = NULL; populations = NULL; sample_size = NULL;}
SFS::~SFS(){ if(frequency_spectrum){ delete [] frequency_spectrum; frequency_spectrum = NULL; } if(populations){ delete[] populations; populations = NULL; } if(sample_size){ delete[] sample_size; sample_size = NULL; }}

//frequency histogram of mutations at a single time point in a single population
void population_frequency_histogram(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, int cuda_device){
	using namespace Spectrum_details;
	set_cuda_device(cuda_device);

	cudaStream_t stream;
	cudaCheckErrors(cudaStreamCreate(&stream),-1,-1);

	float * d_mutations_freq;
	float * d_histogram, * h_histogram;
	transfer_allele_trajectories sample(all_results);
	if(!(sample_index >= 0 && sample_index < sample.num_samples) || !(population_index >= 0 && population_index < sample.sim_run_constants.num_populations)){
		fprintf(stderr,"site_frequency_spectrum error: requested indices out of bounds: sample %d [0 %d), population %d [0 %d)\n",sample_index,sample.num_samples,population_index,sample.sim_run_constants.num_populations); exit(1);
	}

	int population_size = sample.time_samples[sample_index]->Nchrom_e[population_index];
	int num_mutations = sample.time_samples[sample_index]->num_mutations;
	float num_sites = sample.sim_run_constants.num_sites;

	cudaCheckErrorsAsync(cudaMalloc((void**)&d_mutations_freq, sample.time_samples[sample_index]->num_mutations*sizeof(float)),-1,-1);
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_histogram, population_size*sizeof(float)),-1,-1);
	cudaCheckErrors(cudaHostRegister(&sample.time_samples[sample_index]->mutations_freq[population_index*num_mutations], num_mutations*sizeof(float), cudaHostRegisterPortable),-1,-1);
	cudaCheckErrorsAsync(cudaMemcpyAsync(d_mutations_freq, &sample.time_samples[sample_index]->mutations_freq[population_index*num_mutations], num_mutations*sizeof(float), cudaMemcpyHostToDevice, stream),-1,-1);
	cudaCheckErrors(cudaHostUnregister(&sample.time_samples[sample_index]->mutations_freq[population_index*num_mutations]),-1,-1);

	unsigned int * d_pop_histogram;
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_pop_histogram, population_size*sizeof(unsigned int)),-1,-1);
	cudaCheckErrorsAsync(cudaMemsetAsync(d_pop_histogram, 0, population_size*sizeof(unsigned int), stream),-1,-1);
	population_hist<<<50,1024,0,stream>>>(d_pop_histogram, d_mutations_freq, population_size, num_mutations, num_sites);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);

	int num_threads = 1024;
	if(population_size < 1024){ num_threads = 256; if(population_size < 256){  num_threads = 128; } }
	int num_blocks = max(population_size/num_threads,1);
	uint_to_float<<<num_blocks,num_threads,0,stream>>>(d_histogram, d_pop_histogram, population_size);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);
	cudaCheckErrorsAsync(cudaFree(d_pop_histogram),-1,-1);

	h_histogram = new float[population_size];
	cudaCheckErrors(cudaHostRegister(h_histogram, sizeof(float)*population_size, cudaHostRegisterPortable),-1,-1);
	cudaCheckErrorsAsync(cudaMemcpyAsync(h_histogram, d_histogram, population_size*sizeof(float), cudaMemcpyDeviceToHost, stream),-1,-1);
	cudaCheckErrors(cudaHostUnregister(h_histogram),-1,-1);

	if(cudaStreamQuery(stream) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(stream), -1, -1); } //wait for writes to host to finish

	mySFS.frequency_spectrum = h_histogram;
	mySFS.num_populations = 1;
	mySFS.sample_size = new int[1];
	mySFS.sample_size[0] = population_size;
	mySFS.num_sites = sample.sim_run_constants.num_sites;
	mySFS.num_mutations = mySFS.num_sites - mySFS.frequency_spectrum[0];
	mySFS.populations = new int[1];
	mySFS.populations[0] = population_index;
	mySFS.sampled_generation = sample.time_samples[sample_index]->sampled_generation;

	cudaCheckErrorsAsync(cudaFree(d_mutations_freq),-1,-1);
	cudaCheckErrorsAsync(cudaFree(d_histogram),-1,-1);
	cudaCheckErrorsAsync(cudaStreamDestroy(stream),-1,-1);
}

//single-population SFS
void site_frequency_spectrum(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, const int sample_size, int cuda_device){
	using namespace Spectrum_details;
	set_cuda_device(cuda_device);

	cudaStream_t stream;
	cudaCheckErrors(cudaStreamCreate(&stream),-1,-1);

	float * d_mutations_freq;
	float * d_histogram, * h_histogram;
	transfer_allele_trajectories sample(all_results);
	if(!(sample_index >= 0 && sample_index < sample.num_samples) || !(population_index >= 0 && population_index < sample.sim_run_constants.num_populations)){
		fprintf(stderr,"site_frequency_spectrum error: requested indices out of bounds: sample %d [0 %d), population %d [0 %d)\n",sample_index,sample.num_samples,population_index,sample.sim_run_constants.num_populations); exit(1);
	}

	int num_levels = sample_size;
	int population_size = sample.time_samples[sample_index]->Nchrom_e[population_index];
	if((sample_size <= 0) || (sample_size >= population_size)){ fprintf(stderr,"site_frequency_spectrum error: requested sample_size out of range [1,population_size): sample_size %d [1,%d)",sample_size,population_size); exit(1); }

	if(sample_size == 0){ num_levels = population_size; }
	int num_mutations = sample.time_samples[sample_index]->num_mutations;
	float num_sites = sample.sim_run_constants.num_sites;

	cudaCheckErrorsAsync(cudaMalloc((void**)&d_mutations_freq, sample.time_samples[sample_index]->num_mutations*sizeof(float)),-1,-1);
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_histogram, num_levels*sizeof(float)),-1,-1);
	cudaCheckErrors(cudaHostRegister(&sample.time_samples[sample_index]->mutations_freq[population_index*num_mutations], num_mutations*sizeof(float), cudaHostRegisterPortable),-1,-1);
	cudaCheckErrorsAsync(cudaMemcpyAsync(d_mutations_freq, &sample.time_samples[sample_index]->mutations_freq[population_index*num_mutations], num_mutations*sizeof(float), cudaMemcpyHostToDevice, stream),-1,-1);
	cudaCheckErrors(cudaHostUnregister(&sample.time_samples[sample_index]->mutations_freq[population_index*num_mutations]),-1,-1);

	int half_n;
	if((num_levels) % 2 == 0){ half_n = (num_levels)/2+1; }
	else{ half_n = (num_levels+1)/2; }

	float * d_binom_partial_coeff;
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_binom_partial_coeff, half_n*sizeof(float)),-1,-1);
	int num_threads = 1024;
	if(half_n < 1024){ num_threads = 256; if(half_n < 256){  num_threads = 128; } }
	int num_blocks = max(num_levels/num_threads,1);
	binom_coeff<<<num_blocks,num_threads,0,stream>>>(d_binom_partial_coeff, half_n, num_levels);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);

	float * d_binom_coeff;
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_binom_coeff, half_n*sizeof(float)),-1,-1);

	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cudaCheckErrorsAsync(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_binom_partial_coeff, d_binom_coeff, half_n, stream),-1,-1);
	cudaCheckErrorsAsync(cudaMalloc(&d_temp_storage, temp_storage_bytes),-1,-1);
	cudaCheckErrorsAsync(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_binom_partial_coeff, d_binom_coeff, half_n, stream),-1,-1);
	cudaCheckErrorsAsync(cudaFree(d_temp_storage),-1,-1);
	cudaCheckErrorsAsync(cudaFree(d_binom_partial_coeff),-1,-1);
	//print_Device_array_double<<<1,1,0,stream>>>(d_binom, 0, half_n);

	const dim3 gridsize(200,20,1);
	const int num_threads_binom = 1024;
	cudaCheckErrorsAsync(cudaMemsetAsync(d_histogram, 0, num_levels*sizeof(float), stream),-1,-1);
	binom<num_threads_binom><<<gridsize,num_threads_binom,0,stream>>>(d_histogram, d_mutations_freq, d_binom_coeff, half_n, num_levels, num_sites, num_mutations);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);
	cudaCheckErrorsAsync(cudaFree(d_binom_coeff),-1,-1);
	//slight differences in each run of the above reduction are due to different floating point error accumulations as different blocks execute in different orders each time
	//can be ignored, might switch back to using doubles (at least for summing into d_histogram & not calculating d_binom_coeff, speed difference was negligble for the former)

	h_histogram = new float[num_levels];
	cudaCheckErrors(cudaHostRegister(h_histogram, sizeof(float)*num_levels, cudaHostRegisterPortable),-1,-1);
	cudaCheckErrorsAsync(cudaMemcpyAsync(h_histogram, d_histogram, num_levels*sizeof(float), cudaMemcpyDeviceToHost, stream),-1,-1);
	cudaCheckErrors(cudaHostUnregister(h_histogram),-1,-1);

	if(cudaStreamQuery(stream) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(stream), -1, -1); } //wait for writes to host to finish

	mySFS.frequency_spectrum = h_histogram;
	mySFS.num_populations = 1;
	mySFS.sample_size = new int[1];
	mySFS.sample_size[0] = num_levels;
	mySFS.num_sites = sample.sim_run_constants.num_sites;
	mySFS.num_mutations = mySFS.num_sites - mySFS.frequency_spectrum[0];
	mySFS.populations = new int[1];
	mySFS.populations[0] = population_index;
	mySFS.sampled_generation = sample.time_samples[sample_index]->sampled_generation;

	cudaCheckErrorsAsync(cudaFree(d_mutations_freq),-1,-1);
	cudaCheckErrorsAsync(cudaFree(d_histogram),-1,-1);
	cudaCheckErrorsAsync(cudaStreamDestroy(stream),-1,-1);
}

} /*----- end namespace SPECTRUM ----- */
