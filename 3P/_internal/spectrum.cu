/*
 * spectrum.cu
 *
 *      Author: David Lawrie
 */

#include "../spectrum.h"
#include "../_internal/ppp_types.hpp"
#include "../_internal/ppp_cuda.cuh"
#include "../_outside_libraries/cub/device/device_scan.cuh"
#include "../_outside_libraries/cub/block/block_reduce.cuh"
#include <math_constants.h>

//!\cond
namespace spectrum_details{

__global__ void population_hist(unsigned int * out_histogram, const uint * const in_mutation_freq, uint Nchrome_e, int num_mutations, int num_sites){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;

	for(uint id = myID; id < num_mutations; id+= blockDim.x*gridDim.x){
		uint index = in_mutation_freq[id];
		if(index == Nchrome_e){ index = 0; }
		atomicAdd(&out_histogram[index],1U);
	}
	if(myID == 0){  atomicAdd(&out_histogram[0], (num_sites - num_mutations));  }
}

__global__ void uint_to_float(float * out_array, const unsigned int * const in_array, uint N){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;
	for(int id = myID; id < N; id+= blockDim.x*gridDim.x){ out_array[id] = in_array[id]; }
}

__global__ void binom_coeff(float * binom_coeff, uint half_n, uint n){
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
__global__ void binom(float * d_histogram, const uint * const d_mutations_freq, const float * const d_binom_coeff, const uint Nchrom_e, const float log_Nchrom_e, const uint half_n, const uint num_levels, float num_sites, int num_mutations){
	int myIDx =  blockIdx.x*blockDim.x + threadIdx.x;
	int myIDy = blockIdx.y;
	typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduceT;
	__shared__ typename BlockReduceT::TempStorage temp_storage;
	float thread_data[1];

	for(int idy = myIDy; idy <= num_levels; idy+= blockDim.y*gridDim.y){
		thread_data[0] = 0;
		float coeff;
		if(idy < half_n){ coeff = d_binom_coeff[idy]; } else{ coeff = d_binom_coeff[num_levels-idy]; }
		for(int idx = myIDx; idx < num_mutations; idx+= blockDim.x*gridDim.x){
			uint pf = d_mutations_freq[idx];
			if(pf > 0 && pf < Nchrom_e){ //segregating in this population
				uint qf = Nchrom_e-pf;
				float powp = idy*(logf(pf)-log_Nchrom_e);
				float powq = (num_levels-idy)*(logf(qf)-log_Nchrom_e);
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

} /*----- end namespace spectrum_details ----- */
//!\endcond

/** To use Spectrum functions and objects, include header file: spectrum.h
 *
 */
namespace Spectrum{

SFS::SFS(): num_populations(0), num_sites(0), num_mutations(0), sampled_generation(0) {frequency_spectrum = nullptr; populations = nullptr; sample_size = nullptr;}

//frequency histogram of mutations at a single time point in a single population
void population_frequency_histogram(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, int cuda_device){
	using namespace spectrum_details;
	ppp::set_cuda_device(cuda_device);

	ppp::raii_stream_array stream(1);

	GO_Fish::sim_constants sim_run_constants = all_results.last_run_constants();
	int num_samples = all_results.num_time_samples();
	if(!(sample_index >= 0 && sample_index < num_samples) || !(population_index >= 0 && population_index < sim_run_constants.num_populations)){
		fprintf(stderr,"population_frequency_histogram error: requested indices out of bounds: sample %d [0 %d), population %d [0 %d)\n",sample_index,num_samples,population_index,sim_run_constants.num_populations); exit(1);
	}

	uint population_size = all_results.effective_number_of_chromosomes(sample_index,population_index);
	int num_mutations = all_results.num_mutations_time_sample(sample_index);
	float num_sites = sim_run_constants.num_sites;

	auto d_mutations_freq = ppp::make_unique_device_ptr<unsigned int>(num_mutations,-1,-1);
	auto d_histogram = ppp::make_unique_device_ptr<float>(population_size,-1,-1);
	auto h_pointer = all_results.allele_count_span(sample_index,population_index,1);
	cudaCheckErrors(cudaMemcpy(d_mutations_freq.get(), h_pointer.data(), h_pointer.size_bytes(), cudaMemcpyHostToDevice),-1,-1);

	auto d_pop_histogram = ppp::make_unique_device_ptr<unsigned int>(population_size,-1,-1);
	cudaCheckErrorsAsync(cudaMemsetAsync(d_pop_histogram.get(), 0, population_size*sizeof(unsigned int), stream[0]),-1,-1);
	population_hist<<<50,1024,0,stream[0]>>>(d_pop_histogram.get(), d_mutations_freq.get(), population_size, num_mutations, num_sites);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);

	int num_threads = 1024;
	if(population_size < 1024){ num_threads = 256; if(population_size < 256){  num_threads = 128; } }
	int num_blocks = max(population_size/num_threads,1);
	uint_to_float<<<num_blocks,num_threads,0,stream[0]>>>(d_histogram.get(), d_pop_histogram.get(), population_size);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);

	if(cudaStreamQuery(stream[0]) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(stream[0]), -1, -1); } //wait for writes to hist to finish

	mySFS.frequency_spectrum = std::make_unique<float[]>(population_size);
	cudaCheckErrors(cudaMemcpy(mySFS.frequency_spectrum.get(), d_histogram.get(), population_size*sizeof(float), cudaMemcpyDeviceToHost),-1,-1);

	mySFS.num_populations = 1;
	mySFS.sample_size = std::make_unique<unsigned int[]>(1);
	mySFS.sample_size[0] = population_size;
	mySFS.num_sites = num_sites;
	mySFS.num_mutations = mySFS.num_sites - mySFS.frequency_spectrum[0];
	mySFS.populations = std::make_unique<int[]>(1);
	mySFS.populations[0] = population_index;
	mySFS.sampled_generation = all_results.sampled_generation(sample_index);
}

//single-population SFS
void site_frequency_spectrum(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, const unsigned int sample_size, int cuda_device){
	using namespace spectrum_details;
	ppp::set_cuda_device(cuda_device);

	ppp::raii_stream_array stream(1);

	GO_Fish::sim_constants sim_run_constants = all_results.last_run_constants();
	int num_samples = all_results.num_time_samples();
	if(!(sample_index >= 0 && sample_index < num_samples) || !(population_index >= 0 && population_index < sim_run_constants.num_populations)){
		fprintf(stderr,"site_frequency_spectrum error: requested indices out of bounds: sample %d [0 %d), population %d [0 %d)\n",sample_index,num_samples,population_index,sim_run_constants.num_populations); exit(1);
	}

	uint num_levels = sample_size;
	uint population_size = all_results.effective_number_of_chromosomes(sample_index,population_index);
	if((sample_size <= 0) || (sample_size >= population_size)){ fprintf(stderr,"site_frequency_spectrum error: requested sample_size out of range [1,population_size): sample_size %d [1,%d)",sample_size,population_size); exit(1); }

	if(sample_size == 0){ num_levels = population_size; }
	int num_mutations = all_results.num_mutations_time_sample(sample_index);
	float num_sites = sim_run_constants.num_sites;

	auto d_mutations_freq = ppp::make_unique_device_ptr<unsigned int>(num_mutations,-1,-1);
	auto d_histogram = ppp::make_unique_device_ptr<float>(num_levels,-1,-1);
	auto h_pointer = all_results.allele_count_span(sample_index,population_index,1);
	cudaCheckErrors(cudaMemcpy(d_mutations_freq.get(), h_pointer.data(), h_pointer.size_bytes(), cudaMemcpyHostToDevice),-1,-1);

	uint half_n;
	if((num_levels) % 2 == 0){ half_n = (num_levels)/2+1; }
	else{ half_n = (num_levels+1)/2; }

	auto d_binom_partial_coeff = ppp::make_unique_device_ptr<float>(half_n,-1,-1);
	int num_threads = 1024;
	if(half_n < 1024){ num_threads = 256; if(half_n < 256){  num_threads = 128; } }
	int num_blocks = max(num_levels/num_threads,1);
	binom_coeff<<<num_blocks,num_threads,0,stream[0]>>>(d_binom_partial_coeff.get(), half_n, num_levels);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);

	auto d_binom_coeff = ppp::make_unique_device_ptr<float>(half_n,-1,-1);

	auto d_temp_storage = ppp::make_unique_device_ptr<void>(0,-1,-1);
	size_t temp_storage_bytes = 0;
	cudaCheckErrorsAsync(cub::DeviceScan::InclusiveSum(d_temp_storage.get(), temp_storage_bytes, d_binom_partial_coeff.get(), d_binom_coeff.get(), half_n, stream[0]),-1,-1);
	ppp::reset_device_ptr(d_temp_storage,temp_storage_bytes,-1,-1);
	cudaCheckErrorsAsync(cub::DeviceScan::InclusiveSum(d_temp_storage.get(), temp_storage_bytes, d_binom_partial_coeff.get(), d_binom_coeff.get(), half_n, stream[0]),-1,-1);
	//print_Device_array_double<<<1,1,0,stream[0]>>>(d_binom, 0, half_n);

	const dim3 gridsize(200,20,1);
	const int num_threads_binom = 1024;
	cudaCheckErrorsAsync(cudaMemsetAsync(d_histogram.get(), 0, num_levels*sizeof(float), stream[0]),-1,-1);
	binom<num_threads_binom><<<gridsize,num_threads_binom,0,stream[0]>>>(d_histogram.get(), d_mutations_freq.get(), d_binom_coeff.get(), population_size, logf(population_size), half_n, num_levels, num_sites, num_mutations);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);
	//slight differences in each run of the above reduction are due to different floating point error accumulations as different blocks execute in different orders each time
	//can be ignored, might switch back to using doubles (at least for summing into d_histogram & not calculating d_binom_coeff, speed difference was negligble for the former)

	if(cudaStreamQuery(stream[0]) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(stream[0]), -1, -1); } //wait for writes to hist to finish

	mySFS.frequency_spectrum = std::make_unique<float[]>(num_levels);
	cudaCheckErrors(cudaMemcpy(mySFS.frequency_spectrum.get(), d_histogram.get(), num_levels*sizeof(float), cudaMemcpyDeviceToHost),-1,-1);

	mySFS.num_populations = 1;
	mySFS.sample_size = std::make_unique<unsigned int[]>(1);
	mySFS.sample_size[0] = num_levels;
	mySFS.num_sites = num_sites;
	mySFS.num_mutations = mySFS.num_sites - mySFS.frequency_spectrum[0];
	mySFS.populations = std::make_unique<int[]>(1);
	mySFS.populations[0] = population_index;
	mySFS.sampled_generation = all_results.sampled_generation(sample_index);
}

} /*----- end namespace SPECTRUM ----- */
