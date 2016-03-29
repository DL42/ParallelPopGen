/*
 * sfs.cu
 *
 *      Author: David Lawrie
 */

#include "sfs.h"
#include "shared.cuh"

namespace SFS{

sfs::sfs(): num_populations(0), num_sites(0), sampled_generation(0) {frequency_spectrum = NULL; populations = NULL; num_samples = NULL;}
sfs::~sfs(){ if(frequency_spectrum){ cudaCheckErrors(cudaFreeHost(frequency_spectrum),-1,-1); } if(populations){ delete[] populations; } if(num_samples){ delete[] num_samples; }}

__global__ void simple_hist(int * out_histogram, float * in_mutation_freq, int num_samples, int num_mutations, int num_sites){
	int myID =  blockIdx.x*blockDim.x + threadIdx.x;

	for(int id = myID; id < num_mutations; id+= blockDim.x*gridDim.x){
		int index = round(num_samples*in_mutation_freq[id]);
		atomicAdd(&out_histogram[index],1);
	}
	if(myID == 0){  out_histogram[0] = num_sites - num_mutations;  }
}

//single-population sfs, only segregating mutations
__host__ sfs site_frequency_spectrum(GO_Fish::sim_result & sim, int population, int cuda_device /*= -1*/){

	set_cuda_device(cuda_device);

	cudaStream_t stream;

	cudaCheckErrors(cudaStreamCreate(&stream),-1,-1);

	float * d_mutations_freq;
	int * d_histogram, * h_histogram;

	int num_levels = sim.Nchrom_e[population]+1;

	cudaCheckErrorsAsync(cudaMalloc((void**)&d_mutations_freq, sim.num_mutations*sizeof(float)),-1,-1);
	cudaCheckErrorsAsync(cudaMalloc((void**)&d_histogram, num_levels*sizeof(int)),-1,-1);
	cudaCheckErrorsAsync(cudaMemsetAsync(d_histogram, 0, num_levels*sizeof(int), stream),-1,-1);
	cudaCheckErrorsAsync(cudaMemcpyAsync(d_mutations_freq, &sim.mutations_freq[population*sim.num_mutations], sim.num_mutations*sizeof(float), cudaMemcpyHostToDevice, stream),-1,-1);

	simple_hist<<<50,1024,0,stream>>>(d_histogram, d_mutations_freq, sim.Nchrom_e[population], sim.num_mutations, sim.num_sites);
	cudaCheckErrorsAsync(cudaPeekAtLastError(),-1,-1);

	cudaCheckErrors(cudaMallocHost((void**)&h_histogram, num_levels*sizeof(int)),-1,-1);
	cudaCheckErrorsAsync(cudaMemcpyAsync(h_histogram, d_histogram, num_levels*sizeof(int), cudaMemcpyDeviceToHost, stream),-1,-1);

	if(cudaStreamQuery(stream) != cudaSuccess){ cudaCheckErrors(cudaStreamSynchronize(stream), -1, -1); } //wait for writes to host to finish

	sfs mySFS;
	mySFS.frequency_spectrum = h_histogram;
	mySFS.num_populations = 1;
	mySFS.num_samples = new int[1];
	mySFS.num_samples[0] = num_levels;
	mySFS.num_sites = sim.num_sites;
	mySFS.populations = new int[1];
	mySFS.populations[0] = population;
	mySFS.sampled_generation = sim.sampled_generation;

	//cudaCheckErrorsAsync(cudaFree(d_temp_storage),-1,-1);
	cudaCheckErrorsAsync(cudaFree(d_mutations_freq),-1,-1);
	cudaCheckErrorsAsync(cudaFree(d_histogram),-1,-1);
	cudaCheckErrorsAsync(cudaStreamDestroy(stream),-1,-1)

	return mySFS;
}

/*__host__ GO_Fish::sim_result sequencing_sample(GO_Fish::sim_result sim, int * population, int * num_samples, const int seed){

	return GO_Fish::sim_result();
}

//multi-time point, multi-population sfs
__host__ sfs temporal_site_frequency_spectrum(GO_Fish::sim_result sim, int * population, int * num_samples, int num_sfs_populations, const int seed){

	return sfs();
}

//trace frequency trajectories of mutations from generation start to generation end in a (sub-)population
//can track an individual mutation or groups of mutation by specifying when the mutation was "born", in which population, with what threadID
__host__ float ** trace_mutations(GO_Fish::sim_result * sim, int generation_start, int generation_end, int population, int generation_born = -1, int population_born = -1, int threadID = -1){

	return NULL;
}*/

} /*----- end namespace SFS ----- */
