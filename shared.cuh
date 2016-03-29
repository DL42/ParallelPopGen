/*
 * shared.cuh
 *
 *      Author: David Lawrie
 *      for structures and functions used by both go_fish and by sfs
 */

#ifndef SHARED_CUH_
#define SHARED_CUH_

//includes below in sfs & go_fish
#include <cuda_runtime.h>
#include <helper_math.h>
#include <limits.h>
#include <math.h>
#include <iostream>
#include <Random123/philox.h>
#include <Random123/features/compilerfeatures.h>

/* ----- cuda error checking & device setting ----- */
#define __DEBUG__ false
#define cudaCheckErrors(expr1,expr2,expr3) { cudaError_t e = expr1; int g = expr2; int p = expr3; if (e != cudaSuccess) { fprintf(stderr,"error %d %s\tfile %s\tline %d\tgeneration %d\t population %d\n", e, cudaGetErrorString(e),__FILE__,__LINE__, g,p); exit(1); } }
#define cudaCheckErrorsAsync(expr1,expr2,expr3) { cudaCheckErrors(expr1,expr2,expr3); if(__DEBUG__){ cudaCheckErrors(cudaDeviceSynchronize(),expr2,expr3); } }

__forceinline__ cudaDeviceProp set_cuda_device(int & cuda_device){
	int cudaDeviceCount;
	cudaCheckErrorsAsync(cudaGetDeviceCount(&cudaDeviceCount),-1,-1);
	if(cuda_device >= 0 && cuda_device < cudaDeviceCount){ cudaCheckErrors(cudaSetDevice(cuda_device),-1,-1); } //unless user specifies, driver auto-magically selects free GPU to run on
	int myDevice;
	cudaCheckErrorsAsync(cudaGetDevice(&myDevice),-1,-1);
	cudaDeviceProp devProp;
	cudaCheckErrors(cudaGetDeviceProperties(&devProp, myDevice),-1,-1);
	cuda_device = myDevice;
	return devProp;
}

/* ----- end cuda error checking ----- */

/* ----- random number generation ----- */

namespace RNG{

// uint_float_01: Input is a W-bit integer (unsigned).  It is multiplied
// by Float(2^-W) and added to Float(2^(-W-1)).  A good compiler should
// optimize it down to an int-to-float conversion followed by a multiply
// and an add, which might be fused, depending on the architecture.
//
// If the input is a uniformly distributed integer, then the
// result is a uniformly distributed floating point number in [0, 1].
// The result is never exactly 0.0.
// The smallest value returned is 2^-W.
// Let M be the number of mantissa bits in Float.
// If W>M  then the largest value retured is 1.0.
// If W<=M then the largest value returned is the largest Float less than 1.0.
__host__ __device__ __forceinline__ float uint_float_01(unsigned int in){
	//(mostly) stolen from Philox code "uniform.hpp"
	R123_CONSTEXPR float factor = float(1.)/(UINT_MAX + float(1.));
	R123_CONSTEXPR float halffactor = float(0.5)*factor;
    return in*factor + halffactor;
}


__host__ __device__ __forceinline__  uint4 Philox(int2 seed, int k, int step, int population, int round){
	typedef r123::Philox4x32_R<8> P; //can change the 10 rounds of bijection down to 8 (lowest safe limit) to get possible extra speed!
	P rng;

	P::key_type key = {{seed.x, seed.y}}; //random int to set key space + seed
	P::ctr_type count = {{k, step, population, round}};

	union {
		P::ctr_type c;
		uint4 i;
	}u;

	u.c = rng(count, key);

	return u.i;
}

__host__ __device__ __forceinline__ void pois_iter(float j, float mean, float & emu, float & cdf){
	emu *= mean/j;
	cdf += emu;
}

__host__ __device__ __forceinline__ int poiscdfinv(float p, float mean){
	float emu = expf(-1 * mean);
	float cdf = emu;
	if(cdf >= p){ return 0; }

	pois_iter(1.f, mean, emu, cdf); if(cdf >= p){ return 1; }
	pois_iter(2.f, mean, emu, cdf); if(cdf >= p){ return 2; }
	pois_iter(3.f, mean, emu, cdf); if(cdf >= p){ return 3; }
	pois_iter(4.f, mean, emu, cdf); if(cdf >= p){ return 4; }
	pois_iter(5.f, mean, emu, cdf); if(cdf >= p){ return 5; }
	pois_iter(6.f, mean, emu, cdf); if(cdf >= p){ return 6; }
	pois_iter(7.f, mean, emu, cdf); if(cdf >= p){ return 7; }
	pois_iter(8.f, mean, emu, cdf); if(cdf >= p){ return 8; }
	pois_iter(9.f, mean, emu, cdf); if(cdf >= p){ return 9; }
	pois_iter(10.f, mean, emu, cdf); if(cdf >= p){ return 10; }
	pois_iter(11.f, mean, emu, cdf); if(cdf >= p){ return 11; }
	pois_iter(12.f, mean, emu, cdf); if(cdf >= p){ return 12; }
	pois_iter(13.f, mean, emu, cdf); if(cdf >= p){ return 13; }
	pois_iter(14.f, mean, emu, cdf); if(cdf >= p){ return 14; }
	pois_iter(15.f, mean, emu, cdf); if(cdf >= p){ return 15; }
	pois_iter(16.f, mean, emu, cdf); if(cdf >= p){ return 16; }
	pois_iter(17.f, mean, emu, cdf); if(cdf >= p){ return 17; }
	pois_iter(18.f, mean, emu, cdf); if(cdf >= p){ return 18; }
	pois_iter(19.f, mean, emu, cdf); if(cdf >= p){ return 19; }
	pois_iter(20.f, mean, emu, cdf); if(cdf >= p){ return 20; }
	pois_iter(21.f, mean, emu, cdf); if(cdf >= p){ return 21; }
	pois_iter(22.f, mean, emu, cdf); if(cdf >= p){ return 22; }
	pois_iter(23.f, mean, emu, cdf); if(cdf >= p){ return 23; }

	return 24; //limit for mean <= 6, 32 limit for mean <= 10, max float between 0 and 1 is 0.99999999
}

/*__host__ __device__ __forceinline__ int ExactRandBinom(float p, float N, int2 seed, int k, int step, int population, int start_round){
 //only for use when N is small
	int j = 0;
	int counter = 0;

	union {
		uint h[4];
		uint4 i;
	}u;

	while(counter < N){
		u.i = Philox(seed, k, step, population, start_round+counter);

		int counter2 = 0;
		while(counter < N && counter2 < 4){
			if(uint_float_01(u.h[counter2]) <= p){ j++; }
			counter++;
			counter2++;
		}
	}

	return j;
}*/

__host__ __device__ __forceinline__ int ApproxRandBinom1(float mean, float var, float p, float N, int2 seed, int id, int generation, int population){
	uint4 i = Philox(seed, id, generation, population, 0);
	if(mean <= 6){ return poiscdfinv(uint_float_01(i.x), mean); }
	else if(mean >= N-6){ return N - poiscdfinv(uint_float_01(i.x), N-mean); } //flip side of binomial, when 1-p is small
	return round(normcdfinv(uint_float_01(i.x))*sqrtf(var)+mean);
}

//faster on 780M if don't inline!
__device__ int ApproxRandBinomHelper(unsigned int i, float mean, float var, float N);

__device__ __forceinline__ int4 ApproxRandBinom4(float4 mean, float4 var, float4 p, float N, int2 seed, int id, int generation, int population){
	uint4 i = Philox(seed, id, generation, population, 0);
	return make_int4(ApproxRandBinomHelper(i.x, mean.x, var.x, N), ApproxRandBinomHelper(i.y, mean.y, var.y, N), ApproxRandBinomHelper(i.z, mean.z, var.z, N), ApproxRandBinomHelper(i.w, mean.w, var.w, N));
}
/* ----- end random number generation ----- */

} /* ----- end namespace RNG ----- */


namespace GO_Fish{

/* ----- sim result output ----- */
struct mutID{
	int generation,population,threadID,preserve; //generation mutation appeared in simulation, population in which mutation first arose, threadID that generated mutation, flag to preserve mutation in simulation (not filter out if lost or fixed)
};

//for final sim result output
struct sim_result{
	float * mutations_freq; //allele frequency of mutations in final generation
	mutID * mutations_ID; //unique ID consisting of generation, population, threadID, and device
	bool * extinct; //extinct[pop] == true, flag if population is extinct by end of simulation
	int * Nchrom_e; //effective number of chromosomes in each population
	int num_populations; //number of populations in freq array (array length, rows)
	int num_mutations; //number of mutations in array (array length for age/freq, columns)
	int num_sites; //number of sites in simulation
	int sampled_generation; //number of generations in the simulation at point of sampling

	sim_result();
	~sim_result();
};
/* ----- end sim result output ----- */

} /* ----- end namespace GO_Fish ----- */

#endif /* SHARED_CUH_ */
