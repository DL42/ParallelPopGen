/*
 * shared.cu
 *
 *      Author: David Lawrie
 *      for cuda and rand functions used by both go_fish and by sfs
 */

#include "shared.cuh"

__device__ int RNG::ApproxRandBinomHelper(unsigned int i, float mean, float var, float N){
	if(mean <= RNG_MEAN_BOUNDARY_NORM){
		if(N < RNG_N_BOUNDARY_POIS_BINOM){ return binomcdfinv(uint_float_01(i), mean, mean/N, N); } else{ return poiscdfinv(uint_float_01(i), mean); }
	}
	else if(mean >= N-RNG_MEAN_BOUNDARY_NORM){ //flip side of binomial, when 1-p is small
		if(N < RNG_N_BOUNDARY_POIS_BINOM){ return N - binomcdfinv(uint_float_01(i), N-mean, (N-mean)/N, N); } else{ return N - poiscdfinv(uint_float_01(i), N-mean); }
	}
	return round(normcdfinv(uint_float_01(i))*sqrtf(var)+mean);
}
