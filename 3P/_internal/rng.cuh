/*
 * rng.cuh
 *
 *      Author: David Lawrie
 *      for RNG functions
 */

#ifndef RNG_CUH_
#define RNG_CUH_

#include "../_internal/ppp_cuda.cuh"
#include <limits.h>
#include "../_outside_libraries/Random123/philox.h"
#include "../_outside_libraries/Random123/features/compilerfeatures.h"

/* ----- random number generation ----- */

namespace RNG{
constexpr int RNG_MEAN_BOUNDARY_NORM = 6; //max 33
constexpr int RNG_N_BOUNDARY_POIS_BINOM = 100;  //min 70, binomial calculation starts to become numerically unstable for large values of N, not sure where that starts but is between 200 and 200,000
constexpr uint MAX_UINT = std::numeric_limits<uint>::max();


// uint_float_01: Input is a W-bit integer (unsigned).  It is multiplied
// by Float(2^-W) and added to Float(2^(-W-1)).  A good compiler should
// optimize it down to an int-to-float conversion followed by a multiply
// and an add, which might be fused, depending on the architecture.
//
// If the input is a uniformly distributed integer, then the
// result is a uniformly distributed floating point number in (0, 1].
// The result is never exactly 0.0.
// The smallest value returned is 2^-W.
// Let M be the number of mantissa bits in Float.
// If W>M  then the largest value retured is 1.0.
// If W<=M then the largest value returned is the largest Float less than 1.0.
//__host__ __device__ __forceinline__ float uint_float_01(uint in){
//	//(mostly) stolen from Philox code "uniform.hpp"
//	R123_CONSTEXPR float factor = float(1.)/(UINT_MAX + float(1.));
//	R123_CONSTEXPR float halffactor = float(0.5)*factor;
//    return in*factor + halffactor;
//}
//above is old way of doing things, occasionally returns 1 which caused normcdfinv to return inf, check commit 756266bb4fb9522eee2348823c6c6d05ec793534


// uint_float_01:  Return a "fixed point" number in (0,1).  Let:
//   W = width of Itype, e.g., 32 or 64, regardless of signedness.
//   M = mantissa bits of Ftype, e.g., 24, 53 or 64
//   B = min(M, W)
// Then the 2^(B-1) possible output values are:
//    2^-B*{1, 3, 5, ..., 2^B - 1}
// The smallest output is: 2^-B
// The largest output is:  1 - 2^-B
// The output is never exactly 0.0, nor 0.5, nor 1.0.
// The 2^(B-1) possible outputs:
//   - are equally likely,
//   - are uniformly spaced by 2^-(B-1),
//   - are balanced around 0.5

__host__ __device__ __forceinline__ float uint_float_01(uint in){
	//(mostly) stolen from Philox code "uniform.hpp" bit-shifts 'in' and MAX_UINT by the difference W-M (std::numeric_limits<T>::digits)
	constexpr float factor = 1.f/(1.f + ((MAX_UINT>>8)));
    return (1 | (in>>8)) * factor;
}

__host__ __device__ __forceinline__  uint4 Philox(uint2 seed, uint k, uint step, uint population, uint round){
	typedef r123::Philox4x32_R<10> P; //can change the 10 rounds of bijection down to 7 (lowest safe limit) to get possible extra speed!
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

__host__ __device__ __forceinline__ void binom_iter(float j, float x, float n, float & emu, float & cdf){
	emu *= ((n+1.f-j)*x)/(j*(1.f-x));
	cdf += emu;
}

//is fine that uses x as frequency rather than count as this is only for when N is small anyway
__host__ __device__ __forceinline__ uint binomcdfinv(float r, float mean, float x, float n){
	float emu = powf(1.f-x,n);
	if(emu == 1) { emu = expf(-1.f*mean);  }
	float cdf = emu;
	if(cdf >= r){ return 0U; }

	binom_iter(1.f, x, n, emu, cdf); if(cdf >= r){ return 1U; }
	binom_iter(2.f, x, n, emu, cdf); if(cdf >= r){ return 2U; }
	binom_iter(3.f, x, n, emu, cdf); if(cdf >= r){ return 3U; }
	binom_iter(4.f, x, n, emu, cdf); if(cdf >= r){ return 4U; }
	binom_iter(5.f, x, n, emu, cdf); if(cdf >= r){ return 5U; }
	binom_iter(6.f, x, n, emu, cdf); if(cdf >= r){ return 6U; }
	binom_iter(7.f, x, n, emu, cdf); if(cdf >= r){ return 7U; }
	binom_iter(8.f, x, n, emu, cdf); if(cdf >= r){ return 8U; }
	binom_iter(9.f, x, n, emu, cdf); if(cdf >= r){ return 9U; }
	binom_iter(10.f, x, n, emu, cdf); if(cdf >= r){ return 10U; }
	binom_iter(11.f, x, n, emu, cdf); if(cdf >= r || mean <= 1){ return 11U; }
	binom_iter(12.f, x, n, emu, cdf); if(cdf >= r){ return 12U; }
	binom_iter(13.f, x, n, emu, cdf); if(cdf >= r){ return 13U; }
	binom_iter(14.f, x, n, emu, cdf); if(cdf >= r || mean <= 2){ return 14U; }
	binom_iter(15.f, x, n, emu, cdf); if(cdf >= r){ return 15U; }
	binom_iter(16.f, x, n, emu, cdf); if(cdf >= r){ return 16U; }
	binom_iter(17.f, x, n, emu, cdf); if(cdf >= r || mean <= 3){ return 17U; }
	binom_iter(18.f, x, n, emu, cdf); if(cdf >= r){ return 18U; }
	binom_iter(19.f, x, n, emu, cdf); if(cdf >= r){ return 19U; }
	binom_iter(20.f, x, n, emu, cdf); if(cdf >= r || mean <= 4){ return 20U; }
	binom_iter(21.f, x, n, emu, cdf); if(cdf >= r){ return 21U; }
	binom_iter(22.f, x, n, emu, cdf); if(cdf >= r || mean <= 5){ return 22U; }
	binom_iter(23.f, x, n, emu, cdf); if(cdf >= r){ return 23U; }
	binom_iter(24.f, x, n, emu, cdf); if(cdf >= r || mean <= 6){ return 24U; }
	binom_iter(25.f, x, n, emu, cdf); if(cdf >= r){ return 25U; }
	binom_iter(26.f, x, n, emu, cdf); if(cdf >= r || mean <= 7){ return 26U; }
	binom_iter(27.f, x, n, emu, cdf); if(cdf >= r){ return 27U; }
	binom_iter(28.f, x, n, emu, cdf); if(cdf >= r || mean <= 8){ return 28U; }
	binom_iter(29.f, x, n, emu, cdf); if(cdf >= r){ return 29U; }
	binom_iter(30.f, x, n, emu, cdf); if(cdf >= r || mean <= 9){ return 30U; }
	binom_iter(31.f, x, n, emu, cdf); if(cdf >= r){ return 31U; }
	binom_iter(32.f, x, n, emu, cdf); if(cdf >= r || mean <= 10){ return 32U; }
	binom_iter(33.f, x, n, emu, cdf); if(cdf >= r){ return 33U; }
	binom_iter(34.f, x, n, emu, cdf); if(cdf >= r || mean <= 11){ return 34U; }
	binom_iter(35.f, x, n, emu, cdf); if(cdf >= r){ return 35U; }
	binom_iter(36.f, x, n, emu, cdf); if(cdf >= r || mean <= 12){ return 36U; }
	binom_iter(37.f, x, n, emu, cdf); if(cdf >= r){ return 37U; }
	binom_iter(38.f, x, n, emu, cdf); if(cdf >= r || mean <= 13){ return 38U; }
	binom_iter(39.f, x, n, emu, cdf); if(cdf >= r){ return 39U; }
	binom_iter(40.f, x, n, emu, cdf); if(cdf >= r || mean <= 14){ return 40U; }
	binom_iter(41.f, x, n, emu, cdf); if(cdf >= r || mean <= 15){ return 41U; }
	binom_iter(42.f, x, n, emu, cdf); if(cdf >= r){ return 42U; }
	binom_iter(43.f, x, n, emu, cdf); if(cdf >= r || mean <= 16){ return 43U; }
	binom_iter(44.f, x, n, emu, cdf); if(cdf >= r){ return 44U; }
	binom_iter(45.f, x, n, emu, cdf); if(cdf >= r || mean <= 17){ return 45U; }
	binom_iter(46.f, x, n, emu, cdf); if(cdf >= r || mean <= 18){ return 46U; }
	binom_iter(47.f, x, n, emu, cdf); if(cdf >= r){ return 47U; }
	binom_iter(48.f, x, n, emu, cdf); if(cdf >= r || mean <= 19){ return 48U; }
	binom_iter(49.f, x, n, emu, cdf); if(cdf >= r){ return 49U; }
	binom_iter(50.f, x, n, emu, cdf); if(cdf >= r || mean <= 20){ return 50U; }
	binom_iter(51.f, x, n, emu, cdf); if(cdf >= r || mean <= 21){ return 51U; }
	binom_iter(52.f, x, n, emu, cdf); if(cdf >= r){ return 52U; }
	binom_iter(53.f, x, n, emu, cdf); if(cdf >= r || mean <= 22){ return 53U; }
	binom_iter(54.f, x, n, emu, cdf); if(cdf >= r){ return 54U; }
	binom_iter(55.f, x, n, emu, cdf); if(cdf >= r || mean <= 23){ return 55U; }
	binom_iter(56.f, x, n, emu, cdf); if(cdf >= r || mean <= 24){ return 56U; }
	binom_iter(57.f, x, n, emu, cdf); if(cdf >= r){ return 57U; }
	binom_iter(58.f, x, n, emu, cdf); if(cdf >= r || mean <= 25){ return 58U; }
	binom_iter(59.f, x, n, emu, cdf); if(cdf >= r || mean <= 26){ return 59U; }
	binom_iter(60.f, x, n, emu, cdf); if(cdf >= r){ return 60U; }
	binom_iter(61.f, x, n, emu, cdf); if(cdf >= r || mean <= 27){ return 61U; }
	binom_iter(62.f, x, n, emu, cdf); if(cdf >= r || mean <= 28){ return 62U; }
	binom_iter(63.f, x, n, emu, cdf); if(cdf >= r){ return 63U; }
	binom_iter(64.f, x, n, emu, cdf); if(cdf >= r || mean <= 29){ return 64U; }
	binom_iter(65.f, x, n, emu, cdf); if(cdf >= r || mean <= 30){ return 65U; }
	binom_iter(66.f, x, n, emu, cdf); if(cdf >= r){ return 66U; }
	binom_iter(67.f, x, n, emu, cdf); if(cdf >= r || mean <= 31){ return 67U; }
	binom_iter(68.f, x, n, emu, cdf); if(cdf >= r || mean <= 32){ return 68U; }
	binom_iter(69.f, x, n, emu, cdf); if(cdf >= r){ return 69U; }

	return 70U; //17 for mean <= 3, 24 limit for mean <= 6, 32 limit for mean <= 10, 36 limit for mean <= 12, 41 limit for mean <= 15, 58 limit for mean <= 25, 70 limit for mean <= 33; max float between 0 and 1 is 0.99999999
}

__host__ __device__ __forceinline__ void pois_iter(float j, float mean, float & emu, float & cdf){
	emu *= mean*j;
	cdf += emu;
}

__host__ __device__ __forceinline__ uint poiscdfinv(float r, float mean){
	float emu = expf(-1.f*mean);
	float cdf = emu;
	if(cdf >= r){ return 0U; }

	pois_iter(1.f, mean, emu, cdf); if(cdf >= r){ return 1U; }
	pois_iter(1.f/2.f, mean, emu, cdf); if(cdf >= r){ return 2U; }
	pois_iter(1.f/3.f, mean, emu, cdf); if(cdf >= r){ return 3U; }
	pois_iter(1.f/4.f, mean, emu, cdf); if(cdf >= r){ return 4U; }
	pois_iter(1.f/5.f, mean, emu, cdf); if(cdf >= r){ return 5U; }
	pois_iter(1.f/6.f, mean, emu, cdf); if(cdf >= r){ return 6U; }
	pois_iter(1.f/7.f, mean, emu, cdf); if(cdf >= r){ return 7U; }
	pois_iter(1.f/8.f, mean, emu, cdf); if(cdf >= r){ return 8U; }
	pois_iter(1.f/9.f, mean, emu, cdf); if(cdf >= r){ return 9U; }
	pois_iter(1.f/10.f, mean, emu, cdf); if(cdf >= r){ return 10U; }
	pois_iter(1.f/11.f, mean, emu, cdf); if(cdf >= r || mean <= 1){ return 11U; }
	pois_iter(1.f/12.f, mean, emu, cdf); if(cdf >= r){ return 12U; }
	pois_iter(1.f/13.f, mean, emu, cdf); if(cdf >= r){ return 13U; }
	pois_iter(1.f/14.f, mean, emu, cdf); if(cdf >= r || mean <= 2){ return 14U; }
	pois_iter(1.f/15.f, mean, emu, cdf); if(cdf >= r){ return 15U; }
	pois_iter(1.f/16.f, mean, emu, cdf); if(cdf >= r){ return 16U; }
	pois_iter(1.f/17.f, mean, emu, cdf); if(cdf >= r || mean <= 3){ return 17U; }
	pois_iter(1.f/18.f, mean, emu, cdf); if(cdf >= r){ return 18U; }
	pois_iter(1.f/19.f, mean, emu, cdf); if(cdf >= r){ return 19U; }
	pois_iter(1.f/20.f, mean, emu, cdf); if(cdf >= r || mean <= 4){ return 20U; }
	pois_iter(1.f/21.f, mean, emu, cdf); if(cdf >= r){ return 21U; }
	pois_iter(1.f/22.f, mean, emu, cdf); if(cdf >= r || mean <= 5){ return 22U; }
	pois_iter(1.f/23.f, mean, emu, cdf); if(cdf >= r){ return 23U; }
	pois_iter(1.f/24.f, mean, emu, cdf); if(cdf >= r || mean <= 6){ return 24U; }
	pois_iter(1.f/25.f, mean, emu, cdf); if(cdf >= r){ return 25U; }
	pois_iter(1.f/26.f, mean, emu, cdf); if(cdf >= r || mean <= 7){ return 26U; }
	pois_iter(1.f/27.f, mean, emu, cdf); if(cdf >= r){ return 27U; }
	pois_iter(1.f/28.f, mean, emu, cdf); if(cdf >= r || mean <= 8){ return 28U; }
	pois_iter(1.f/29.f, mean, emu, cdf); if(cdf >= r){ return 29U; }
	pois_iter(1.f/30.f, mean, emu, cdf); if(cdf >= r || mean <= 9){ return 30U; }
	pois_iter(1.f/31.f, mean, emu, cdf); if(cdf >= r){ return 31U; }
	pois_iter(1.f/32.f, mean, emu, cdf); if(cdf >= r || mean <= 10){ return 32U; }
	pois_iter(1.f/33.f, mean, emu, cdf); if(cdf >= r){ return 33U; }
	pois_iter(1.f/34.f, mean, emu, cdf); if(cdf >= r || mean <= 11){ return 34U; }
	pois_iter(1.f/35.f, mean, emu, cdf); if(cdf >= r){ return 35U; }
	pois_iter(1.f/36.f, mean, emu, cdf); if(cdf >= r || mean <= 12){ return 36U; }
	pois_iter(1.f/37.f, mean, emu, cdf); if(cdf >= r){ return 37U; }
	pois_iter(1.f/38.f, mean, emu, cdf); if(cdf >= r || mean <= 13){ return 38U; }
	pois_iter(1.f/39.f, mean, emu, cdf); if(cdf >= r){ return 39U; }
	pois_iter(1.f/40.f, mean, emu, cdf); if(cdf >= r || mean <= 14){ return 40U; }
	pois_iter(1.f/41.f, mean, emu, cdf); if(cdf >= r || mean <= 15){ return 41U; }
	pois_iter(1.f/42.f, mean, emu, cdf); if(cdf >= r){ return 42U; }
	pois_iter(1.f/43.f, mean, emu, cdf); if(cdf >= r || mean <= 16){ return 43U; }
	pois_iter(1.f/44.f, mean, emu, cdf); if(cdf >= r){ return 44U; }
	pois_iter(1.f/45.f, mean, emu, cdf); if(cdf >= r || mean <= 17){ return 45U; }
	pois_iter(1.f/46.f, mean, emu, cdf); if(cdf >= r || mean <= 18){ return 46U; }
	pois_iter(1.f/47.f, mean, emu, cdf); if(cdf >= r){ return 47U; }
	pois_iter(1.f/48.f, mean, emu, cdf); if(cdf >= r || mean <= 19){ return 48U; }
	pois_iter(1.f/49.f, mean, emu, cdf); if(cdf >= r){ return 49U; }
	pois_iter(1.f/50.f, mean, emu, cdf); if(cdf >= r || mean <= 20){ return 50U; }
	pois_iter(1.f/51.f, mean, emu, cdf); if(cdf >= r || mean <= 21){ return 51U; }
	pois_iter(1.f/52.f, mean, emu, cdf); if(cdf >= r){ return 52U; }
	pois_iter(1.f/53.f, mean, emu, cdf); if(cdf >= r || mean <= 22){ return 53U; }
	pois_iter(1.f/54.f, mean, emu, cdf); if(cdf >= r){ return 54U; }
	pois_iter(1.f/55.f, mean, emu, cdf); if(cdf >= r || mean <= 23){ return 55U; }
	pois_iter(1.f/56.f, mean, emu, cdf); if(cdf >= r || mean <= 24){ return 56U; }
	pois_iter(1.f/57.f, mean, emu, cdf); if(cdf >= r){ return 57U; }
	pois_iter(1.f/58.f, mean, emu, cdf); if(cdf >= r || mean <= 25){ return 58U; }
	pois_iter(1.f/59.f, mean, emu, cdf); if(cdf >= r || mean <= 26){ return 59U; }
	pois_iter(1.f/60.f, mean, emu, cdf); if(cdf >= r){ return 60U; }
	pois_iter(1.f/61.f, mean, emu, cdf); if(cdf >= r || mean <= 27){ return 61U; }
	pois_iter(1.f/62.f, mean, emu, cdf); if(cdf >= r || mean <= 28){ return 62U; }
	pois_iter(1.f/63.f, mean, emu, cdf); if(cdf >= r){ return 63U; }
	pois_iter(1.f/64.f, mean, emu, cdf); if(cdf >= r || mean <= 29){ return 64U; }
	pois_iter(1.f/65.f, mean, emu, cdf); if(cdf >= r || mean <= 30){ return 65U; }
	pois_iter(1.f/66.f, mean, emu, cdf); if(cdf >= r){ return 66U; }
	pois_iter(1.f/67.f, mean, emu, cdf); if(cdf >= r || mean <= 31){ return 67U; }
	pois_iter(1.f/68.f, mean, emu, cdf); if(cdf >= r || mean <= 32){ return 68U; }
	pois_iter(1.f/69.f, mean, emu, cdf); if(cdf >= r){ return 69U; }

	return 70U; //17 for mean <= 3, 24 limit for mean <= 6, 32 limit for mean <= 10, 36 limit for mean <= 12, 41 limit for mean <= 15, 58 limit for mean <= 25, 70 limit for mean <= 33; max float between 0 and 1 is 0.99999999
}

//draws 4 random numbers, takes the first one, distributes the randomly drawn uint into an approximate/exact Poisson distribution
__host__ __device__ __forceinline__ uint rand1_approx_pois(float mean, float var, float N, uint2 seed, int id, unsigned int generation, unsigned int population){
	uint4 i = Philox(seed, id, generation, population, 0);
	if(mean <= RNG_MEAN_BOUNDARY_NORM){ return poiscdfinv(uint_float_01(i.x), mean); }
	else if(mean >= N-RNG_MEAN_BOUNDARY_NORM){ return ppp::fround_u(N) - poiscdfinv(uint_float_01(i.x), N-mean); } //flip side of poisson, when 1-p is small
	return ppp::rfclamp_u((normcdfinv(uint_float_01(i.x))*sqrtf(var)+mean),N);
}

//distributes the randomly drawn uint into an approximate/exact Binomial distribution
 __device__ __forceinline__ uint approx_binom(uint i, float mean, float var, float N){
	if(mean <= RNG_MEAN_BOUNDARY_NORM){
		if(N < RNG_N_BOUNDARY_POIS_BINOM){ return ppp::clamp_u(binomcdfinv(uint_float_01(i), mean, mean/N, N),N); } else{ return poiscdfinv(uint_float_01(i), mean); } //poisson don't need to be clamped because only called when N is bigger than the largest poisson draw so it can never exceed
	}
	else if(mean >= N-RNG_MEAN_BOUNDARY_NORM){ //flip side of binomial, when 1-p is small
		if(N < RNG_N_BOUNDARY_POIS_BINOM){ return ppp::fround_u(N) - ppp::clamp_u(binomcdfinv(uint_float_01(i), N-mean, (N-mean)/N, N),N); } else{ return ppp::fround_u(N) - poiscdfinv(uint_float_01(i), N-mean); }
	}
	return ppp::rfclamp_u((normcdfinv(uint_float_01(i))*sqrtf(var)+mean),N);
}
/* ----- end random number generation ----- */

} /* ----- end namespace RNG ----- */

#endif /* SHARED_CUH_ */
