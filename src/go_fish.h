/*
 * go_fish.h
 *
 *      Author: David Lawrie
 */

#ifndef FW_SIM_API_H_
#define FW_SIM_API_H_
#include <cuda_runtime.h>
#include "shared.cuh"

namespace GO_Fish{

/* ----- selection models ----- */
struct const_selection
{
	float s;
	const_selection();
	const_selection(float s);
	__host__ __device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const;
};

struct linear_frequency_dependent_selection
{
	float slope;
	float intercept;
	linear_frequency_dependent_selection();
	linear_frequency_dependent_selection(float slope, float intercept);
	__host__ __device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const;
};

//models selection as a sine wave through time
struct seasonal_selection
{
	float A; //Amplitude
	float pi; //Frequency
	float rho; //Phase
	float D; //Offset
	int generation_shift;

	seasonal_selection();
	seasonal_selection(float A, float pi, float D, float rho = 0, int generation_shift = 0);
	__host__ __device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const;
};

//one population, pop, has a different, selection functor, s_pop
template <typename Functor_sel, typename Functor_sel_pop>
struct population_specific_selection
{
	int pop, generation_shift;
	Functor_sel s;
	Functor_sel_pop s_pop;
	population_specific_selection();
	population_specific_selection(Functor_sel s_in, Functor_sel_pop s_pop_in, int pop, int generation_shift = 0);
	__host__ __device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const;
};

//selection function changes at inflection_point
template <typename Functor_sel1, typename Functor_sel2>
struct piecewise_selection
{
	int inflection_point, generation_shift;
	Functor_sel1 s1;
	Functor_sel2 s2;
	piecewise_selection();
	piecewise_selection(Functor_sel1 s1_in, Functor_sel2 s2_in, int inflection_point, int generation_shift = 0);
	__host__ __device__ __forceinline__ float operator()(const int population, const int generation, const float freq) const;
};
/* ----- end selection models ----- */

/* ----- mutation, dominance, & inbreeding models ----- */
struct const_parameter
{
	float p;
	const_parameter();
	const_parameter(float p);
	__host__ __forceinline__ float operator()(const int population, const int generation) const;
};

//models parameter as a sine wave through time
struct seasonal_parameter
{
	float A; //Amplitude
	float pi; //Frequency
	float rho; //Phase
	float D; //Offset
	int generation_shift;

	seasonal_parameter();
	seasonal_parameter(float A, float pi, float D, float rho = 0, int generation_shift = 0);
	__host__ __forceinline__ float operator()(const int population, const int generation) const;
};

//one population, pop, has a different, parameter functor, p_pop
template <typename Functor_p, typename Functor_p_pop>
struct population_specific_parameter
{
	int pop, generation_shift;
	Functor_p p;
	Functor_p_pop p_pop;
	population_specific_parameter();
	population_specific_parameter(Functor_p p_in, Functor_p_pop p_pop_in, int pop, int generation_shift = 0);
	__host__ __forceinline__ float operator()(const int population, const int generation) const;
};

//parameter function changes at inflection_point
template <typename Functor_p1, typename Functor_p2>
struct piecewise_parameter
{
	int inflection_point, generation_shift;
	Functor_p1 p1;
	Functor_p2 p2;
	piecewise_parameter();
	piecewise_parameter(Functor_p1 p1_in, Functor_p2 p2_in, int inflection_point, int generation_shift = 0);
	__host__ __forceinline__ float operator()(const int population, const int generation) const;
};
/* ----- end of mutation, dominance, & inbreeding models ----- */

/* ----- demography models ----- */
struct const_demography
{
	int p;
	const_demography();
	const_demography(int p);
	__host__ __device__  __forceinline__ int operator()(const int population, const int generation) const;
};

//models demography as a sine wave through time
struct seasonal_demography
{
	float A; //Amplitude
	float pi; //Frequency
	float rho; //Phase
	int D; //Offset
	int generation_shift;

	seasonal_demography();
	seasonal_demography(float A, float pi, int D, float rho = 0, int generation_shift = 0);
	__host__ __device__  __forceinline__ int operator()(const int population, const int generation) const;
};

//models exponential growth of population size over time
struct exponential_growth
{
	float rate;
	int initial_population_size;
	int generation_shift;

	exponential_growth();
	exponential_growth(float rate, int initial_population_size, int generation_shift = 0);
	__host__ __device__ __forceinline__ int operator()(const int population, const int generation) const;
};

//models logistic growth of population size over time
struct logistic_growth
{
	float rate;
	int initial_population_size;
	int carrying_capacity;
	int generation_shift;

	logistic_growth();
	logistic_growth(float rate, int initial_population_size, int carrying_capacity, int generation_shift = 0);
	__host__ __device__ __forceinline__ int operator()(const int population, const int generation) const;
};

//one population, pop, has a different, demography functor, p_pop
template <typename Functor_p, typename Functor_p_pop>
struct population_specific_demography
{
	int pop, generation_shift;
	Functor_p p;
	Functor_p_pop p_pop;
	population_specific_demography();
	population_specific_demography(Functor_p p_in, Functor_p_pop p_pop_in, int pop, int generation_shift = 0);
	__host__ __device__ __forceinline__ int operator()(const int population, const int generation) const;
};

//demography function changes at inflection_point
template <typename Functor_p1, typename Functor_p2>
struct piecewise_demography
{
	int inflection_point, generation_shift;
	Functor_p1 p1;
	Functor_p2 p2;
	piecewise_demography();
	piecewise_demography(Functor_p1 p1_in, Functor_p2 p2_in, int inflection_point, int generation_shift = 0);
	__host__ __device__ __forceinline__ int operator()(const int population, const int generation) const;
};
/* ----- end of demography models ----- */

/* ----- migration models ----- */
struct const_equal_migration
{
	float m;
	int num_pop;
	const_equal_migration();
	const_equal_migration(int n);
	const_equal_migration(float m, int n);
	__host__ __device__ __forceinline__ float operator()(const int pop_FROM, const int pop_TO, const int generation) const;
};

//migration flows at rate m from pop1 to pop2 and Functor_m1 for the rest
template <typename Functor_m1>
struct const_directional_migration
{
	float m;
	int pop1, pop2;
	Functor_m1 rest;
	const_directional_migration();
	const_directional_migration(float m, int pop1, int pop2, Functor_m1 rest_in);
	__host__ __device__ __forceinline__ float operator()(const int pop_FROM, const int pop_TO, const int generation) const;
};

//migration function changes at inflection_point
template <typename Functor_m1, typename Functor_m2>
struct piecewise_migration
{
	int inflection_point, generation_shift;
	Functor_m1 m1;
	Functor_m2 m2;
	piecewise_migration();
	piecewise_migration(Functor_m1 m1_in, Functor_m2 m2_in, int inflection_point, int generation_shift = 0);
	__host__ __device__ __forceinline__ int operator()(const int pop_FROM, const int pop_TO, const int generation) const;
};

/* ----- end of migration models ----- */

/* ----- preserving & sampling functions ----- */
struct do_nothing{ __host__ __forceinline__ bool operator()(const int generation) const; };

struct do_something{__host__ __forceinline__ bool operator()(const int generation) const; };

//returns the result of Functor_stable except at time Fgen(-generation_shift) returns the result of Functor_action
template <typename Functor_stable, typename Functor_action>
struct do_something_else{
	int Fgen, generation_shift;
	Functor_stable f1;
	Functor_action f2;
	do_something_else();
	do_something_else(Functor_stable f1_in, Functor_action f2_in, int Fgen, int generation_shift = 0);
	__host__ __forceinline__ bool operator()(const int generation) const;
};
/* ----- end of preserving & sampling functions ----- */

/* ----- go_fish_impl  ----- */
template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_preserve, typename Functor_timesample>
__host__ sim_result * run_sim(const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding FI, const Functor_dominance dominance, const int num_generations, const float num_sites, const int num_populations, const int seed1, const int seed2, Functor_preserve preserve_mutations, Functor_timesample take_sample, int max_samples = 0, const bool init_mse = true, const sim_result & prev_sim = sim_result(), const int compact_rate = 35, int cuda_device = -1);
/* ----- end go_fish_impl ----- */

} /* ----- end namespace GO_Fish ----- */

/* ----- importing functor implementations ----- */
#include "simulation_functors.cuh"
/* ----- end importing functor implementations ----- */

/* ----- importing go_fish_impl  ----- */
#include "go_fish_impl.cuh"
/* ----- end importing go_fish_impl ----- */


#endif /* GO_FISH_H_ */
