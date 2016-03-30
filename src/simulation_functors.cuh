/*
 * simulation_functors.cuh
 *
 *      Author: David Lawrie
 */

#ifndef SIMULATION_FUNCTORS_CUH_
#define SIMULATION_FUNCTORS_CUH_

namespace GO_Fish{

/* ----- selection models ----- */
/* ----- constant selection model ----- */
const_selection::const_selection() : s(0) {}
const_selection::const_selection(float s) : s(s){ }
__host__ __device__ __forceinline__ float const_selection::operator()(const int population, const int generation, const float freq) const{ return s; }
/* ----- end constant selection model ----- */

/* ----- linear frequency dependent selection model ----- */
linear_frequency_dependent_selection::linear_frequency_dependent_selection() : slope(0), intercept(0) {}
linear_frequency_dependent_selection::linear_frequency_dependent_selection(float slope, float intercept) : slope(slope), intercept(intercept) { }
__host__ __device__ __forceinline__ float linear_frequency_dependent_selection::operator()(const int population, const int generation, const float freq) const{ return slope*freq+intercept; }
/* ----- end linear frequency dependent selection model ----- */

/* ----- seasonal selection model ----- */
seasonal_selection::seasonal_selection() : A(0), pi(0), rho(0), D(0), generation_shift(0) {}
seasonal_selection::seasonal_selection(float A, float pi, float D, float rho /*= 0*/, int generation_shift /*= 0*/) : A(A), pi(pi), rho(rho), D(D), generation_shift(generation_shift) {}
__host__ __device__ __forceinline__ float seasonal_selection::operator()(const int population, const int generation, const float freq) const{ return A*sin(pi*(generation-generation_shift) + rho) + D;}
/* ----- end seasonal selection model ----- */

/* ----- population specific selection model ----- */
template <typename Functor_sel, typename Functor_sel_pop>
population_specific_selection<Functor_sel,Functor_sel_pop>::population_specific_selection() : pop(0), generation_shift(0) { s = Functor_sel(); s_pop = Functor_sel_pop(); }
template <typename Functor_sel, typename Functor_sel_pop>
population_specific_selection<Functor_sel,Functor_sel_pop>::population_specific_selection(Functor_sel s_in, Functor_sel_pop s_pop_in, int pop, int generation_shift /*= 0*/) : pop(pop), generation_shift(generation_shift){  s = s_in; s_pop = s_pop_in; }
template <typename Functor_sel, typename Functor_sel_pop>
__host__ __device__ __forceinline__ float population_specific_selection<Functor_sel,Functor_sel_pop>::operator()(const int population, const int generation, const float freq) const{
	if(pop == population) return s_pop(population, generation-generation_shift, freq);
	return s(population, generation-generation_shift, freq);
}
/* ----- end population specific selection model ----- */

/* ----- piecewise selection model ----- */
template <typename Functor_sel1, typename Functor_sel2>
piecewise_selection<Functor_sel1, Functor_sel2>::piecewise_selection() : inflection_point(0), generation_shift(0) { s1 = Functor_sel1(); s2 = Functor_sel2(); }
template <typename Functor_sel1, typename Functor_sel2>
piecewise_selection<Functor_sel1, Functor_sel2>::piecewise_selection(Functor_sel1 s1_in, Functor_sel2 s2_in, int inflection_point, int generation_shift /* = 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { s1 = s1_in; s2 = s2_in; }
template <typename Functor_sel1, typename Functor_sel2>
__host__ __device__ __forceinline__ float piecewise_selection<Functor_sel1, Functor_sel2>::operator()(const int population, const int generation, const float freq) const{
	if(generation >= inflection_point+generation_shift){ return s2(population, generation-generation_shift, freq) ; }
	return s1(population, generation-generation_shift, freq);
};
/* ----- end piecewise selection model ----- */
/* ----- end selection models ----- */

/* ----- mutation, dominance, & inbreeding models ----- */
/* ----- constant parameter model ----- */
const_parameter::const_parameter() : p(0) {}
const_parameter::const_parameter(float p) : p(p){ }
__host__ __forceinline__ float const_parameter::operator()(const int population, const int generation) const{ return p; }
/* ----- end constant parameter model ----- */

/* ----- seasonal parameter model ----- */
seasonal_parameter::seasonal_parameter() : A(0), pi(0), rho(0), D(0), generation_shift(0) {}
seasonal_parameter::seasonal_parameter(float A, float pi, float D, float rho /*= 0*/, int generation_shift /*= 0*/) : A(A), pi(pi), rho(rho), D(D), generation_shift(generation_shift) {}
__host__ __forceinline__ float seasonal_parameter::operator()(const int population, const int generation) const{ return A*sin(pi*(generation-generation_shift) + rho) + D;}
/* ----- end seasonal parameter model ----- */

/* ----- population specific parameter model ----- */
template <typename Functor_p, typename Functor_p_pop>
population_specific_parameter<Functor_p,Functor_p_pop>::population_specific_parameter() : pop(0), generation_shift(0) { p = Functor_p(); p_pop = Functor_p_pop(); }
template <typename Functor_p, typename Functor_p_pop>
population_specific_parameter<Functor_p,Functor_p_pop>::population_specific_parameter(Functor_p p_in, Functor_p_pop p_pop_in, int pop, int generation_shift /*= 0*/) : pop(pop), generation_shift(generation_shift){  p = p_in; p_pop = p_pop_in; }
template <typename Functor_p, typename Functor_p_pop>
__host__ __forceinline__ float population_specific_parameter<Functor_p,Functor_p_pop>::operator()(const int population, const int generation) const{
	if(pop == population) return p_pop(population, generation-generation_shift);
	return p(population, generation-generation_shift);
}
/* ----- end population specific parameter model ----- */

/* ----- piecewise parameter model ----- */
template <typename Functor_p1, typename Functor_p2>
piecewise_parameter<Functor_p1, Functor_p2>::piecewise_parameter() : inflection_point(0), generation_shift(0) { p1 = Functor_p1(); p2 = Functor_p2(); }
template <typename Functor_p1, typename Functor_p2>
piecewise_parameter<Functor_p1, Functor_p2>::piecewise_parameter(Functor_p1 p1_in, Functor_p2 p2_in, int inflection_point, int generation_shift /* = 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { p1 = p1_in; p2 = p2_in; }
template <typename Functor_p1, typename Functor_p2>
__host__ __forceinline__ float piecewise_parameter<Functor_p1, Functor_p2>::operator()(const int population, const int generation) const{
	if(generation >= inflection_point+generation_shift){ return p2(population, generation-generation_shift) ; }
	return p1(population, generation-generation_shift);
};
/* ----- end piecewise parameter model ----- */
/* ----- end of mutation, dominance, & inbreeding models ----- */

/* ----- demography models ----- */
const_demography::const_demography() : p(0) {}
const_demography::const_demography(int p) : p(p){ }
__host__ __device__  __forceinline__ int const_demography::operator()(const int population, const int generation) const{ return p; }

/* ----- seasonal demography model ----- */
seasonal_demography::seasonal_demography() : A(0), pi(0), rho(0), D(0), generation_shift(0) {}
seasonal_demography::seasonal_demography(float A, float pi, int D, float rho /*= 0*/, int generation_shift /*= 0*/) : A(A), pi(pi), rho(rho), D(D), generation_shift(generation_shift) {}
__host__ __device__  __forceinline__ int seasonal_demography::operator()(const int population, const int generation) const{ return (int)A*sin(pi*(generation-generation_shift) + rho) + D;}
/* ----- end seasonal parameter model ----- */

/* ----- exponential growth model ----- */
exponential_growth::exponential_growth() : rate(0), initial_population_size(0), generation_shift(0) {}
exponential_growth::exponential_growth(float rate, int initial_population_size, int generation_shift /*= 0*/) : rate(rate), initial_population_size(initial_population_size), generation_shift(generation_shift) {}
__host__ __device__  __forceinline__ int exponential_growth::operator()(const int population, const int generation) const{ return (int)round(initial_population_size*exp(rate*(generation-generation_shift))); }
/* ----- end exponential growth model ----- */

/* ----- exponential growth model ----- */
logistic_growth::logistic_growth() : rate(0), initial_population_size(0), carrying_capacity(0), generation_shift(0) {}
logistic_growth::logistic_growth(float rate, int initial_population_size, int carrying_capacity, int generation_shift /*= 0*/) : rate(rate), initial_population_size(initial_population_size), carrying_capacity(carrying_capacity), generation_shift(generation_shift) {}
__host__ __device__  __forceinline__ int logistic_growth::operator()(const int population, const int generation) const{
	float term = exp(rate*(generation-generation_shift));
	return (int)round(carrying_capacity*initial_population_size*term/(carrying_capacity + initial_population_size*(term-1)));
}
/* ----- end exponential growth model ----- */

/* ----- population specific demography model ----- */
template <typename Functor_p, typename Functor_p_pop>
population_specific_demography<Functor_p,Functor_p_pop>::population_specific_demography() : pop(0), generation_shift(0) { p = Functor_p(); p_pop = Functor_p_pop(); }
template <typename Functor_p, typename Functor_p_pop>
population_specific_demography<Functor_p,Functor_p_pop>::population_specific_demography(Functor_p p_in, Functor_p_pop p_pop_in, int pop, int generation_shift /*= 0*/) : pop(pop), generation_shift(generation_shift){  p = p_in; p_pop = p_pop_in; }
template <typename Functor_p, typename Functor_p_pop>
__host__ __device__  __forceinline__ int population_specific_demography<Functor_p,Functor_p_pop>::operator()(const int population, const int generation) const{
	if(pop == population) return p_pop(population, generation-generation_shift);
	return p(population, generation-generation_shift);
}
/* ----- end population specific demography model ----- */

/* ----- piecewise demography model ----- */
template <typename Functor_p1, typename Functor_p2>
piecewise_demography<Functor_p1, Functor_p2>::piecewise_demography() : inflection_point(0), generation_shift(0) { p1 = Functor_p1(); p2 = Functor_p2(); }
template <typename Functor_p1, typename Functor_p2>
piecewise_demography<Functor_p1, Functor_p2>::piecewise_demography(Functor_p1 p1_in, Functor_p2 p2_in, int inflection_point, int generation_shift /* = 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { p1 = p1_in; p2 = p2_in; }
template <typename Functor_p1, typename Functor_p2>
__host__ __device__  __forceinline__ int piecewise_demography<Functor_p1, Functor_p2>::operator()(const int population, const int generation) const{
	if(generation >= inflection_point+generation_shift){ return p2(population, generation-generation_shift) ; }
	return p1(population, generation-generation_shift);
};
/* ----- end piecewise demography model ----- */

/* ----- end of demography models ----- */

/* ----- migration models ----- */
/* ----- constant equal migration model ----- */
const_equal_migration::const_equal_migration() : m(0), num_pop(0){ }
const_equal_migration::const_equal_migration(int n) : m(0), num_pop(n){ }
const_equal_migration::const_equal_migration(float m, int n) : m(m), num_pop(n){ }
__host__ __device__ __forceinline__ float const_equal_migration::operator()(const int pop_FROM, const int pop_TO, const int generation) const{
		if(pop_FROM == pop_TO){ return 1-(num_pop-1)*m; }
		return (num_pop > 1) * m;
}
/* ----- end constant equal migration model ----- */

/* ----- constant directional migration model ----- */
template <typename Functor_m1>
const_directional_migration<Functor_m1>::const_directional_migration() : m(0), pop1(0), pop2(0) { rest = Functor_m1(); }
template <typename Functor_m1>
const_directional_migration<Functor_m1>::const_directional_migration(float m, int pop1, int pop2, Functor_m1 rest_in) : m(m), pop1(pop1), pop2(pop2) { rest = rest_in; }
template <typename Functor_m1>
__host__ __device__ __forceinline__ float const_directional_migration<Functor_m1>::operator()(const int pop_FROM, const int pop_TO, const int generation) const{
	if(pop_FROM == pop1 && pop_TO == pop2) return m;
	return rest(pop_FROM, pop_TO, generation);
}
/* ----- end constant directional migration model ----- */

/* ----- piecewise migration model ----- */
template <typename Functor_m1, typename Functor_m2>
piecewise_migration<Functor_m1,Functor_m2>::piecewise_migration() : inflection_point(0), generation_shift(0) { m1 = Functor_m1(); m2 = Functor_m2(); }
template <typename Functor_m1, typename Functor_m2>
piecewise_migration<Functor_m1,Functor_m2>::piecewise_migration(Functor_m1 m1_in, Functor_m2 m2_in, int inflection_point, int generation_shift /*= 0*/) : inflection_point(inflection_point), generation_shift(generation_shift) { m1 = m1_in; m2 = m2_in; }
template <typename Functor_m1, typename Functor_m2>
__host__ __device__ __forceinline__ int piecewise_migration<Functor_m1,Functor_m2>::operator()(const int pop_FROM, const int pop_TO, const int generation) const{
	if(generation >= inflection_point+generation_shift){ return m2(pop_FROM,pop_TO,generation); }
	return m1(pop_FROM,pop_TO,generation);
}
/* ----- end piecewise migration model ----- */
/* ----- end of migration models ----- */

/* ----- preserving & sampling functions ----- */
__host__ __forceinline__ bool do_nothing::operator()(const int generation) const{ return false; }

__host__ __forceinline__ bool do_something::operator()(const int generation) const{ return true; }

template <typename Functor_stable, typename Functor_action>
do_something_else<Functor_stable,Functor_action>::do_something_else() : Fgen(0), generation_shift(0) { f1 = Functor_stable(); f2 = Functor_action(); }
template <typename Functor_stable, typename Functor_action>
do_something_else<Functor_stable,Functor_action>::do_something_else(Functor_stable f1_in, Functor_action f2_in, int Fgen, int generation_shift/*= 0*/) : Fgen(Fgen), generation_shift(generation_shift) { f1 = f1_in; f2 = f2_in; }
template <typename Functor_stable, typename Functor_action>
__host__ __forceinline__ bool do_something_else<Functor_stable,Functor_action>::operator()(const int generation) const{ if(generation-generation_shift == Fgen){ return f2(generation); } return f1(generation); }
/* ----- end of preserving & sampling functions ----- */

}/* ----- end namespace GO_Fish ----- */

#endif /* SIMULATION_FUNCTORS_CUH_ */
