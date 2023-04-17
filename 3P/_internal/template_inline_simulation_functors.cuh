/*
 * template_simulation_functors.cuh
 *
 *      Author: David Lawrie
 *      implementation of template and inline functions for GO Fish evolutionary models and sampling schemes
 */

#ifndef TEMPLATE_INLINE_SIMULATION_FUNCTORS_CUH_
#define TEMPLATE_INLINE_SIMULATION_FUNCTORS_CUH_

/** Namespace contains functions for modeling selection, inbreeding, mutation, dominance, demography, migration. \n
* \n Use of these functions is optional as users can supply their own, for details on how to write your own simulation functions, go to the <a href="modules.html">Modules</a> page, click on the simulation function group which describes the function you wish to write, and read its detailed description.
* \n\n To use Sim_Model functions and objects, include header file: go_fish.cuh
*/
namespace Sim_Model{

namespace details{

struct population_specific{
	static constexpr std::size_t num_params = 1;
	unsigned int population;
	population_specific(unsigned int pop) : population(pop) {}
	template<typename... Args>
	__host__ __device__  bool operator()(const unsigned int gen, const unsigned int pop, Args... rest) const { return population == pop; }
};

struct directional_migration{
	static constexpr std::size_t num_params = 2;
	unsigned int pop_FROM, pop_TO;
	directional_migration(unsigned int pop_from, unsigned int pop_to) : pop_FROM(pop_from), pop_TO(pop_to) {}
	__host__ __device__  bool operator()(const unsigned int gen, const unsigned int pop_from, const unsigned int pop_to) const { return (pop_FROM == pop_from && pop_TO == pop_to); }
};

struct piecewise{
	static constexpr std::size_t num_params = 1;
	unsigned int generation;
	piecewise(unsigned int gen) : generation(gen) {}
	template<typename... Args>
	__host__ __device__  bool operator()(const unsigned int gen, Args... rest) const { return generation <= gen; }
};

struct piecewise_pop_specific{
	static constexpr std::size_t num_params = 2;
	unsigned int generation, population;
	piecewise_pop_specific(unsigned int gen, unsigned int pop) : generation(gen), population(pop) {}
	template<typename... Args>
	__host__ __device__  bool operator()(const unsigned int gen, const unsigned int pop, Args... rest) const { return (population == pop && generation <= gen); }
};

struct piecewise_directional_migration{
	static constexpr std::size_t num_params = 3;
	unsigned int generation, pop_FROM, pop_TO;
	piecewise_directional_migration(unsigned int gen, unsigned int pop_from, unsigned int pop_to) : generation(gen), pop_FROM(pop_from), pop_TO(pop_to) {}
	__host__ __device__  bool operator()(const unsigned int gen, const unsigned int pop_from, const unsigned int pop_to) const { return (pop_FROM == pop_from && pop_TO == pop_to && generation <= gen); }
};

template <typename Pred_function, typename Default_evo_function, typename... Evol_functions>
struct list_of_functions{
	Default_evo_function myDefault;
	std::tuple<std::pair<Evol_functions,Pred_function>...> func_list;

	template <typename... Args>
	__host__ __device__ auto operator()(Args... args) const{ //need to reverse if statement order for piecewise generation or institute intervals
		constexpr std::size_t my_tuple_size = std::tuple_size<decltype(func_list)>::value;
		return operator_helper_overall<my_tuple_size>(std::make_index_sequence<my_tuple_size>{},args...);
	}

	private:
		template <std::size_t N, std::size_t... I, typename... Args>
		__host__ __device__  auto operator_helper_overall(std::index_sequence<I...>, Args... args) const {
			return operator_helper<N-1,I...>(args...);
		}

		template <std::size_t Last, std::size_t First, std::size_t Second, std::size_t... rest, typename... Args>
		__host__ __device__  auto operator_helper(Args... args) const {
			if(std::get<1>(std::get<Last-First>(func_list))(args...)){ return std::get<0>(std::get<Last-First>(func_list))(args...); }
			else if(std::get<1>(std::get<Last-Second>(func_list))(args...)){ return std::get<0>(std::get<Last-Second>(func_list))(args...); }
			return operator_helper<Last,rest...>(args...);
		}

		template <std::size_t Last, std::size_t First, typename... Args>
		__host__ __device__  auto operator_helper(Args... args) const {
			if(std::get<1>(std::get<Last-First>(func_list))(args...)){ return std::get<0>(std::get<Last-First>(func_list))(args...); }
			return myDefault(args...);
		}

		template <std::size_t Last, typename... Args>
		__host__ __device__  auto operator_helper(Args... args) const {
			return myDefault(args...);
		}
};

template<std::size_t Start, typename Pred_function, typename... Args, std::size_t... J>
auto make_pair_helper(const std::tuple<Args...> & helper_tuple, std::index_sequence<J...>){
	return std::make_pair(std::get<Start+sizeof...(J)>(helper_tuple), Pred_function(std::get<Start+J>(helper_tuple)...));
}

template <typename Pred_function, typename Default_evo_function, typename... Args, std::size_t... I>
auto make_struct_helper(Default_evo_function defaultFun, const std::tuple<Args...> & helper_tuple, std::index_sequence<I...>){
	constexpr std::size_t P = Pred_function::num_params+1;
	return list_of_functions<Pred_function, Default_evo_function, std::tuple_element_t<P*I+P-1,std::tuple<Args...>>...>{defaultFun, std::make_tuple(make_pair_helper<P*I,Pred_function>(helper_tuple, std::make_index_sequence<(P-1)>{})...)};
}

template <typename Pred_function, typename Default_evo_function, typename... Args>
auto make_master_helper(Default_evo_function defaultFun, Args... args_in){
	constexpr std::size_t P = Pred_function::num_params+1;
	return make_struct_helper<Pred_function>(defaultFun, std::make_tuple(args_in...), std::make_index_sequence<(sizeof...(Args)/P)>{});
}


} /* ----- end namespace Sim_Model::details ----- */

/* ----- translate effective parameter to coefficient ----- */
/**\param population_size population size in individuals \n \param inbreeding inbreeding coefficient 0 (diploid) to 1 (haploid) */
inline effective_parameter::effective_parameter(float population_size, float inbreeding): N(round(population_size)), F(inbreeding) {}
template <typename Functor_demography, typename Functor_inbreeding>
/**`N = population_size(generation,population)` and `F = inbreeding(generation,population)`
 * \param generation and \param population
 * */
inline effective_parameter::effective_parameter(Functor_demography population_size, Functor_inbreeding inbreeding, unsigned int generation, unsigned int population) : effective_parameter(population_size(generation,population), inbreeding(generation,population)) {}
/** returns `effective_parameter/(2*N/(1+F))` */
inline float effective_parameter::operator()(float effective_parameter) const { return (1.f+F)*effective_parameter/(2.f*N); }
/* ----- end translate effective parameter to coefficient ----- */

/* ----- constant parameter model ----- */
inline constant_parameter::constant_parameter() : p(0) {}
inline constant_parameter::constant_parameter(float p) : p(p){ }
template<typename... Args>
__host__ __device__ __forceinline__ float constant_parameter::operator()(Args...) const { return p; }
/* ----- end constant parameter model ----- */

/* ----- constant effective parameter model ----- */
template <typename Functor_demography, typename Functor_inbreeding>
inline constant_effective_parameter<Functor_demography,Functor_inbreeding>::constant_effective_parameter(float p, float reference_population_size, float reference_inbreeding, Functor_demography population_size, Functor_inbreeding inbreeding) : p(p), reference_Nchrom_e(2*reference_population_size/(1+reference_inbreeding)), N(population_size), F(inbreeding){ }
template <typename Functor_demography, typename Functor_inbreeding>
template <typename... Args>
__host__ __device__ __forceinline__ float constant_effective_parameter<Functor_demography,Functor_inbreeding>::operator()(const unsigned int generation, const unsigned int population, Args...) const {
	return p*reference_Nchrom_e*(1+F(generation, population))/2*N(generation, population);
}
/* ----- end constant effective parameter model ----- */

template <typename Functor_demography, typename Functor_inbreeding>
auto make_constant_effective_parameter(float p, float reference_population_size, float reference_inbreeding, Functor_demography population_size, Functor_inbreeding inbreeding){
	return constant_effective_parameter<Functor_demography,Functor_inbreeding>(p, reference_population_size, reference_inbreeding, population_size, inbreeding);
}

/* ----- linear generation model ----- */
inline linear_generation_parameter::linear_generation_parameter(): slope(0), intercept(0), start_generation(0){}
inline linear_generation_parameter::linear_generation_parameter(float start_parameter, float end_parameter, unsigned int start_generation, unsigned int end_generation): slope((end_parameter-start_parameter)/(end_generation-start_generation)), intercept(start_parameter), start_generation(start_generation){};
template <typename... Args>
__host__ __device__ __forceinline__ float linear_generation_parameter::operator()(const unsigned int generation, Args...) const { return slope*(generation-start_generation)+intercept; }
/* ----- end linear generation model ----- */

/* ----- seasonal parameter model ----- */
inline sine_generation_parameter::sine_generation_parameter() : A(0), pi(0), rho(0), D(0), start_generation(0) {}
/**\param rho (optional input) default `0` \param start_generation (optional input) default `0` */
inline sine_generation_parameter::sine_generation_parameter(float A, float pi, float D, float rho /*= 0*/, unsigned int start_generation /*= 0*/) : A(A), pi(pi), rho(rho), D(D), start_generation(start_generation) {}
/** return `A*sin(pi*(generation-start_generation) + rho) + D` */
template <typename... Args>
__host__ __device__ __forceinline__ float sine_generation_parameter::operator()(const unsigned int generation, Args...) const{ return A*sin(pi*(generation-start_generation) + rho) + D;}
/* ----- end seasonal parameter model ----- */

/* ----- exponential generation model ----- */
inline exponential_generation_parameter::exponential_generation_parameter() : rate(0), initial_parameter_value(0), start_generation(0) {}
/** \param start_generation (optional input) default `0` */
inline exponential_generation_parameter::exponential_generation_parameter(float initial_parameter_value, float rate, unsigned int start_generation /*= 0*/) : initial_parameter_value(initial_parameter_value), rate(rate), start_generation(start_generation) { }
/** `rate = log(final_parameter_value/initial_parameter_value)/(end_generation-start_generation)` */
inline exponential_generation_parameter::exponential_generation_parameter(float initial_parameter_value, float final_parameter_value, unsigned int start_generation, unsigned int end_generation) : initial_parameter_value(initial_parameter_value), start_generation(start_generation) { rate = log(final_parameter_value/initial_parameter_value)/(end_generation-start_generation); }
/** `N = round(initial_population_size*`\f$e^{\textrm{rate*(generation-start_generation)}} \f$`)` */
template <typename... Args>
__host__ __device__  __forceinline__ float exponential_generation_parameter::operator()(const unsigned int generation, Args...) const{ return initial_parameter_value*exp(rate*(generation-start_generation)); }
/* ----- end exponential generation model ----- */

/* ----- logistic generation model ----- */
inline logistic_generation_parameter::logistic_generation_parameter() : rate(0), initial_parameter_value(0), carrying_capacity(0), start_generation(0) {}
/** \param start_generation (optional input) default `0` */
inline logistic_generation_parameter::logistic_generation_parameter(float initial_parameter_value, float carrying_capacity, float rate, unsigned int start_generation /*= 0*/) : initial_parameter_value(initial_parameter_value), carrying_capacity(carrying_capacity), rate(rate), start_generation(start_generation) { }
/** `rate = log(Ai/Af)/(end_generation-start_generation)` where `Ax` := `(carrying_capacity - x)/x` and `i` := `initial_parameter_value` and `f` := final_parameter_value` */
inline logistic_generation_parameter::logistic_generation_parameter(float initial_parameter_value, float carrying_capacity, float final_parameter_value, unsigned int start_generation, unsigned int end_generation) : initial_parameter_value(initial_parameter_value), carrying_capacity(carrying_capacity), start_generation(start_generation) {
	float Ai = (carrying_capacity - initial_parameter_value)/initial_parameter_value;
	float Af = (carrying_capacity - final_parameter_value)/final_parameter_value;
	rate = log(Ai/Af)/(end_generation-start_generation);
}
/** `exp_term = `\f$e^{\textrm{rate*(generation-start_generation)}} \f$ \n
 * `N = round((carrying_capacity*initial_population_size*exp_term)`\f$\div\f$`(carrying_capacity + initial_population_size*(exp_term-1)))` */
template <typename... Args>
__host__ __device__  __forceinline__ float logistic_generation_parameter::operator()(const unsigned int generation, Args...) const{
	float term = exp(rate*(generation-start_generation));
	return carrying_capacity*initial_parameter_value*term/(carrying_capacity + initial_parameter_value*(term-1));
}
/* ----- end logistic generation model ----- */

/* ----- linear frequency dependent dominance and selection model ----- */
/**\struct linear_frequency_h_s
 *  when modeling selection:
 * `(slope < 0)` := balancing selection model (negative frequency-dependent selection) \n
 * `(slope = 0)` := constant selection \n
 * `(slope > 0)` := reinforcement selection model (positive frequency-dependent selection) \n
 * */
inline linear_frequency_h_s::linear_frequency_h_s() : slope(0), intercept(0) {}
/** \param start_parameter := parameter at frequency 0
 *  \param start_parameter := parameter at frequency 1
 *   `slope = end_parameter - start_parameter`\n
 *   `intercept = start_parameter`
 */
inline linear_frequency_h_s::linear_frequency_h_s(float start_parameter, float end_parameter_slope, bool is_slope): intercept(start_parameter), slope(is_slope ? end_parameter_slope : end_parameter_slope - start_parameter) { }
/** return `slope*freq + intercept` */
__host__ __device__ __forceinline__ float linear_frequency_h_s::operator()(const unsigned int population, const unsigned int generation, const float freq) const { return slope*freq+intercept; }
/* ----- end linear frequency dependent dominance and selection model ----- */

linear_frequency_h_s make_robertson_stabilizing_selection_model(float effect_size, float variance){
	float e_s = effect_size*effect_size/(2*variance);
	return linear_frequency_h_s(-1*e_s,2*e_s,true);
}

/* ----- hyperbola frequency dependent dominance and selection model ----- */
inline hyperbola_frequency_h_s::hyperbola_frequency_h_s() : A(0), B(-1), C(0.5) {}
inline hyperbola_frequency_h_s::hyperbola_frequency_h_s(float A, float B, float C): A(A), B(B), C(C) {}
inline hyperbola_frequency_h_s::hyperbola_frequency_h_s(linear_frequency_h_s numerator, linear_frequency_h_s denominator){
	A = numerator.intercept/denominator.slope - denominator.intercept*numerator.slope/pow(denominator.slope,2.f);
	B = -1.f*denominator.intercept/denominator.slope;
	C = numerator.slope/denominator.slope;
}
__host__ __device__ __forceinline__ float hyperbola_frequency_h_s::operator()(const unsigned int generation, const unsigned int population, const float freq) const {
	if(freq == B) return 0.f; //doesn't matter here, normally used for dominance => when freq = B, selection = 0 anyway
	return A/(freq-B) + C;
}
/* ----- end hyperbola frequency dependent dominance and selection model ----- */

hyperbola_frequency_h_s make_robertson_stabilizing_dominance_model(){ return hyperbola_frequency_h_s(1.f/8.f, 1.f/2.f, 1.f/2.f); }

template <typename Default_fun, typename... List>
auto make_population_specific_evolution_model(Default_fun defaultFun, List... list){ return details::make_master_helper<details::population_specific>(defaultFun, list...); }

template <typename Default_fun, typename... List>
auto make_directional_migration_model(Default_fun defaultFun, List... list){ return details::make_master_helper<details::directional_migration>(defaultFun, list...); }

template <typename Start_fun, typename... List>
auto make_piecewise_evolution_model(Start_fun defaultFun, List... list){ return details::make_master_helper<details::piecewise>(defaultFun, list...); }

template <typename Default_Start_fun, typename... List>
auto make_piecewise_population_specific_model(Default_Start_fun defaultFun, List... list){ return details::make_master_helper<details::piecewise_pop_specific>(defaultFun, list...); }

template <typename Default_Start_fun, typename... List>
auto make_piecewise_directional_migration_model(Default_Start_fun defaultFun, List... list){ return details::make_master_helper<details::piecewise_directional_migration>(defaultFun, list...); }

standard_mse_integrand::standard_mse_integrand(): N(0), s(0), h(0), F(0) {}
template <typename Functor_demography, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance>
standard_mse_integrand::standard_mse_integrand(const Functor_demography dem, const Functor_selection sel_coeff, const Functor_inbreeding f_inbred, const Functor_dominance dominance, unsigned int gen, unsigned int pop): N(round(dem(gen,pop))), F(f_inbred(gen,pop)), s(max(sel_coeff(gen,pop,0.5),-1.f)), h(dominance(gen,pop,0.5)) { } //this model doesn't allow frequency dependent selection
__host__ __device__ double standard_mse_integrand::operator()(double i) const{ return exp(-2*N*s*i*((2*h+(1-2*h)*i)*(1-F)+2*F)/(1+F)); } //exponent term in integrand is negative inverse, //works for either haploid or diploid, N should be number of individuals, for haploid, F = 1
__host__ __device__ bool standard_mse_integrand::neutral() const{ return s==0; }

robertson_stabilizing_mse_integrand::robertson_stabilizing_mse_integrand():constant(0) {}
template <typename Functor_demography, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance>
robertson_stabilizing_mse_integrand::robertson_stabilizing_mse_integrand(const Functor_demography dem, const Functor_selection sel_coeff, const Functor_inbreeding f_inbred, const Functor_dominance dominance, unsigned int gen, unsigned int pop) {
	float N(round(dem(gen,pop))), F(f_inbred(gen,pop)), e_s((max(sel_coeff(gen,pop,1),-1.f) - max(sel_coeff(gen,pop,0),-1.f))/2.f);

	constant = -4*N*e_s*((1-F)/4+F)/(1+F);
}
__host__ __device__ double robertson_stabilizing_mse_integrand::operator()(double i) const{ return exp(constant*(i*i-i)); } //exponent term in integrand is negative inverse, //works for either haploid or diploid, N should be number of individuals, for haploid, F = 1
__host__ __device__ bool robertson_stabilizing_mse_integrand::neutral() const{ return constant==0; }

/** \addtogroup selection
*  \brief Functions that model selection coefficients (\p s) across populations and over time
*
*  Selection coefficients are defined by:
*
*  AA | Aa | aa
*  -- | -- | --
*  `1` | `1+hs` | `1+s`
*
*  where \p h is the dominance coefficient and AA, Aa, aa represent the various alleles.
*  Thus Effective Selection, \p gamma, is defined by `N_chromosome_e*s` which for outbreeding diploids is `2*N*s` and haploid is `N*s`
*  where N is the number of individuals (as returned by demography functions).
*  Diploids with inbreeding, `F`, will have an effective strength of selection, \p gamma, of `2*N*s/(1+F)`. See \ref demography for more about effective population size in the simulation.
*  Side note: if `gamma` is set to some float, `S*(1+F)`, `h=0.5` (co-dominant), and the population size is similarly scaled (i.e. `N*(1+F)`), then the effective selection in the simulation will be invariant with respect to inbreeding.
*
*  Minimum selection is `s >= -1` (lethal). Currently the program will not throw an error if the selection is less than -1, but will simply take the `max(s,-1)`.
*
*  ###Writing your own Selection functions###
*  These can be functions or functors (or soon, with C++11 support, lambdas). However, the selection function must be of the form:\n
*  \code __device__ float your_function(int population, unsigned int generation, float freq){ ... return selection_coeff; } \endcode
*  This returns the selection coefficient in population \p population at generation \p generation for a mutation at frequency \p freq.
*  The `__device__` flag is to ensure the nvcc compiler knows the function must be compiled for the device (GPU).
*  Because of this flag, the function must be defined in CUDA source file (*.cu) or declared/defined header file (*.h, *.hpp, *.cuh, etc...) which is included in a CUDA source file.
*  Since this code will be compiled on the GPU, do not use dynamically allocated arrays in your function (e.g. `float * f = new float[5]`) unless you know CUDA.
*  And even then avoid them as they will slow the code down (parameters have to be pulled from the GPU's global memory (vRAM), which is slow).
*  Statically allocated arrays (e.g. `float f[5]`) are fine.
*/

/* ----- population specific selection model ----- */
/** \struct selection_population_specific
 * Takes in two template types: the function to be returned for the rest of the populations and the function for the specific population, `pop`. \n
 * Population specific selection functors can be nested within each other and with piecewise selection functors for multiple populations and multiple time functions, e.g.:\n\n
 * 3 populations with different selection coefficients where mutations in the 1st are neutral, in the 2nd are deleterious, and in the 3rd are beneficial up to generation 300 when selection relaxes and they become neutral
 * \code typedef Sim_Model::selection_constant constant;
	typedef Sim_Model::selection_piecewise<constant,constant> piecewise_constant;
	typedef Sim_Model::selection_population_specific<constant,constant> population_specific_constant;
	constant neutral;
	constant purifying(-0.01);
	constant positive(0.01);
	population_specific_constant first_second(neutral,purifying,1);
	piecewise_constant third(positive,neutral,300);
	Sim_Model::selection_population_specific<population_specific_constant,piecewise_constant> selection_model(first_second,third,2); \endcode
	The modularity of these functor templates allow selection models to be extended to any number of populations and piecewise selection functions (including user defined functions).
 * */
/* ----- end population specific selection model ----- */

/* ----- piecewise selection model ----- */
/** \struct selection_piecewise
 *  Takes in two template types: the function to be returned before the `inflection_point` and the function for after the `inflection_point`. \n
 * Piecewise selection functors can be nested within each other and with population specific selection functors for multiple populations and multiple time functions, e.g.:\n\n
 * 3 populations with different selection coefficients where mutations in the 1st are neutral, in the 2nd are deleterious, and in the 3rd are beneficial up to generation 300 when selection relaxes and they become neutral
 * \code typedef Sim_Model::selection_constant constant;
	typedef Sim_Model::selection_piecewise<constant,constant> piecewise_constant;
	typedef Sim_Model::selection_population_specific<constant,constant> population_specific_constant;
	constant neutral;
	constant purifying(-0.01);
	constant positive(0.01);
	population_specific_constant first_second(neutral,purifying,1);
	piecewise_constant third(positive,neutral,300);
	Sim_Model::selection_population_specific<population_specific_constant,piecewise_constant> selection_model(first_second,third,2);\endcode
	The modularity of these functor templates allow selection models to be extended to any number of populations and piecewise selection functions (including user defined functions).
 */
/* ----- end piecewise selection model ----- */

/** \addtogroup in_mut_dom
*  \brief Functions that model inbreeding coefficients (\p F), per-site mutation rates (\p mu), and dominance coefficients (\p h) across populations and over time
*
*  Inbreeding coefficients, \p F, must be between [0,1]:
*  > `(F = 0)` := diploid, outbred; `(F = 1)` := haploid, inbred
*  Per-site mutation rate, \p mu, must be `> 0`. \n\n Dominance coefficients, \p h, can be any real number:
*  > `(h < 0)` := underdominance; `(h > 1)` := overdominance; `(h = 0)` := recessive; `(h = 1)` := dominant; `(0 < h < 1)` := co-dominant
*  \n
*  ###Writing your own Inbreeding, Mutation, and Dominance functions###
*  These can be functions or functors (or soon, with C++11 support, lambdas). However, the parameter function must be of the form:\n
*  \code float your_function(int population, unsigned int generation){ ... return parameter; } \endcode
*  This returns the parameter in population \p population at generation \p generation.
*  Adding a `__host__` flag is optional, but if done, the function must be defined in CUDA source file (*.cu) or declared/defined header file (*.h, *.hpp, *.cuh, etc...) which is included in a CUDA source file.
*  If no flag added, function can be defined in a regular C/C++ source file (e.g. *.c, *.cpp).
*  Note: run_sim is required to be in a CUDA source file to be compiled.
*/

/* ----- population specific parameter model ----- */
/** \struct F_mu_h_population_specific
 * Takes in two template types: the function to be returned for the rest of the populations and the function for the specific population, `pop`. \n
 * Population specific parameter functors can be nested within each other and with piecewise parameter functors for multiple populations and multiple time functions, e.g.:\n\n
 * 3 populations with different Inbreeding coefficients where populations in the 1st are outbred, in the 2nd are partially inbred, and in the 3rd are completely inbred until population becomes completely outcrossing at generation 300
 * \code typedef Sim_Model::F_mu_h_constant F_constant;
	typedef Sim_Model::F_mu_h_piecewise<F_constant,F_constant> F_piecewise_constant;
	typedef Sim_Model::F_mu_h_population_specific<F_constant,F_constant> F_population_specific_constant;
	F_constant outbred;
	F_constant inbred(1);
	F_constant mixed(0.5);
	F_population_specific_constant first_second(outbred,mixed,1);
	F_piecewise_constant third(inbred,outbred,300);
	Sim_Model::F_mu_h_population_specific<F_population_specific_constant,F_piecewise_constant> inbreeding_model(first_second,third,2); \endcode
	The modularity of these functor templates allow parameter models to be extended to any number of populations and piecewise parameter functions (including user defined functions).
 * */
/* ----- end population specific parameter model ----- */

/* ----- piecewise parameter model ----- */
/** \struct F_mu_h_piecewise
 * Takes in two template types: the function to be returned before the `inflection_point` and the function for after the `inflection_point`. \n
 * Piecewise parameter functors can be nested within each other and with population specific parameter functors for multiple populations and multiple time functions, e.g.:\n\n
 * 3 populations with different Inbreeding coefficients where populations in the 1st are outbred, in the 2nd are partially inbred, and in the 3rd are completely inbred until population becomes completely outcrossing at generation 300
 * \code typedef Sim_Model::F_mu_h_constant F_constant;
	typedef Sim_Model::F_mu_h_piecewise<F_constant,F_constant> F_piecewise_constant;
	typedef Sim_Model::F_mu_h_population_specific<F_constant,F_constant> F_population_specific_constant;
	F_constant outbred;
	F_constant inbred(1);
	F_constant mixed(0.5);
	F_population_specific_constant first_second(outbred,mixed,1);
	F_piecewise_constant third(inbred,outbred,300);
	Sim_Model::F_mu_h_population_specific<F_population_specific_constant,F_piecewise_constant> inbreeding_model(first_second,third,2); \endcode
	The modularity of these functor templates allow parameter models to be extended to any number of populations and piecewise parameter functions (including user defined functions).
 * */

/** \addtogroup demography
*  \brief Functions that model population size in number of individuals across populations and over time
*
*  If a population with a previously positive number of individuals hits 0 individuals over the course of a simulation, it will be declared extinct.
*  If the population size of an extinct population becomes non-zero after the extinction, an error will be thrown.
*  Populations that start the simulation at 0 individuals are not considered extinct. As population size is stored as an integer currently, the max population size is ~2x10<sup>9</sup> individuals. \n\n
*  The effective number of chromosomes, `N_chromosome_e` - which is the effective population size in the simulation - is defined as `2*N/(1+F)`. Thus to equate two populations with differing inbreeding coefficients,
*  multiply the number of individuals, `N`, in each by the inbreeding coefficient in each, `1+F`. Side note: if population size is set to `N*(1+F)` in the simulation, the effective population size will be invariant with respect to inbreeding.
*
*  ###Writing your own Demography functions###
*  These can be functions or functors (or soon, with C++11 support, lambdas). However, the demographic function must be of the form:\n
*  \code __host__ __device__ int your_function(int population, unsigned int generation){ ... return number_of_individuals; } \endcode
*  This returns the number of individuals in population \p population at generation \p generation.
*  The `__host__` and `__device__` flags are to ensure the nvcc compiler knows the function must be compiled for both the host (CPU) and device (GPU).
*  Because of these flags, the function must be defined in CUDA source file (*.cu) or declared/defined header file (*.h, *.hpp, *.cuh, etc...) which is included in a CUDA source file.
*  Since this code will be compiled on the GPU, do not use dynamically allocated arrays in your function (e.g. `int * f = new int[5]`) unless you know CUDA.
*  And even then avoid them as they will slow the code down (parameters have to be pulled from the GPU's global memory (vRAM), which is slow).
*  Statically allocated arrays (e.g. `int f[5]`) are fine.
*/

/* ----- population specific demography model ----- */
/** \struct demography_population_specific
 * Takes in two template types: the function to be returned for the rest of the populations and the function for the specific population, `pop`. \n
 * Population specific demographic functors can be nested within each other and with piecewise demographic functors for multiple populations and multiple time functions, e.g.:\n\n
 * Using both demographic and migration functors, population 0 splits in two, forming population 1 in the first generation. Population 1's size increases exponentially afterwards with no further migration between the groups
 * \code
   typedef Sim_Model::demography_constant dem_constant;
   typedef Sim_Model::demography_exponential_growth dem_exponential;
   typedef Sim_Model::demography_population_specific<dem_constant,dem_constant> dem_pop_constant_constant;
   typedef Sim_Model::demography_population_specific<dem_constant,dem_exponential> dem_pop_constant_exponential;

   dem_constant d_pop0(100000), d_pop1(0);
   dem_pop_constant_constant d_generation_0(d_pop0,d_pop1,1); //at the start of the simulation, the first population pop0 starts out at 100,000 individuals, pop1 doesn't exist yet
   dem_constant d_pop0_1(90000); dem_exponential d_pop1_1(0.01, 10000, 1); //shifts exponential back one generation so it starts at 10,000
   dem_pop_constant_exponential d_remaining_generations(d_pop0_1,d_pop1_1,1); //in the first generation, 10,000 individuals from pop0 move to start pop1, which grows exponentially afterwards at a rate of 1%
   Sim_Model::demography_piecewise<dem_pop_constant_constant,dem_pop_constant_exponential> demography_model(d_generation_0,d_remaining_generations,1);

   typedef Sim_Model::migration_constant_equal mig_const_equal;
   typedef Sim_Model::migration_constant_directional<mig_const_equal> mig_const_equal_const_dir;
   typedef Sim_Model::migration_constant_directional<mig_const_equal_const_dir> mig_const_equal_const_dir_const_dir;
   typedef Sim_Model::migration_piecewise<mig_const_equal,mig_const_equal_const_dir_const_dir> split_pop0_gen1;
   mig_const_equal m0; //no migration
   mig_const_equal_const_dir m_pop0_pop1(1.f,0,1,m0); //pop1 made up entirely of individuals from pop0, no other population contributing to pop0
   mig_const_equal_const_dir_const_dir m_pop1_pop1(0.f,1,1,m_pop0_pop1); //pop1 made up entirely of individuals from pop0 (since pop1 previously did not exist, no migration from previous pop1 generation!)
   split_pop0_gen1 m_generation_1(m0,m_pop1_pop1,1); //no migration in generation 0, splits pop1 off from pop0 in generation 1
   Sim_Model::migration_piecewise<split_pop0_gen1,mig_const_equal> migration_model(m_generation_1,m0,2); //no further migration between groups \endcode
   The modularity of these functor templates allow parameter models to be extended to any number of populations and piecewise parameter functions (including user defined functions).
 **/
/* ----- end population specific demography model ----- */

/* ----- piecewise demography model ----- */
/** \struct demography_piecewise
 * Takes in two template types: the function to be returned before the `inflection_point` and the function for after the `inflection_point`. \n
 * Piecewise demographic functors can be nested within each other and with population specific demographic functors for multiple populations and multiple time functions, e.g.:\n\n
 * Using both demographic and migration functors, population 0 splits in two, forming population 1 in the first generation. Population 1's size increases exponentially afterwards with no further migration between the groups
 * \code
   typedef Sim_Model::demography_constant dem_constant;
   typedef Sim_Model::demography_exponential_growth dem_exponential;
   typedef Sim_Model::demography_population_specific<dem_constant,dem_constant> dem_pop_constant_constant;
   typedef Sim_Model::demography_population_specific<dem_constant,dem_exponential> dem_pop_constant_exponential;

   dem_constant d_pop0(100000), d_pop1(0);
   dem_pop_constant_constant d_generation_0(d_pop0,d_pop1,1); //at the start of the simulation, the first population pop0 starts out at 100,000 individuals, pop1 doesn't exist yet
   dem_constant d_pop0_1(90000); dem_exponential d_pop1_1(0.01, 10000, 1); //shifts exponential back one generation so it starts at 10,000
   dem_pop_constant_exponential d_remaining_generations(d_pop0_1,d_pop1_1,1); //in the first generation, 10,000 individuals from pop0 move to start pop1, which grows exponentially afterwards at a rate of 1%
   Sim_Model::demography_piecewise<dem_pop_constant_constant,dem_pop_constant_exponential> demography_model(d_generation_0,d_remaining_generations,1);

   typedef Sim_Model::migration_constant_equal mig_const_equal;
   typedef Sim_Model::migration_constant_directional<mig_const_equal> mig_const_equal_const_dir;
   typedef Sim_Model::migration_constant_directional<mig_const_equal_const_dir> mig_const_equal_const_dir_const_dir;
   typedef Sim_Model::migration_piecewise<mig_const_equal,mig_const_equal_const_dir_const_dir> split_pop0_gen1;
   mig_const_equal m0; //no migration
   mig_const_equal_const_dir m_pop0_pop1(1.f,0,1,m0); //pop1 made up entirely of individuals from pop0, no other population contributing to pop0
   mig_const_equal_const_dir_const_dir m_pop1_pop1(0.f,1,1,m_pop0_pop1); //pop1 made up entirely of individuals from pop0 (since pop1 previously did not exist, no migration from previous pop1 generation!)
   split_pop0_gen1 m_generation_1(m0,m_pop1_pop1,1); //no migration in generation 0, splits pop1 off from pop0 in generation 1
   Sim_Model::migration_piecewise<split_pop0_gen1,mig_const_equal> migration_model(m_generation_1,m0,2); //no further migration between groups \endcode
   The modularity of these functor templates allow parameter models to be extended to any number of populations and piecewise parameter functions (including user defined functions).
 **/
/* ----- end piecewise demography model ----- */

/** \addtogroup migration
*  \brief Functions that model migration rates over time (conservative model of migration)
*
*  In the conservative model of migration, migration rate from population i to population j is expressed as the fraction of population `j` originally from `i`:\n
*  > e.g. in a 2 population model, a migration rate of \p m<sub>ij</sub> `= 0.1` ==> 10% of population `j` is originally from population `i` \n
*  > and the frequency, \f$x_{mig,j}\f$, in the next generation of an allele is \f$x_{mig,j} = 0.1*x_i + 0.9*x_j\f$
*  Thus, in general, the following must be true: \f$\sum_{i=1}^n\f$ \p m<sub>ij</sub> `= 1` (program will throw error elsewise).
*  However the sum of migration rates FROM a population need not sum to anything in particular.
*  This is also the set of functions that are used to specify a single population splitting into (two or more) populations. \n \n
*
*  ###Writing your own Migration functions###
*  These can be functions or functors (or soon, with C++11 support, lambdas). However, the migration function must be of the form:\n
*  \code __host__ __device__ float your_function(int population_FROM, int population_TO, unsigned int generation){ ... return migration_rate; } \endcode
*  This returns the rate of migration from population \p population_FROM to population \p population_TO at generation \p generation.
*  The `__host__` and `__device__` flags are to ensure the nvcc compiler knows the function must be compiled for both the host (CPU) and device (GPU).
*  Because of these flags, the function must be defined in CUDA source file (*.cu) or declared/defined header file (*.h, *.hpp, *.cuh, etc...) which is included in a CUDA source file.
*  Since this code will be compiled on the GPU, do not use dynamically allocated arrays in your function (e.g. `float * f = new float[5]`) unless you know CUDA.
*  And even then avoid them as they will slow the code down (parameters have to be pulled from the GPU's global memory (vRAM), which is slow).
*  Statically allocated arrays (e.g. `float f[5]`) are fine.
*/


/* ----- constant directional migration model ----- */
/** \struct migration_constant_directional
 * Takes in one template type: the migration function to be returned for all other migration directions than from pop1 to pop2. \n
 * Constant directional migration functors can be nested within each other and with piecewise migration functors for multiple migration directions and multiple time functions, e.g.:\n\n
 * Using both demographic and migration functors, population 0 splits in two, forming population 1 in the first generation. Population 1's size increases exponentially afterwards with no further migration between the groups
 * \code
   typedef Sim_Model::demography_constant dem_constant;
   typedef Sim_Model::demography_exponential_growth dem_exponential;
   typedef Sim_Model::demography_population_specific<dem_constant,dem_constant> dem_pop_constant_constant;
   typedef Sim_Model::demography_population_specific<dem_constant,dem_exponential> dem_pop_constant_exponential;

   dem_constant d_pop0(100000), d_pop1(0);
   dem_pop_constant_constant d_generation_0(d_pop0,d_pop1,1); //at the start of the simulation, the first population pop0 starts out at 100,000 individuals, pop1 doesn't exist yet
   dem_constant d_pop0_1(90000); dem_exponential d_pop1_1(0.01, 10000, 1); //shifts exponential back one generation so it starts at 10,000
   dem_pop_constant_exponential d_remaining_generations(d_pop0_1,d_pop1_1,1); //in the first generation, 10,000 individuals from pop0 move to start pop1, which grows exponentially afterwards at a rate of 1%
   Sim_Model::demography_piecewise<dem_pop_constant_constant,dem_pop_constant_exponential> demography_model(d_generation_0,d_remaining_generations,1);

   typedef Sim_Model::migration_constant_equal mig_const_equal;
   typedef Sim_Model::migration_constant_directional<mig_const_equal> mig_const_equal_const_dir;
   typedef Sim_Model::migration_constant_directional<mig_const_equal_const_dir> mig_const_equal_const_dir_const_dir;
   typedef Sim_Model::migration_piecewise<mig_const_equal,mig_const_equal_const_dir_const_dir> split_pop0_gen1;
   mig_const_equal m0; //no migration
   mig_const_equal_const_dir m_pop0_pop1(1.f,0,1,m0); //pop1 made up entirely of individuals from pop0, no other population contributing to pop0
   mig_const_equal_const_dir_const_dir m_pop1_pop1(0.f,1,1,m_pop0_pop1); //pop1 made up entirely of individuals from pop0 (since pop1 previously did not exist, no migration from previous pop1 generation!)
   split_pop0_gen1 m_generation_1(m0,m_pop1_pop1,1); //no migration in generation 0, splits pop1 off from pop0 in generation 1
   Sim_Model::migration_piecewise<split_pop0_gen1,mig_const_equal> migration_model(m_generation_1,m0,2); //no further migration between groups \endcode
   The modularity of these functor templates allow parameter models to be extended to any number of populations and piecewise parameter functions (including user defined functions).
 **/
/* ----- end constant directional migration model ----- */

/* ----- piecewise migration model ----- */
/** \struct migration_piecewise
 * Takes in two template types: the function to be returned before the `inflection_point` and the function for after the `inflection_point`. \n
 * Piecewise migration functors can be nested within each other and with constant directional migration functors for multiple migration directions and multiple time functions, e.g.:\n\n
 * Using both demographic and migration functors, population 0 splits in two, forming population 1 in the first generation. Population 1's size increases exponentially afterwards with no further migration between the groups
 * \code
   typedef Sim_Model::demography_constant dem_constant;
   typedef Sim_Model::demography_exponential_growth dem_exponential;
   typedef Sim_Model::demography_population_specific<dem_constant,dem_constant> dem_pop_constant_constant;
   typedef Sim_Model::demography_population_specific<dem_constant,dem_exponential> dem_pop_constant_exponential;

   dem_constant d_pop0(100000), d_pop1(0);
   dem_pop_constant_constant d_generation_0(d_pop0,d_pop1,1); //at the start of the simulation, the first population pop0 starts out at 100,000 individuals, pop1 doesn't exist yet
   dem_constant d_pop0_1(90000); dem_exponential d_pop1_1(0.01, 10000, 1); //shifts exponential back one generation so it starts at 10,000
   dem_pop_constant_exponential d_remaining_generations(d_pop0_1,d_pop1_1,1); //in the first generation, 10,000 individuals from pop0 move to start pop1, which grows exponentially afterwards at a rate of 1%
   Sim_Model::demography_piecewise<dem_pop_constant_constant,dem_pop_constant_exponential> demography_model(d_generation_0,d_remaining_generations,1);

   typedef Sim_Model::migration_constant_equal mig_const_equal;
   typedef Sim_Model::migration_constant_directional<mig_const_equal> mig_const_equal_const_dir;
   typedef Sim_Model::migration_constant_directional<mig_const_equal_const_dir> mig_const_equal_const_dir_const_dir;
   typedef Sim_Model::migration_piecewise<mig_const_equal,mig_const_equal_const_dir_const_dir> split_pop0_gen1;
   mig_const_equal m0; //no migration
   mig_const_equal_const_dir m_pop0_pop1(1.f,0,1,m0); //pop1 made up entirely of individuals from pop0, no other population contributing to pop0
   mig_const_equal_const_dir_const_dir m_pop1_pop1(0.f,1,1,m_pop0_pop1); //pop1 made up entirely of individuals from pop0 (since pop1 previously did not exist, no migration from previous pop1 generation!)
   split_pop0_gen1 m_generation_1(m0,m_pop1_pop1,1); //no migration in generation 0, splits pop1 off from pop0 in generation 1
   Sim_Model::migration_piecewise<split_pop0_gen1,mig_const_equal> migration_model(m_generation_1,m0,2); //no further migration between groups \endcode
   The modularity of these functor templates allow parameter models to be extended to any number of populations and piecewise parameter functions (including user defined functions).
 **/
/* ----- end piecewise migration model ----- */
}/* ----- end namespace Sim_Model ----- */

/** Namespace contains functions to sample time points in the simulation.
* \n Use of these functions is optional as users can supply their own, for details on how to write your own simulation functions, go to the <a href="modules.html">Modules</a> page, click on the sample function group, and read its detailed description.
* \n\n To use Sampling functions and objects, include header file: go_fish.cuh
 */
namespace Sampling{
/** \addtogroup pres_samp
*  \brief Functions that control at when to flag currently segregating mutations as preserved and when to sample the mutation frequencies
*
*  When the preserve function returns true for a given generation, the simulation will compact out any lost or fixed mutations that are not already flagged preserved - provided that `GO_Fish::allele_trajectories::sim_constants::compact_interval > 0`.
*  (Though if `compact_interval == 0`, then there is no reason to employ the preserve function.)
*  The remaining mutations will then be flagged as preserved so that compact will not remove them from the simulation.\n\n
*
*  Sampling a simulation writes out the mutation frequencies from the GPU's memory vRAM to the CPU's RAM for storage and later analysis. Sampling a simulation at a time point other than the final generation (the final generation is always sampled) will also trigger the preserve flag (and thus compact if `compact_interval > 0`).
*  Thus, those mutations sampled at generation \p t will be preserved in all samples after generation \p t in order to ease frequency comparisons and track allele frequencies.
*
*  \n
*  ###Writing your own Preserving and Sampling Functions###
*  These can be functions or functors (or soon, with C++11 support, lambdas). However, the preserving/sampling function must be of the form:
*
*  \code bool your_function(unsigned int generation){ ... return true/false; } \endcode
*  This returns true or false (preserve or not preserve, sample or do not sample) at generation \p generation.
*  Adding a `__host__` flag is optional, but if done, the function must be defined in CUDA source file (*.cu) or declared/defined header file (*.h, *.hpp, *.cuh, etc...) which is included in a CUDA source file.
*  If no flag added, function can be defined in a regular C/C++ source file (e.g. *.c, *.cpp).
*  Note: run_sim is required to be in a CUDA source file to be compiled.
*/

/* ----- preserving & sampling functions ----- */
/* ----- off ----- */
__host__ __forceinline__ bool off::operator()(const unsigned int generation) const{ return false; }
/* ----- end off ----- */

/* ----- on ----- */
__host__ __forceinline__ bool on::operator()(const unsigned int generation) const{ return true; }
/* ----- end on ----- */

/* ----- bool pulse ----- */
/** \struct pulse
* Takes in a vector of generations that determines when to flip the return from `default_state` to `!default_state` - turning sampling on or off for those specific generations. \n
* e.g.:\n\n
* Sampling strategy that takes time samples of generations 0, 65, & 110 (& final generation is always sampled).
*
* \code
* Sampling::pulse sample_gen_0_65_110({0,65,110}); //sampling strategy, will take time samples of generations 0, 65, 110 (& final generation is always sampled)
* \endcode
* Note mutations present in these generations will be preserved until the final generation of the simulation.
 **/

/**
 * \param pulses, vector of generations that determines when to flip the return from `default_state` to `!default_state` - turning sampling on or off for those specific generations. \n
 * \param start_generation, shifts generation in operator by start_generation (generation-start_generation), useful for starting a new simulation from the results of a previous allele_trajectory as it shifts all the values in pulses by a set amount
 * \param default_state, default boolean (false := sample off, true := sample on)
 *  */
inline pulse::pulse(std::vector<unsigned int> pulses, unsigned int start_generation /*= 0*/, bool default_state /*= false*/) : pulses(pulses), start_generation(start_generation), default_state(default_state) {}
__host__ __forceinline__ bool pulse::operator()(const unsigned int generation) const{
	auto shift_gen = generation - start_generation;
	for(auto gen : pulses){
		if(gen == shift_gen){ return !default_state; }
		if(gen > shift_gen){ return default_state; }
	}
	return default_state;
}
/* ----- end pulse ----- */

/* ----- intervals ----- */
/** \struct intervals
 * Takes in a vector of generations that determines when to flip the `current_state` to `!current_state` - turning sampling on or off for those and subsequent generations. \n
 * e.g.:\n\n
 * Sampling strategy that takes time samples of generation 0 & all generations [65,110] inclusive (& final generation is always sampled).
 *
 * \code
 Sampling::intervals sample_gen_0_65_110({1,65,111},0,true); //sampling strategy, will take time samples of generation 0 & generations [65,110] inclusive (& final generation is always sampled)
 * \endcode
 * Sampling starts on at generation 0, turns off at generation 1 through generation 64, turns back on generation 65, and off again after generation 111 (& final generation is always sampled). \n
 * Note mutations present in these generations will be preserved until the final generation of the simulation.
 **/
/**
 * \param change_points, vector of generations that determines when to flip `current_state` to `!current_state` - turning sampling on or off for those and subsequent generations. \n
 * \param start_generation, shifts generation in operator by start_generation (generation-start_generation), useful for starting a new simulation from the results of a previous allele_trajectory as it shifts all the values in pulses by a set amount
 * \param start_state, whether sampling starts in the on or off state (false := sample off, true := sample on)
 *  */
inline intervals::intervals(std::vector<unsigned int> change_points, unsigned int start_generation /*= 0*/, bool start_state /*= false*/) : change_points(change_points), start_generation(start_generation), start_state(start_state) {}
__host__ __forceinline__ bool intervals::operator()(const unsigned int generation) const{
	bool current_state = start_state;
	auto shift_gen = generation - start_generation;
	for(auto gen : change_points){
		if(gen <= shift_gen){
			current_state = !current_state;
			if(gen == shift_gen){ return current_state; }
		}
		if(gen > shift_gen){ return current_state; }
	}
	return current_state;
}
/* ----- end intervals ----- */
} /* ----- end namespace Sampling ----- */



#endif /* TEMPLATE_INLINE_SIMULATION_FUNCTORS_CUH_ */
