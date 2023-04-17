/*!\file
* \brief GO Fish Standard Simulation Functions (contains namespaces Sim_Model and Sampling)
*
*/
/*
 * 		sim_model.cuh
 *
 *      Author: David Lawrie
 *      GO Fish Standard Simulation Functions
 */

#include <vector>

///Namespace of functions for controlling GO_Fish simulations
namespace Sim_Model{

/** \defgroup selection Simulation Models: Selection Group*//**@{*/
/* ----- end selection models ----- *//** @} */

///functor: translates effective parameters to coefficients (i.e. gamma to s) for simulation construction
struct effective_parameter
{
	float N; /**<\brief population size in individuals */ /**<\t*/
	float F; /**<\brief inbreeding coefficient */ /**<\t*/
	
	inline effective_parameter(float population_size, float inbreeding); /**<\brief constructor */
	template <typename Functor_demography, typename Functor_inbreeding>
	inline effective_parameter(Functor_demography population_size, Functor_inbreeding inbreeding, unsigned int generation, unsigned int population); /**<\brief constructor */
	inline float operator()(float effective_parameter) const; /**<\brief returns parameter coefficient, for a given `effective_parameter, population_size, inbreeding` */
};

///functor: models parameter \p p as a constant across populations and over time
struct constant_parameter
{
	float p; /**<\brief parameter */ /**<\t*/
	
	inline constant_parameter(); /**<\brief default constructor */ /**<`s = 0`*/
	inline constant_parameter(float p); /**<\brief constructor */ /**<\t*/
	template <typename... Args>
	__host__ __device__ __forceinline__ float operator()(Args ...) const; /**<\brief operator, returns parameter, for a given `population, generation, ...` */ /**<\t*/
};

///functor: scales parameter \p p so the effective parameter (Ne_chrome*p) is constant across populations and over time
template <typename Functor_demography, typename Functor_inbreeding>
struct constant_effective_parameter
{
	float p; /**<\brief parameter */ /**<\t*/
	float reference_Nchrom_e; /**<\brief reference number of effective chromosomes against which to scale effective parameter */ /**<\t*/
	Functor_demography N; /**<\brief population size in individuals */ /**<\t*/
	Functor_inbreeding F; /**<\brief inbreeding coefficient */ /**<\t*/

	inline constant_effective_parameter(float p, float reference_population_size, float reference_inbreeding, Functor_demography population_size, Functor_inbreeding inbreeding); /**<\brief constructor */ /**<\t*/
	template <typename... Args>
	__host__ __device__ __forceinline__ float operator()(const unsigned int generation, const unsigned int population, Args...) const; /**<\brief operator, returns parameter, for a given `population, generation, ...` */ /**<\t*/
};

template <typename Functor_demography, typename Functor_inbreeding>
auto make_constant_effective_parameter(float p, float reference_population_size, float reference_inbreeding, Functor_demography population_size, Functor_inbreeding inbreeding); /**<\brief constructs  template <typename Functor_demography, typename Functor_inbreeding>  Sim_Model::constant_effective_parameter */ /**<\t*/

/**\brief functor: models parameter as linearly dependent on generation */
struct linear_generation_parameter
{
	float slope; /**<\brief slope of parameter's linear dependence on generation */ /**<\t*/
	float intercept; /**<\brief parameter's intercept with generation 0 */ /**<\t*/
	unsigned int start_generation; //!<\copydoc Sim_Model::selection_sine_wave::start_generation

	inline linear_generation_parameter(); /**<\brief default constructor */ /**<`slope = 0, intercept = 0`*/
	inline linear_generation_parameter(float start_parameter, float end_parameter, unsigned int start_generation, unsigned int end_generation); /**<\brief constructor */ /**<\t*/
	template <typename... Args>
	__host__ __device__ __forceinline__ float operator()(const unsigned int generation, Args...) const; //!<\copybrief template<typename... Args> Sim_Model::constant_parameter::operator()(Args ...) const
};

/**\brief functor: models parameter as a sine wave through time */ /**useful for modeling cyclical/seasonal behavior over time*/
struct sine_generation_parameter
{
	float A; /**<\brief Amplitude of sine wave */ /**<\t*/
	float pi; /**<\brief Frequency of sine wave */ /**<\t*/
	float rho; /**<\brief Phase of sine wave */ /**<\t*/
	float D; /**<\brief Offset of sine wave */ /**<\t*/
	unsigned int start_generation; /**<\brief start_generation of the function dependent on time */ /**<\details shifts function back by `start_generation` - i.e. `f(generation-start_generation)` */
	
	inline sine_generation_parameter(); /**<\brief default constructor */ /**<all parameters set to `0`*/
	inline sine_generation_parameter(float A, float pi, float D, float rho = 0, unsigned int start_generation = 0); /**<\brief constructor */
	template <typename... Args>
	__host__ __device__ __forceinline__ float operator()(const unsigned int generation, Args...) const; //!<\copybrief template<typename... Args> Sim_Model::constant_parameter::operator()(Args ...) const
};

///functor: models parameter as an exponential function of time
struct exponential_generation_parameter
{
	float initial_parameter_value; /**<\brief initial population size */ /**<\t*/
	float rate; /**<\brief exponential growth rate */ /**<\t*/
	unsigned int start_generation; //!<\copydoc Sim_Model::selection_sine_wave::start_generation
	
	inline exponential_generation_parameter(); //!<\copydoc Sim_Model::selection_sine_wave::selection_sine_wave()
	inline exponential_generation_parameter(float initial_parameter_value, float rate, unsigned int start_generation = 0); //!< constructor
	inline exponential_generation_parameter(float initial_parameter_value, float final_parameter_value, unsigned int start_generation, unsigned int end_generation); //!< constructor
	template <typename... Args>
	__host__ __device__ __forceinline__ float operator()(const unsigned int generation, Args...) const; //!<\copybrief template<typename... Args> Sim_Model::constant_parameter::operator()(Args ...) const
};

///functor: models parameter as logistic function of time
struct logistic_generation_parameter
{
	float rate; /**<\brief logistic growth rate */ /**<\t*/
	float initial_parameter_value; /**<\brief initial population size */ /**<\t*/
	float carrying_capacity; /**<\brief carrying capacity */ /**<\t*/
	unsigned int start_generation; //!<\copydoc Sim_Model::selection_sine_wave::start_generation
	
	inline logistic_generation_parameter(); //!<\copydoc Sim_Model::selection_sine_wave::selection_sine_wave()
	inline logistic_generation_parameter(float initial_parameter_value, float carrying_capacity, float rate, unsigned int start_generation = 0); //!<\copydoc Sim_Model::demography_exponential_growth::demography_exponential_growth(float rate, int initial_population_size, unsigned int start_generation = 0)
	inline logistic_generation_parameter(float initial_parameter_value, float carrying_capacity, float final_parameter_value, unsigned int start_generation, unsigned int end_generation); //!<\copydoc Sim_Model::demography_exponential_growth::demography_exponential_growth(float rate, int initial_population_size, unsigned int start_generation = 0)
	template <typename... Args>
	__host__ __device__ __forceinline__ float operator()(const unsigned int generation, Args...) const; //!<\copybrief template<typename... Args> Sim_Model::constant_parameter::operator()(Args ...) const
};

/**\brief functor: models dominance/selection coefficient as linearly dependent on allele frequency */
struct linear_frequency_h_s
{
	float slope; /**<\brief slope of selection coefficient's linear dependence on frequency */ /**<\t*/
	float intercept; /**<\brief selection coefficient's intercept with frequency 0 */ /**<\t*/
	
	inline linear_frequency_h_s(); /**<\brief default constructor */ /**<`slope = 0, intercept = 0`*/
	inline linear_frequency_h_s(float start_parameter, float end_parameter_slope, bool is_slope = false); /**<\brief constructor */ /**<`intercept` := \param start_parameter  && if \param is_slope `slope` := \param end_parameter_slope, else `slope` := \param end_parameter_slope `-` \param start_parameter */
	template <typename... Args>
	__host__ __device__ __forceinline__ float operator()(const unsigned int generation, const unsigned int population, const float freq, Args...) const; //!<\copybrief template<typename... Args> Sim_Model::constant_parameter::operator()(Args ...) const
};

/**\brief function: returns a functor modeling selection as in Robertson, 1955: a coefficient linearly dependent on frequency with slope and intercept determined by effect size \param effect_size and population variance \param variance := `sigma^2` */
linear_frequency_h_s make_robertson_stabilizing_selection_model(float effect_size, float variance);

/**\brief functor: models dominance/selection coefficient as a hyperbola dependent on allele frequency */
struct hyperbola_frequency_h_s
{
	float A; /**<\brief y = A/(freq-B) + C */ /**<\t*/
	float B; /**<\copybrief Sim_Model::hyperbola_frequency_dependent_h_s::A */ /**<\t*/
	float C; /**<\copybrief Sim_Model::hyperbola_frequency_dependent_h_s::A */ /**<\t*/
	
	inline hyperbola_frequency_h_s(); /**<\brief default constructor */ /**<co-dominant `A = 0, B = -1, C = 0.5`*/
	inline hyperbola_frequency_h_s(float A, float B, float C); /**<\brief constructor */ /**<\t*/
	inline hyperbola_frequency_h_s(linear_frequency_h_s numerator, linear_frequency_h_s denominator); /**<\brief constructor */ /**<\t*/
	template <typename... Args>
	__host__ __device__ __forceinline__ float operator()(const unsigned int generation, const unsigned int population, const float freq, Args...) const; //!<\copybrief template<typename... Args> Sim_Model::constant_parameter::operator()(Args ...) const
};

/**\brief function: returns a functor modeling dominance as in Robertson, 1955: a hyperbola dependent on allele frequency with `A` = 1/8, `B` = 1/2, and `C` = 1/2 */
hyperbola_frequency_h_s make_robertson_stabilizing_dominance_model();

/**\brief functor: returns the selection coefficient stored in the mutation ID from a DFE, optionally shifted and scaled*/
struct DFE_s
{
	float scale; /**<\brief *reinterpret_cast<float*>(&mutationID.w))*scale + shift */ /**<\t*/
	float shift;/**<\copybrief Sim_Model::DFE_s::scale */ /**<\t*/
		
	inline DFE_s(); /**<\brief default constructor */ /**<no shift = 0, no multiple = 1`*/
	inline DFE_s(float scale, float shift); /**<\brief constructor */ /**<\t*/
	__host__ __device__ __forceinline__ float operator()(const unsigned int generation, const unsigned int population, const float freq, const uint4 mutID) const; //!<\copybrief template<typename... Args> Sim_Model::constant_parameter::operator()(Args ...) const
};

template <typename Default_fun, typename... List>
auto make_population_specific_evolution_model(Default_fun defaultFun, List... list);

template <typename Default_fun, typename... List>
auto make_directional_migration_model(Default_fun defaultFun, List... list);

template <typename Start_fun, typename... List>
auto make_piecewise_evolution_model(Start_fun defaultFun, List... list);

template <typename Default_Start_fun, typename... List>
auto make_piecewise_population_specific_model(Default_Start_fun defaultFun, List... list);

template <typename Default_Start_fun, typename... List>
auto make_piecewise_directional_migration_model(Default_Start_fun defaultFun, List... list);

struct standard_mse_integrand{
	float N,s,F,h;

	standard_mse_integrand();
	template <typename Functor_demography, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance>
	standard_mse_integrand(const Functor_demography dem, const Functor_selection sel_coeff, const Functor_inbreeding f_inbred, const Functor_dominance dominance, unsigned int gen, unsigned int pop);
	__host__ __device__ double operator()(double i) const;
	__host__ __device__ bool neutral() const;
};

struct robertson_stabilizing_mse_integrand{
	float constant;

	robertson_stabilizing_mse_integrand();
	template <typename Functor_demography, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance>
	robertson_stabilizing_mse_integrand(const Functor_demography dem, const Functor_selection sel_coeff, const Functor_inbreeding f_inbred, const Functor_dominance dominance, unsigned int gen, unsigned int pop);
	__host__ __device__ double operator()(double i) const;
	__host__ __device__ bool neutral() const;
};

__host__ __device__ __forceinline__ float null_DFE_inv_func(float in);

template <typename DFE_inv_function = decltype(&null_DFE_inv_func)>
struct DFE{
	DFE_inv_function inverse_distribution;
	uint2 seed;
	
	inline DFE();
	inline DFE(unsigned int seed1, unsigned int seed2, DFE_inv_function inverse_distribution);
	__host__ __device__ __forceinline__ unsigned int operator()(const unsigned int generation, const unsigned int population, const unsigned int tID) const;
};

//!\cond
// MAKE THESE LATER

struct linear_s_mse_integrand{
	float N,s,F,h;

	linear_s_mse_integrand();
	template <typename Functor_demography, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance>
	linear_s_mse_integrand(const Functor_demography dem, const Functor_selection sel_coeff, const Functor_inbreeding f_inbred, const Functor_dominance dominance, unsigned int gen, unsigned int pop);
	__host__ __device__ double operator()(double i) const;
	__host__ __device__ bool neutral() const;
};

struct linear_s_hyperbola_h_mse_integrand{
	float N,s,F,h;

	linear_s_hyperbola_h_mse_integrand();
	template <typename Functor_demography, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance>
	linear_s_hyperbola_h_mse_integrand(const Functor_demography dem, const Functor_selection sel_coeff, const Functor_inbreeding f_inbred, const Functor_dominance dominance, unsigned int gen, unsigned int pop);
	__host__ __device__ double operator()(double i) const;
	__host__ __device__ bool neutral() const;
};

template <std::size_t num_pops>
auto make_constant_directional_migration_model(std::array<float,num_pops> migration_to);

template <std::size_t num_pops>
auto make_constant_migration_model(std::array<std::array<float,num_pops>,num_pops> migration_matrix);

template <std::size_t gen_size>
struct static_array_generation_parameter
{
	float parameter_array[gen_size];
	unsigned int start_generation;

	inline static_array_generation_parameter(std::array<float,gen_size> parameter_array);
	template <typename Function, typename... Args>
	inline static_array_generation_parameter(Function f, unsigned int start_generation, Args... rest_f_params);
	template<typename... Args>
	__host__ __device__ __forceinline__ float operator()(const unsigned int generation, Args...) const;
};

struct dynamic_array_pop_gen_freq_parameter
{
	ppp::unique_device_ptr<float> d_pop_gen_freq;
	std::unique_ptr<float[]> h_pop_gen_freq;
	unsigned int num_gen, num_pop, num_freq;

	template<typename Function>
	inline dynamic_array_pop_gen_freq_parameter(Function f, unsigned int num_gen, unsigned int num_pop, unsigned int num_freq = 0);
	__host__ __device__ __forceinline__ float operator()(const unsigned int generation, const unsigned int population, const float freq = -1.f) const;  /**<\brief Migration operator, returns migration rate, \p mig_rate, which is the proportion of chromosomes in `pop_TO` from `pop_FROM` for a given `generation` */
};

struct dynamic_array_migration
{
	ppp::unique_device_ptr<float> d_migration;
	std::unique_ptr<float[]> h_migration;
	unsigned int num_pop, num_gen;
	template<typename Function>
	inline dynamic_array_migration(Function f, unsigned int num_gen, unsigned int num_pop);
	__host__ __device__ __forceinline__ float operator()(const unsigned int generation, const unsigned int pop_FROM, const unsigned int pop_TO) const; /**<\brief Migration operator, returns migration rate, \p mig_rate, which is the proportion of chromosomes in `pop_TO` from `pop_FROM` for a given `generation` */
};
//!\endcond

} /* ----- end namespace Sim_Model ----- */

///Namespace for sampling GO_Fish simulations
namespace Sampling{

/**\brief functor: turns sampling off (for every generation except the final one which is always sampled) */
struct off
{
	__host__ __forceinline__ bool operator()(const unsigned int generation) const; /**<\brief Preserving and Sampling operator, returns boolean \p b to turn on/off preserving and sampling in generation \p generation of the simulation*//**<`b = false`*/
};

/**\brief functor: turns sampling on (for every generation except the final one which is always sampled) */
struct on
{
	__host__ __forceinline__ bool operator()(const unsigned int generation) const; /**<\copybrief Sim_Model::bool_off::operator()(const unsigned int generation) const *//**<`b = true`*/
};

/**\brief functor: returns default state except during a pulse generation and the final generation which is always sampled */
struct pulse
{
	bool default_state; /**<\brief default boolean to return */ /**<\t*/
	std::vector<unsigned int> pulses; /**<\brief list of generations at which pulses (!default) occur */ /**<\t*/
	unsigned int start_generation; //!<\copydoc Sim_Model::selection_sine_wave::start_generation

	inline pulse(std::vector<unsigned int> pulses, unsigned int start_generation = 0, bool default_state = false); /**<\brief constructor */
	__host__ __forceinline__ bool operator()(const unsigned int generation) const;
};

/**\brief functor: returns current state, sample */
struct intervals
{
	bool start_state;  /**<\brief current boolean to return */ /**<\t*/
	std::vector<unsigned int> change_points; /**<\brief list of generations at which the current boolean state flips */ /**<\t*/
	unsigned int start_generation; //!<\copydoc Sim_Model::selection_sine_wave::start_generation

	inline intervals(std::vector<unsigned int> change_points, unsigned int start_generation = 0, bool start_state = false); /**<\brief constructor */
	__host__ __forceinline__ bool operator()(const unsigned int generation) const;
};

} /* ----- end namespace Sampling ----- */

