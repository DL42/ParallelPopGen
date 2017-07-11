/*!\file
* \brief Subset of go_fish.cuh (the GO_Fish data structures)
*
* go_fish_data_struct.h contains the structures and associated functions for storing and outputting a GO_Fish simulation run.
* When go_fish.cuh is already included into a source file, go_fish_data_struct.h is automatically included - no need to include it separately.
* However, go_fish_data_struct.h can be included by itself - the advantage being that it can be included in C, C++ (*.c, *.cpp) source files
* as well as CUDA source files (*.cu). This allows data from a simulation run to be passed from a CUDA project to an already established
* C/C++ project compiled with a standard C/C++ compiler (e.g. clang, g++, msvc, etc ...) using structures GO_Fish::allele_trajectories and GO_Fish::mutID.
* See \ref Example3-Compilation.\n\n
*/
/* go_fish_data_struct.h
 *
 * Author: David Lawrie
 * Subset of go_fish.cuh: GO_Fish data structures
 */

#ifndef GO_FISH_DATA_H_
#define GO_FISH_DATA_H_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <cstring>

//!\cond
namespace Spectrum_details{ class transfer_allele_trajectories; } //for passing data to SPECTRUM functions
//!\endcond

namespace GO_Fish{

/* ----- sim result output ----- */
//!structure specifying the ID for a mutation in a GO_Fish simulation
struct mutID{
	int origin_generation; /**<\brief generation in which mutation appeared in simulation */ /**<\t*/
	int origin_population; /**<\brief population in which mutation first arose */ /**<\t*/
	int origin_threadID;/**<\brief threadID that generated mutation */ /**<\t*/
    int reserved; /**<\brief reserved for later use, currently 0 */ /**<\t*/

    //!default constructor
    inline mutID();

    //!constructor
	inline mutID(int origin_generation, int origin_population, int origin_threadID, int reserved);

    //!returns a string constant of mutID
    inline std::string toString();
    //!\copydoc GO_Fish::mutID::toString()
    inline std::string toString() const;
};

//!control and output data structure for GO_Fish simulation
struct allele_trajectories{
	//----- initialization parameters -----
	//!specification of simulation constants
	struct sim_constants{
		int seed1; /**<\brief random number seed 1 of 2 */ /**< default: 0xbeeff00d */
		int seed2; /**<\brief random number seed 2 of 2 */ /**< default: 0xdecafbad */
		int num_generations; /**<\brief number of generations in simulation */ /**< default: 0 */
		float num_sites; /**<\brief number of sites in simulation*/ /**<default: 1000*/
		int num_populations; /**<\brief number of populations in simulation */ /**< default: 1 */
		bool init_mse; /**<\brief true: initialize simulation in mutation_selection_equilibrium; false: initialize blank simulation or using previous simulation time sample */ /**< default: true */
		int prev_sim_sample; /**<\brief time sample of previous simulation to use for initializing current simulation */ /**< overridden by init_mse if init_mse = true \n default: -1 (if init_mse = false, ignore previous simulation & initialize blank simulation) */
		int compact_interval; /**<\brief how often to compact the simulation and remove fixed or lost mutations */ /**< default: 35 := compact every 35 generations\n compact_interval = 0 turns off compact (mutations will not be removed even if lost or fixed) \n\n **Note:** Changing the compact
                              * interval will change the result of the simulation run for the same seed numbers. However, these are not independent simulation runs! Changing the compact interval produces new random, but correlated simulation results. */
		int device; /**<\brief GPU identity to run simulation on, if -1 next available GPU will be assigned */ /**< default: -1 */

		inline sim_constants();
	};

	sim_constants sim_input_constants; /**<\brief constants for initializing the next simulation */ /**<\t*/
	//----- end -----

	/**\brief default constructor */ /**\t*/
	inline allele_trajectories();

	/**\brief copy constructor */ /**\t*/
	inline allele_trajectories(const allele_trajectories & in);

	/**\brief copy assignment */ /**\t*/
	inline allele_trajectories & operator=(allele_trajectories in);

	/**\brief returns sim_constants of the simulation currently held by allele_trajectories */ /**\t*/
	inline sim_constants last_run_constants();

	/**\brief returns the number of sites in the simulation */ /**\t*/
	inline int num_sites();

	/**\brief returns the number of populations in the simulation */ /**maximum population_index*/
	inline int num_populations();

	/**\brief returns number of time samples taken during simulation run */ /**maximum sample_index*/
	inline int num_time_samples();

	/**\brief returns number of reported mutations in the final time sample (maximal number of stored mutations in the allele_trajectories) */ /**maximum mutation_index*/
	inline int maximal_num_mutations();

	/**\brief number of reported mutations in the time sample \p sample_index */ /**\t*/
	inline int num_mutations_time_sample(int sample_index);

	/**\brief returns final generation of simulation */ /**\t*/
	inline int final_generation();

	/**\brief return generation of simulation in the time sample \p sample_index */ /**\t*/
	inline int sampled_generation(int sample_index);

	/**\brief returns whether or not population \p population_index is extinct in time sample \p sample_index */ /**\t*/
	inline bool extinct(int sample_index, int population_index);

	/**\brief returns the effective number of chromosomes of population \p population_index in time sample \p sample_index */ /**\t*/
	inline int effective_number_of_chromosomes(int sample_index, int population_index);

	/**\brief returns the frequency of the mutation at time sample \p sample_index, population \p population_index, mutation \p mutation_index */ /**\t*/
	inline float frequency(int sample_index, int population_index, int mutation_index);

	/**\brief returns the mutation ID at \p mutation_index */ /**\t*/
	inline mutID mutation_ID(int mutation_index);

	/**\brief deletes a single time sample, \p sample_index */ /**\t*/
	inline void delete_time_sample(int sample_index);

	/**\brief deletes all memory held by allele_trajectories, resets constants to default */
	inline void reset();

	/**\brief destructor */ /**calls reset()*/
	inline ~allele_trajectories();

	//!\cond
	friend void swap(allele_trajectories & a, allele_trajectories & b);

	friend std::ostream & operator<<(std::ostream & stream, allele_trajectories & A);

	template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_preserve, typename Functor_timesample>
	friend void run_sim(allele_trajectories & all_results, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding FI, const Functor_dominance dominance, const Functor_preserve preserve_mutations, const Functor_timesample take_sample, const allele_trajectories & prev_sim);

	friend class Spectrum_details::transfer_allele_trajectories;
	//!\endcond
private:

	struct time_sample{
		float * mutations_freq; //allele frequency of mutations in final generation
		bool * extinct; //extinct[pop] == true, flag if population is extinct by time sample
		int * Nchrom_e; //effective number of chromosomes in each population
		int num_mutations; //number of mutations in frequency array (columns array length for freq)
		int sampled_generation; //number of generations in the simulation at point of sampling

		time_sample();
		~time_sample();
	};

	inline void initialize_run_constants();

	inline void initialize_sim_result_vector(int new_length);

	sim_constants sim_run_constants; //stores constants of the simulation run currently held by time_samples
	time_sample ** time_samples; //the actual allele trajectories output from the simulation
	int num_samples; //number of time samples taken from the simulation
	mutID * mutations_ID; //unique ID for each mutation in simulation
	int all_mutations; //number of mutations in mutation ID array - maximal set of mutations stored in allele_trajectories
}; /**< Stores the constants, mutation IDs (mutID), and time samples of a simulation run. Each time sample holds the frequencies of each mutation at the time the sample was taken, the size of each population in chromosomes and which population were extinct for a time sample, the number of mutations in the sample, and of which simulation generation is the sample. Data is accessed through the member functions. **/

/**\brief insertion operator: sends `mutID id` into the `ostream stream` */
inline std::ostream & operator<<(std::ostream & stream, const mutID & id);

//! insertion operator: sends `allele_trajectories A` into the `ostream stream`
inline std::ostream & operator<<(std::ostream & stream, allele_trajectories & A);

/**\brief swaps data held by allele_trajectories a and b */ /**\t*/
inline void swap(allele_trajectories & a, allele_trajectories & b);

/* ----- end sim result output ----- */

} /* ----- end namespace GO_Fish ----- */

/* ----- import inline function definitions ----- */
#include "../3P/_internal/inline_go_fish_data_struct.hpp"
/* ----- end ----- */

#endif /* GO_FISH_DATA_H_ */
