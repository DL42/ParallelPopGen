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
#include <string>
#include <memory>
#include <vector>
#include <tuple>
#include <limits>
#include "../3P/span.hpp"
#include "../3P/_internal/ppp_types.hpp"

namespace GO_Fish{

/* ----- sim result input/output ----- */
//!structure specifying the ID for a mutation in a GO_Fish simulation
struct mutID{
	unsigned int origin_generation; /**<\brief generation in which mutation appeared in simulation */ /**<\t*/
	unsigned int origin_population; /**<\brief population in which mutation first arose */ /**<\t*/
	unsigned int origin_threadID;/**<\brief threadID that generated mutation */ /**<\t*/
	unsigned int reserved; /**<\brief reserved for later use, currently 0 */ /**<\t*/
};

//!enum specifying type of compact: compact_all fixed and lost mutations, only compact_losses, only compact_fixations, and turn compact_off
enum class compact_scheme{ compact_all, compact_losses, compact_fixations, compact_off };

//!specification of simulation constants
struct sim_constants{
	unsigned int seed1{0xbeeff00d}; /**<\brief random number seed 1 of 2 */ /**< default: 0xbeeff00d */
	unsigned int seed2{0xdecafbad}; /**<\brief random number seed 2 of 2 */ /**< default: 0xdecafbad */
	unsigned int num_generations{0}; /**<\brief number of generations in simulation */ /**< default: 0 */
	float num_sites{1000}; /**<\brief number of sites in simulation*/ /**<default: 1000*/
	unsigned int num_populations{1}; /**<\brief number of populations in simulation */ /**< default: 1 */
	bool init_mse{true}; /**<\brief true: initialize simulation in mutation_selection_equilibrium; false: initialize blank simulation or using previous simulation time sample */ /**< default: true */
	unsigned int prev_sim_sample{0}; /**<\brief time sample of previous simulation to use for initializing current simulation */ /**< default: 0 */
	compact_scheme compact_type{compact_scheme::compact_all}; /**<\brief compact scheme used in simulation */ /**< default: compact_all */
	unsigned int compact_interval{35}; /**<\brief how often to compact the simulation and remove fixed or lost mutations */ /**< default: 35 := compact every 35 generations, must be >= 1 if compact_type != compact_off */
	int device{-1}; /**<\brief GPU identity to run simulation on, if -1 next available GPU will be assigned */ /**< default: -1 */
};

//!output data structure for GO_Fish simulation
struct allele_trajectories{

	/**\brief default constructor */ /**\t*/
	inline allele_trajectories() noexcept = default;

	inline allele_trajectories(const sim_constants & run_constants, size_t _num_samples);

	template<typename cContainer_sample, typename cContainer_numgen_mut, typename cContainer_pop, typename cContainer_extinct, typename cContainer_allele, typename cContainer_mutID>
	inline allele_trajectories(const sim_constants & run_constants, const cContainer_sample & sampled_generation, const cContainer_numgen_mut & total_generated_mutations, const cContainer_pop & pop_span_view, const cContainer_extinct & extinct_span_view, const cContainer_allele & allele_span_view, const cContainer_mutID & mutID_span);

	inline void initialize_time_sample(unsigned int sample_index, unsigned int sampled_generation, unsigned int num_mutations, unsigned long total_generated_mutations);

	inline void initialize_time_sample(unsigned int sample_index, unsigned int sampled_generation, unsigned long total_generated_mutations, std::span<const unsigned int> pop_span, std::span<const unsigned int> extinct_span, std::span<const unsigned int> allele_span, std::span<const mutID> mutID_span);

	/**\brief copy constructor */ /**\t*/
	inline allele_trajectories(const allele_trajectories & in);

	/**\brief move constructor */ /**\t*/
	inline allele_trajectories(allele_trajectories && in) noexcept;

	/**\brief move/copy assignment */ /**\t*/
	inline allele_trajectories & operator=(allele_trajectories in) noexcept;

	/**\brief swaps data held by this allele_trajectories and in */ /**\t*/
	inline void swap(allele_trajectories & in) noexcept;

	/**\brief frees memory held in allele_trajectories, resets constants to default */
	inline void reset() noexcept;

	/**\brief default destructor */
	inline ~allele_trajectories() noexcept = default;

	/**\brief returns sim_constants of the simulation currently held by allele_trajectories */ /**\t*/
	inline sim_constants last_run_constants() const noexcept;

	/**\brief returns the number of sites in the simulation */ /**\t*/
	inline float num_sites() const noexcept;

	/**\brief returns the number of populations in the simulation */ /**maximum population_index*/
	inline unsigned int num_populations() const noexcept;

	/**\brief returns number of time samples taken during simulation run */ /**maximum sample_index*/
	inline unsigned int num_time_samples() const noexcept;

	/**\brief returns final generation of simulation */ /**\t*/
	inline unsigned int final_generation() const;

	/**\brief returns number of reported mutations in the final time sample (maximal number of stored mutations in the allele_trajectories) */ /**maximum mutation_index*/
	inline unsigned int maximal_num_mutations() const noexcept;

	/**\brief number of reported mutations in the time sample \p sample_index */ /**\t*/
	inline unsigned int num_mutations_time_sample(unsigned int sample_index) const;

	/**\brief total number of mutations generated in the simulation by time sample \p sample_index */ /**this includes lost, fixed, and segregating mutations \n if compacting is turned off will be equal to num_mutations_time_sample*/
	inline unsigned long total_generated_mutations_time_sample(unsigned int sample_index) const;

	/**\brief return generation of simulation in the time sample \p sample_index */ /**\t*/
	inline unsigned int sampled_generation(unsigned int sample_index) const;

	/**\brief returns whether or not population \p population_index is extinct in time sample \p sample_index */ /**\t*/
	inline bool extinct(unsigned int sample_index, unsigned int population_index) const;

	/**\brief returns the effective number of chromosomes of population \p population_index in time sample \p sample_index */ /**\t*/
	inline unsigned int effective_number_of_chromosomes(unsigned int sample_index, unsigned int population_index) const;

	/**\brief returns the frequency of the mutation at time sample \p sample_index, population \p population_index, mutation \p mutation_index */ /**\t*/
	inline unsigned int allele_count(unsigned int sample_index, unsigned int population_index, unsigned int mutation_index) const;

	/**\brief returns the mutation ID at \p mutation_index */ /**\t*/
	inline const mutID & mutation_ID(unsigned int mutation_index) const;

	inline std::vector<unsigned int> dump_sampled_generations() const;

	inline std::vector<unsigned int> dump_num_mutations_samples() const;

	inline std::vector<unsigned long> dump_total_generated_mutations_samples() const;

	inline std::span<unsigned int> popsize_span(unsigned int sample_index);

	inline std::span<const unsigned int> popsize_span(unsigned int sample_index) const;

	inline std::vector<std::span<const unsigned int>> popsize_view() const;

	inline std::vector<std::vector<unsigned int>> dump_popsize() const;

	inline std::span<unsigned int> extinct_span(unsigned int sample_index);

	inline std::span<const unsigned int> extinct_span(unsigned int sample_index) const;

	inline std::vector<std::span<const unsigned int>> extinct_view() const;

	inline std::vector<std::vector<unsigned int>> dump_extinct() const;

	inline std::span<unsigned int> allele_count_span(unsigned int sample_index, unsigned int start_population_index = 0, unsigned int num_contig_pop = 0);

	inline std::span<const unsigned int> allele_count_span(unsigned int sample_index, unsigned int start_population_index = 0, unsigned int num_contig_pop = 0) const;

	inline std::vector<std::span<const unsigned int>> allele_count_view() const;

	inline std::vector<std::vector<unsigned int>> dump_allele_counts() const;

	inline std::vector<std::vector<std::vector<unsigned int>>> dump_padded_allele_counts() const;

	inline std::span<mutID> mutID_span() noexcept;

	inline std::span<const mutID> mutID_span() const noexcept;

	inline std::vector<mutID> dump_mutID() const;

private:

	struct time_sample{
		ppp::unique_host_span<unsigned int> mutations_freq; //allele counts of mutations in sampled generation
		ppp::unique_host_span<unsigned int> extinct; //extinct[pop] > 0 if population is extinct by time sample, 0 still alive
		ppp::unique_host_span<unsigned int> Nchrom_e; //effective number of chromosomes in each population
		unsigned int num_mutations; //number of mutations in frequency array (columns array length for freq)
		unsigned int sampled_generation; //number of generations in the simulation at point of sampling
		unsigned long total_generated_mutations; //total number of generated mutations at point of sampling (including those compacted away)
	};

	sim_constants sim_run_constants; //stores the simulation input constants of the results currently held by allele_trajectories
	ppp::unique_host_span<time_sample> time_samples; //the allele trajectories, sample generation, and population information for each time sample from the simulation
	ppp::unique_host_span<mutID> mutations_ID; //unique ID for each mutation in simulation
}; /**< Stores the constants, mutation IDs (mutID), and time samples of a simulation run. Each time sample holds the frequencies of each mutation at the time the sample was taken, the size of each population in chromosomes and which population were extinct for a time sample, the number of mutations in the sample, and of which simulation generation is the sample. Data is accessed through the member functions. **/

/**\brief insertion operator: sends `mutID id` into the `ostream stream` */
inline std::ostream & operator<<(std::ostream & stream, const mutID & id);

/**\brief insertion operator: sends `compact_scheme ctype` into the `ostream stream` */
inline std::ostream & operator<<(std::ostream & stream, const compact_scheme & ctype);

/**\brief insertion operator: sends `sim_constants constants` into the `ostream stream` */
inline std::ostream & operator<<(std::ostream & stream, const sim_constants & constants);

//! insertion operator: sends `allele_trajectories A` into the `ostream stream`
inline std::ostream & operator<<(std::ostream & stream, const allele_trajectories & A);

/**\brief extraction operator: sends `istream stream` into the `mutID id` */
inline std::istream & operator>>(std::istream & stream, mutID & id);

/**\brief extraction operator: sends `istream stream` into the `sim_constants constants` */
inline std::istream & operator>>(std::istream & stream, sim_constants & constants);

//! extraction operator: sends `istream stream` into the `allele_trajectories A`
inline std::istream & operator>>(std::istream & stream, allele_trajectories & A);

/**\brief swaps data held by allele_trajectories lhs and rhs */
inline void swap(allele_trajectories & lhs, allele_trajectories & rhs) noexcept;

/* ----- end sim result input/output ----- */

} /* ----- end namespace GO_Fish ----- */

/* ----- import inline function definitions ----- */
#include "../3P/_internal/inline_go_fish_data_struct.hpp"
/* ----- end ----- */

#endif /* GO_FISH_DATA_H_ */
