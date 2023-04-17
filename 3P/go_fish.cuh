/*!\file
* \brief GO Fish Simulation API (contains namespaces GO_Fish and Sim_Model)
*
* go_fish.cuh contains all structures and functions necessary for running, storing, and outputting a single locus Wright_fisher simulation.
* When including go_fish.cuh into a source file, go_fish_data_struct.h is automatically included - no need to include it separately.
* Unlike go_fish_data_struct.h and spectrum.h, go_fish.cuh can only be included in CUDA source files (*.cu).
* CUDA source files are identical to C, C++ source files (*.c, *.cpp) except that the the CUDA suffix (*.cu) indicates for NVCC (the CUDA compiler)
* that the file contains CUDA-specific `__global__` and `__device__` functions/variables meant for the GPU. `__global__` and `__device__` functions/variables
* cannot be compiled in normal C/C++ source files. \n\n

*/
/*
 * 		go_fish.cuh
 *
 *      Author: David Lawrie
 *      GO Fish Simulation API
 */

#ifndef GO_FISH_API_H_
#define GO_FISH_API_H_
#include "_internal/ppp_cuda.cuh"
#include "../3P/go_fish_data_struct.h"
#include "../3P/sim_model.cuh"

//!Namespace for single-locus, forward, Monte-Carlo Wright-Fisher simulation and output data structures
namespace GO_Fish{

/* ----- go_fish_impl  ----- */
///runs a single-locus Wright-Fisher simulation specified by the given simulation functions and sim_constants, storing the results into returned allele_trajectories
template <typename Functor_mutation, typename Functor_demography, typename Functor_migration, typename Functor_selection, typename Functor_inbreeding, typename Functor_dominance, typename Functor_timesample, typename Functor_mse = Sim_Model::standard_mse_integrand, typename Functor_dfe = Sim_Model::DFE<>>
__host__ allele_trajectories run_sim(sim_constants sim_input_constants, const Functor_mutation mu_rate, const Functor_demography demography, const Functor_migration mig_prop, const Functor_selection sel_coeff, const Functor_inbreeding f_inbred, const Functor_dominance dominance, const Functor_timesample take_sample, const allele_trajectories & prev_sim = allele_trajectories(), Functor_mse = Sim_Model::standard_mse_integrand(), Functor_dfe DFE = Sim_Model::DFE<>{});
/* ----- end go_fish_impl ----- */

} /* ----- end namespace GO_Fish ----- */

/* ----- importing go_fish_impl  ----- */
#include "../3P/_internal/go_fish_impl.cuh"
/* ----- end importing go_fish_impl ----- */

/* ----- importing functor implementations ----- */
#include "../3P/_internal/template_inline_simulation_functors.cuh"
/* ----- end importing functor implementations ----- */


#endif /* GO_FISH_API_H_ */
