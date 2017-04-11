/*!\file
* \brief proto-API for building site frequency spectra (contains the titular namespace Spectrum)
*
* spectrum.h is a prototype API for accelerating site frequency spectrum analysis on the GPU.
* Though functions Spectrum::population_frequency_histogram and Spectrum::site_frequency_spectrum are accelerated on the GPU,
* the CUDA specific code is not in the spectrum.h header file and thus, like go_fish_data_struct.h, spectrum.h can be included
* in either CUDA (*.cu) or standard C, C++ (*.c, *.cpp) source files.
*/
/*
 * spectrum.h
 *
 *      Author: David Lawrie
 */

#ifndef SPECTRUM_H_
#define SPECTRUM_H_

#include "../include/go_fish_data_struct.h"

///Namespace for site frequency spectrum data structure and functions. (in prototype-phase)
namespace Spectrum{

///site frequency spectrum data structure (at the moment, functions only generate SFS for a single population at a single time point)
struct SFS{
	float * frequency_spectrum; ///<site frequency spectrum data structure
	int * populations; ///<which populations are in SFS
	int * sample_size; ///<number of samples taken for each population
	int num_populations; ///<number of populations in SFS
	float num_sites;  ///<number of sites in SFS
	float num_mutations; ///<number of segregating mutations in SFS
	int sampled_generation; ///<number of generations in the simulation at time of sampling

	//!default constructor
	SFS();
	//!default destructor
	~SFS();
};

///create a frequency histogram of mutations at a single time point \p sample_index in a single population \p population_index store in \p mySFS
void population_frequency_histogram(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, int cuda_device = -1);

///create a single-population SFS of size \p sample_size from a single time point \p sample_index in a single population \p population_index from allele trajectory \p all_results, store in \p mySFS
void site_frequency_spectrum(SFS & mySFS, const GO_Fish::allele_trajectories & all_results, const int sample_index, const int population_index, const int sample_size, int cuda_device = -1);

} /*----- end namespace SPECTRUM ----- */

#endif /* SPECTRUM_H_ */
