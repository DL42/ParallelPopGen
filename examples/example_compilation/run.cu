/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include "go_fish.cuh"
#include "run.h"

//scenario: two populations start in mutation-selection-migration equilibrium with weakly deleterious mutations, then simulate what happens after mutations in population 1 become beneficial (1,000 generations)
//migration set in main.cpp
void run_migration_equilibrium_simulation(GO_Fish::allele_trajectories & a, float migration_rate){
	using namespace Sim_Model;								//using namespace Sim_Model (avoids Sim_Model::, but compiler can get confused if multiple functions from different namespaces have same name and Sim_Model:: shows ownership)
	using namespace GO_Fish;								//using namespace GO_Fish (avoids GO_Fish::, but compiler can get confused if multiple functions from different namespaces have same name and GO_Fish:: shows ownership)

	a.sim_input_constants.num_sites = 20*pow(10.f,7); 		//number of sites
	a.sim_input_constants.num_populations = 2;				//number of populations

	a.sim_input_constants.init_mse = false; 				//start from blank simulation
	F_mu_h_constant mutation(pow(10.f,-9)); 				//per-site mutation rate
	F_mu_h_constant inbreeding(1.f); 						//constant inbreeding
	demography_constant demography(1000); 					//number of individuals in both populations
	migration_constant_equal migration(migration_rate,a.sim_input_constants.num_populations); //constant migration rate
	selection_constant deleterious(-1.f/1000.f); 			//constant selection coefficient (weakly deleterious)
	F_mu_h_constant dominance(0.f); 						//constant allele dominance (ignored as population is fully inbred)
	bool_off dont_preserve; 								//don't preserve mutations
	bool_off dont_sample; 									//don't sample generations
	a.sim_input_constants.compact_interval = 100;			//interval between compacts

	a.sim_input_constants.num_generations = 2*pow(10.f,4); 	//burn-in simulation to achieve migration equilibrium 20,0000 generations
	run_sim(a,mutation,demography,migration,deleterious,inbreeding,dominance,dont_preserve,dont_sample); //only sample final generation

	allele_trajectories c(a); 								//copy constructor, copies a to c (not actually needed for this simulation, just showing it is possible)

	bool_on sample; 										//sample generation
	bool_pulse<bool_off,bool_on> sample_strategy(dont_sample,sample,0,a.sim_input_constants.num_generations); //sample starting generation of second simulation (i.e. last generation of burn-in simulation)
	a.sim_input_constants.num_generations = pow(10.f,3);	//scenario simulation 1,0000 generations
	a.sim_input_constants.prev_sim_sample = 0; 				//start from previous simulation time sample 0
	selection_constant beneficial(20.f/1000.f);				//constant selection coefficient (beneficial)
	selection_population_specific<selection_constant,selection_constant> selection_model(deleterious,beneficial,1); //selection in population 1
	run_sim(a,mutation,demography,migration,selection_model,inbreeding,dominance,dont_preserve,sample_strategy,a); //scenario simulation, start from migration equilibrium, sample both start and final generations
}
