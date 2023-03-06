/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include "go_fish.cuh"
#include "run.h"

//scenario: two populations start in mutation-selection-migration equilibrium with weakly deleterious mutations, then simulate what happens after mutations in population 1 become beneficial (1,000 generations)
//migration set in main.cpp
GO_Fish::allele_trajectories run_migration_equilibrium_simulation(float migration_rate){
	using namespace Sim_Model;								//using namespace Sim_Model (avoids Sim_Model::, but compiler can get confused if multiple functions from different namespaces have same name and Sim_Model:: shows ownership)
	using namespace Sampling;								//using namespace Sampling (avoids Sampling::, but compiler can get confused if multiple functions from different namespaces have same name and Sampling:: shows ownership)
	using namespace GO_Fish;								//using namespace GO_Fish (avoids GO_Fish::, but compiler can get confused if multiple functions from different namespaces have same name and GO_Fish:: shows ownership)

	sim_constants input;

	//burn-in simulation
	input.num_sites = 20*pow(10.f,7); 						//number of sites
	input.num_populations = 2;								//number of populations
	input.init_mse = false; 				    			//start from blank simulation
	input.compact_interval = 100;							//interval between compacts
	input.num_generations = 2*pow(10.f,4); 					//burn-in simulation to achieve migration equilibrium 20,0000 generations

	constant_parameter mutation(pow(10.f,-9)); 				//per-site mutation rate
	constant_parameter inbreeding(1.f); 					//constant inbreeding
	constant_parameter demography(1000); 					//number of individuals in both populations
	constant_parameter migration(migration_rate); 			//constant migration rate
	constant_parameter deleterious(-1.f/1000.f); 			//constant selection coefficient (weakly deleterious)
	constant_parameter dominance(0.f); 						//constant allele dominance (ignored as population is fully inbred)
	off dont_sample; 										//don't sample generations

	auto a = run_sim(input,mutation,demography,migration,deleterious,inbreeding,dominance,dont_sample); //only sample final generation

	allele_trajectories c = std::move(a); 					//demonstrates move-assignment, moves a to c, a is now reset

	//second simulation
	input.num_generations = pow(10.f,3);					//scenario simulation 1,000 generations
	input.prev_sim_sample = 0; 								//start from previous simulation time sample 0
	pulse sample_strategy({0},c.sampled_generation(input.prev_sim_sample)); //sample starting generation of second simulation (i.e. last generation of burn-in simulation)
	constant_parameter beneficial(20.f/1000.f);				//constant selection coefficient (beneficial)
	auto selection_model = make_population_specific_evolution_model(deleterious,1,beneficial); //selection in population 1
	return run_sim(std::move(input),mutation,demography,migration,selection_model,inbreeding,dominance,sample_strategy,c); //scenario simulation, start from migration equilibrium, sample both start and final generations
	//input no longer needed, can move it into run_sim
}
