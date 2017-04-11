/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include "go_fish.cuh"
#include "spectrum.h"
#include <vector>

/*
 * population 0 starts of in mutation-selection equilibrium at size N_ind
 * population 0 grows to size 2*N_ind at generation 1
 * at generation 0.01*N_ind, population 1 splits off from population 0
 * population 1's initial size is 0.05*N_ind
 * population 1 grows exponentially to size 5*N_ind for 0.09*N_ind generations
 *
 * migration between populations is at rate 1/(2*N_ind) starting at generation 0.01*N_ind+1
 *
 * selection is weakly deleterious (gamma = -4), mutations are co-dominant (h = 0.5), populations are outbred (F = 0)
 */

void run_validation_test(){
	typedef Sim_Model::demography_constant dem_const;
	typedef Sim_Model::demography_population_specific<dem_const,dem_const> dem_pop_const;
	typedef Sim_Model::demography_piecewise<dem_pop_const,dem_pop_const> init_expansion;
	typedef Sim_Model::demography_exponential_growth exp_growth;
	typedef Sim_Model::demography_population_specific<dem_const,exp_growth> dem_pop_const_exp;

	typedef Sim_Model::migration_constant_equal mig_const;
	typedef Sim_Model::migration_constant_directional<mig_const> mig_dir;
	typedef Sim_Model::migration_constant_directional<mig_dir> mig_split;
	typedef Sim_Model::migration_piecewise<mig_const,mig_split> split_pop0;
	float scale_factor = 1.0f;											//entire simulation can be scaled up or down with little to no change in resulting normalized SFS

	GO_Fish::allele_trajectories b;
	b.sim_input_constants.num_populations = 2; 							//number of populations
	b.sim_input_constants.num_generations = scale_factor*pow(10.f,3)+1;	//1,000 generations

	Sim_Model::F_mu_h_constant codominant(0.5f); 						//dominance (co-dominant)
	Sim_Model::F_mu_h_constant outbred(0.f); 							//inbreeding (outbred)
	Sim_Model::F_mu_h_constant mutation(pow(10.f,-9)/scale_factor); 	//per-site mutation rate 10^-9

	int N_ind = scale_factor*pow(10.f,4);								//initial number of individuals in population
	dem_const pop0(N_ind);
	dem_const pop1(0);
	dem_pop_const gen0(pop0,pop1,1);									//intial population size of N_ind for pop 0 and 0 for pop 1
	dem_const pop0_final(2*N_ind);
	dem_pop_const gen1(pop0_final,pop1,1);
	init_expansion gen_0_1(gen0,gen1,1);								//population 0 grows to size 2*N_ind
	exp_growth pop1_gen100((log(100.f)/(scale_factor*900.f)),0.05*N_ind,scale_factor*100);
	dem_pop_const_exp gen100(pop0_final,pop1_gen100,1);					//population 1 grows exponentially from size 0.05*N_ind to 5*N_ind
	Sim_Model::demography_piecewise<init_expansion,dem_pop_const_exp> demography_model(gen_0_1,gen100,scale_factor*100);

	mig_const no_mig_pop0;
	mig_dir no_pop1_gen0(0.f,1,1,no_mig_pop0);
	mig_split create_pop1(1.f,0,1,no_pop1_gen0);						//individuals from population 0 migrate to form population 1
	split_pop0 migration_split(no_mig_pop0,create_pop1,scale_factor*100);
	float mig = 1.f/(2.f*N_ind);
	mig_const mig_prop(mig,b.sim_input_constants.num_populations);		//constant and equal migration between populations
	Sim_Model::migration_piecewise<split_pop0,mig_const> mig_model(migration_split,mig_prop,scale_factor*100+1);

	float gamma = -4; 													//effective selection
	Sim_Model::selection_constant weak_del(gamma,demography_model,outbred);

	b.sim_input_constants.compact_interval = 30;						//compact interval
	b.sim_input_constants.num_sites = 100*2*pow(10.f,7); 				//number of sites
	int sample_size = 1001;												//number of samples in SFS

	int num_iter = 50;													//number of iterations
    Spectrum::SFS my_spectra;

    cudaEvent_t start, stop;											//CUDA timing functions
    float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float avg_num_mutations = 0;
	float avg_num_mutations_sim = 0;
	std::vector<std::vector<float> > results(num_iter); 				//storage for SFS results
	for(int j = 0; j < num_iter; j++){ results[j].reserve(sample_size); }

	for(int j = 0; j < num_iter; j++){
		if(j == num_iter/2){ cudaEventRecord(start, 0); } 				//use 2nd half of the simulations to time simulation runs + SFS creation

		b.sim_input_constants.seed1 = 0xbeeff00d + 2*j; 				//random number seeds
		b.sim_input_constants.seed2 = 0xdecafbad - 2*j;
		GO_Fish::run_sim(b, mutation, demography_model, mig_model, weak_del, outbred, codominant, Sim_Model::bool_off(), Sim_Model::bool_off());
		Spectrum::site_frequency_spectrum(my_spectra,b,0,1,sample_size);

		avg_num_mutations += ((float)my_spectra.num_mutations)/num_iter;
		avg_num_mutations_sim += b.maximal_num_mutations()/num_iter;
		for(int i = 0; i < sample_size; i++){ results[j][i] = my_spectra.frequency_spectrum[i]; }
	}

	elapsedTime = 0;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//output SFS simulation results
	std::cout<<"SFS :"<<std::endl<< "allele count\tavg# mutations\tstandard dev\tcoeff of variation (aka relative standard deviation)"<< std::endl;
	for(int i = 1; i < sample_size; i++){
		double avg = 0;
		double std = 0;
		float num_mutations;
		for(int j = 0; j < num_iter; j++){ num_mutations = b.num_sites() - results[j][0]; avg += results[j][i]/(num_iter*num_mutations); }
		for(int j = 0; j < num_iter; j++){ num_mutations = b.num_sites() - results[j][0]; std += 1.0/(num_iter-1)*pow(results[j][i]/num_mutations-avg,2); }
		std = sqrt(std);
		std::cout<<i<<"\t"<<avg<<"\t"<<std<<"\t"<<(std/avg)<<std::endl;
	}

	std::cout<<"\nnumber of sites in simulation: "<< b.num_sites() <<"\ncompact interval: "<< b.last_run_constants().compact_interval;
	std::cout<<"\naverage number of mutations in simulation: "<<avg_num_mutations_sim<<"\naverage number of mutations in SFS: "<<avg_num_mutations<<"\ntime elapsed (ms): "<< 2*elapsedTime/num_iter<<std::endl;
}

////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

int main(int argc, char **argv) { run_validation_test(); }
