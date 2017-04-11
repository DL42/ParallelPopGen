/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include "go_fish.cuh"

void run_speed_test()
{
	//----- speed test scenario parameters -----
	GO_Fish::allele_trajectories a;
	a.sim_input_constants.num_populations = 1; 									//1 population
	Sim_Model::F_mu_h_constant mutation(pow(10.f,-9)); 							//per-site mutation rate 10^-9
	Sim_Model::F_mu_h_constant inbreeding(1.f); 								//constant inbreeding (fully inbred)
	Sim_Model::demography_constant demography(pow(10.f,5)*(1+inbreeding(0,0)));	//200,000 haploid individuals in population, set to maintain consistent effective number of chromosomes invariant w.r.t. inbreeding
	Sim_Model::migration_constant_equal migration; 								//constant, 0, migration rate
	Sim_Model::selection_constant selection(0); 								//constant, neutral, selection coefficient
	Sim_Model::F_mu_h_constant dominance(0.f); 									//constant allele dominance (effectively ignored since F = 1)
    a.sim_input_constants.num_generations = pow(10.f,3);						//1,000 generations in simulation
    a.sim_input_constants.seed1 = 0xbeeff00d; 									//random number seeds
    a.sim_input_constants.seed2 = 0xdecafbad;

    a.sim_input_constants.num_sites = 20*2*pow(10.f,7);							//number of sites
    a.sim_input_constants.compact_interval = 15;								//compact interval (in general: decrease compact interval for larger number of sites)
	//----- end speed test scenario parameters -----

    //----- speed test -----
    cudaEvent_t start, stop;													//CUDA timing functions
    float elapsedTime;
    int num_iter = 20;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for(int i = 0; i < num_iter; i++){
		if(i == num_iter/2){ cudaEventRecord(start, 0); }						//use half of the simulations to warm-up GPU, the other half to time simulation runs
		GO_Fish::run_sim(a,mutation,demography,migration,selection,inbreeding,dominance,Sim_Model::bool_off(),Sim_Model::bool_off());
	}

	elapsedTime = 0;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	std::cout<<"number of sites:\t"<< a.sim_input_constants.num_sites<<std::endl<< "compact interval:\t"<< a.sim_input_constants.compact_interval<<std::endl<<"number of mutations:\t"<<a.maximal_num_mutations()<<std::endl<<"time elapsed (ms):\t"<<2*elapsedTime/num_iter<<std::endl;
	//----- end speed test -----
	//
}

////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

int main(int argc, char **argv){ run_speed_test(); }
