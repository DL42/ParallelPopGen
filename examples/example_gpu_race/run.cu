/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include "go_fish.cuh"
#include <chrono>
#include <fstream>
#include <future>

void run_multipop_speed_test(int device, int num_populations, int length_factor, int compact_interval)
{
	//----- speed test scenario parameters -----
	GO_Fish::allele_trajectories a;
	GO_Fish::sim_constants input;
	input.num_populations = num_populations; 									//1 population
	Sim_Model::constant_parameter mutation(pow(10.f,-9)); 						//per-site mutation rate 10^-9
	Sim_Model::constant_parameter inbreeding(1.f); 								//constant inbreeding (fully inbred)
	Sim_Model::constant_parameter demography(pow(10.f,5)*(1+inbreeding(0,0)));	//200,000 haploid individuals in population, set to maintain consistent effective number of chromosomes invariant w.r.t. inbreeding
	Sim_Model::constant_parameter migration(0.05f); 							//constant migration rate
	Sim_Model::constant_parameter selection(0); 								//constant, neutral, selection coefficient
	Sim_Model::constant_parameter dominance(0.f); 								//constant allele dominance (effectively ignored since F = 1)
	input.num_generations = pow(10.f,3);										//1,000 generations in simulation
	input.seed1 = 0xbeeff00d; 													//random number seeds
	input.seed2 = 0xdecafbad;
	input.device = device;														//GPU device to run simulation on

	input.num_sites = length_factor*2*pow(10.f,7)/input.num_populations;		//number of sites
	input.compact_interval = compact_interval;									//compact interval (in general: decrease compact interval for larger number of sites)
	//----- end speed test scenario parameters -----

    //----- speed test -----
    int num_iter = 20;
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

	for(int i = 0; i < num_iter; i++){
		if(i == num_iter/2){ start = std::chrono::high_resolution_clock::now(); }						//use half of the simulations to warm-up GPU, the other half to time simulation runs
		a = GO_Fish::run_sim(input,mutation,demography,migration,selection,inbreeding,dominance,Sampling::off());
	}

	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> elapsed_ms = end - start;
	std::cout<<"device:\t"<<input.device<<std::endl<<"number of populations:\t"<<input.num_populations<<std::endl<<"number of sites:\t"<<input.num_sites<<std::endl<< "compact interval:\t"<<input.compact_interval<<std::endl<<"number of mutations:\t"<<a.maximal_num_mutations()<<std::endl<<"time elapsed (ms):\t"<< 2*elapsed_ms.count()/num_iter<<std::endl<<std::endl;

	//----- end speed test -----
}

////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

int main(int argc, char **argv){
	auto func = [](int device){ run_multipop_speed_test(device, 2, 90, 40); };
	auto future_0 = std::async(std::launch::async,func,0);
	auto future_1 = std::async(std::launch::async,func,1);
	future_0.wait();
	future_1.wait();
}
