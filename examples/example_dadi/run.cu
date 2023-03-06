/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include "go_fish.cuh"
#include "spectrum.h"
#include <chrono>

/*
 * initial population size of N_ind for pop 0 scales the simulation up or down with small changes in resulting normalized SFS
 * (the larger the population size, N_ind, the closer the results are to the DaDi diffusion approximation)
 * since we are comparing shape of SFS, effective per-site mutation rate, mu, and number of sites, num_sites likewise scale the simulation
 * (although the total number of mutations does affect the variance in the estimation of the SFS)
 *
 * population 0 starts of in mutation-selection equilibrium at size N_ind
 * population 0 grows to size nu1F*N_ind at generation 1
 * at generation 2*Tp*N_ind+1, population 1 splits off from population 0
 * population 1's initial size is nu2B*N_ind
 * population 1 grows exponentially to size nu2F*N_ind for 2*T*N_ind generations
 *
 * migration between populations is at rate m/(2*N_ind) starting at generation 2*Tp*N_ind+2
 * mutation rate per site is at rate mu/(2*N_ind)
 *
 * selection is weakly deleterious (gamma = -4), populations are outbred (F = 0), mutations are co-dominant (h = 0.5)
 * Sampling::off() means only the last generation is sampled
 */

auto run_model(float nu1F, float nu2B, float nu2F, float m, float Tp, float T, float N_ind, float num_sites, float mu, unsigned int seed1, unsigned int seed2){
	using cp = Sim_Model::constant_parameter;
	using exp_growth = Sim_Model::exponential_generation_parameter;

	Sim_Model::effective_parameter eff(N_ind,0);
	unsigned int start_growth = round(2*Tp*N_ind)+1;							//DaDi generations are scaled in units of chromosomes
	unsigned int num_gens = round(2*T*N_ind)+start_growth;	    				//scale_factor = 1.f, Tp = 0.005, T = 0.045, then num_gens = 1,001 simulation generations => initialize population 0 in MSE @ generation 0, simulate for generation [1,1001]

	//default population size (population 0) N_ind, generation 0 population 1 starts at size 0, at generation 1 population 0 size changes from N_ind to nu1F*N_ind, at generation start_growth population 1 grows exponentially from nu2B*N_ind to nu2F*N_ind
	auto demography_model = Sim_Model::make_piecewise_population_specific_model(cp(N_ind),0,1,cp(0),1,0,cp(nu1F*N_ind),start_growth,1,exp_growth(nu2B*N_ind, nu2F*N_ind, start_growth, num_gens));
	//default is no migration, at generation start_growth population 1 splits off from population 0 and they have constant migration at effective rate m between them thereafter
	auto migration_model =  Sim_Model::make_piecewise_directional_migration_model(cp(0),start_growth,0,1,cp(1),start_growth+1,0,1,cp(eff(m)),start_growth+1,1,0,cp(eff(m)));

	return GO_Fish::run_sim({seed1,seed2,num_gens,num_sites,2,true,0,GO_Fish::compact_scheme::compact_all,35,1}, cp(eff(mu)), demography_model, migration_model, cp(eff(-4.f)), cp(0), cp(0.5), Sampling::off());
}

void print_sfs(float nu1F, float nu2B, float nu2F, float m, float Tp, float T, float N_ind, float num_sites, float mu){
	auto num_iter = 1;															//number of iterations
	auto sample_size = 1001;													//number of samples in SFS
    Spectrum::SFS my_spectra;
    GO_Fish::allele_trajectories b;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();

	for(auto j = 0; j < num_iter; j++){
		b = run_model(nu1F, nu2B, nu2F, m, Tp, T, N_ind, num_sites, mu, 0xbeeff00d + 2*j, 0xdecafbad - 2*j);
		Spectrum::site_frequency_spectrum(my_spectra,b,0,1,sample_size);
	}
	
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> elapsed_ms = end - start;
	std::cout<<"\nnumber of mutations in simulation: " << b.maximal_num_mutations() << "\nnumber of mutations in SFS: "<< my_spectra.num_mutations <<"\ntime elapsed (ms): "<< elapsed_ms.count()/num_iter << std::endl << std::endl;
	for(int i = 1; i < sample_size; i++){ std::cout<< my_spectra.frequency_spectrum[i]/my_spectra.num_mutations << std::endl; }
}

////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

int main(int argc, char **argv) { print_sfs(2, 0.05, 5, 1, 0.005, 0.045, 1*pow(10.f,4), 2*pow(10.f,7), 2*pow(10.f,-5)); }
