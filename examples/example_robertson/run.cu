/*
 * run.cu
 *
 *      Author: David Lawrie
 */

#include "go_fish.cuh"
#include "spectrum.h"
#include <chrono>
#include <cstdlib>

auto run_mse_robertson_model(float effect_size, float variance, float N, float num_sites, float mu, unsigned int seed1, unsigned int seed2){
	using cp = Sim_Model::constant_parameter;
	return GO_Fish::run_sim({seed1,seed2,0,num_sites,1}, cp(mu), cp(N), cp(0), Sim_Model::make_robertson_stabilizing_selection_model(effect_size,variance), cp(0), Sim_Model::make_robertson_stabilizing_dominance_model(), Sampling::off(), GO_Fish::allele_trajectories(), Sim_Model::robertson_stabilizing_mse_integrand());
}

void print_mse_robertson_sfs(int sample_size, float effect_size, float variance, float N, float num_sites, float mu){												
    Spectrum::SFS my_spectra_mse;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();
    //GO_Fish::allele_trajectories b = run_mse_robertson_model(effect_size, variance, N, num_sites, mu, rand(), rand());
    GO_Fish::allele_trajectories b = run_mse_robertson_model(effect_size, variance, N, num_sites, mu, 0xbeeff00d, 0xdecafbad);
	Spectrum::site_frequency_spectrum(my_spectra_mse,b,0,0,sample_size);

	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> elapsed_ms = end - start;
	std::cout<<"\n"<< variance << ",";
	for(int i = 1; i < sample_size; i++){ std::cout<< my_spectra_mse.frequency_spectrum[i]/my_spectra_mse.num_mutations << ","; }
}

auto run_mse_model(float S, float N, float num_sites, float mu, unsigned int seed1, unsigned int seed2){
	using cp = Sim_Model::constant_parameter;
	Sim_Model::effective_parameter eff(N,0);
	return GO_Fish::run_sim({seed1,seed2,0,num_sites,1}, cp(mu), cp(N), cp(0), cp(eff(S)), cp(0), cp(0.5), Sampling::off());
}

void print_mse_sfs(int sample_size, float selection, float N, float num_sites, float mu){												
    Spectrum::SFS my_spectra_mse;

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();

    GO_Fish::allele_trajectories b = run_mse_model(selection, N, num_sites, mu, 0xbeeff00d, 0xdecafbad);
	Spectrum::site_frequency_spectrum(my_spectra_mse,b,0,0,sample_size);

	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double,std::milli> elapsed_ms = end - start;
	std::cout<<"\n"<< selection << ",";
	for(int i = 1; i < sample_size; i++){ std::cout<< my_spectra_mse.frequency_spectrum[i]/my_spectra_mse.num_mutations << ","; }
}

////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////

int main(int argc, char **argv) { 
	std::vector<float> variance = {25, 58, 100, 200, 500};
	//srand(0xbeeff00d);
	//std::vector<float> variance = {50, 50, 50, 50, 50, 50, 50, 50, 50};
	for(const auto & vs: variance){
		print_mse_robertson_sfs(51, 1, vs, 1*pow(10.f,4), 100*pow(10.f,7), 10*pow(10.f,-9)); 
	}
	
	std::vector<float> selection = {25, 0, -10, -25, -85, -200};
	for(const auto & s: selection){
		print_mse_sfs(51, s, 1*pow(10.f,4), 100*pow(10.f,7), 10*pow(10.f,-9)); 
	}
}
