/*
 * run.cu
 *
 *      Author: David Lawrie
 */

//currently using separate compilation which is a little slower than whole program compilation because a Rand1 function is not inlined and used in multiple sources (objects)
#include "go_fish.h"
#include "sfs.h"
#include "run.h"

using namespace std;
using namespace GO_Fish;
using namespace SFS;

void run_speed_test()
{
	//----- warm up scenario parameters -----
    float gamma = 0; //effective selection
	float h = 0.5; //dominance
	float F = 0.0; //inbreeding
	int N_ind = pow(10.f,5)*(1+F); //number of individuals in population, set to maintain consistent effective number of chromosomes
	float s = gamma/(2*N_ind); //selection coefficient
	float mu = pow(10.f,-9); //per-site mutation rate
	int total_number_of_generations = pow(10.f,5);//36;//50;//
	float L = 2*pow(10.f,7); //number of sites
	float m = 0.00; //migration rate
	int num_pop = 1; //number of populations
	int seed1 = 0xbeeff00d; //random number seeds
	int seed2 = 0xdecafbad;
	bool printSFS = true; //calculate and print out the SFS
	//----- end warm up scenario parameters -----

	//----- warm up GPU -----
	sim_result * a = run_sim(const_parameter(mu), const_demography(N_ind), const_equal_migration(m,num_pop), const_selection(s), const_parameter(F), const_parameter(h), total_number_of_generations, L, num_pop, seed1, seed2, do_nothing(), do_nothing(), 0, true);
	cout<<endl<<"final number of mutations: " << a[0].num_mutations << endl;

	//----- print allele counts x to x+y of warm up GPU scenario -----
	int start_index = 0;
	int print_num = 50;
	if(printSFS){
		SFS::sfs mySFS = SFS::site_frequency_spectrum(a[0],0);
		cout<< "allele count\t# mutations"<< endl;
		for(int printIndex = start_index; printIndex < min((mySFS.num_samples[0]-start_index),start_index+print_num); printIndex++){ cout<< (printIndex) << "\t" << mySFS.frequency_spectrum[printIndex] <<endl;}
	}
	//----- end print allele counts x to x+y of warm up GPU scenario -----
	delete [] a;

	a = run_sim(const_parameter(mu), const_demography(N_ind), const_equal_migration(m,num_pop), const_selection(s), const_parameter(F), const_parameter(h), total_number_of_generations, L, num_pop, seed1, seed2, do_nothing(), do_nothing(), 0, true);
	delete [] a;
	//----- end warm up GPU -----

	//----- speed test scenario parameters -----
    cudaEvent_t start, stop;
    float elapsedTime;
    int num_iter = 10;
    int compact_rate = 35;

    gamma = 0;
    h = 0.0;
    F = 1.0;
    N_ind = pow(10.f,5)*(1+F);
    s = gamma/(2*N_ind);
    mu = pow(10.f,-9);
    total_number_of_generations = pow(10.f,3);
    L = 1*2*pow(10.f,7);
    num_pop = 1;
    m = 0.0;
	seed1 = 0xbeeff00d; //random number seeds
	seed2 = 0xdecafbad;
	//----- end speed test scenario parameters -----

    //----- speed test -----
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for(int i = 0; i < num_iter; i++){
		sim_result * b = run_sim(const_parameter(mu), const_demography(N_ind), const_equal_migration(m,num_pop), const_selection(s), const_parameter(F), const_parameter(h), total_number_of_generations, L, num_pop, seed1, seed2, do_nothing(), do_nothing(), 0, true, sim_result(), compact_rate);
		if(i==0){ cout<<endl<<"final number of mutations: " << b[0].num_mutations << endl; }
		delete [] b;
	}

	elapsedTime = 0;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("time elapsed: %f\n\n", elapsedTime/num_iter);
	//----- end speed test -----

	cudaDeviceSynchronize();
	cudaDeviceReset();
}
