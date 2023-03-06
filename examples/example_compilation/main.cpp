/*
 * main.cpp
 *
 *      Author: David Lawrie
 */

#include "run.h"
#include <fstream>

int main(int argc, char **argv){

	float migration_rate = 0.005;								    //migration proportion (equal between populations)
	auto a = run_migration_equilibrium_simulation(migration_rate); 	//run simulation
	auto b = a; 						                            //demonstrates copy-assignment, copies a to b
	a.reset();														//demonstrates the freeing of memory held by a
	GO_Fish::allele_trajectories c(b.last_run_constants(), b.dump_sampled_generations(), b.dump_total_generated_mutations_samples(), b.dump_popsize(), b.extinct_view(), b.allele_count_view(), b.mutID_span());
																	//^ demonstrates constructing allele_trajectories c through contiguous containers generated from b
	/* --- output simulation information --- */
	std::ofstream outfile;
	outfile.open("allele_traj.out");
	outfile<<c;														//demonstrates printing allele_trajectory c to file
	outfile.close();

	std::cout<<std::endl<<"number of time samples: " << c.num_time_samples();
	std::cout<<std::endl<<"mutations in first time sample: " << c.num_mutations_time_sample(0) <<std::endl<<"mutations in final time sample: " << c.maximal_num_mutations() << std::endl; //mutations in final time sample >= mutations in first time sample as all mutations in the latter were preserved by sampling

	//prints the first 10 mutations output from burn-in simulation
	std::cout<<std::endl<<"mutations from burn-in simulation\n";
	int mutation_range_begin = 0; int mutation_range_end = mutation_range_begin+10;
	std::cout<<"ID\tstart count pop 1\tstart count pop 2\tfinal count pop 1\tfinal count pop 2"<<std::endl;
	for(int i = mutation_range_begin; i < mutation_range_end; i++){ std::cout<<c.mutation_ID(i)<<"\t"<<c.allele_count(0,0,i)<<"\t"<<c.allele_count(0,1,i)<<"\t"<<c.allele_count(1,0,i)<<"\t"<<c.allele_count(1,1,i)<<std::endl; }

	//prints the first 10 mutations output from scenario simulation
	std::cout<<std::endl<<"mutations from scenario simulation\n";
	mutation_range_begin = c.num_mutations_time_sample(0); mutation_range_end = mutation_range_begin+10;
	std::cout<<"ID\tstart count pop 1\tstart count pop 2\tfinal count pop 1\tfinal count pop 2"<<std::endl;
	for(int i = mutation_range_begin; i < mutation_range_end; i++){ std::cout<<c.mutation_ID(i)<<"\t"<<c.allele_count(0,0,i)<<"\t"<<c.allele_count(0,1,i)<<"\t"<<c.allele_count(1,0,i)<<"\t"<<c.allele_count(1,1,i)<<std::endl; }

	GO_Fish::allele_trajectories d;
	std::ifstream infile;
	infile.open("allele_traj.out");
	infile>>d;														//demonstrates reading allele_trajectory d from file
	infile.close();

	std::cout<<std::endl<<"number of time samples: " << d.num_time_samples();
	std::cout<<std::endl<<"mutations in first time sample: " << d.num_mutations_time_sample(0) <<std::endl<<"mutations in final time sample: " << d.maximal_num_mutations() << std::endl; //mutations in final time sample >= mutations in first time sample as all mutations in the latter were preserved by sampling

	//prints the first 10 mutations output from burn-in simulation
	std::cout<<std::endl<<"mutations from burn-in simulation\n";
	mutation_range_begin = 0; mutation_range_end = mutation_range_begin+10;
	std::cout<<"ID\tstart count pop 1\tstart count pop 2\tfinal count pop 1\tfinal count pop 2"<<std::endl;
	for(int i = mutation_range_begin; i < mutation_range_end; i++){ std::cout<<d.mutation_ID(i)<<"\t"<<d.allele_count(0,0,i)<<"\t"<<d.allele_count(0,1,i)<<"\t"<<d.allele_count(1,0,i)<<"\t"<<d.allele_count(1,1,i)<<std::endl; }

	//prints the first 10 mutations output from scenario simulation
	std::cout<<std::endl<<"mutations from scenario simulation\n";
	mutation_range_begin = d.num_mutations_time_sample(0); mutation_range_end = mutation_range_begin+10;
	std::cout<<"ID\tstart count pop 1\tstart count pop 2\tfinal count pop 1\tfinal count pop 2"<<std::endl;
	for(int i = mutation_range_begin; i < mutation_range_end; i++){ std::cout<<d.mutation_ID(i)<<"\t"<<d.allele_count(0,0,i)<<"\t"<<d.allele_count(0,1,i)<<"\t"<<d.allele_count(1,0,i)<<"\t"<<d.allele_count(1,1,i)<<std::endl; }
	/* --- end output simulation information --- */
}
