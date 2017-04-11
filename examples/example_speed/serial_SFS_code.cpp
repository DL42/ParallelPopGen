//============================================================================
// Name        : simulator.cpp
// Author      : David Lawrie
// Version     :
// Copyright   : Your copyright notice
// Description : Wright-Fisher simulation
//============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <time.h>
#include <chrono>

using namespace std;


struct mutation{
	double sel_coeff;
	double freq;
	bool track;
	int ID;
};

int demography_model(int, int);
vector<double> mut_sel_equil_freq(double, double, int);
void initialize_equilibrium(vector<mutation>&, vector<double> &, double, double, int, bool, int*, gsl_rng*);
mutation new_mutation(double, double, bool, int);
void add_new_mutations(vector<mutation>&, double, double, double, int, bool, int*,gsl_rng*);
int binom_approx(gsl_rng*, double, double, double);
int binom_approx_2(gsl_rng*, double, double, double);
mutation new_freq(mutation, int, gsl_rng*, bool, bool);


int main() {
	for(int iter = 0; iter < 1; iter++){

	auto start = std::chrono::system_clock::now();
	vector<mutation>* current_generation = new vector<mutation>;
	vector<mutation>* next_generation = new vector<mutation>;

	int num_ben_fixations = 0;
	int num_del_fixations = 0;
	int num_neu_fixations = 0;
	int* ID = new int;
	*ID = 0;

	gsl_rng* rng;
	rng = gsl_rng_alloc (gsl_rng_taus); //gsl_rng_alloc(gsl_rng_mrg);//
	gsl_rng_set(rng,iter*3);//11);//1000);//
	int generations = 0;
	int N = demography_model(generations, 0); //population size
	double initial_gamma = 0.0;
	double sel_coeff = initial_gamma/(4.0*N);
	int runs = 1000; //100000;//number of total runs after burn-in
	int burn_in = 0;
	double mu_site = 1*pow(10,-9);
	double sites = 10*2*pow(10,7);//100*2*pow(10,7);//7.5*2*pow(10,7);//
	double mu = sites*mu_site;
	bool binom_approx = true;
	bool track = false;

	//initialize current generation with mutation-selection equilibrium
	vector<double> frequencies;
	frequencies = mut_sel_equil_freq(mu, sel_coeff, N);
	initialize_equilibrium(*current_generation, frequencies, sites, sel_coeff, N, track, ID, rng);
	unsigned int start_popsize = (*current_generation).size();
	int end = burn_in+runs;

	int track_ben_gained = 0;
	int track_del_gained = 0;
	int track_neu_gained = 0;

	int track_ben_lost = 0;
	int track_del_lost = 0;
	int track_neu_lost = 0;

	//run sim
	for(generations = 1; generations <= end; generations++){
		N = demography_model(generations, N);
		while(!current_generation->empty()){
			mutation temp = new_freq(current_generation->back(), N, rng, false, binom_approx);
			current_generation->pop_back();

			if(temp.track == true){
				if(temp.freq == 1){
					if(temp.sel_coeff > 0){ track_ben_gained++; }
					else if(temp.sel_coeff < 0){ track_del_gained++; }
					else{ track_neu_gained++; }
				}else if(temp.freq == 0){
					if(temp.sel_coeff > 0){ track_ben_lost++; }
					else if(temp.sel_coeff < 0){ track_del_lost++; }
					else{ track_neu_lost++; }
				}
			}

			if(temp.freq == 1 && generations > burn_in){
				if(temp.sel_coeff > 0){ num_ben_fixations++; }
				else if(temp.sel_coeff < 0){ num_del_fixations++; }
				else{ num_neu_fixations++; }
			}else if(temp.freq > 0 && temp.freq < 1){ next_generation->push_back(temp); }
		}

		add_new_mutations(*next_generation, mu, sites, sel_coeff, N, track, ID, rng);
		if(generations == N){ track = false; } //track first N generations
		vector<mutation>* temp = current_generation;
		current_generation = next_generation;
		next_generation = temp;
	}

	auto duration = std::chrono::duration_cast< std::chrono::milliseconds> (std::chrono::system_clock::now() - start);
	cout <<iter << "\t" << start_popsize << "\t" << (*current_generation).size() << "\t" << duration.count() <<endl;

	gsl_rng_free(rng);
	delete current_generation;
	delete next_generation;
	delete ID;
	}
	return 0;
}

int demography_model(int generations, int current_N){
	return (pow(10,5));//2*pow(10,4);//3000;//(500*pow(10,5));//
}

vector<double> mut_sel_equil_freq(double mu, double sel_coeff, int N){
	vector<double> frequencies((2*N-1),0);
	double p = mu;

	for(int j = 1; j < 2.0*N; j++){
		double freq = ((double)j)/(2.0*N);
		double q = 0;
		if(sel_coeff != 0){
			double r =  (1-exp(-4.0*N*sel_coeff*(1-freq)));
			double s = (1-exp(-4.0*N*sel_coeff))*freq*(1-freq);
			q = r/s;
		}else{ q = 1/freq; }

		frequencies[(j-1)] = 2*p*q;
	}

	return frequencies;
}

void initialize_equilibrium(vector<mutation>& generation, vector<double> & frequencies, double sites, double sel_coeff, int N, bool track, int* ID, gsl_rng* rng){
	int length = frequencies.size();

	for(int j = 0; j < length; j++){
		int num = binom_approx_2(rng,frequencies[j],frequencies[j],2*N*sites);
		for(int k = 0; k < num; k++){
			mutation temp = new_mutation(sel_coeff, ((double)j)/(2.0*N), track, *ID);
			generation.push_back(temp);
			(*ID)++;
		}
	}

}


mutation new_mutation(double selection, double freq, bool trackme, int ID){
	mutation temp;
	temp.sel_coeff = selection;
	temp.track = trackme;
	temp.freq = freq;
	temp.ID = ID;
	return temp;
}

void add_new_mutations(vector<mutation>& generation, double mu, double sites, double sel_coeff, int N, bool track, int* ID, gsl_rng* rng){
	double p = mu;
	int num = gsl_ran_poisson(rng, (2*N)*p);//binom_approx_2(rng, (2*N)*p, (2*N)*p, 2*N*sites);//
	//cout<<endl<<2*N*sites<<" "<< (2*N)*p << " "<< p << " "<< num;
	for(int k = 0; k < num; k++){
		mutation temp = new_mutation(sel_coeff, (1.0/(2.0*N)), track, *ID);
		generation.push_back(temp);
		(*ID)++;
	}
}

int binom_approx(gsl_rng* rng, double mean, double var, double N){
	if(mean <= 6){ return gsl_ran_poisson(rng, mean); }
	else if(mean >= N-6){ return N - gsl_ran_poisson(rng, N-mean); } //flip side of binomial, when 1-p is small
	return round(gsl_ran_gaussian(rng,sqrt(var))+mean);
}

int binom_approx_2(gsl_rng* rng, double mean, double var, double N){
	if(mean <= 6){ return gsl_ran_poisson(rng, mean); }
	else if(mean >= N-6){ return N - gsl_ran_poisson(rng, N-mean); } //flip side of binomial, when 1-p is small
	return round(gsl_cdf_gaussian_Pinv(gsl_rng_uniform(rng),sqrt(var))+mean);
}


mutation new_freq(mutation a, int N, gsl_rng* rng, bool sample, bool binom_approx){
	mutation temp;
	temp.sel_coeff = a.sel_coeff;
	temp.track = a.track;
	double sel = a.sel_coeff;
	temp.freq = 0;
	double i = a.freq;
	double p;
	if(!sample){ p = (1+sel)*i/((1+sel)*i + 1*(1.0-i)); }
	else{ p = i; }
	//double p = ((1+sel)*pow(i,2)+(1+0.5*sel)*i*(1-i))/((1+sel)*pow(i,2) + 2*(1+0.5*sel)*i*(1-i) + pow((1-i),2));
	int j;
	if(binom_approx){ j = binom_approx_2(rng, (2*N)*p, (2*N)*p*(1-p), 2*N); }
	else{ j = gsl_ran_binomial (rng, p, (2*N)); }

	temp.freq = ((double)j)/(2.0*N);

	return temp;
}
