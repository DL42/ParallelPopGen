/*
 * inline_go_fish_data_struct.hpp
 *
 *      Author: David Lawrie
 *      inline function definitions for GO Fish data structures
 */

#ifndef INLINE_GOFISH_DATA_FUNCTIONS_HPP_
#define INLINE_GOFISH_DATA_FUNCTIONS_HPP_

#include "../_internal/ppp_types.hpp"

namespace GO_Fish{

inline allele_trajectories::allele_trajectories(const sim_constants & run_constants, size_t num_samples) : time_samples{num_samples}, mutations_ID{} { sim_run_constants = run_constants; }

template<typename cContainer_sample, typename cContainer_numgen_mut, typename cContainer_pop, typename cContainer_extinct, typename cContainer_allele, typename cContainer_mutID>
inline allele_trajectories::allele_trajectories(const sim_constants & run_constants, const cContainer_sample & sampled_generation, const cContainer_numgen_mut & total_generated_mutations, const cContainer_pop & pop_span_view, const cContainer_extinct & extinct_span_view, const cContainer_allele & allele_span_view, const cContainer_mutID & mutID_span) : allele_trajectories(run_constants, sampled_generation.size()) {
	if(sampled_generation.size() != total_generated_mutations.size() || total_generated_mutations.size() != pop_span_view.size() || pop_span_view.size() != extinct_span_view.size() || extinct_span_view.size() != allele_span_view.size()) { fprintf(stderr,"allele_trajectories error: different number of samples in each view\n"); exit(1); }
	if(run_constants.num_populations == 0){ fprintf(stderr,"allele_trajectories error: run_constants.num_populations == 0, must have at least 1 population"); exit(1); }

	for(int i = 0; i < time_samples.size(); i++){
		initialize_time_sample(i, sampled_generation[i], total_generated_mutations[i], pop_span_view[i], extinct_span_view[i], allele_span_view[i], mutID_span);
	}
}

inline allele_trajectories::allele_trajectories(const allele_trajectories & in) : allele_trajectories(in.sim_run_constants, in.num_time_samples()){
	//not using shared pointers: requires deep copy
	for(int i = 0; i < time_samples.size(); i++){
		initialize_time_sample(i, in.time_samples[i].sampled_generation, in.time_samples[i].total_generated_mutations, in.time_samples[i].Nchrom_e, in.time_samples[i].extinct, in.time_samples[i].mutations_freq, in.mutations_ID);
	}
}

inline void allele_trajectories::initialize_time_sample(unsigned int sample_index, unsigned int sampled_generation, unsigned int num_mutations, unsigned long total_generated_mutations){
	if(sample_index < time_samples.size() && time_samples[sample_index].Nchrom_e.size() == 0){
		time_samples[sample_index].sampled_generation = sampled_generation;
		auto num_populations = sim_run_constants.num_populations;
		time_samples[sample_index].Nchrom_e.reset(num_populations);
		time_samples[sample_index].extinct.reset(num_populations);
		time_samples[sample_index].mutations_freq.reset(num_mutations*num_populations);
		time_samples[sample_index].num_mutations = num_mutations;
		time_samples[sample_index].total_generated_mutations = total_generated_mutations;
		if(sample_index == time_samples.size()-1){ mutations_ID.reset(num_mutations); }
	}
	else{
		if(time_samples.size() == 0){ fprintf(stderr,"initialize_time_sample error: in sample index %d: empty allele_trajectories\n",sample_index); exit(1); }
		else if(time_samples[sample_index].Nchrom_e.size() == 0){ fprintf(stderr,"initialize_time_sample error: index out of bounds: sample %d [0 %ld)\n",sample_index,time_samples.size()); exit(1); }
		else{ fprintf(stderr,"initialize_time_sample error: sample index %d already initialized\n",sample_index); exit(1); }
	}
}

inline void allele_trajectories::initialize_time_sample(unsigned int sample_index, unsigned int sampled_generation, unsigned long total_generated_mutations, std::span<const unsigned int> pop_span, std::span<const unsigned int> extinct_span, std::span<const unsigned int> allele_span, std::span<const mutID> mutID_span){
	auto distance = allele_span.size();
	auto num_populations = sim_run_constants.num_populations;
	if(distance % num_populations != 0){ fprintf(stderr,"initialize_time_sample error: in sample index %d: number of alleles in allele_span, %ld, not a multiple of the number of populations, %d\n",sample_index,distance,num_populations); exit(1); }
	auto my_num_mutations = distance/num_populations;
	initialize_time_sample(sample_index, sampled_generation, my_num_mutations, total_generated_mutations);
	if(distance > 0){ ppp::pcopy(time_samples[sample_index].mutations_freq.data(),allele_span.data(),distance); }
	time_samples[sample_index].num_mutations = my_num_mutations;

	distance = pop_span.size();
	if(distance != num_populations){ fprintf(stderr,"initialize_time_sample error: in sample index %d: number of populations in pop_span, %ld, does not match number of populations in sim_run_constants, %d\n",sample_index,distance,num_populations); exit(1); }
	ppp::pcopy(time_samples[sample_index].Nchrom_e.data(), pop_span.data(), distance);

	distance = extinct_span.size();
	if(distance != num_populations){ fprintf(stderr,"initialize_time_sample error: in sample index %d: number of populations in extinct_span, %ld, does not match number of populations in sim_run_constants, %d\n",sample_index,distance,num_populations); exit(1); }
	ppp::pcopy(time_samples[sample_index].extinct.data(), extinct_span.data(), distance);

	if(sample_index == time_samples.size()-1){
		distance = mutID_span.size();
		if(distance != my_num_mutations){ fprintf(stderr,"initialize_time_sample error: in final sample index %d: number of mutIDs in mut_span, %ld, does not match num_mutations in final sample, %ld\n",sample_index,distance,my_num_mutations); exit(1); }
		ppp::pcopy(mutations_ID.data(),mutID_span.data(),distance);
	}
}

inline allele_trajectories::allele_trajectories(allele_trajectories && in) noexcept : allele_trajectories() { this->swap(in); } //http://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom

inline allele_trajectories & allele_trajectories::operator=(allele_trajectories in) noexcept { this->swap(in); return *this; } //is both copy and move https://stackoverflow.com/questions/3106110/what-are-move-semantics

inline void allele_trajectories::swap(allele_trajectories & in) noexcept {
	auto temp = this->sim_run_constants;
	this->sim_run_constants = in.sim_run_constants;
	in.sim_run_constants = temp;
	using std::swap; //good for ADL
	swap(this->mutations_ID,in.mutations_ID);
	swap(this->time_samples,in.time_samples);
}

/**Useful for when the allele_trajectories object is still in scope, but memory needs to be free and the data held by the object is no longer needed. Note: Not necessary to call as destructor will be called when object leaves scope.
 * \n\n Warning: will cause references, pointers, and iterators to be invalidated.
 * */
inline void allele_trajectories::reset() noexcept {
	sim_run_constants = sim_constants{ };
	time_samples.reset();
	mutations_ID.reset();
}

inline GO_Fish::sim_constants allele_trajectories::last_run_constants() const noexcept { return sim_run_constants; }

inline float allele_trajectories::num_sites() const noexcept { return sim_run_constants.num_sites; }

inline unsigned int allele_trajectories::num_populations() const noexcept { return sim_run_constants.num_populations; }

inline unsigned int allele_trajectories::num_time_samples() const noexcept { return time_samples.size(); }

inline unsigned int allele_trajectories::final_generation() const { return sampled_generation(time_samples.size()-1); }

inline unsigned int allele_trajectories::maximal_num_mutations() const noexcept { return mutations_ID.size(); }

inline unsigned int allele_trajectories::num_mutations_time_sample (unsigned int sample_index) const {
	if(sample_index < time_samples.size()){ return time_samples[sample_index].num_mutations; }
	else if(time_samples.size() == 0){ fprintf(stderr,"num_mutations_time_sample error: empty allele_trajectories\n"); exit(1); }
	else{ fprintf(stderr,"num_mutations_time_sample error: requested sample index out of bounds: sample %d [0 %ld)\n",sample_index,time_samples.size()); exit(1); }
}

inline unsigned long allele_trajectories::total_generated_mutations_time_sample(unsigned int sample_index) const {
	if(sample_index < time_samples.size()){ return time_samples[sample_index].total_generated_mutations; }
	else if(time_samples.size() == 0){ fprintf(stderr,"total_generated_mutations_time_sample error: empty allele_trajectories\n"); exit(1); }
	else{ fprintf(stderr,"total_generated_mutations_time_sample error: requested sample index out of bounds: sample %d [0 %ld)\n",sample_index,time_samples.size()); exit(1); }
}

inline unsigned int allele_trajectories::sampled_generation(unsigned int sample_index) const{
	if(sample_index < time_samples.size()){ return time_samples[sample_index].sampled_generation; }
	else if(time_samples.size() == 0){ fprintf(stderr,"sampled_generation error: empty allele_trajectories\n"); exit(1); }
	else{ fprintf(stderr,"sampled_generation error: requested sample index out of bounds: sample %d [0 %ld)\n",sample_index,time_samples.size()); exit(1); }
}

inline bool allele_trajectories::extinct(unsigned int sample_index, unsigned int population_index) const {
	auto num_populations = sim_run_constants.num_populations;
	if(sample_index < time_samples.size() && population_index < num_populations){ return (time_samples[sample_index].extinct[population_index] > 0); }
	else if(time_samples.size() == 0){ fprintf(stderr,"extinct error: empty allele_trajectories\n"); exit(1); }
	else{ fprintf(stderr,"extinct error: index out of bounds: sample %d [0 %ld), population %d [0 %d)\n",sample_index,time_samples.size(),population_index,num_populations); exit(1); }

}

inline unsigned int allele_trajectories::effective_number_of_chromosomes(unsigned int sample_index, unsigned int population_index) const {
	auto num_populations = sim_run_constants.num_populations;
	if(sample_index < time_samples.size() && population_index < num_populations){ return time_samples[sample_index].Nchrom_e[population_index]; }
	else if(time_samples.size() == 0){ fprintf(stderr,"effective_number_of_chromosomes error: empty allele_trajectories\n"); exit(1); }
	else{ fprintf(stderr,"effective_number_of_chromosomes error: index out of bounds: sample %d [0 %ld), population %d [0 %d)\n",sample_index,time_samples.size(),population_index,num_populations); exit(1); }
}

/*!if the \p mutation_index is of a mutation that is in the simulation, but which had not arisen as of /p sample_index, the reported allele count will be 0*/
inline unsigned int allele_trajectories::allele_count(unsigned int sample_index, unsigned int population_index, unsigned int mutation_index) const {
	auto num_populations = sim_run_constants.num_populations;
	if(sample_index < time_samples.size() && population_index < num_populations && mutation_index < mutations_ID.size()){
		auto num_mutations_in_sample = time_samples[sample_index].num_mutations;
		if(mutation_index >= num_mutations_in_sample){ return 0; }
		return time_samples[sample_index].mutations_freq[mutation_index+population_index*num_mutations_in_sample];
	}
	else{
		if(time_samples.size() == 0){ fprintf(stderr,"allele_count error: empty allele_trajectories\n"); exit(1); }
		fprintf(stderr,"allele_count error: index out of bounds: sample %d [0 %ld), population %d [0 %d), mutation %d [0 %ld)\n",sample_index,time_samples.size(),population_index,num_populations,mutation_index,mutations_ID.size()); exit(1);
	}
}

inline const mutID & allele_trajectories::mutation_ID(unsigned int mutation_index) const {
	if(time_samples.size() > 0){
		if(mutation_index < mutations_ID.size()){ return mutations_ID[mutation_index]; }
		fprintf(stderr,"mutation_ID error: requested mutation index out of bounds: mutation %d [0 %d)\n",mutation_index,maximal_num_mutations()); exit(1);
	}else{ fprintf(stderr,"mutation_ID error: empty allele_trajectories\n"); exit(1); }
}

inline std::vector<unsigned int> allele_trajectories::dump_sampled_generations() const {
	std::vector<unsigned int> sampled_gen_vec(time_samples.size());
	int i = 0;
	for(auto & sample: time_samples.span()){ //will output vectors of length 0 if no time samples
		sampled_gen_vec[i] = sample.sampled_generation;
	 	i++;
	 }
	return sampled_gen_vec;
}

inline std::vector<unsigned int> allele_trajectories::dump_num_mutations_samples() const {
	std::vector<unsigned int> num_mut_vec(time_samples.size());
	int i = 0;
	for(auto & sample: time_samples.span()){ //will output vectors of length 0 if no time samples
		num_mut_vec[i] = sample.num_mutations;
	 	i++;
	 }
	return num_mut_vec;
}

inline std::vector<unsigned long> allele_trajectories::dump_total_generated_mutations_samples() const {
	std::vector<unsigned long> num_mut_vec(time_samples.size());
	int i = 0;
	for(auto & sample: time_samples.span()){ //will output vectors of length 0 if no time samples
		num_mut_vec[i] = sample.total_generated_mutations;
	 	i++;
	 }
	return num_mut_vec;
}

inline std::span<unsigned int> allele_trajectories::popsize_span(unsigned int sample_index){
	if(sample_index < time_samples.size()){ return time_samples[sample_index].Nchrom_e; }
	else if(time_samples.size() == 0){ fprintf(stderr,"Nchrom_e_span error: empty allele_trajectories\n"); exit(1); }
	else { fprintf(stderr,"Nchrom_e_span error: requested sample index out of bounds: sample %d [0 %ld)\n",sample_index,time_samples.size()); exit(1); }
}

inline std::span<const unsigned int> allele_trajectories::popsize_span(unsigned int sample_index) const {
	if(sample_index < time_samples.size()){ return time_samples[sample_index].Nchrom_e; }
	else if(time_samples.size() == 0){ fprintf(stderr,"Nchrom_e_span error: empty allele_trajectories\n"); exit(1); }
	else { fprintf(stderr,"Nchrom_e_span error: requested sample index out of bounds: sample %d [0 %ld)\n",sample_index,time_samples.size()); exit(1); }
}

inline std::vector<std::span<const unsigned int>> allele_trajectories::popsize_view() const {
	std::vector<std::span<const unsigned int>> Ne_span_view(time_samples.size());
	int i = 0;
	for(auto & sample: time_samples.span()){ //will output vectors of length 0 if no time samples
		Ne_span_view[i] = sample.Nchrom_e.span();
		i++;
	}
	return Ne_span_view;
}

inline std::vector<std::vector<unsigned int>> allele_trajectories::dump_popsize() const {
	std::vector<std::vector<unsigned int>> Ne_vec(time_samples.size());
	int i = 0;
	for(auto & sample: time_samples.span()){ //will output vectors of length 0 if no time samples
		Ne_vec[i].resize(sample.Nchrom_e.size());
		ppp::pcopy(Ne_vec[i].data(), sample.Nchrom_e.data(), Ne_vec[i].size());
		i++;
	}
	return Ne_vec;
}

inline std::span<unsigned int> allele_trajectories::extinct_span(unsigned int sample_index){
	if(sample_index < time_samples.size()){ return time_samples[sample_index].extinct; }
	else if(time_samples.size() == 0){ fprintf(stderr,"extinct_span error: empty allele_trajectories\n"); exit(1); }
	else { fprintf(stderr,"extinct_span error: requested sample index out of bounds: sample %d [0 %ld)\n",sample_index,time_samples.size()); exit(1); }
}

inline std::span<const unsigned int> allele_trajectories::extinct_span(unsigned int sample_index) const {
	if(sample_index < time_samples.size()){ return time_samples[sample_index].extinct; }
	else if(time_samples.size() == 0){ fprintf(stderr,"extinct_span error: empty allele_trajectories\n"); exit(1); }
	else { fprintf(stderr,"extinct_span error: requested sample index out of bounds: sample %d [0 %ld)\n",sample_index,time_samples.size()); exit(1); }
}

inline std::vector<std::span<const unsigned int>> allele_trajectories::extinct_view() const {
	std::vector<std::span<const unsigned int>> extinct_span_view(time_samples.size());
	int i = 0;
	for(auto & sample: time_samples.span()){ //will output vectors of length 0 if no time samples
		extinct_span_view[i] = sample.extinct.span();
		i++;
	}
	return extinct_span_view;
}

inline std::vector<std::vector<unsigned int>> allele_trajectories::dump_extinct() const {
	std::vector<std::vector<unsigned int>> extinct_vec(time_samples.size());
	int i = 0;
	for(auto & sample: time_samples.span()){ //will output vectors of length 0 if no time samples
		for(int j = 0; j < extinct_vec[i].size(); j++){ extinct_vec[i][j] = sample.extinct[j]; }
		i++;
	}
	return extinct_vec;
}

inline std::span<unsigned int> allele_trajectories::allele_count_span(unsigned int sample_index, unsigned int start_population_index, unsigned int num_contig_pop){
	auto num_populations = sim_run_constants.num_populations;
	auto _num_contig_pop = (num_contig_pop > 0) ? num_contig_pop : num_populations - start_population_index;
	if(sample_index < time_samples.size() && start_population_index < num_populations && (start_population_index + _num_contig_pop <= num_populations)){
		auto num_mutations = time_samples[sample_index].num_mutations;
		return time_samples[sample_index].mutations_freq.subspan(start_population_index*num_mutations, num_mutations*_num_contig_pop);
	}else{
		if(time_samples.size() == 0){ fprintf(stderr,"allele_count_data error: empty allele_trajectories\n"); exit(1); }
		fprintf(stderr,"allele_count_data error: index out of bounds: sample %d [0 %ld), population %d - %d [0 %d)\n",sample_index,time_samples.size(),start_population_index,start_population_index+_num_contig_pop,num_populations); exit(1);
	}
}

inline std::span<const unsigned int> allele_trajectories::allele_count_span(unsigned int sample_index, unsigned int start_population_index, unsigned int num_contig_pop) const {
	auto num_populations = sim_run_constants.num_populations;
	auto _num_contig_pop = (num_contig_pop > 0) ? num_contig_pop : num_populations - start_population_index;
	if(sample_index < time_samples.size() && start_population_index < num_populations && (start_population_index + _num_contig_pop <= num_populations)){
		auto num_mutations = time_samples[sample_index].num_mutations;
		return time_samples[sample_index].mutations_freq.subspan(start_population_index*num_mutations, num_mutations*_num_contig_pop);
	}else{
		if(time_samples.size() == 0){ fprintf(stderr,"allele_count_data error: empty allele_trajectories\n"); exit(1); }
		fprintf(stderr,"allele_count_data error: index out of bounds: sample %d [0 %ld), population %d - %d [0 %d)\n",sample_index,time_samples.size(),start_population_index,start_population_index+_num_contig_pop,num_populations); exit(1);
	}
}

inline std::vector<std::span<const unsigned int>> allele_trajectories::allele_count_view() const {
	std::vector<std::span<const unsigned int>> allele_count_span_view(time_samples.size());
	int i = 0;
	for(auto & sample: time_samples.span()){ //will output vectors of length 0 if no time samples
		allele_count_span_view[i] = sample.mutations_freq.span();
		i++;
	}
	return allele_count_span_view;
}

inline std::vector<std::vector<unsigned int>> allele_trajectories::dump_allele_counts() const {
	std::vector<std::vector<unsigned int>> allele_count_vec(time_samples.size());
	int i = 0;
	for(auto & sample: time_samples.span()){ //will output vectors of length 0 if no time samples
		allele_count_vec[i].resize(sample.Nchrom_e.size());
		ppp::pcopy(allele_count_vec[i].data(), sample.mutations_freq.data(), allele_count_vec[i].size());
		i++;
	}
	return allele_count_vec;
}

inline std::vector<std::vector<std::vector<unsigned int>>> allele_trajectories::dump_padded_allele_counts() const {
	std::vector<std::vector<std::vector<unsigned int>>> allele_count_vec(time_samples.size());
	int i = 0;
	for(auto & sample: time_samples.span()){ //will output vectors of length 0 if no time samples
		allele_count_vec[i].resize(sample.Nchrom_e.size());
		int j = 0;
		for(auto & vec : allele_count_vec[i]){
			vec.resize(mutations_ID.size());
			int mut = 0;
			for(auto & v : vec){
				if(mut >= sample.num_mutations){ v = 0; }
				else{ v = sample.mutations_freq[mut + j*sample.num_mutations]; }
				mut++;
			}
			j++;
		}
		i++;
	}
	return allele_count_vec;
}

inline std::span<mutID> allele_trajectories::mutID_span() noexcept { return mutations_ID; } //will output empty span if no mutations

inline std::span<const mutID> allele_trajectories::mutID_span() const noexcept { return mutations_ID; } //will output empty span if no mutations

inline std::vector<mutID> allele_trajectories::dump_mutID() const {
	std::vector<mutID> mutID_vec(mutations_ID.size());
	if(mutations_ID.size() > 0){ ppp::pcopy(mutID_vec.data(), mutations_ID.data(), mutID_vec.size()); } //will output empty vector if no mutations
	return mutID_vec;
}

/** returns `ostream stream` containing mutID formatted: (origin_generation,origin_population,origin_thread,DFE_selection) \n\n Stream can be fed into terminal output, file output, or, if an `iostream`, for extraction with the `>>` operator. */
inline std::ostream & operator<<(std::ostream & stream, const mutID & id){ stream <<"("<<id.origin_generation<<","<<id.origin_population<<","<<id.origin_threadID<<","<<id.DFE_selection<<")"; return stream; }

inline std::ostream & operator<<(std::ostream & stream, const compact_scheme & ctype){
    switch(ctype)
    {
        case compact_scheme::compact_all   		: stream << "compact_all";    break;
        case compact_scheme::compact_losses 	: stream << "compact_losses"; break;
        case compact_scheme::compact_fixations 	: stream << "compact_fixations";  break;
        case compact_scheme::compact_off  		: stream << "compact_off";   break;
        default    								: stream.setstate(std::ios_base::failbit);
    }
    return stream;
}

/** returns `ostream stream` containing sim_constants constants \n\n Stream can be fed into terminal output, file output, or, if an `iostream`, for extraction with the `>>` operator. */
inline std::ostream & operator<<(std::ostream & stream, const sim_constants & constants){
	stream << "seed1" << "\t" << constants.seed1 << std::endl;
	stream << "seed2" << "\t" << constants.seed2 << std::endl;
	stream << "num_generations" << "\t" << constants.num_generations << std::endl;
	stream << "num_sites" << "\t" << constants.num_sites << std::endl;
	stream << "num_populations" << "\t" << constants.num_populations << std::endl;
	stream << "init_mse" << "\t" << constants.init_mse << std::endl;
	stream << "prev_sim_sample" << "\t" << constants.prev_sim_sample << std::endl;
	stream << "compact_type" << "\t" << constants.compact_type << std::endl;
	stream << "compact_interval" << "\t" << constants.compact_interval << std::endl;
	stream << "device" << "\t" << constants.device << std::endl << std::endl;
	return stream;
}

/** returns `ostream stream` containing the last simulation run information stored by `allele_trajectories A` \n\n
 * First function inserts the run constants (not input constants) held by `A` into the output stream with the variable name tab-delimited from its value.
 * This is followed by the feature information (e.g. generation, number of mutations, population size, population extinction) from each time sample (if any).
 * Each feature of a time sample is a row in the stream while each time sample is a major column and each population is a minor column. Finally, the allele trajectory of each
 * mutation (if any) is added to the stream. The allele trajectories are mutation row-ordered (by `origin_generation` then `origin_population` then `origin_threadID`),
 * where each major column is a time sample and each minor column is a population. All columns are tab-delimited. An example is provided in example_compilation/allele_traj.dat. \n\n
 * Stream can be fed into terminal output, file output, or, if an `iostream`, for extraction with the `>>` operator.
 *  */
inline std::ostream & operator<<(std::ostream & stream, const allele_trajectories & A){
	stream << A.last_run_constants();

	auto num_samples = A.num_time_samples();
	stream << "number of time samples" << "\t" << num_samples <<std::endl;
	if(num_samples == 0){ return stream; }

	auto num_populations = A.num_populations();

	stream << "time sample";
	for(int j = 0; j < num_samples; j++){
		stream << "\t" << j;
		if(j < num_samples-1){ for(int k = 0; k < num_populations-1; k++){ stream << "\t"; } } //don't pad the ends
	} stream << std::endl;

	stream << "generation";
	for(int j = 0; j < num_samples; j++){
		stream << "\t" << A.sampled_generation(j);
		if(j < num_samples-1){ for(int k = 0; k < num_populations-1; k++){ stream << "\t"; } } //don't pad the ends
	} stream << std::endl << "number of mutations reported";

	for(int j = 0; j < num_samples; j++){
		stream << "\t" << A.num_mutations_time_sample(j);
		if(j < num_samples-1){ for(int k = 0; k < num_populations-1; k++){ stream << "\t"; } } //don't pad the ends
	} stream << std::endl << "total number of mutations generated";

	for(int j = 0; j < num_samples; j++){
		stream << "\t" << A.total_generated_mutations_time_sample(j);
		if(j < num_samples-1){ for(int k = 0; k < num_populations-1; k++){ stream << "\t"; } } //don't pad the ends
	} stream << std::endl << "population" ;

	for(int j = 0; j < num_samples; j++){
		for(int k = 0; k < num_populations; k++){ stream << "\t" << k; }
	} stream << std::endl << "effective population size (chromosomes)";

	for(int j = 0; j < num_samples; j++){
		for(int k = 0; k < num_populations; k++){ stream << "\t" << A.effective_number_of_chromosomes(j,k); }
	} stream << std::endl << "population extinct";

	for(int j = 0; j < num_samples; j++){
		for(int k = 0; k < num_populations; k++){ stream << "\t" << A.extinct(j,k); }
	} stream << std::endl;

	auto total_mutations = A.maximal_num_mutations();

	if(total_mutations == 0){ stream << std::endl << "no mutations stored" << std::endl; return stream; }

	stream << "mutation ID (origin_generation,origin_population,origin_threadID,DFE_selection)";
	for(int j = 0; j < num_samples; j++){ for(int k = 0; k < num_populations; k++){ stream << "\t" << "allele counts"; } }
	stream << std::endl;

	for(int i = 0; i < total_mutations; i++) {
		stream << A.mutation_ID(i);
		for(int j = 0; j < num_samples; j++){
			for(int k = 0; k < num_populations; k++){ stream << "\t" << A.allele_count(j,k,i); }
		}
		stream << std::endl;
	}

	return stream;
}

/** returns `istream stream` after extracting mutID */
inline std::istream & operator>>(std::istream & stream, mutID & id){
	char a;
	stream >> a >> id.origin_generation >> a >> id.origin_population >> a >> id.origin_threadID >> a >> id.DFE_selection >> a;
	return stream;
}

inline std::istream & operator>>(std::istream & stream, compact_scheme & ctype){
	std::string s;
	stream >> s;
    if(s == "compact_all"){ ctype = compact_scheme::compact_all; }
    else if (s == "compact_losses"){ ctype = compact_scheme::compact_losses; }
    else if (s == "compact_fixations"){ ctype = compact_scheme::compact_fixations; }
    else if (s == "compact_off"){ ctype = compact_scheme::compact_off; }
    else { stream.setstate(std::ios_base::failbit); }
    return stream;
}

/** returns `istream stream` after extracting sim_constants constants */
inline std::istream & operator>>(std::istream & stream, sim_constants & constants){
	stream.ignore(sizeof("seed1"));  stream >> constants.seed1;
	stream.ignore(sizeof("seed2"));  stream >> constants.seed2;
	stream.ignore(sizeof("num_generations")); stream >> constants.num_generations;
	stream.ignore(sizeof("num_sites")); stream >> constants.num_sites;
	stream.ignore(sizeof("num_populations")); stream >> constants.num_populations;
	stream.ignore(sizeof("init_mse")); stream >> constants.init_mse;
	stream.ignore(sizeof("prev_sim_sample")); stream >> constants.prev_sim_sample;
	stream.ignore(sizeof("compact_type")); stream >> constants.compact_type;
	stream.ignore(sizeof("compact_interval")); stream >> constants.compact_interval;
	stream.ignore(sizeof("device")); stream >> constants.device;
	return stream;
}

/** returns `istream stream` after extracting `allele_trajectories A` */
inline std::istream & operator>>(std::istream & stream, allele_trajectories & A){
	sim_constants constants;
	stream >> constants;

	stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	size_t num_samples;
	stream.ignore(sizeof("number of time samples")); stream >> num_samples;

	A = allele_trajectories(constants,num_samples);
	if(num_samples == 0){ return stream; }

	ppp::unique_host_span<unsigned int> sampled_generation(num_samples);
	ppp::unique_host_span<unsigned int> num_mutations(num_samples);
	ppp::unique_host_span<unsigned long> total_gen_mutations(num_samples);

	stream.ignore(sizeof("time sample")); //get stream on to the next line
	stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); //ignore rest of line

	stream.ignore(sizeof("generation")); for(auto & sg : sampled_generation.span()){ stream >> sg;  }
	stream.ignore(sizeof("number of mutations reported")); for(auto & nm : num_mutations.span()){ stream >> nm; }
	stream.ignore(sizeof("total number of mutations generated")); for(auto & tgm : total_gen_mutations.span()){ stream >> tgm; }

	for(int i = 0; i < num_samples; i++){ A.initialize_time_sample(i, sampled_generation[i], num_mutations[i], total_gen_mutations[i]); }
	stream.ignore(sizeof("population"));
	stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	stream.ignore(sizeof("effective population size (chromosomes)"));
	for(int j = 0; j < num_samples; ++j){
		auto pop_span = A.popsize_span(j);
		for(auto ps: pop_span){ stream >> ps; }
	}

	stream.ignore(sizeof("population extinct"));
	for(int j = 0; j < num_samples; ++j){
		auto extinct_span = A.extinct_span(j);
		for(auto es: extinct_span){ stream >> es; }
	}

	auto total_mutations = num_mutations[num_mutations.size()-1];
	if(total_mutations == 0){ return stream; }
	stream.ignore(sizeof("mutation"));
	stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

	auto mutID_span = A.mutID_span();
	for(int i = 0; i < total_mutations; ++i) {
		stream >> mutID_span[i];
		for(int j = 0; j < num_samples; ++j){
			if(i >= num_mutations[j]){
				unsigned int temp; //do not store these
				for(int k = 0; k < constants.num_populations; k++){ stream >> temp; }
			}
			else{
				for(int k = 0; k < constants.num_populations; k++){
					auto allele_count_span = A.allele_count_span(j,k,1);
					stream >> allele_count_span[i];
				}
			}
		}
	}

	return stream;
}

/** calls `lhs.swap(rhs)` */
inline void swap(allele_trajectories & lhs, allele_trajectories & rhs) noexcept { lhs.swap(rhs); }

} /* ----- end namespace GO_Fish ----- */

#endif /* INLINE_GOFISH_DATA_FUNCTIONS_HPP_ */
