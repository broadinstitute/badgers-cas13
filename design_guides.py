#!/usr/bin/env python3
"""
Script for running the algorithms to design diagnostic Cas13a 
diagnostic guide RNA spacer sequences.
"""

# Importing standard packages
import argparse
import pandas as pd
import multiprocessing
from multiprocessing import set_start_method
import os
import os.path
from collections import Counter 
from operator import itemgetter 
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# Most machines don't have dedicated GPUs
# So, to avoid CUDA's error messages, we disable the GPU
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    pass

# Importing DRAGON packages
from badgers.models.cas13_mult import Cas13Mult
from badgers.models.cas13_diff import Cas13Diff 

from badgers.utils.cas13_landscape import Cas13Landscape
from badgers.utils import prepare_sequences as prep_seqs
from badgers.explorers.evolutionary import EvolutionaryExplorer 
from badgers.explorers.wgan_am import WGANExplorer
from badgers.utils import import_fasta

def run_wgan(args, site_df):
    """Uses the WGAN-AM algorithm to generate optimal guides.
    Args:
        results_path: path to the results file 
        results_path: path to the directory where the temporary files can be written
        site_df: a pandas dataframe which includes the target set 
                and start position of the genomic site being designed for
    """  

    landscape = Cas13Landscape()
    s_time = time.time()

    if(args.objective == 'multi'): 
        model = Cas13Mult(landscape, site_df.target_set)
        wgan_explorer = WGANExplorer(
            model,
            rounds=15, 
            starting_sequence=model.target_cons_nt,
            inner_iterations=144, 
            adam_lr = 1.540127,
            optimizer = 'adam',
            sequences_batch_size=100, 
            model_queries_per_batch=2000, 
            track_perf = args.track_performance,
            local_s = args.local_s
            ) 
    else:
        grid = {'c': 1.0, 'a' : 3.769183, 'k':  -3.833902, 'o': -2.134395, 't2w' : 2.973052} 
        model = Cas13Diff(landscape, site_df.target_set1_nt, site_df.target_set2_nt, grid)
    
        wgan_explorer = WGANExplorer(
            model,
            rounds=14,  
            starting_sequence=model.target_cons_nt,
            inner_iterations=147, 
            adam_lr = 0.632998,
            optimizer = 'adam',
            sequences_batch_size=30, 
            model_queries_per_batch=2000,
            track_perf = args.track_performance,
            local_s = args.local_s
            ) 

    gen_guides, metadata = wgan_explorer.run(landscape, verbose = False)
    
    save_results('wgan_am', gen_guides, site_df, args, model, s_time)

def run_evolutionary(args, site_df):
    """Uses the evolutionary algorithm to generate optimal guides.
    Args:
        results_path: path to the results file 
        results_path: path to the directory where the temporary files can be written
        site_df: a pandas dataframe which includes the target set 
                and start position of the genomic site being designed for
    """  

    landscape = Cas13Landscape()
    s_time = time.time()

    if(args.objective == 'multi'):
        model = Cas13Mult(landscape, site_df.target_set)
        evolutionary_explorer = EvolutionaryExplorer(
            model,
            S=87,
            beta=0.077373,
            j=0.794996,
            rounds=1,
            gamma = 0.003362,
            sequences_batch_size=87,
            model_queries_per_batch=2000, 
            track_perf = args.track_performance,
            local_s = args.local_s 
            )

    else:
        grid = {'c': 1.0, 'a' : 5.897292, 'k':  -2.857755, 'o': -2.510856, 't2w' : 1.736507}
        model = Cas13Diff(landscape, site_df.target_set1_nt, site_df.target_set2_nt, grid)

        evolutionary_explorer = EvolutionaryExplorer(
            model,
            S=119,
            beta=2.201796,
            j=0.893401,
            rounds=1,
            gamma = 0.029049,
            sequences_batch_size=119,
            model_queries_per_batch=2000, 
            track_perf = args.track_performance,
            local_s = args.local_s
            )

    gen_guides, metadata = evolutionary_explorer.run(landscape, verbose = False)
    
    save_results('evolutionary', gen_guides, site_df, args, model, s_time)


def save_results(explorer_name, gen_seqs, site_df, args, model, s_time):
    """Saves the results of the exploration algorithm for a single genomic site.
    Args:
        explorer_name: name of explorer used to generate guides
        gen_seqs: a dataframe of sequences and their corresponding fitness designed by the algorithms
        site_df: a pandas dataframe which includes the target set 
                and start position of the genomic site being designed for
        args: arguments passed in by the user
        model: the model object that was used to guide the search process
    """  
    
    output_file = os.path.join(args.results_path, f'results_by_site/{explorer_name}/' + site_df.seq_id + '.tsv')
    print(f"\nSaving {explorer_name} design for site {site_df.seq_id}\n\n")

    gen_seqs = gen_seqs.drop('true_score', axis =1).dropna().rename(columns={'model_score': 'fitness'})

    if(args.output_to_single_directory):
            output_file = os.path.join(args.results_path, f'results_by_site/' + explorer_name + "_" + site_df.seq_id + '.tsv')

    results_df = pd.DataFrame()
    context_nt = 10
    runtime = (time.time() - s_time)/60

    if(args.objective == 'multi'):
        num_targets = len(model.target_set_nt)
        num_utargets = len(model.target_set_utargets_onehot)
        shannon_entropy_site = prep_seqs.shannon_entropy_msa(site_df.target_set)
        PFS = Counter(map(itemgetter(38), model.target_set_nt))
        G_PFS = (max(PFS, key=PFS.get) == 'G')

        results_df = pd.DataFrame()
        for index, gen_seq in gen_seqs.iterrows():
            results_df = results_df.append(pd.DataFrame({
            'algo': [explorer_name],
            'start_pos': [site_df.start_pos],
            'guide_sequence': [gen_seq.sequence],
            'fitness': [gen_seq.fitness],
            'perc_highly_active': [model._fitness_function(gen_seq.sequence, output_type = 'perc_highly_active')[0]],
            'hd_cons': [prep_seqs.hamming_dist(gen_seq.sequence, model.target_cons_no_context_nt)],
            'shannon_entropy': [shannon_entropy_site],
            'hd_min_targets': [min([prep_seqs.hamming_dist(gen_seq.sequence, target) for target in model.unique_target_set_no_context_nt])],
            'num_targets': [num_targets],
            'num_utargets': [num_utargets],
            'runtime': [runtime]
            })).reset_index(drop = True)          
        
        if(args.benchmarking):
            for baseline_method, baseline_sequence in zip(['adapt', 'consensus'], [site_df['adapt_guide'], model.target_cons_no_context_nt]):
                results_df = results_df.append(pd.DataFrame({
                'algo': [baseline_method],
                'start_pos': [site_df.start_pos],
                'guide_sequence': [baseline_sequence],
                'fitness': [model._fitness_function(baseline_sequence)[0]], 
                'perc_highly_active': [model._fitness_function(baseline_sequence, output_type = 'perc_highly_active')[0]],
                'hd_cons': [prep_seqs.hamming_dist(baseline_sequence, model.target_cons_no_context_nt)],
                'shannon_entropy': [shannon_entropy_site],
                'hd_min_targets': [min([prep_seqs.hamming_dist(baseline_sequence, target) for target in model.unique_target_set_no_context_nt])],
                'num_targets': [num_targets],
                'num_utargets': [num_utargets],
                'runtime': [runtime]
                })).reset_index(drop = True)         

        if(not args.verbose_results):
            results_df = results_df.drop(columns = ['hd_cons', 'hd_min_targets', 'num_targets', 'num_utargets', 'runtime'])
 
    else:
        
        for index, gen_seq in gen_seqs.iterrows():
            perf = model._fitness_function(gen_seq.sequence, output_type='save')[0]
            results_df = results_df.append(pd.DataFrame({ 
            'algo': [explorer_name],
            'on_target_name' : [site_df.target1_name],
            'off_target_name': [site_df.target2_name], 
            'start_pos': [site_df.start_pos],
            'guide_sequence': [gen_seq.sequence],
            'fitness':[gen_seq.fitness],
            'hd_cons_on_target': [prep_seqs.hamming_dist(gen_seq.sequence, prep_seqs.consensus_seq(model.target_set1_nt, nt = True)[context_nt:-context_nt])],
            'mean_on_target_act' : [perf[0]], 
            'mean_off_target_act' : [perf[1]],
            'hd_min_targets': [min([prep_seqs.hamming_dist(gen_seq.sequence, target[context_nt:-context_nt]) for target in site_df.target_set1_nt + site_df.target_set2_nt])],
            'hd_min_target_set1': [min([prep_seqs.hamming_dist(gen_seq.sequence, target[context_nt:-context_nt]) for target in site_df.target_set1_nt])],
            'hd_min_target_set2': [min([prep_seqs.hamming_dist(gen_seq.sequence, target[context_nt:-context_nt]) for target in site_df.target_set1_nt])],
            't1_t2_diff_activity' : [perf[2]],
            't1_cost_input' : [perf[3]], 
            't2_cost_input' : [perf[4]], 
            't1cost' : [perf[5]], 
            't2cost' : [perf[6]],  
            'pred_perf_t1' : [perf[7]], 
            'pred_perf_t2' : [perf[8]], 
            'pred_act_t1' : [perf[9]], 
            'pred_act_t2' : [perf[10]],
            'runtime': [runtime]
            })).reset_index(drop = True)            

        if(not args.verbose_results):
            results_df = results_df.drop(columns = ['hd_cons_on_target', 'hd_min_targets', 'hd_min_target_set1', 'hd_min_target_set2',
            't1_t2_diff_activity', 't1_cost_input', 't2_cost_input', 't1cost', 't2cost', 'pred_perf_t1', 'pred_perf_t2', 'pred_act_t1', 'pred_act_t2', 'runtime'])

    results_df.start_pos = results_df.start_pos.astype(int)
    results_df = results_df.drop_duplicates(ignore_index=True).sort_values(by = 'fitness', ascending=False)
 
    if(args.save_pickled_results):
        results_df.to_pickle(output_file.replace('.tsv', '.pkl'))
    else:
        results_df.to_csv(output_file, sep = '\t', index=False)


def compile_results_across_sites(explorer_name, args):
    """Compiles results across genomic sites into a single file.
    Args:
        explorer_name: name of explorer used to generate guides
        results_path: path to the directory where the per-site results files are stored in a subdirectory
    """

    print(f"\nCompiling Results for {explorer_name}")

    results_dir = os.path.join(args.results_path, f'results_by_site/{explorer_name}/') 
    results_files = os.listdir(results_dir)

    all_results_df = pd.DataFrame()
    target_set_results_df = pd.DataFrame()

    for file in results_files:
        if(args.n_top_guides_per_site == 0):
            # If we are only saving the top guide for each site, then we can just extract the top one for each site
            all_results_df = all_results_df.append(pd.read_csv(results_dir + file, sep = '\t').iloc[0]).reset_index(drop = True)
        else:
            # If we are saving multiple guides per site, then we cannot just extract the top one
            all_results_df = all_results_df.append(pd.read_csv(results_dir + file, sep = '\t')).reset_index(drop = True)
     
    # Compiling results for multi-target detection
    if(args.objective == 'multi'):
        if(args.n_top_guides_per_site == 0):
            target_set_results_df = all_results_df.sort_values(by = 'fitness', ascending=False).reset_index(drop = True).iloc[0:args.n_top_guides]
        else:
            collected_results = pd.DataFrame()
            for name, genomic_site_df in target_set_results_df.groupby('start_pos'):
                reset_genomic_site_df = genomic_site_df.reset_index(drop = True)
                indices = prep_seqs.edit_distance_index(genomic_site_df, args.n_top_guides_per_site, edit_dist_thresh = 3)
                collected_results = collected_results.append(reset_genomic_site_df.iloc[indices])
            target_set_results_df = collected_results.sort_values(by = 'fitness', ascending=False)

    # Compiling results for differential identification
    # Results are compiled for each unique on-target name
    else:
        unique_target_sets = all_results_df.on_target_name.unique()
        for target_set in unique_target_sets:
            per_target_result = all_results_df[all_results_df.on_target_name == target_set].sort_values(by = 'fitness', ascending=False)

            if(args.n_top_guides_per_site == 0):
                # Save the top n guides for each target set
                proc_per_target_results = per_target_result.reset_index(drop = True).iloc[0:args.n_top_guides] 
  
            else:    
                # Group the target_set_results_df by the start_pos and then take the top n guides for each start_pos
                collected_results = pd.DataFrame()
                for name, genomic_site_df in per_target_result.groupby('start_pos'):
                    reset_genomic_site_df = genomic_site_df.reset_index(drop = True)
                    indices = prep_seqs.edit_distance_index(genomic_site_df, args.n_top_guides_per_site, edit_dist_thresh = 3)
                    collected_results = collected_results.append(reset_genomic_site_df.iloc[indices])
                proc_per_target_results = collected_results

            target_set_results_df = target_set_results_df.append(proc_per_target_results).sort_values(by = ['on_target_name', 'fitness'], ascending=False)

    target_set_results_df.start_pos = target_set_results_df.start_pos.astype(int) 
    target_set_results_df = target_set_results_df.drop_duplicates(ignore_index=True)
    target_set_results_df.reset_index(drop = True).to_csv(os.path.join(args.results_path, f'results_compiled_{explorer_name}.tsv'), sep = '\t', index=False)
 
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('objective',
            help=("""The objective function to use to generate guide sequences. 
            'multi' uses the objective function for multi-target detection, while 'diff' uses the objective function for differential identification"""),
            default = 'multi')

    parser.add_argument('exploration_algorithm',
        type = str,
        help=("""The exploration algorithm that will be used to generate the guide sequences. 
        'wgan-am' runs the WGAN-AM algorithm, 'evolutionary' runs the evolutionary algorithm, and 'both' runs both algorithms"""))
            
    parser.add_argument('fasta_path',
        type = str,
        help=("""Path to a FASTA file providing the genomic sequences that the guides will be designed for.
        If the specified objective is 'multi', then this should be a path to a single FASTA file
        If the specified objective is 'diff', then this should be a path to a folder of FASTA files."""),
            default = False)

    parser.add_argument('results_path',
        type = str,
        help=("""Path to a directory where the output of the algorithms will go.
        The algorithms output a .pkl file for each genomic site that they design for, and these files are stored in a subdirectory (/results_by_site/) of this directory.
        The algorithms will also output a results.tsv file that provides data for the top-scoring guides, and this file is stored in the main directory of this path."""),
            default = False)

    parser.add_argument('--use_range',
        help=("""Path to a .tsv file that specifies  the ranges of genomic sites that should be considered. This is an inclusive range."""),
            default = False)
    
    parser.add_argument('--all_targets_one_file',
        help=("""If all the targets for the variant identification task are in one file rather than multiple aligned fasta files, set to True"""),
            default = False) 

    parser.add_argument('--num_cpu',
        type = int,
        help=("Number of CPUs to parallelize the jobs over. Each genomic site can be designed for in parallel."""),
            default = 0)
    
    parser.add_argument('--verbose_results',
    help=("Determines if the methods output the standard results file or the verbose results file with additional fields."""),
        action='store_true')
    
    parser.add_argument('--processed_sites_path',
    help=("""If true, the pickled file at the fasta_path path will directly be read in and used for design.
    Otherwise, by default the FASTA file at the fasta_path will be read in and processed to generate the sites that will be designed for."""),
        action='store_true')
    
    parser.add_argument('--benchmarking',
    help=("""If this is true, the script will assume that ADAPT's designs are provided as part of the input dataframe, and will use
        the predictive models to evaluate the baseline designs (adapt/consensus) and add them to the results dataframe."""), 
        action='store_true')
    
    parser.add_argument('--output_to_single_directory',
    help=("""If this is true, all the design results for both explorers will be written to a single directory."""),
        action='store_true')
    
    parser.add_argument('--save_pickled_results',
    help=("""If this is true, the design results will be written to pickled files."""),
      action='store_true')
    
    parser.add_argument('--n_top_guides',
    type = int,
    help=("""This argument tells the algorithms how many of the top-scoring guides to save for across the genomic sites."""),
        default = 20) 
    
    parser.add_argument('--local_s',
    type = bool,
    help=("""This argument tells the algorithms to optimize the mismatches with the baseline seq at the end of the run."""),
       default = True)
     
    parser.add_argument('--n_top_guides_per_site', 
    type = int,
    help=("""If this argument is specified when the algorithms are used for the variant objective, then the algorithms will save multiple guides
    for each positioning of the guide on the target sequence, using an edit distance threshold to ensure that the resulting guides are diverse. 
    This can be useful if a user is interested in targeting a specific genomic site for a variant identification task and wants multiple design options.
    If this option is specified, then the algorithms will save n_top_guides * n_top_guides_per_site guides."""),
        default = 0)
    
    parser.add_argument('--track_performance',
    type = str,
    help=("""This argument tells the script to save the performance of the maximally-fit guide at each step during the search process.
    The file is saved to the path specified in the argument."""),
        default = '')


    args = parser.parse_args()
    set_start_method("spawn")

    # If the user specifies a range, then only read in genomic sites in that range from the FASTA
    # Otherwise, read in all the genomic sites
    if(not args.use_range):
        included_sites_list = None
    else:
        included_sites_df = pd.read_csv(args.use_range, sep  = '\t', header = None)
        included_sites_list = []

        print('Designing guides for the following genomic ranges:')
        
        for index, row in included_sites_df.iterrows():
            print(f"{row[0]} to {row[1]}")
            for site in range(row[0], row[1] + 1):
                included_sites_list.append(site)
        

    # Creating the directories where the results will be stored in
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)
        os.makedirs(os.path.join(args.results_path, 'results_by_site/'))


    if(args.processed_sites_path):
        # Reading in the pickled dataframe of sites that will be designed for if it is directly provided
        seqs_df = pd.read_pickle(args.fasta_path) 
    else:
        # Reading in the FASTA file and processing it to generate the dataframe of sites that will be designed for
        if(args.objective == 'multi'):
            seqs_df = import_fasta.read_multi_detection(args.fasta_path, args.results_path, included_sites_list)
        elif(args.objective == 'diff'):
            seqs_df = import_fasta.read_diff_identification(args.fasta_path, args.results_path, included_sites_list, args.all_targets_one_file)
        else:
            print("Invalid objective type entered. Please enter either 'multi' or 'diff'")
            exit(1) 

    # Selecting the exploration algorithms to run based on the user's input
    if(args.exploration_algorithm == 'wgan_am'):
        explorers_to_run = [run_wgan]
        exploration_algos = ['wgan_am'] 
    elif(args.exploration_algorithm == 'evolutionary'):
        explorers_to_run = [run_evolutionary]
        exploration_algos = ['evolutionary']
    elif(args.exploration_algorithm == 'both'):
        explorers_to_run = [run_wgan, run_evolutionary]
        exploration_algos = ['wgan_am', 'evolutionary']
    else:
        print("Invalid exploration algorithm type entered. Please enter either 'wgan_am', 'evolutionary', or 'both'")
        exit(1)

    for explorer in exploration_algos:
        if not os.path.exists(os.path.join(args.results_path, f'results_by_site/{explorer}/')):
            os.makedirs(os.path.join(args.results_path, f'results_by_site/{explorer}/'))
    
    print('Running the following exploration algorithms: {} across {} genomic sites'.format(exploration_algos, len(seqs_df)))

    # Using Python's multiprocessing module to run the exploration algorithms across the genomic sites in parallel
    num_cpu = args.num_cpu
    if num_cpu == 0:
        num_cpu = max(1, multiprocessing.cpu_count() - 1)
    jobs = []
    for site, site_df in seqs_df.iterrows():
        for explorer in explorers_to_run:
            while len(jobs) >= num_cpu:
                jobs = [job for job in jobs if job.is_alive()]
                time.sleep(.1)

            p = multiprocessing.Process(target=explorer, args=(args, site_df)) 
            jobs.append(p)
            p.start()
            print('Started Explorer {} for Genomic Site {}'.format(explorer, site_df.seq_id))
    
    # Waiting for all the jobs to finish and then saving the results across all sites that both explorers considered 
    while len([job for job in jobs if job.is_alive()]) > 0:
        time.sleep(10)
        print('Waiting for all jobs across the different genomic sites to finish, and then will compile results')
 
    for explorer in exploration_algos: 
        compile_results_across_sites(explorer, args)
