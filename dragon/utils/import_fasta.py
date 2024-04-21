import fastaparser
import os
import pandas as pd
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
def valid_seqs_from_list(start_pos, seqs):
    """
    Function to return a list of valid sub-sequences from a list of sequences given a starting position
    Args:
        start_pos: Starting position of the sub-sequence
        seqs: List of sequences
    """

    return [seq[start_pos:start_pos+48] for seq in seqs 
        if all(char in ['A', 'C', 'T', 'G'] for char in seq[start_pos:start_pos+48])]


def read_multi_detection(fasta_path, temp_path, included_sites_list = None):   
    """
    Takes a fasta file for and returns a dataframe with the sequences and their target positions.
    This processes fasta files specifically for the multi-target identification task.

    Args:
        fasta_path: Path to the fasta file
        temp_path: Path to the temporary directory where the dataframe will be stored in pickle format
        included_sites_list: List of sites to include in the dataframe. If None, all sites will be included.
    """
   
    seqs = []
    items = ['target_set', 'start_pos', 'seq_id']
    seqs_dict = {item: [] for item in items}
    
    print(f'Reading in FASTA file {fasta_path}')
    with open(fasta_path) as fasta_file:           
        parser = fastaparser.Reader(fasta_file)
        for seq in parser:
            seqs.append(seq.sequence_as_string())

    min_length = min([len(seq) for seq in seqs])
    start_pos_list = range(min_length)

    # If a list of sites is provided, only include those sites
    if(included_sites_list):
        start_pos_list = [site - 11 for site in included_sites_list]
        if(min(included_sites_list) < 10):
            print("The start_pos of the included_sites_list should be 10 or greater, since the first 10 positions required to provide context for the predictive model.")
            exit(1) 
        
    for start_pos in start_pos_list:
        # Only include sequences that have valid characters and no gaps
        targets = valid_seqs_from_list(start_pos, seqs)

        # Only include the genomic site if a valid sequence is present in 80% of the input targets 
        if(len(targets) < 0.8 * len(seqs)):
            continue

        seqs_dict['target_set'].append(targets)
        seqs_dict['start_pos'].append(int(start_pos + 11))  # Add 11 because the start_pos is where the target (with 10nt context) starts, and is zero-indexed            
        seqs_dict['seq_id'].append(fasta_path.split('/')[-1].split('.fasta')[0] + "_pos" + str(start_pos + 11))
 

    virus_df = pd.DataFrame(seqs_dict)

    virus_df.to_pickle(os.path.join(temp_path, f"processed_mult_sites_{fasta_path.split('/')[-1]}.pkl"))

    print(f'Finished Importing {fasta_path}')
        
    return virus_df


def read_diff_identification(fasta_path, temp_path, included_sites_list = None, all_targets_one_file = False):
    """
    Prepares a dataframe for the differential identification task. 
    This function takes in a directory of fasta files which are already pre-aligned. This means that position 1 in fasta file 1 corresponds
    to position 1 in fasta file 2, and so on.

    Then, it creates a dataframe with all possible on-target and off-target pairs for each position in the alignment.
    For example, if the input fasta files are A.fasta, B.fasta, and C.fasta and there is only one position in the alignment,
    the output dataframe will have rows for A vs B, A vs C, and B vs C.

    Args:
        fasta_path: Path to the directory containing the fasta files
        temp_path: Path to the temporary directory where the dataframe will be stored in pickle format
        included_sites_list: List of sites to include in the dataframe. If None, all sites will be included.
    """
    
    # Creating a list of all fasta files in the directory
    fasta_files = [x for x in os.listdir(fasta_path) if 'fasta' in x and 'split' not in x]
    
    if(all_targets_one_file):
        original_file = fasta_files[0]
        with open(fasta_path + original_file) as fasta_file:           
            parser = fastaparser.Reader(fasta_file)
            for ix, seq in enumerate(parser):
                # Write each seq into a new fasta file
                with open(fasta_path + f"{seq.id}_split.fasta", "w") as f:
                    f.write(f">{seq.id}\n")
                    f.write(f"{seq.sequence_as_string()}\n")
    
        fasta_files = [x for x in os.listdir(fasta_path) if 'fasta' in x and 'split' not in x]
        fasta_files.remove(original_file)

    # Seqs is a list all sequences across all the fasta files
    seqs = []

    # Seqs_diff is a list of list of all sequences in the fasta files, for each fasta file
    seqs_diff = []

    for idx, alignment_file in enumerate(fasta_files):
        print('Reading in FASTA file ' + alignment_file)
        seqs_diff.append([])
        with open(fasta_path + alignment_file) as fasta_file:           
            parser = fastaparser.Reader(fasta_file)
            for ix, seq in enumerate(parser):
                seqs.append(seq.sequence_as_string())
                seqs_diff[idx].append(seq.sequence_as_string())

    min_length = min([len(seq) for seq in seqs])
    seqs_diff = np.array(seqs_diff)
    
    start_pos_list = []
    for start_pos in range(min_length):
        targets = valid_seqs_from_list(start_pos, seqs)
    
        # Only include the genomic site if a valid sequence is present in 80% of the input targets 
        if(len(targets) > len(seqs) * 0.8):
            start_pos_list.append(start_pos) 

        # Previously did thresholding based on a set number of target sequences
        # if(len(targets) > 300):
        #     start_pos_list.append(start_pos) 

    start_pos_list_final = start_pos_list

    # If a list of sites is provided, only include those sites
    if(included_sites_list):
        start_pos_list_final = [site - 11 for site in included_sites_list]
        if(min(included_sites_list) < 10):
            print("The start_pos of the included_sites_list should be 10 or greater, since the first 10 positions required to provide context for the predictive model.")
            exit(1) 

    df_list = []
    for idx, alignment_file in enumerate(fasta_files):
        site_df = pd.DataFrame()
        
        num_files = len(fasta_files)
        file_idx = list(range(num_files))
        file_idx.remove(idx)
        
        for start_pos in start_pos_list_final:
            target_set1_nt = valid_seqs_from_list(start_pos, seqs_diff[idx]) 
                        
            seqs_diff2 = [item for sublist in seqs_diff[file_idx] for item in sublist]
            target_set2_nt = valid_seqs_from_list(start_pos, seqs_diff2)                
            
            # Previously had a threshold on the target set size
            # if(len(target_set2_nt) < 250 or len(target_set1_nt) < 250):  
            #     continue

            # If both sets of sequences are the same, skip this site 
            if(set(target_set1_nt) == set(target_set2_nt)):
                continue
                
            target1_name = alignment_file.split('.fasta')[0]
            v2 = fasta_files.copy()
            v2.remove(alignment_file)
            v2 = [x.split('.fasta')[0] for x in v2]
            target2_name = "-".join(v2)
            site_df = site_df.append(pd.DataFrame({'seq_id': [f"{target1_name}_vs_{target2_name}.{start_pos + 11}"],
                                'start_pos': [int(start_pos + 11)], 
                                'target_set1_nt': [target_set1_nt],
                                'target_set2_nt': [target_set2_nt],
                                'target1_name' : [target1_name]  ,
                                'target2_name': [target2_name]  })).reset_index(drop = True)
    

        df_list.append(site_df)  
                
    master_df = pd.concat(df_list).reset_index(drop = True)
    master_df.to_pickle(os.path.join(temp_path + 'processed_variant_identification_sites.pkl'))
    
    return master_df
