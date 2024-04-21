"""Functions to prepare sequences for other tasks in DRAGON-driven guide design"""
import pandas as pd
import os
import numpy as np
import editdistance

def sub_t_u(seq):
    """
    Function to substitute all thymine bases with uracil bases
    Args:
        seq: DNA sequence
    Returns:   
        seq: RNA sequence
    """
    newseq = []

    for base in seq:
        if(base == 'T' or base == 't'):
            newseq.append('u')
        else:
            newseq.append(base)
    
    return ''.join(newseq).lower()

def mismatch_positions(a, b):
    """Calculate positions of mismatches.
    Args:
        a/b: two strings
    Returns:
        list x of length len(a)==len(b) such that x[i] is 1 if
        a[i] != b[i] and 0 if a[i]==b[i]
    """
    assert len(a) == len(b)
    return [1 if a[i] != b[i] else 0 for i in range(len(a))]


def hamming_dist(a, b):
    """Calculate Hamming distance.
    Args:
        a/b: two strings
    Returns:
        Hamming distance between a and b
    """
    return sum(mismatch_positions(a, b))


def plt_convert(seqs):
    """Convert sequences to numerical representation for plotting."""
    nuc_dict = {'A' : 1, 'C' : 2, 'G': 3, 'T':4}

    return [[nuc_dict[base] for base in seq] for seq in seqs]


def mismatch_positions(a, b):
    """Calculate positions of mismatches.
    Args:
        a/b: two strings
    Returns:
        list x of length len(a)==len(b) such that x[i] is 1 if
        a[i] != b[i] and 0 if a[i]==b[i]
    """
    assert len(a) == len(b)
    return [1 if a[i] != b[i] else 0 for i in range(len(a))]

nuc_dict = {'A' : 1, 'C' : 2, 'G': 3, 'T':4}


def mismatch_nucleotides(a, b):
    """Calculate positions of mismatches.
    Args:
        a/b: two strings
    Returns:
        list x of length len(a)==len(b) where each 
    """
    assert len(a) == len(b)
    colors = [nuc_dict[a[i]] if a[i] != b[i] else 0 for i in range(len(a))]

    for pos in range(len(a)):
        if((a[pos] == 'C' and b[pos] == 'T') or (a[pos] == 'A' and b[pos] == 'G')):
            colors[pos] = 5
    
    return colors


def complement_base(base):
    """Returns the complement of a base."""
    
    if base in 'Aa':
        return 'T'
    elif base in 'Tt':
        return 'A'
    elif base in 'Gg':
        return 'C'
    else:
        return 'G'


def revcomp(seq):
    """Compute reverse complement of a sequence."""
    
    # Initialize reverse complement
    rev_seq = ''
    
    # Loop through and populate list with reverse complement
    for base in reversed(seq):
        rev_seq += complement_base(base)
        
    return rev_seq


def convert_to_nt(x):
    """Convert 4-channel encoding of sequence into nucleotides.
    Args:
        x: array such that len(x) is the sequence length, and x[i] has
            4 numerical elements denoting sequence (x[i][0] is A,
            x[i][1] is C, x[i][2] is G, x[i][3] is T); may be one-hot
            encoded or represent a softmax over the nucleotides; if
            all values at x[i] are 0, this uses '-' as the nucleotide
    Returns:
        string s represented by x, such that s[i] is the nucleotide
        with the max value at x[i]
    """
    onehot_idx = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    def nt(xi):

        if sum(xi) == 0:
            # All values are 0, use '-'
            return '-'
        else:
            assert np.isclose(sum(xi), 1.0) # either one-hot encoded or softmax
            return onehot_idx[np.argmax(xi)] 

    
    assert x.shape[1] == 4


    return ''.join(nt(xi) for xi in np.array(x))
    #changed from np.array(x)

def convert_to_nucleotides(x):
    """Convert 4-channel encoding of sequence into nucleotides."""
    return convert_to_nt(x)

def one_hot_encode(seq):
    """One-hot encode a sequence.
    Args:
        seq: string representing a sequence
    Returns:
        2D numpy array with shape (len(seq), 4) representing the
        one-hot encoding of seq
    """
    mapping = dict(zip("ACGT", range(4)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]

def mismatch_nuc_specific(a, b):
    """Calculate positions of mismatches.
    Args:
        a/b: two strings
    Returns:
        list x of length len(a)==len(b) such that x[i] is 1 if
        a[i] != b[i] and 0 if a[i]==b[i]
    """

    nuc_dict = {'A' : 1, 'C' : 2, 'G': 3, 'T':4}
    assert len(a) == len(b)
    nuc_specific = [nuc_dict[a[i]] if a[i] != b[i] else 0 for i in range(len(a))]
    binary = [1 if a[i] != b[i] else 0 for i in range(len(a))]

    return [binary, nuc_specific, sum(binary)]


def avg_seq(target_set):
    """
    Calculate average sequence from an input target set - this function is used in the following consensus_seq function.
    Args:
        target_set: list of sequences
    Returns:
        array of shape (len(target_set[0]), 4) such that each row is the average
        of the corresponding position in the target set
    """
    target_avg = np.mean(target_set, axis = 0)

    return target_avg

def consensus_seq(target_set, nt = False):
    """
    Calculate consensus sequence from target set.
    Args:
        target_set: list of sequences
        nt: whether to return consensus sequence in nucleotide form, or in one-hot encoding
    """

    if(nt):
        target_avg = avg_seq([one_hot_encode(x) for x in target_set])
        target_cons = np.eye(4)[np.argmax(target_avg, axis=-1)]
        target_cons = convert_to_nucleotides(target_cons)

    else:
        target_avg = avg_seq(target_set)
        target_cons = np.eye(4)[np.argmax(target_avg, axis=-1)]

    return target_cons


def shannon_entropy(list_input):
    """Calculate Shannon's Entropy per column of the alignment (H=-\sum_{i=1}^{M} P_i\,log_2\,P_i
    Citation: https://gist.github.com/jrjhealey/130d4efc6260dd76821edc8a41d45b6a?permalink_comment_id=3195609"""

    import math
    unique_base = set(list_input)
    M   =  len(list_input)
    entropy_list = []
    # Number of residues in column
    for base in unique_base:
        n_i = list_input.count(base) # Number of residues of type i
        P_i = n_i/float(M) # n_i(Number of residues of type i) / M(Number of residues in column)
        entropy_i = P_i*(math.log(P_i,2))
        entropy_list.append(entropy_i)

    sh_entropy = -(sum(entropy_list))

    return sh_entropy


def shannon_entropy_msa(alignment): 
    """Calculate Shannon Entropy across the whole MSA
    Citation: https://gist.github.com/jrjhealey/130d4efc6260dd76821edc8a41d45b6a?permalink_comment_id=3195609"""
    
    alignment = np.array([list(x) for x in alignment])

    shannon_entropy_list = []
    for col_no in range(len(list(alignment[0]))):
        list_input = list(alignment[:, col_no])
        shannon_entropy_list.append(shannon_entropy(list_input))

    return sum(shannon_entropy_list)/len(shannon_entropy_list)


def track_performance(run_id, run_id2, guide, starting, fitness, population, results_path):
    """
    Track performance of the algorithm by saving the run_id, guide, starting sequence, hamming distance, fitness and shannon entropy of the population at each iteration.
    Args:
        run_id: unique identifier for the run
        guide: guide sequence
        starting: starting sequence
        hd: hamming distance
        fitness: fitness of the maximally-fit guide at that iteration
        population: population of guides at each iteration
        results_path: path to save the results  
    """
    hd = hamming_dist(guide, starting)

    if(shannon_entropy_msa(population) == 0):
        method = 'WGAN-AM'
    else:
        method = 'Evolutionary'

    r_dict = {'run_id': run_id, 'run_id2': run_id2,'guide_sequence': guide, 'starting_sequence': starting, 
              'hd': hd, 'fitness': fitness, 'shannon_entropy': shannon_entropy_msa(population), 'method': method}
    
    results = pd.DataFrame(r_dict, index = [0])
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    file_path = f"{results_path}{method}_{starting}.pkl"
 
    if(os.path.exists(file_path)):
        old_results = pd.read_pickle(file_path).reset_index(drop = True)
        upd_results = old_results.append(results, ignore_index = True)
        upd_results.to_pickle(file_path)
    else:
        results.to_pickle(file_path)
        

def edit_distance_index(gen_df, top_n, edit_dist_thresh = 5):
    """
    Function to return the index of the top n sequences in a dataframe based on edit distance
    Args:
        gen_df: dataframe of generated sequences
        top_n: number of sequences to return
        edit_dist_thresh: edit distance threshold
    Returns:
        index: index of the top n sequences based on edit distance"""
    
    index = [0]
    
    i = 1
    while(len(index) < top_n and i < len(gen_df)):
        
        dist = min([editdistance.eval(gen_df.iloc[z].guide_sequence, gen_df.iloc[i].guide_sequence) for z in index])
        
        if(dist > edit_dist_thresh):
            index.append(i)
            
        i += 1
 
    return index

def mismatch_nuc_opt (refseq, genseq, hd = 3):
    """
    Function to generate all possible sequences with a given hamming distance from a reference sequence
    Args:
        refseq: Reference sequence
        genseq: Generated sequence
        hd: Hamming distance
    Returns:
        newseqs: List of all possible sequences with a given hamming distance from the reference sequence
    """

    mismatch_pos = [i for i in range(len(refseq)) if refseq[i] != genseq[i]]

    if(len(mismatch_pos) > hd):
        mismatch_pos = np.random.choice(mismatch_pos, size = hd, replace = False)
        
    newseqs = [refseq]
    for pos in mismatch_pos:
        for nucleotide in ['A', 'T', 'C', 'G']:
            for ref in newseqs:
                ref_list = list(ref)
                ref_list[pos] = nucleotide
                newseqs.append("".join(ref_list))
                newseqs = list(set(newseqs))
            
    return newseqs