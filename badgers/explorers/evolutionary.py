"""Defines the evolutionary algorithm explorer class.
This algorithm was inspired by the evolutionary algorithms described in the follow papers:
https://arxiv.org/abs/2010.02141, https://www.nature.com/articles/s41467-018-03746-3"""

from typing import Optional, Tuple
#from badgers.cas13_explorer import Cas13Explorer
import flexs
from badgers.utils import prepare_sequences as prep_seqs
from badgers.utils.prepare_sequences import track_performance as track_performance
import numpy as np
from numpy.random import choice
import pandas as pd

class EvolutionaryExplorer(flexs.Explorer):
    """
    Defines the evolutionary explorer class for designing Cas13a guides.
    Args:
        rounds: The evolutionary algorithm is only run once for each target site, so this parameter isn't used here
        model_queries_per_batch: Number of times the predictive model is evaluated per round - used to constrain runtime
        sequences_batch_size: Number of guides returned per round
        gamma: Gamma is the rate of mutation when generating children guides
        S: S is the size of the population of guides
        j: J is the fraction of the population that is replaced by children guides each round
        beta: Beta is a parameter that alters the selection intensity when sampling guides
        starting_sequence: FLEXS requires this argument, but since the evolutionary algorithm samples from the alignment to create a list of parent sequences, this argument is not used
    """ 

    def __init__(
        self,
        model,
        rounds: int,
        model_queries_per_batch: int,
        sequences_batch_size: int,
        gamma: float, 
        S: int,
        j: float, 
        beta: float,
        local_s : bool,
        log_file: Optional[str] = None,
        track_perf: Optional[str] = ''):
         
        name = "evolutionary"
        starting_sequence = "not applicable"
        self.gamma = gamma
        self.S = S
        self.j = j
        self.beta = beta
        self.track_perf = track_perf
        self.local_s = local_s

        super().__init__(
        model,
        name,
        rounds,
        sequences_batch_size,
        model_queries_per_batch,
        starting_sequence,
        log_file,)
        

         
    def sample_guides(self, guides, guides_fitness, number_of_guides_to_sample: int):
        """Sample `number_of_guides` guides from the input list of guides proportional to their fitness.
        Args:
            guides: A list of guides.
            guides_fitness: A list of fitnesses for each guide.
            number_of_guides_to_sample: The number of guides to sample.
        Returns:
            A list of sampled guides.
        """
        
        exp_fitnesses = np.exp(guides_fitness / self.beta)

        # Converting this vector to float64 to avoid 'probabilities don't sum to 1' error with np choice function
        exp_fitnesses = np.asarray(exp_fitnesses).astype('float64') 
        probability_distribution = exp_fitnesses / exp_fitnesses.sum()

        idx = choice(range(0, len(guides)), number_of_guides_to_sample, p=probability_distribution, replace = True)
        return [[guides[i] for i in idx], [guides_fitness[i] for i in idx]] 
    
        
    def mutate_guide(self, DNA_guide, gamma):
        """Mutate a DNA guide.
        Args:
            DNA_guide: A string representing a DNA guide.
            mutation_rate: The probability of a mutation.
        Returns:
            A mutated DNA guide.
        """
        DNA_guide = list(DNA_guide)
        new_guide = []
        for idx, base in enumerate(DNA_guide):
            if np.random.rand() < gamma:
                new_guide.append(np.random.choice(list("ATCG")))
            else:
                new_guide.append(base)

        return "".join(new_guide)
         
    def propose_sequences(self, measured_guides: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]: 
        """Generate new guides via the evolutionary algorithm
        Args:
            measured_guides: FLEXS requires this input dataframe, but it is not used in this function.
            Rather, the parent guides are sampled from the sequences present in the target set (those in self.model.parent_df)
        Returns:
            A tuple of numpy arrays containing the generated guides and their fitnesses."""

        guides_df = pd.DataFrame(columns=["guide", "fitness"])

        # Sample guides from the input target guides proportional to their fitness.
        parent_guides, parent_fitness = self.sample_guides(self.model.parent_df.sequence, self.model.parent_df.fitness, self.S)
        
        guides_df = guides_df.append(pd.DataFrame({"guide": parent_guides,"fitness": parent_fitness}), ignore_index = True)

        current_generation = 0
        total_guides_evaluated = 0
        # Mutate the sampled parent guides and to generate children guides
        while total_guides_evaluated < self.model_queries_per_batch:
            parent_guides, parent_fitness = self.sample_guides(guides_df.guide, guides_df.fitness, int(self.j * self.S)) 
            child_guides_to_evaluate = []
            
            for parent_guide in parent_guides:
                child_guide = self.mutate_guide(parent_guide, self.gamma)

                # If the generated guide seq has not been seen before, then evaluate it
                # We previously removed generated guide seqs that happenened to be present in the target set, but don't do that anymore
                if child_guide not in guides_df.guide.to_list():
                    child_guides_to_evaluate.append(child_guide)
                    total_guides_evaluated += 1
            
            # If we have generated new guides, then evaluate them
            if(len(child_guides_to_evaluate) > 0):
                children_fitness = self.model.get_fitness(child_guides_to_evaluate)

                # Sort the dataframe and delete the worst performing guides and add the children guides
                guides_df = guides_df.sort_values(by = "fitness", ascending = False).reset_index(drop = True)
                guides_df = guides_df.iloc[:-len(children_fitness)]
                guides_df = guides_df.append(pd.DataFrame({"guide": child_guides_to_evaluate,"fitness": children_fitness}), ignore_index = True)

            if(self.track_perf):  
                # Tracks guide performance over optimization process
                guides_df = guides_df.reset_index(drop = True)
                max_idx = guides_df.fitness.idxmax()
                track_performance(current_generation, total_guides_evaluated, guides_df.guide[max_idx], 
                self.model.baseline_seq, guides_df.fitness[max_idx], guides_df.guide, self.track_perf)
            
            current_generation += 1
        
        # If enabled, this local search step takes all the positions at which the generated guide differs from the baseline sequence
        # and then mutates the guide at each of these positions to generate a list of mutated guides   
        if(self.local_s):
            mut_seqs = []
            sorted_guides = guides_df.sort_values(by = "fitness", ascending = False).guide.to_list()
            for child in sorted_guides[:5]:
                newseqs = prep_seqs.mismatch_nuc_opt(self.model.baseline_seq, child)
                for seq in newseqs:
                    mut_seqs.append(seq)
                    
            mut_seqs = list(set(mut_seqs))
            mut_perf = self.model.get_fitness(mut_seqs)
            guides_df = guides_df.append(pd.DataFrame({"guide": mut_seqs,"fitness": mut_perf}), ignore_index = True)

        # Sort dataframe by fitness, from highest to lowest
        guides_df = guides_df.sort_values(by = "fitness", ascending = False)
        guides_df = guides_df.iloc[0: self.sequences_batch_size]

        return np.array(guides_df.guide), np.array(guides_df.fitness)  
 