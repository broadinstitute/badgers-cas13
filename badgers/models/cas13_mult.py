"""Define the multi-target detection model class."""
import numpy as np
from badgers.utils import prepare_sequences as prep_seqs
import tensorflow as tf
from badgers.utils import cas13_cnn as cas13_cnn
import pandas as pd 
from flexs import Model

class Cas13Mult(Model):
    """
    Uses CNN models to compute the average activity over sequence diversity for the multi-target
    detection objective function
    """

    def __init__(
        self, landscape, target_set_nt,
        name = 'cas13_mult'):
        """
        Use the CNN model of Cas13 activity to compute expected activity across sequence diversity

        Args:
            target_set_nt: A list of 48 nucleotide-long strings representing the set of targets the guides activity
            is being predicted against.
        """
        super().__init__(name)

        # Each of the targets is 48 nt (28 nt of the probe-binding region and 10 nt of context on each side)
        context_nt = 10

        self.target_set_nt = target_set_nt

        self.landscape = landscape

        # Converting the target set to a one-hot-encoded vector
        self.target_set_onehot = [prep_seqs.one_hot_encode(x) for x in target_set_nt]
        self.target_set_no_context_nt = [x[context_nt:-context_nt] for x in self.target_set_nt]

        # Computing the consensus sequence with and without context
        self.target_cons_no_context_nt = prep_seqs.consensus_seq(self.target_set_no_context_nt, nt = True)
        self.target_cons_nt = prep_seqs.consensus_seq(self.target_set_nt, nt = True)

        # Computing the unique sequences in the target set and their frequencies
        self.unique_target_set_no_context_nt, unique_idx = np.unique(self.target_set_no_context_nt, return_index = True)
        self.unique_target_set_no_context_nt = list(self.unique_target_set_no_context_nt)
        self.target_set_utargets_onehot, self.target_set_counts = np.unique(self.target_set_onehot, return_counts = True, axis = 0)
        self.target_set_freqs = self.target_set_counts/len(self.target_set_onehot)

        # The set of parent sequences the algorithms are seeded on includes the consensus sequence and the 
        # unique sequences in the target set 
        parent_seqs = [self.target_cons_no_context_nt] + self.unique_target_set_no_context_nt

        self.baseline_seq = self.target_cons_no_context_nt

        # Evaluating the parent sequences and creating the dataframe of parent sequence fitness
        # print(f'Creating Parent DF with {len(parent_seqs)} seqs')
        parent_fitness = self.get_fitness(parent_seqs)
        self.parent_df = pd.DataFrame({'sequence' : parent_seqs, 'fitness': parent_fitness})

        # print('Parent Sequences DF Complete')
        # print(f'Cost Before Explorer Run: {self.cost}')

    def _fitness_function(self, sequences, output_type = 'fitness'):
        """
        Use the CNN model of Cas13 activity to compute the fitness for the multi-target detection application

        Args:
            sequences: A 28nt probe sequence or a list of 28nt probe sequences whose fitness is being evaluated
            target_set: A one-hot-encoded list of 48 nt sequences against which to predict the activity of the probe.
            target_freqs: A list of 
        """
        # When we evaluate the fitness of a guide, we use the target set and target frequencies that were provided when the model was initialized
        target_set = self.target_set_utargets_onehot
        target_freqs = self.target_set_freqs 

        # There are three cases for the input to this function:
        # 1. A single guide sequence in a string format
        # 2. A list with a single guide sequence in it
        # 3. A list of guide sequences

        # The input into predict_activity must be a list, and this function must return a list, so we handle each of these cases

        # If the input is a single guide sequence in a string format, one hot encode it and predict activity and return it within a list
        if(len(sequences[0]) == 1 and len(sequences) == 28):
            sequence_onehot = prep_seqs.one_hot_encode(sequences)
            fitness = self.predict_activity([sequence_onehot], target_set, target_freqs = target_freqs, output_type = output_type)
            return np.array([fitness])

        # If the input is a list with a single guide sequence in it, one hot encode the sequence and predict activity and return it within a list
        elif(len(sequences[0]) == 28 and len(sequences) == 1):
            sequence_onehot = prep_seqs.one_hot_encode(sequences[0])
            fitness = self.predict_activity([sequence_onehot], target_set, target_freqs = target_freqs, output_type = output_type)
            return np.array([fitness])

        # If the input is a list of guide sequences, one hot encode the each of these sequences and predict activity
        else:
            sequences_onehot = [prep_seqs.one_hot_encode(sequence) for sequence in sequences]
            fitnesses = self.predict_activity(sequences_onehot, target_set, target_freqs = target_freqs, output_type = output_type) 
            return fitnesses

    def _diff_fitness_function(self, sequences):
        """
        The WGAN-AM explorer requires a differentiable fitness function, so we use this fitness
          function during the WGAN-AM search process and return a tensorflow tensor rather than a numpy object
        """
        # Check that the input sequence is a guide sequence with no context
        assert len(sequences) == 28

        # Return the predicted activity
        return self.predict_activity([sequences], self.target_set_utargets_onehot, self.target_set_freqs, diff = True)

    
    def train(self, sequences, scores):
        """
        FLEXS requires this to be defined, but for our application, the model is already trained on experimental data
        """
        # print("Predictor already trained")

    def predict_activity(self, gen_guide, target_set, target_freqs, output_type = 'fitness', diff = False):
        """Predict fitness of a generated guide against the target set.
        Args:
            gen_guide: generated guide (no context / unpadded)
            target_set: A one-hot-encoded list of 48 nt sequences against which to predict the activity of the probe.
            target_freqs: A list of frequencies for each of the targets in the target set
        Returns:
            weighted performance (expected activity of guide over target set)
        """  

        # Check that the number of targets equals the length of the frequency list provided
        assert len(target_set) == len(target_freqs)

        # Check that the combined frequency of the targets is approximately 1
        assert sum(target_freqs) > 0.99
        assert sum(target_freqs) < 1.01
            
        # Run the model on the generated guide and the target set
        pred_perf_list, pred_act_list  = cas13_cnn.run_full_model(gen_guide, target_set)  
            
        # Compute the combined predictor and classifier activity of the guide for each of the targets in the target set
        pos4 = tf.constant(4.0)
        unweighted_vec = tf.math.subtract(tf.math.multiply(pred_act_list, tf.math.add(pred_perf_list, pos4)), pos4)

        # Compute the final fitness by weighting each of the per-target activity by the frequency of that target in the target set
        if(len(gen_guide) > 1):
            weighted_fitness = tf.math.reduce_sum(tf.math.multiply(unweighted_vec, target_freqs), axis = 1) 
         
        else:
            weighted_fitness = tf.math.reduce_sum(tf.math.multiply(unweighted_vec, target_freqs)) 

        # Return the fitness in the format requested
        if(output_type == 'perf_list'):
            return pred_perf_list

        elif(output_type == 'act_list'):
            return pred_act_list

        elif(output_type == 'perc_active'): 
            return np.sum((np.array(pred_act_list) > 0.577467) * np.array(target_freqs))

        elif(output_type == 'perc_highly_active'): 
            # For a guide to be considered highly active on a target, it needs to be classified active by the classifier model in top 25th percentile of regression model 
            return np.sum(((np.array(pred_act_list) > 0.577467) & (np.array(pred_perf_list) > -1.2801363)) * np.array(target_freqs))

        elif(output_type == 'target_freqs'):
            return target_freqs

        # If we are computing fitness during the WGAN-AM search process, we need to return a tensorflow tensor, not a numpy object
        else:
            if(diff):
                return weighted_fitness
            else:
                return weighted_fitness.numpy()
