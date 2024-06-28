"""Define the model class for the variant identification objective."""
import numpy as np
from badgers.utils import prepare_sequences as prep_seqs
import tensorflow as tf
from badgers.utils import cas13_cnn as cas13_cnn
import pandas as pd 
from flexs import Model

class Cas13Diff(Model):
    """
    Use CNN models to compute the fitness of Cas13a guides designed for the variant identification task 
    """

    def __init__(
        self, landscape, target_set1_nt, target_set2_nt, grid,
        name = 'Cas13-CNN-DiffIdentification'):
        """
        Use CNN models to compute the fitness of Cas13a guides designed for the variant identification task 
        Args:
            landscape: The ground truth landscape.
            target_set1: A list of sequences for the on-target set
            target_set2: A list of sequences for the off-target set

        """
        super().__init__(name)

        # The predictive models require 10 nt of context on each side of the 28 nt probe-binding region
        context_nt = 10

        # Preparing the target sets for the search process
        self.grid = grid
        self.landscape = landscape
        self.target_set1_nt = target_set1_nt
        self.target_set2_nt = target_set2_nt
        self.target_set1_no_context_nt = [x[context_nt:-context_nt] for x in self.target_set1_nt]
        self.target_set2_no_context_nt = [x[context_nt:-context_nt] for x in self.target_set2_nt]

       
        self.target_cons_nt = prep_seqs.consensus_seq(self.target_set1_nt, nt = True) 
        self.baseline_seq = self.target_cons_nt[context_nt:-context_nt]
        self.target_set1_onehot = [prep_seqs.one_hot_encode(x) for x in target_set1_nt]
        self.target_set2_onehot = [prep_seqs.one_hot_encode(x) for x in target_set2_nt]

        self.unique_target_set1_no_context_nt = np.unique(self.target_set1_no_context_nt)
        self.unique_target_set1_no_context_nt = list(self.unique_target_set1_no_context_nt)

        self.unique_target_set2_no_context_nt = np.unique(self.target_set2_no_context_nt)
        self.unique_target_set2_no_context_nt = list(self.unique_target_set2_no_context_nt)

        # Creating the parent dataframe for the evoutionary algorithm, which is simply all of the sequences
        # present in the on-target and off-target set
        parent_seqs =  self.unique_target_set1_no_context_nt + self.unique_target_set2_no_context_nt

        self.utarget_set1_onehot, self.utarget_set1_freqs = np.unique(self.target_set1_onehot, return_counts = True, axis = 0)
        self.utarget_set1_freqs = self.utarget_set1_freqs/len(self.target_set1_onehot)

        self.utarget_set2_onehot, self.utarget_set2_freqs = np.unique(self.target_set2_onehot, return_counts = True, axis = 0)
        self.utarget_set2_freqs = self.utarget_set2_freqs/len(self.target_set2_onehot)

        self.previous_preds = {}

        # print('Predicting Activity of {}'.format(len(parent_seqs)))
        parent_fitness = self._fitness_function(parent_seqs)

        self.parent_df = pd.DataFrame({'sequence' : parent_seqs, 'fitness': parent_fitness})

        # print('Parent Sequences DF:')
        # print(self.parent_df)


    def train(self, sequences, scores):
        """
        FLEXS requires this to be defined, but for our application, the model is already trained on experimental data
        """
        # print("Predictor already trained")

    def _fitness_function(self, sequences, output_type = 'fitness'):
        """
        Compute the fitness of a list of sequences for the variant identification fitness function
        Args:
            sequences: A list of guide sequences to evaluate
            target_set1: A list of sequences for the on-target set. If not provided, the default is the unique targets in the on-target set.
            target_set1_freqs: The frequency of each sequence in target_set1. If not provided, the default is the frequency of unique targets in the on-target set.
            target_set2: A list of sequences for the off-target set. If not provided, the default is the unique targets in the off-target set.
            target_set2_freqs: The frequency of each sequence in target_set2.  If not provided, the default is the frequency of unique off-targets in the off-target set.
            save: Tells the function to return a full set of values benchmarking the guide performance
            output_type: Tells the function to return the fitness or the predicted activity of the guide
            """
        # When we evaluate the fitness of a guide, we use the target set and target frequencies that were provided when the model was initialized
        target_set1 = self.utarget_set1_onehot
        target_set1_freqs = self.utarget_set1_freqs 

        target_set2 = self.utarget_set2_onehot
        target_set2_freqs = self.utarget_set2_freqs 

        # There are three cases for the input to this function:
        # 1. A single guide sequence in a string format
        # 2. A list with a single guide sequence in it
        # 3. A list of guide sequences

        # The input into predict_activity must be a list, and this function must return a list, so we handle each of these cases

        # If the input is a single guide sequence in a string format, one hot encode it and predict activity and return it within a list
        if(len(sequences[0]) == 1 and len(sequences) == 28):
            sequence_onehot = prep_seqs.one_hot_encode(sequences)
            fitness = self.predict_activity([sequence_onehot], target_set1, target_set1_freqs, target_set2, target_set2_freqs, output_type = output_type).numpy()
            return np.array([fitness])

        # If the input is a list with a single guide sequence in it, one hot encode the sequence and predict activity and return it within a list
        elif(len(sequences[0]) == 28 and len(sequences) == 1):
            sequence_onehot = prep_seqs.one_hot_encode(sequences[0])
            fitness = self.predict_activity([sequence_onehot], target_set1, target_set1_freqs, target_set2, target_set2_freqs, output_type = output_type).numpy()
            return np.array([fitness])

        # If the input is a list of guide sequences, one hot encode the each of these sequences and predict activity
        else:
            sequences_onehot = [prep_seqs.one_hot_encode(sequence) for sequence in sequences]
            fitnesses = self.predict_activity(sequences_onehot, target_set1, target_set1_freqs, target_set2, target_set2_freqs, output_type = output_type).numpy() 
            return fitnesses
            
    def _diff_fitness_function(self, sequences):
        """
        The WGAN-AM explorer requires a differentiable fitness function, so we use this fitness
          function during the WGAN-AM search process and return a tensorflow tensor rather than a numpy object
        """
        return self.predict_activity([sequences], self.utarget_set1_onehot, self.utarget_set1_freqs , self.utarget_set2_onehot, self.utarget_set2_freqs )

    def predict_activity(self, gen_guide, target_set1, target_set1_freqs, target_set2, target_set2_freqs, output_type = 'fitness', full_model = False):
            """Predict the fitness of a guide sequence in the variant identification task
            Args:
                gen_guide: A list of one hot encoded guide sequences to evaluate
                target_set1: A list of one hot encoded target sequences for the on-target set. 
                target_set1_freqs: A list of of the frequences of each target sequence in the on-target set.
                target_set2: A list of one hot encoded target sequences for the off-target set. 
                target_set2_freqs: A list of of the frequences of each target sequence in the off-target set.
            Returns:
                Fitness of each guide

            NOTE: Throughout the code, we compute the 'cost' of the guide, but then take the negative at the end to compute the fitness
            """
             # Check that the number of targets equals the length of the frequency list provided
            assert len(target_set1) == len(target_set1_freqs)
            assert len(target_set2) == len(target_set2_freqs)

            # Check that the combined frequency of the targets is approximately 1
            assert sum(target_set1_freqs) > 0.99
            assert sum(target_set1_freqs) < 1.01
            assert sum(target_set2_freqs) > 0.99
            assert sum(target_set2_freqs) < 1.01 

            # Run the predictive models on both the off-target and on-target sets
            pred_perf_t1, pred_act_t1 = cas13_cnn.run_full_model(gen_guide, target_set1)
            pred_perf_t2, pred_act_t2 = cas13_cnn.run_full_model(gen_guide, target_set2)


            # Compute the activity of the gudie using both the classifier and predictor model
            pos4 = tf.constant(4.0)
            weighted_perf_t1 = tf.math.subtract(tf.math.multiply(pred_act_t1, tf.math.add(pred_perf_t1, pos4)), pos4)
            weighted_perf_t2 = tf.math.subtract(tf.math.multiply(pred_act_t2, tf.math.add(pred_perf_t2, pos4)), pos4)
 
            # Compute the cost of the guide using the per-target activity of the guide and the frequency of the targets
            if(len(gen_guide) > 1):
                t1_cost_input = tf.math.reduce_sum(tf.math.multiply(weighted_perf_t1, target_set1_freqs), axis = 1) 
                t2_cost_input = tf.math.reduce_sum(tf.math.multiply(weighted_perf_t2, target_set2_freqs), axis = 1)   
            else:
                t1_cost_input = tf.math.reduce_sum(tf.math.multiply(weighted_perf_t1, target_set1_freqs))  
                t2_cost_input = tf.math.reduce_sum(tf.math.multiply(weighted_perf_t2, target_set2_freqs))     
          
            c = self.grid['c']
            a = self.grid['a']
            k = self.grid['k']
            o = self.grid['o']
            
            # Compute the cost of the guide sequence using the hyperparameters for the fitness function
            t2cost = tf.math.divide(tf.constant(c), 
            tf.math.add(tf.constant(1.0), tf.math.scalar_mul(tf.constant(a), tf.math.exp(tf.math.scalar_mul(tf.constant(k), tf.math.subtract(t2_cost_input, o)
            ))))) 

            t1cost = tf.math.subtract(tf.constant(c), tf.math.divide(tf.constant(c), 
            tf.math.add(tf.constant(1.0), tf.math.scalar_mul(tf.constant(a), tf.math.exp(tf.math.scalar_mul(tf.constant(k), tf.math.subtract(t1_cost_input, o)
            )))))) 

            cost = tf.math.add(tf.math.scalar_mul(tf.constant(self.grid['t2w']), t2cost), t1cost)

            # Return the requested output
            if(output_type == 'eval'):
                return tf.constant([np.mean(weighted_perf_t1), np.mean(weighted_perf_t2),np.mean(weighted_perf_t1) - np.mean(weighted_perf_t2)])
            elif(output_type == 'save'):
                # Return a long vector with various performance characteristics ofd the guide
                return tf.constant([np.mean(weighted_perf_t1), np.mean(weighted_perf_t2), np.mean(weighted_perf_t1) - np.mean(weighted_perf_t2), 
                t1_cost_input.numpy(), t2_cost_input.numpy(), t1cost.numpy(), t2cost.numpy(), 
                np.mean(pred_perf_t1), np.mean(pred_perf_t2), np.mean(pred_act_t1), np.mean(pred_act_t2), tf.math.negative(cost).numpy()])

            else:
                # Return the fitness of the guide
                return tf.math.negative(cost) 
 




