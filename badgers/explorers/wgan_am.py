"""Defines the WGAN-AM explorer class."""
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import os 

import flexs
from badgers.utils.prepare_sequences import mismatch_nuc_specific as mismatch_nuc_specific
from badgers.utils.prepare_sequences import track_performance as track_performance
import badgers.utils.prepare_sequences as prep_seqs
import badgers.utils.gan as gan
import tensorflow as tf

class WGANExplorer(flexs.Explorer):
    """
    Defining the WGAN-AM explorer class
    Algorithm works by randomly sampling from a latent distribution,
    using a Wasserstein generative adversarial network to generate a probe using that latent variable,
    and then iteratively updating that latent variable to explore regions of the sequence landscape that have higher activity.
    Args:
        inner_iterations: Number of steps that each run of the WGAN-AM algorithm takes
        starting_sequence: The sequence conditioned upon by the WGAN-AM, generally the consensus sequence
        adam_lr: Learning rate for the Adam optimizer
        optimizer: Optimizer used for the WGAN-AM algorithm
        rounds: Number of starting points for the WGAN-AM algorithm
        model_queries_per_batch: Required by FLEXS, but not used in this explorer
        sequences_batch_size: Required by FLEXS, but not used in this explorer. Only one guide is returned per round of the WGAN-AM algorithm
    """

    def __init__(
        self,
        model,
        rounds: int,
        inner_iterations: int,
        starting_sequence: str,
        adam_lr: float,
        optimizer: str,
        model_queries_per_batch: int,
        sequences_batch_size : int,
        local_s : bool,
        log_file: Optional[str] = None,
        track_perf: Optional[str] = ''
    ):
        name = "WGAN_AM_outeri={rounds}"
        self.track_perf = track_perf

        super().__init__(
            model,
            name,
            rounds,
            sequences_batch_size,
            model_queries_per_batch,
            starting_sequence,
            log_file,
        )

        absolute_path = os.path.dirname(__file__)
        relative_path = '../' 
        full_path = os.path.join(absolute_path, relative_path)
        
        # Loading in the generator model for the WGAN
        global gen_model
        gen_model, _ = gan.load_models(full_path + 'utils/gan_data')

        self.starting_sequence = starting_sequence
        self.starting_sequence_onehot = prep_seqs.one_hot_encode(starting_sequence)
        self.inner_iterations = inner_iterations    
        self.adam_lr = adam_lr 
        self.rounds = rounds
        self.local_s = local_s
        self.optimizer = optimizer
            
    def propose_sequences(self, measured_sequences: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Generate new guides via the WGAN-AM algorithm
         Args:
            measured_sequences: FLEXS requires this input dataframe, but it is not used in this function. 
            Rather, the parent guides are sampled from the sequences present in the target set (those in self.model.parent_df)
        Returns:
            A tuple of numpy arrays containing the generated guides and their fitnesses."""

        last_round = measured_sequences["round"].max()
        
        # A new random seed is set for each round, so the random starting point is different for each round
        gan.set_seed(last_round + np.random.randint(0, 100))
        run_id = last_round

        # Defining the tensorflow latent variable, which is 10-dimensional
        z = tf.Variable(
            initial_value=tf.random.normal([1, 10]),
            trainable=True,
            name='latent')

        @tf.function
        def tf_activity_loss():
            return self.activity_loss(z)

        # Define the optimizer
        if(self.optimizer == 'adam'):
            opt = tf.optimizers.Adam(learning_rate= self.adam_lr)
        elif(self.optimizer == 'sgd'):
            opt = tf.optimizers.SGD(learning_rate= self.adam_lr)
        elif(self.optimizer == 'rmsprop'):
            opt = tf.optimizers.RMSprop(learning_rate= self.adam_lr)
        
        # Run the optimization process inner_i times
        for inner_i in range(self.inner_iterations):

            # Step z in a direction to minimize activity_loss and generate optimally fit guides
            # Here, we are minimizing the -1*fitness, which is the same as maximizing the fitness
            opt.minimize(tf_activity_loss, var_list=[z])

            if(self.track_perf): 
                # Track guide performance during optimization process
                # Extracting the current guide as a nucleotide sequence
                curr_guide = np.eye(4)[np.argmax(self.generate_guide(z), axis=-1)]
                curr_guide_nt = prep_seqs.convert_to_nt(curr_guide)

                # Computing the fitness of the guide generated at the current step
                weighted_activity = self.model.get_fitness([curr_guide_nt])[0]

                #Saving performance metrics
                track_performance(run_id, inner_i, curr_guide_nt, self.model.baseline_seq, weighted_activity, [self.model.baseline_seq,  self.model.baseline_seq] , self.track_perf)

        # Saving the one-hot-encoded vector of the final guide and the corresponding nt version
        final_guide = np.eye(4)[np.argmax(self.generate_guide(z), axis=-1)]
        final_guide_nt = prep_seqs.convert_to_nt(final_guide)

        # Compute the guide fitness using the fitness function
        fitness = self.model._diff_fitness_function(final_guide).numpy()

        # This local search step takes all the positions at which the generated guide differs from the baseline sequence
        # and then mutates the guide at each of these positions to generate a list of mutated guides.
        # In certain situations, optimizing the nucleotides at these positions can advance guide performance.
        if(self.local_s):  
            newseqs = prep_seqs.mismatch_nuc_opt(self.model.baseline_seq, final_guide_nt, hd = 2)
            newseqs.append(final_guide_nt)
            newseqs_fitness = self.model.get_fitness(newseqs)
            best_newseq_idx = np.argmax(np.array(newseqs_fitness))
            final_guide_nt = newseqs[best_newseq_idx]
            fitness = newseqs_fitness[best_newseq_idx]  
     
        return [final_guide_nt], fitness
 
    def activity_loss(self, z):
            """Loss function for the WGAN-AM algorithm
            Args:
                z: sample from the latent space
            Returns:
                Negative predicted activity of the generated guide
                (The WGAN-AM algorithm minimizes this loss function, 
                so we want to minimize the negative predicted activity)
            """
            
            gen_guide = self.generate_guide(z)
            pred_activity = self.model._diff_fitness_function(gen_guide)

            return -1 * pred_activity

    def generate_guide(self, z):
        """Generate a guide using z.
        Note that gen_model operates on a batch; z has a batch size of
        1 and this passes `[target]` to gen_model (1 target per batch). In
        the output of gen_model, which is a batch of guides (just 1 in the
        batch), this pulls out the one guide.
        Args:
            z: sample from the latent space
        Returns:
            generated guide (unpadded) from the latent sample z
        """
        gen_guide = gen_model([z, [self.starting_sequence_onehot]], training=False,
                pad_to_target_length=False)
        assert gen_guide.shape[0] == 1
        return gen_guide[0]