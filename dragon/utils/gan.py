"""
Generator and discriminator in a GAN for guide sequences.
This implements a conditional GAN to generate guide sequences conditional
on a target and to discriminate guide sequences conditional on a
target.

This script includes the training functions and the
function to import the generator model for the DRAGON package.
"""

import argparse
from collections import defaultdict
import gzip
import os
import pickle

import numpy as np 
import tensorflow as tf

def parse_args():
    """Parse arguments.
    Returns:
        argument namespace
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
            choices=['cas13'],
            default='cas13',
            help=("Dataset to use."))
    parser.add_argument('--cas13-subset',
            choices=['exp', 'pos', 'neg', 'exp-and-pos'],
            help=("Use a subset of the Cas13 data. See parse_data module "
                  "for descriptions of the subsets. To use all data, do not "
                  "set."))
    parser.add_argument('--cas13-only-active',
            action="store_true",
            help=("If set, only use Cas13 data from the active class"))
    parser.add_argument('--context-nt',
            type=int,
            default=10,
            help=("nt of target sequence context to include alongside each "
                  "guide"))
    parser.add_argument('--num-gen-iter',
            type=int,
            default=10000,
            help=("Number of generator iterations to train for"))
    parser.add_argument('--test-split-frac',
            type=float,
            default=0.3,
            help=("Fraction of the dataset to use for testing the final "
                  "model; this module does not perform any testing, but "
                  "reserves this fraction for evaluate_gan"))
    parser.add_argument('--seed',
            type=int,
            default=1,
            help=("Random seed"))
    parser.add_argument('--save-path',
            help=("If set, path to directory in which to save parameters, "
                  "model weights, and train/validation metrics"))
    args = parser.parse_args()

    # Print the arguments provided
    print(args)

    return args


def set_seed(seed):
    """Set tensorflow and numpy seed.
    Args:
        seed: random seed
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)


def read_data(args):
    """Read input/output data.
    Args:
        args: argument namespace
    Returns:
        data parser object from parse_data
    """
    # Read data
    if args.dataset == 'cas13':
        # Since we want the dataset to consist of 'true' guide/target pairs,
        # do not allow the 'neg' subset (if None, it includes all data, which
        # will include 'neg')
        if (args.cas13_subset is None or
                args.cas13_subset not in ['exp', 'pos', 'exp-and-pos']):
            raise Exception(("cas13_subset must be 'exp', 'pos', or "
                "'exp-and-pos'"))
        print('\nCas13 Subset in gan.py is: ' + str(args.cas13_subset))
        parser_class = parse_data.Cas13ActivityParser
        subset = args.cas13_subset
    test_frac = args.test_split_frac
    train_frac = (1.0 - test_frac) * (4.0/5.0)
    validation_frac = (1.0 - test_frac) * (1.0/5.0)
    data_parser = parser_class(
            subset=subset,
            context_nt=args.context_nt,
            split=(train_frac, validation_frac, test_frac),
            shuffle_seed=args.seed,
            stratify_by_pos=True)
    if args.dataset == 'cas13' and args.cas13_only_active:
        # Set parser mode to only read 'active' points
        data_parser.set_activity_mode(False, False, True)
        print('\n\nCas13_Only_Active mode is active in gan.py')
    data_parser.read()

    x_train, y_train = data_parser.train_set()
    x_validate, y_validate = data_parser.validate_set()
    x_test, y_test = data_parser.test_set()

    # Print the size of each data set
    data_sizes = 'DATA SIZES - Train: {}, Validate: {}, Test: {}'
    print(data_sizes.format(len(x_train), len(x_validate), len(x_test)))

    return data_parser

#sampling replicates of G:T

def make_dataset_and_batch(X, batch_size=8):
    """Make tensorflow dataset and batch.
    Args:
        X: input data
        batch_size: batch size
    Returns:
        batched tf.data.Dataset object
    """
    return tf.data.Dataset.from_tensor_slices(X).batch(batch_size)


class CasResNetUtilities():
    """Functions for building a residual network discriminator or generator.
    This is based loosely on the discriminator in Killoran et al. 2017
    ( https://arxiv.org/pdf/1712.06148.pdf /
    https://github.com/co9olguy/Generating-and-designing-DNA )
    """

    def __init__(self, params):
        """
        Args:
            params: dict of hyperparameters
        """
        self.params = params

    def conv_layer_up(self, num_filters):
        """Build convolutional layer from sequence (4 channel) to a high
        number of output channels.
        Args:
            num_filters: number of output channels
        Returns:
            tf.keras.layers.Conv1D object
        """
        # Construct a convolutional layer with filter width of 1
        # (width 1 will make it straightforward for the generator to produce
        # output)
        filter_width = 1
        conv = tf.keras.layers.Conv1D(
                num_filters,
                filter_width,
                strides=1,  # stride by 1
                padding='same', # pad input so output has same length
                activation='linear',
                name='conv_up')
        return conv

    def conv_layer_down(self, num_filters=4):
        """Build convolutional layer from some high number of channels into
        sequence (4 channel).
        Args:
            num_filters: number of output channels (4 for one-hot encoded
                sequence)
        Returns:
            tf.keras.layers.Conv1D object
        """
        # Construct a convolutional layer with filter width of 1
        # (width 1 will make it straightforward for the generator to produce
        # output)
        filter_width = 1
        conv = tf.keras.layers.Conv1D(
                num_filters,
                filter_width,
                strides=1,  # stride by 1
                padding='same', # pad input so output has same length
                activation='linear',
                name='conv_down')
        return conv

    def residual_block(self, dim, use_batchnorm=False,
            use_leakyrelu=False):
        """Build components of a residual block.
        The layers this returns only computes the residual; it does not add
        the input (shortcut) to the residual.
        This is based partly on the proposed scheme here:
        https://github.com/raghakot/keras-resnet
        Note that this performs batchnorm after the ReLU activation,
        not before, which seems to have become more standard practice.
        Args:
            dim: number of dimensions (channels) read in and output by the
                convolutional filters
            use_batchnorm: if set, use batch normalization layers; since
                the rest of this module implements a WGAN-GP, which
                should not incorporate batch normalization (see the
                paper cited below for why), it is off by default
            use_leakyrelu: if set, use LeakyReLU instead of ReLU for
                activations
        Returns:
            list of layers to compute the residual
        """
        if use_leakyrelu:
            rb_relu1 = tf.keras.layers.LeakyReLU()
            rb_relu2 = tf.keras.layers.LeakyReLU()
        else:
            rb_relu1 = tf.keras.layers.ReLU()
            rb_relu2 = tf.keras.layers.ReLU()
        rb_conv1 = tf.keras.layers.Conv1D(
                dim,
                3, # filter width
                strides=1,  # stride by 1
                padding='same', # pad input so output has same length
                activation='linear')
        rb_conv2 = tf.keras.layers.Conv1D(
                dim,
                3, # filter width
                strides=1,  # stride by 1
                padding='same', # pad input so output has same length
                activation='linear')

        if use_batchnorm:
            rb_bn1 = tf.keras.layers.BatchNormalization()
            rb_bn2 = tf.keras.layers.BatchNormalization()
            rb = [rb_relu1, rb_bn1, rb_conv1, rb_relu2, rb_bn2, rb_conv2]
        else:
            rb = [rb_relu1, rb_conv1, rb_relu2, rb_conv2]

        return rb

    def residual_layer(self, dim, num_blocks=3,
            use_leakyrelu=False):
        """Build layer of multiple residual blocks.
        Args:
            dim: number of dimensions (channels) read in and output by the
                convolutional filters
            num_blocks: number of residual blocks to stack
            use_leakyrelu: if set, use LeakyReLU instead of ReLU for
                activations
        Returns:
            list of output of residual_block()
        """
        layer = []
        for i in range(num_blocks):
            layer += [self.residual_block(dim, use_leakyrelu=use_leakyrelu)]
        return layer

    def residual_layer_call(self, x, residual_blocks, training=False):
        """Compute output of a residual layer.
        Note that this merges the two paths with equal weight (i.e.,
        does not weight the convolutional output (residual) when adding it to
        the input (shortcut)).
        If the channels do not match, see
        https://github.com/raghakot/keras-resnet/blob/5e9bcca7e467f7baf3459d809ef16bb75e53f115/resnet.py#L70
        as a reference for adding the shortcut and residual.
        Args:
            x: input
            residual_blocks: list l such that each l[i] is a list of
                layers that compute the residual of a residual block
            training: whether in training mode
        """
        # Define function to compute the output of a single residual block
        def residual_block_call(curr_x, res_block):
            # Save the input, to shortcut to the end
            shortcut = curr_x

            # Compute the residual
            residual = curr_x
            for l in res_block:
                if isinstance(l, tf.keras.layers.BatchNormalization):
                    residual = l(residual, training=training)
                else:
                    residual = l(residual)

            # Add the shortcut and residual
            return tf.keras.layers.add([shortcut, residual])

        num_blocks = len(residual_blocks)
        for i in range(num_blocks):
            res_block = residual_blocks[i]
            x = residual_block_call(x, res_block)
        return x


class CasResNetDiscriminator(tf.keras.Model):
    """Discriminator based on a residual network.
    This accepts both the target and guide (8-channel input, if each is
    one-hot encoded). Another way of looking at this is that it accepts a
    guide g and is conditional on a target t. This classifies whether g
    is a 'real' or 'fake' guide for the target t.
    """
    def __init__(self, params):
        """
        Args:
            params: dict of hyperparameters
        """
        super(CasResNetDiscriminator, self).__init__()

        self.util = CasResNetUtilities(params)

        # Create layer to add some Gaussian noise
        # This might be helpful because the real input guides are 1-hot
        # encoded, whereas the generated guides are continuous over 4-channels
        # at each position (but softmax-d, so summing to 1); the discriminator
        # might be able to distinguish real/fake based on whether the 4
        # channels at a position are, e.g., [1,0,0,0] vs. [0.97,0.01,0.01,0.01].
        # We could just apply noise to only the real input guides, but it should
        # be fine to apply the noise to both real/fake input, as done here.
        # This also provides regularization, which is a reason for
        # applying it to both real/fake guides
        # We also apply this to the targets for regularization during training
        self.input_noise = tf.keras.layers.GaussianNoise(stddev=0.2)

        # Create layer to concatenate target and guide sequence
        # The axes are
        #   (batch axis, width, filters)
        # We want to concatenate along the filters axis, which is axis=2 or
        # axis=-1
        self.guide_target_concat = tf.keras.layers.Concatenate(axis=-1)

        # Create a conv layer
        self.conv = self.util.conv_layer_up(params['disc_conv_num_filters'])

        # Create the residual layer; use LeakyReLU activations in the
        # discriminator
        self.res_layer = self.util.residual_layer(params['disc_conv_num_filters'],
                use_leakyrelu=True)

        # Flatten the output while preserving the batch axis
        self.flatten = tf.keras.layers.Flatten()

        # Add dropout for additional regularization
        self.dropout = tf.keras.layers.Dropout(
                params['disc_dropout_rate'],
                name='dropout')

        # Combine the above values (linearly) to a single score; leave
        # this score unbounded (i.e., do not pass through a sigmoid)
        self.final = tf.keras.layers.Dense(
                1,
                activation='linear',
                name='disc_final')
    
    def call(self, inputs, training=False):
        assert isinstance(inputs, list)
        guides, targets = inputs

        # self.input_noise is a regularization layer and therefore should only
        # be active at training time; when calling, it accepts a `training`
        # argument
        if training:
            guides = self.input_noise(guides, training=training)
            targets = self.input_noise(targets, training=training)

        # Assume that the guides are already padded with 0s on the two ends
        # so that they are the same length as the targets
        x = self.guide_target_concat([guides, targets])

        # Run the discriminator
        x = self.conv(x)
        x = self.util.residual_layer_call(
                x,
                self.res_layer,
                training=training)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.final(x)

#S: So the discriminator is a 1D convolution and then a flatten and dropout?

class CasResNetGenerator(tf.keras.Model):
    """Generator based on a residual network.
    This accepts a vector z drawn from a latent space, and generates a guide
    g. It is conditional on a target t, and tries to generate g that makes
    sense for t.
    We can perhaps think of z as representing some variation on t (just
    the part of t to which the guide would align). There are convolutions,
    below as done in self.res_layer, to create sequence that combines t and
    z.
    """
    def __init__(self, params, guide_len, target_len, guide_loc):
        """
        Args:
            params: dict of hyperparameters
            guide_len: length of guide sequence in nt
            target_len: length of target sequence in nt
            guide_loc: location of the guide along the target sequence
                (0-based)
        """
        super(CasResNetGenerator, self).__init__()

        self.params = params
        self.util = CasResNetUtilities(params)

        # Set the number of channels that will be read in by the convolutional
        # layer at the end; this matters early on for setting dimensionality
        conv_layer_num_in_channels = params['gen_conv_num_in_channels']

        # The target will use 4 channels, so calculate how many channels to use
        # for the latent input
        assert conv_layer_num_in_channels > 4
        latent_channels = conv_layer_num_in_channels - 4

        # Transform the latent vector (of any arbitrary dimension) into a
        # (latent_channels*guide_len)-length vector; this will
        # probably mean upsampling
        self.upsample = tf.keras.layers.Dense(
                latent_channels * guide_len,
                activation='linear',
                name='gen_upsample')

        # Reshape the vector to a (guide_len x latent_channels)
        # matrix so that the residual blocks (and the convolutional layers
        # they contain) can read it
        self.reshape = tf.keras.layers.Reshape((guide_len, latent_channels))

        # Pad the guide matrix with 0s on each side to cover the length of
        # the target (i.e., so there are nonzero values where the guide aligns
        # with the target, and 0s elsewhere)
        # The input/output shape is (batch axis, width, channels) where
        # the padded axis is axis=1 (the width dimension)
        left_pad = guide_loc
        right_pad = target_len - guide_loc - guide_len
        self.guide_pad = tf.keras.layers.ZeroPadding1D(
                padding=(left_pad, right_pad))

        # Create layer to concatenate target and guide sequence
        # The axes are
        #   (batch axis, width, filters)
        # We want to concatenate along the filters axis, which is axis=2 or
        # axis=-1
        self.guide_target_concat = tf.keras.layers.Concatenate(axis=-1)

        # Create the residual layer; use ReLU activations (non-leaky) in the
        # generator
        self.res_layer = self.util.residual_layer(conv_layer_num_in_channels,
                use_leakyrelu=False)

        # Perform a convolution to create 4-channel encoded guide sequence
        self.conv = self.util.conv_layer_down(num_filters=4)

        # Perform a softmax to have a 'probability' of each base at each
        # position
        # The softmax needs to happen along an axis; the axes are:
        #   (batch size, width, filters)
        # We want to softmax along the filters axis, so use axis=2 or axis=-1
        self.softmax = tf.keras.layers.Softmax(axis=-1)

        # The generated sequence is actually along the whole target (both
        # guide and sequence context); hopefully the context it creates it
        # learns to be identical to the target sequence, but we really only
        # care about the guide sequence
        # (A way to avoid this would be to do the convolutions in
        # self.res_layer only along the part of the width that will become the
        # guide sequence, but it is perhaps easier to just do it along the
        # whole target width, and then cut out only the part corresponding to
        # the guide.)
        # Pick out only the guide_len part at guide_loc (along the width axis,
        # which is axis=1); tf.keras.layers.Cropping1D crops along axis=1 and
        # will trim left_pad units from the left and right_pad units from
        # the right
        self.guide_slice = tf.keras.layers.Cropping1D((left_pad, right_pad))

    def call(self, inputs, training=False, pad_to_target_length=False):
        """Generate guide.
        Args:
            inputs: list [z, target] where z is random noise and target
                represents target sequences to design for
            training: is in training mode
            pad_to_target_length: if False, return only a guide of length
                guide_len (as specified to the initializer) - i.e., the
                vector generated has shape (batch size, guide_len, 4); if True,
                vector generated has shape (batch size, target_len, 4) where
                target_len is specified to the initializer and all the entries
                to the left/right of the guide is 0s
        Returns:
            vector representing generated guide sequences
        """
        assert isinstance(inputs, list)
        z, target = inputs

        # Create a matrix representing guide variation
        z = self.upsample(z)
        z = self.reshape(z)
        z = self.guide_pad(z)

        # Concatenate z with the target input
        x = self.guide_target_concat([z, target])

        # Generate a guide sequence
        x = self.util.residual_layer_call(
                x,
                self.res_layer,
                training=training)
        x = self.conv(x)
        x = self.softmax(x)
        g = self.guide_slice(x)

        if pad_to_target_length:
            # Re-apply self.guide_pad to pad the guide with zeros on both
            # sides (self.guide_slice will have cropped the context on
            # both sides of the guide, but those elements may have been
            # nonzero)
            g = self.guide_pad(g)

        return g


def construct_discriminator():
    """Construct discriminator model.
    Returns:
        CasResNetDiscriminator object
    """
    params = {'disc_conv_num_filters': 20, 'disc_dropout_rate': 0.25}
    disc = CasResNetDiscriminator(params)

    return disc


def construct_generator(guide_len, target_len, guide_loc):
    """Construct generator model.
    Args:
        guide_len: length of guide sequence in nt
        target_len: length of target sequence in nt
        guide_loc: location of the guide along the target sequence
            (0-based)
    Returns:
        CasResNetGenerator object
    """
    params = {'gen_conv_num_in_channels': 20}
    gen = CasResNetGenerator(params, guide_len, target_len, guide_loc)

    return gen


#####################################################################
# Perform training and testing
#####################################################################

# Define a loss function for the generator
def gen_loss_fn(fake_predictions):
    """Compute loss for generator.
    For the usual GAN, this is binary cross-entropy where we say the
    true value is 1 (what we want the discriminator to predict - i.e., real)
    Equivalently, it is sum {-log D(G(z))} where G(z) is a generated
    data point and D is the discriminator.
    Here we implement a WGAN, so the loss is sum {-f(G(z))} where f is
    a 1-Lipschitz function; since we apply a gradient penalty to the
    discriminator loss, we use that for f and the loss is sum {-D(G(z))}.
    (Note as well that we do not pass the discriminator output through a
    sigmoid with a WGAN, so its output is any real number and therefore we
    cannot take the log.)
    Just to be consistent with the below, we take the average across the
    batch (i.e., divide the sum by the batch size, or number of data points
    here); since the batch size is constant this should be equivalent to
    taking the sum.
    Args:
        fake_predictions: output of discriminator on generated data
    Returns:
        loss value
    """
    return tf.math.reduce_mean(-1.0 * fake_predictions)

# Define a loss function for the discriminator
def disc_loss_fn(real_predictions, fake_predictions, penalty_predictions,
        penalty_data, tape, lmbd=10.0):
    """Compute loss for discriminator.
    This is based on Algorithm 1 in https://arxiv.org/pdf/1704.00028.pdf
    (Gulrajani et al. 2017) for training WGANs.
    The usual GAN is trained to minimize the cross-entropy of assigning the
    correct label, which could be seen as:
      [cross-entropy for real_predictions with true value of 1 + cross-
       entropy for fake_predictions with true value of 0]
    or, equivalently, minimizing:
      sum_{real value x} {-log D(x)} + sum_{generated value g from G(z)} {-log 1-D(g)}
    Here we are implementing a Wasserstein GAN (WGAN), and to minimize the
    Wasserstein distance (above is JS divergence) the loss function changes.
    We instead minimize:
      sum_{real value x} {-f(x)} + sum_{generated value g from G(z)} {f(g)}
    where f is a 1-Lipschitz function. Since we train the discriminator with
    a gradient penalty (see below) to satisfy the Lipschitz constraint, the
    loss function is:
      sum_{real value x} {-D(x)} + sum_{generated value g from G(z)} {D(g)}
    (Note also that We do not pass the discriminator output through a sigmoid,
    so its output is any real number and therefore we cannot take the log.)
    Following Algorithm 1 in the above paper we also add a gradient penalty
    to improve learning; we add:
      lambda * (|| gradient wrt \hat{x} of D(\hat{x}) || - 1)^2
    where \hat{x} = eps*x + (1-eps)*g where g is drawn from G(z), eps is a
    uniformly random number in [0,1] and x is a real data point. This is
    enforcing the Lipschitz constraint. The model we are implementing is
    a WGAN with gradient penalty, or WGAN-GP.
    To be consistent with Algorithm 1 in the above paper, we take the average
    across the batch (i.e., divide the sum by the batch size, or number of
    data points here); since the batch size is constant this should be
    equivalent to taking the sum.
    Args:
        real_predictions: output of discriminator on real data (i.e., the D(x)
            above)
        fake_predictions: output of discriminator on generated data (i.e., the
            D(g) above)
        penalty_predictions: output of discriminator on the data points
            eps*x + (1-eps)*g (i.e., D(\hat{x}) above)
        penalty_data: the data points eps*x + (1-eps)*g (i.e., \hat{x} above)
        tape: tf.GradientTape object for computing gradients
        lmbd: weight of gradient penalty (for Lipschitz constraint)
    Returns:
        tuple (a, b) where a is loss value before adding the penalty and b
        includes the penalty"""
    loss = (tf.math.reduce_mean(-1.0 * real_predictions) +
            tf.math.reduce_mean(fake_predictions))

    # Compute the gradient of D(\hat{x}) wrt \hat{x}
    penalty_grad = tape.gradient(penalty_predictions, [penalty_data])[0]
    
    # Comput the norm of the gradient over the dimensions of each \hat{x};
    # the axes should be (0: across the different \hat{x}; 1: across the
    # width of each \hat{x}; 2: across the dimensions (channels) of each
    # \hat{x} at each position)
    assert len(tf.shape(penalty_grad)) == 3 # check there are 3 axes
    penalty_grad_norm = tf.norm(penalty_grad, axis=[1,2])
    penalty = tf.math.reduce_mean(lmbd * tf.math.square(penalty_grad_norm - 1))

    return (loss, loss + penalty)


# Define metrics for evaluating performance
# Note the predictions are not in [0,1] (the discriminator's output is not
# passed through a sigmoid) and therefore the AUC and accuracy metrics may
# not be meaningful; even if we were to pass validation predictions through
# a sigmoid just before computing accuracy/AUC, this may still not be
# meaningful because they were not passed through a sigmoid during training
train_disc_loss_metric = tf.keras.metrics.Mean(name='train_disc_loss')
train_disc_loss_without_penalty_metric = tf.keras.metrics.Mean(name='train_disc_loss_without_penalty')
validate_disc_loss_without_penalty_metric = tf.keras.metrics.Mean(name='validate_disc_loss_without_penalty')
train_gen_loss_metric = tf.keras.metrics.Mean(name='train_gen_loss')
validate_gen_loss_metric = tf.keras.metrics.Mean(name='validate_gen_loss')


def make_penalty_data(batch_size, real_data_guides, fake_data_guides):
    """Make data for enforcing Lipschitz constraint.
    Args:
        batch_size: batch size
        real_data_guides: guides from real data
        fake_data_guides: guides from generated data
    Returns:
        guides weighted between real and fake
    """
    assert len(real_data_guides) == batch_size
    assert len(fake_data_guides) == batch_size

    # Sample epsilon for generating penalty data (weighted between real and
    # fake data); this is uniform in [0,1] and we should have a separate
    # epsilon for each data point in the batch (note that each data point
    # has 2 dimensions)
    eps = tf.random.uniform([batch_size, 1, 1])

    penalty_data_guides = eps*real_data_guides + (1.0 - eps)*fake_data_guides

    return penalty_data_guides


# Train the discriminator; this is called on each batch
def train_disc_step(gen_model, disc_model,
        real_data_guides, real_data_targets, disc_optimizer, latent_dim):
    """Train the discriminator for one step.
    Args:
        gen_model: generator model
        disc_model: discriminator model
        real_data_guides: guides from real data
        real_data_targets: targets corresponding to the guides from real
            data (real_data_guides[i] is the guide for target
            real_data_targets[i])
        disc_optimizer: discriminator optimizer
        latent_dim: dimension of the latent space of the generator
    Returns:
        (discriminator predictions on the real data,
         discriminator predictions on the generated data,
         discriminator predictions on the penalty (interpolated) data)
    """
    # This is based on Algorithm 1 in https://arxiv.org/pdf/1704.00028.pdf
    # (Gulrajani et al. 2017) for training WGANs (the part on lines 3-9); it
    # is called for each iteration of training the discriminator

    batch_size = len(real_data_guides)

    # We are only working with real targets (conditional on them) and never
    # generating targets, so just use 'targets' to refer to them
    targets = real_data_targets

    # Use `persistent=True` to allow computing gradients (via
    # `tape.gradient()`) multiple times
    with tf.GradientTape(persistent=True) as tape:
        # Sample latent variable
        z = tf.random.normal([batch_size, latent_dim])

        # Make generated data (guides)
        fake_data_guides = gen_model([z, targets], training=False,
                pad_to_target_length=True)

        # Make data for enforcing Lipschitz constraint
        penalty_data_guides = make_penalty_data(batch_size,
                real_data_guides, fake_data_guides)

        # Compute predictions
        fake_predictions = disc_model([fake_data_guides, targets],
                training=True)
        real_predictions = disc_model([real_data_guides, targets],
                training=True)

        # For batch norm during penalty discrimination, use mean/variance
        # of the moving statistics learned during training by setting
        # `training=False`; note that using `training=True` seems to make
        # a big difference in the penalty component of the loss
        # computed by disc_loss_fn()
        penalty_predictions = disc_model([penalty_data_guides, targets],
                training=True)
        
        # Compute loss
        disc_loss_without_penalty, disc_loss = disc_loss_fn(
                real_predictions, fake_predictions,
                penalty_predictions, penalty_data_guides, tape)

    # Compute gradients in discriminator and optimize parameters
    gradients = tape.gradient(disc_loss, disc_model.trainable_variables)
    disc_optimizer.apply_gradients(zip(gradients, disc_model.trainable_variables))

    # Record metrics
    train_disc_loss_metric(disc_loss)
    train_disc_loss_without_penalty_metric(disc_loss_without_penalty)

    return real_predictions, fake_predictions, penalty_predictions


# Validate the discriminator on a batch
def validate_disc_step(gen_model, disc_model,
        real_data_guides, real_data_targets, latent_dim):
    """Validate the discriminator for one step.
    This updates the validation metrics.
    Args:
        gen_model: generator model
        disc_model: discriminator model
        real_data_guides: guides from real data
        real_data_targets: targets corresponding to the guides from real
            data (real_data_guides[i] is the guide for target
            real_data_targets[i])
        latent_dim: dimension of the latent space of the generator
    Returns:
        (discriminator predictions on the real data,
         discriminator predictions on the generated data)
    """
    batch_size = len(real_data_guides)

    # We are only working with real targets (conditional on them) and never
    # generating targets, so just use 'targets' to refer to them
    targets = real_data_targets

    # Sample latent variable
    z = tf.random.normal([batch_size, latent_dim])

    # Make generated data (guides)
    fake_data_guides = gen_model([z, targets], training=False,
                pad_to_target_length=True)

    # Compute predictions
    fake_predictions = disc_model([fake_data_guides, targets], training=False)
    real_predictions = disc_model([real_data_guides, targets], training=False)

    loss = (tf.math.reduce_mean(-1.0 * real_predictions) +
            tf.math.reduce_mean(fake_predictions))
    validate_disc_loss_without_penalty_metric(loss)

    return real_predictions, fake_predictions


# Train the generator; this is called on each batch
def train_gen_step(gen_model, disc_model, real_data_targets,
        gen_optimizer, latent_dim):
    """Train the generator for one step.
    Args:
        gen_model: generator model
        disc_mode: discriminator model
        real_data_targets: targets corresponding to guides from real data
        gen_optimizer: generator optimizer
        latent_dim: dimension of the latent space of the generator
    Returns:
        discriminator predictions on the generated data
    """
    # This is based on Algorithm 1 in https://arxiv.org/pdf/1704.00028.pdf
    # (Gulrajani et al. 2017) for training WGANs (the part on lines 11-12);
    # it is called for each iteration of training the generator

    batch_size = len(real_data_targets)

    # We are only working with real targets (conditional on them) and never
    # generating targets, so just use 'targets' to refer to them
    targets = real_data_targets

    with tf.GradientTape() as tape:
        # Sample latent variable
        z = tf.random.normal([batch_size, latent_dim])

        # Make generated data
        fake_data_guides = gen_model([z, targets], training=True,
                pad_to_target_length=True)

        # Compute predictions
        fake_predictions = disc_model([fake_data_guides, targets],
                training=False)

        # Compute loss
        gen_loss = gen_loss_fn(fake_predictions)

    # Compute gradients in generator and optimize parameters
    gradients = tape.gradient(gen_loss, gen_model.trainable_variables)
    gen_optimizer.apply_gradients(zip(gradients, gen_model.trainable_variables))

    # Record metrics
    train_gen_loss_metric(gen_loss)

    return fake_predictions


# Validate the generator on a batch
def validate_gen_step(gen_model, disc_model, real_data_targets,
        latent_dim):
    """Validate the generator for one step.
    This updates the validation metrics.
    Args:
        gen_model: generator model
        disc_mode: discriminator model
        real_data_targets: targets corresponding to guides from real data
        latent_dim: dimension of the latent space of the generator
    Returns:
        discriminator predictions on the generated data
    """

    batch_size = len(real_data_targets)

    # We are only working with real targets (conditional on them) and never
    # generating targets, so just use 'targets' to refer to them
    targets = real_data_targets

    # Sample latent variable
    z = tf.random.normal([batch_size, latent_dim])

    # Make generated data (guides)
    fake_data_guides = gen_model([z, targets], training=False,
                pad_to_target_length=True)

    # Compute predictions
    fake_predictions = disc_model([fake_data_guides, targets],
            training=False)

    # Compute and record loss
    gen_loss = gen_loss_fn(fake_predictions)
    validate_gen_loss_metric(gen_loss)

    return fake_predictions


def split_data_into_guides_and_targets(data):
    """Split data into guide and targets.
    Note that parsers in parse_data return one-hot encoded sequence with
    target coming first and then guide.
    Args:
        data: Tensor of shape (batch size, sequence width, channels) where
            there are 8 channels
    Returns:
        tuple (guides, targets)
    """
    # Split data into guides and target
    # The shape is
    #   (batch size, width, channels)
    # and we want to split evenly along the channels dimension (axis=2 or
    # axis=-1)
    # This assumes there are 8 channels, and checks that there are 4 after
    # splitting
    assert data.get_shape().as_list()[2] == 8
    data_targets, data_guides = tf.split(data, 2, axis=-1)
    assert data_guides.get_shape().as_list()[2] == 4
    assert data_targets.get_shape().as_list()[2] == 4
    assert len(data_guides) == len(data_targets)
    return data_guides, data_targets


def data_iter(dataset):
    """Iterate over batches of data, indefinitely.
    Args:
        dataset: tf.data.Dataset that has been batched
    Yields:
        tuple (epoch, batch of data)
    """
    epoch = 0
    while True:
        for X in dataset:
            yield (epoch, X)
        epoch += 1


def train_and_validate(gen_model, disc_model, x_train,
        x_validate, n_disc_iter_per_gen=5, num_gen_iter=10000,
        latent_dim=10):
    """Train the GAN and validate on each epoch.
    This is based on Algorithm 1 in https://arxiv.org/pdf/1704.00028.pdf
    (Gulrajani et al. 2017) for training WGANs (n_disc_iter_per_gen here
    corresponds to n_critic in that pseudocode).
    Args:
        gen_model: generator model
        disc_model: discriminator model
        x_train: data for training
        x_validate: data for validation
        n_disc_iter_per_gen: number of discriminator iterations per generator
            iteration
        num_gen_iter: number of generator iterations to train for
        latent_dim: dimension of the latent space for the generator
    Returns:
        metric values as dict {metric: [list of values, one per generator
        iteration]}
    """
    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    tf_train_gen_step = tf.function(train_gen_step)
    tf_train_disc_step = tf.function(train_disc_step)

    train_ds = make_dataset_and_batch(x_train)
    train_ds_iter = data_iter(train_ds)
    validate_ds = make_dataset_and_batch(x_validate)

    # Save metrics for each generator iteration:
    #   {metric: [list of values, one per generator iteration]}
    metric_vals = defaultdict(list)

    curr_gen_iter = 0
    curr_epoch = 0
    while curr_gen_iter < num_gen_iter:
        # Perform one generator iteration

        # Go through n_disc_iter batches of training data to train the
        # discriminator
        for _ in range(n_disc_iter_per_gen):
            curr_epoch, X = next(train_ds_iter)
            X_guides, X_targets = split_data_into_guides_and_targets(X)
            train_disc_step(gen_model, disc_model,
                    X_guides, X_targets, disc_optimizer, latent_dim)

        # Train the generator on a batch
        curr_epoch, X = next(train_ds_iter)
        X_guides, X_targets = split_data_into_guides_and_targets(X)
        train_gen_step(gen_model, disc_model, X_targets,
                gen_optimizer, latent_dim)
        curr_gen_iter += 1

        # Validate and log metrics for this generator iteration
        for X in validate_ds:
            X_guides, X_targets = split_data_into_guides_and_targets(X)
            validate_disc_step(gen_model, disc_model,
                    X_guides, X_targets, latent_dim)
            validate_gen_step(gen_model, disc_model, X_targets,
                    latent_dim)

        # Save metrics
        metric_vals['train_disc_loss'].append(train_disc_loss_metric.result().numpy())
        metric_vals['train_disc_loss_without_penalty'].append(train_disc_loss_without_penalty_metric.result().numpy())
        metric_vals['train_gen_loss'].append(train_gen_loss_metric.result().numpy())
        metric_vals['validate_disc_loss_without_penalty'].append(validate_disc_loss_without_penalty_metric.result().numpy())
        metric_vals['validate_gen_loss'].append(validate_gen_loss_metric.result().numpy())

        # Print out metrics
        
        # print('Generator iteration {} of {}'.format(
        #     curr_gen_iter, num_gen_iter))
        # print('  On epoch {}'.format(curr_epoch+1))
        # print('  Train metrics:')
        # print('    Discriminator loss: {}'.format(
        #     train_disc_loss_metric.result()))
        # print('    Discriminator loss (without penalty): {}'.format(
        #     train_disc_loss_without_penalty_metric.result()))
        # print('    Generator loss: {}'.format(
        #     train_gen_loss_metric.result()))
        # print('  Validate metrics:')
        # print('    Discriminator loss (without penalty): {}'.format(
        #     validate_disc_loss_without_penalty_metric.result()))
        # print('    Generator loss: {}'.format(
        #     validate_gen_loss_metric.result()))

        # Reset states so they are not cumulative over generator iterations
        train_disc_loss_metric.reset_states()
        train_disc_loss_without_penalty_metric.reset_states()
        validate_disc_loss_without_penalty_metric.reset_states()
        train_gen_loss_metric.reset_states()
        validate_gen_loss_metric.reset_states()

    # Print model summaries
    # It is easier to do this here, after the models have been called,
    # rather than earlier; if we were to do it earlier, we would have to
    # compute the input shapes (to pass to model.build(..)) which are
    # tricky here as there are multiple inputs
    # The models should have determined their input shape, after having been
    # called above

    # print("Generator model summary:")
    # print(gen_model.summary())
    # print("Discriminator model summary:")
    # print(disc_model.summary())

    return metric_vals
 

def construct_models(params):
    """Construct generator and discriminator models.
    This also modifies params, adding parameters that can be computed from
    existing ones.
    Args:
        params: dict of parameters
    Returns:
        tuple (generator model, discriminator model)
    """
    params['guide_len'] = 28
    params['target_len'] = params['guide_len'] + 2*params['context_nt']
    params['guide_loc'] = params['context_nt']

    gen_model = construct_generator(params['guide_len'], params['target_len'],
            params['guide_loc'])
    disc_model = construct_discriminator()

    return gen_model, disc_model


def save_and_plot_metric_values(metric_vals, save_path):
    """Save and plot metric values, one per generator iteration.
    Args:
        metric_vals: dict {metric: [list of metric values, one per generator
            iteration]}
        save_path: path to directory in which to save values as a TSV and
            in which to save plot as a PDF
    """
    # Note that this records and plots generator iterations as 0-based (first
    # is 0, even though when printing in `train_and_validate()` the first is
    # printed as 1)
    with gzip.open(os.path.join(save_path, 'metrics.tsv.gz'), 'wt') as fw:
        header = ['metric', 'gen_iter', 'value']
        fw.write('\t'.join(header) + '\n')
        for metric in metric_vals:
            for gen_iter in range(len(metric_vals[metric])):
                val = metric_vals[metric][gen_iter]
                fw.write('\t'.join([metric, str(gen_iter), str(val)]) + '\n')

    import matplotlib.pyplot as plt
    plt.figure(1)
    for metric in metric_vals:
        if 'disc_loss' not in metric:
            # Only plot the discriminator losses (the generator losses can be
            # large relative to the discriminator losses and enlargen the
            # y-axis limits)
            continue
        gen_iters = list(range(len(metric_vals[metric])))
        plt.plot(gen_iters, metric_vals[metric], label=metric)
    plt.legend(loc='best')
    plt.xlabel('Generator iteration')
    plt.ylabel('Metric value')
    plt.title('Metrics during training')
    plt.show()
    plt.savefig(os.path.join(save_path, 'metrics.pdf'))


gen_saved_params = {'dataset': 'cas13', 'cas13_subset': 'exp-and-pos', 'cas13_only_active': True, 'context_nt': 10, 'generator_path': 'models/cas13/gan/2500-gen-iter', 'predictor_path': 'models/cas13/regress/model-f8b6fd5d', 'seed': 1, 'batch': [5, 1], 'data_frac': 1, 'csv_path': '', 'inner_i': 150, 'outer_i': 1, 'learn_rate': 1.0, 'num_gen_iter': 2500, 'test_split_frac': 0.3, 'save_path': 'models/cas13/gan/2500-gen-iter', 'guide_len': 28, 'target_len': 48, 'guide_loc': 10, 'conv_filter_width': [1, 2], 'conv_num_filters': 25, 'pool_window_width': 2, 'fully_connected_dim': [53], 'pool_strategy': 'avg', 'locally_connected_width': [1, 2], 'locally_connected_dim': 3, 'skip_batch_norm': True, 'add_gc_content': False, 'activation_fn': 'relu', 'dropout_rate': 0.29845230312675486, 'l2_factor': 2.6309419773217563e-06, 'sample_weight_scaling_factor': 0, 'batch_size': 229, 'learning_rate': 0.0017914755444431493, 'max_num_epochs': 1000, 'regression': True}

def load_models(load_path, params = gen_saved_params):
    """Construct models and load weights.
    Args:
        load_path: path containing model weights
        params: dict of parameters
        x_train, x_validate: train and validate data (only needed to initialize
            variables)
    Returns:
        tuple (generator model, discriminator model)
    """
    # First construct the models
    gen_model, disc_model = construct_models(params)

    x_train = np.load(load_path + '/saved_loadmodel_traindata.npy')
    x_validate = np.load(load_path + '/saved_loadmodel_xvalidationdata.npy')

    # See https://www.tensorflow.org/beta/guide/keras/saving_and_serializing
    # for details on loading a serialized subclassed model
    # To initialize variables used by the optimizers and any stateful metric
    # variables, we need to train it on some data before calling `load_weights`;
    # note that it appears this is necessary (otherwise, there are no variables
    # in the model, and nothing gets loaded)
    # Only train the models on one data point, and for 1 epoch
    train_and_validate(gen_model, disc_model, x_train[:1], x_validate[:1],
            num_gen_iter=1)

    def copy_weights(model):
        # Copy weights, so we can verify that they changed after loading
        return [tf.Variable(w) for w in model.weights]

    def weights_are_eq(weights1, weights2):
        # Determine whether weights1 == weights2
        for w1, w2 in zip(weights1, weights2):
            # 'w1' and 'w2' are each collections of weights (e.g., the kernel
            # for some layer); they are tf.Variable objects (effectively,
            # tensors)
            # Make a tensor containing element-wise boolean comparisons (it
            # is a 1D tensor with True/False)
            elwise_eq = tf.equal(w1, w2)
            # Check if all elements in 'elwise_eq' are True (this will make a
            # Tensor with one element, True or False)
            all_are_eq_tensor = tf.reduce_all(elwise_eq)
            # Convert the tensor 'all_are_eq_tensor' to a boolean
            all_are_eq = all_are_eq_tensor.numpy()
            if not all_are_eq:
                return False
        return True

    def load_weights(model, fn):
        # Load weights
        # There are some concerns about whether weights are actually being
        # loaded (e.g., https://github.com/tensorflow/tensorflow/issues/27937),
        # so check that they have changed after calling `load_weights`
        w_before = copy_weights(model)
        w_before2 = copy_weights(model)
        model.load_weights(os.path.join(load_path, fn))
        w_after = copy_weights(model)
        w_after2 = copy_weights(model)

        assert (weights_are_eq(w_before, w_before2) is True)
        assert (weights_are_eq(w_before, w_after) is False)
        assert (weights_are_eq(w_after, w_after2) is True)

    load_weights(gen_model, 'gen.weights')
    load_weights(disc_model, 'disc.weights')

    return gen_model, disc_model


def main():
    # Read arguments and data
    args = parse_args()
    set_seed(args.seed)
    data_parser = read_data(args)
    x_train, y_train = data_parser.train_set()
    x_validate, y_validate = data_parser.validate_set()
    x_test, y_test = data_parser.test_set()

    # Construct models
    params = vars(args)
    gen_model, disc_model = construct_models(params)

    # Check data shape
    assert x_train.shape[1] == params['target_len']

    # Train the model, with validation
    metric_vals = train_and_validate(gen_model, disc_model, x_train,
            x_validate, num_gen_iter=args.num_gen_iter)

    # Save the model weights to args.save_path
    # Also save the arguments, so that later uses of the models know the
    # arguments that were used for building/training them
    # Also save the metric values as a TSV and in a plot
    if args.save_path:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        # Note that we can only save the model weights, not the model itself
        # (incl. architecture), because they are subclassed models and
        # therefore are described by code (Keras only allows saving
        # the full model if they are Sequential or Functional models)
        # See https://www.tensorflow.org/beta/guide/keras/saving_and_serializing
        # for details on saving subclassed models
        disc_model.save_weights(
                os.path.join(args.save_path, 'disc.weights'),
                save_format='tf')
        gen_model.save_weights(
                os.path.join(args.save_path, 'gen.weights'),
                save_format='tf')
        print('Saved model weights to {}'.format(args.save_path))

        save_path_params = os.path.join(args.save_path, 'params.pkl')
        with open(save_path_params, 'wb') as f:
            pickle.dump(params, f)
        print('Saved parameters to {}'.format(args.save_path))

        save_and_plot_metric_values(metric_vals, args.save_path)
        print('Saved metric values to {}'.format(args.save_path))

    # Note that model testing is done in the evaluate_gan module


if __name__ == "__main__":
    main()