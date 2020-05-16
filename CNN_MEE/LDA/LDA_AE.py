# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:34:48 2019

@author: Xi Yu
"""

import numpy as np
import scipy.sparse
from six.moves import cPickle as pickle
from six.moves import urllib
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import functools
#%%

""" Clips the Dirichlet parameters to the numerically stable KL region."""
def _clip_dirichlet_parameters(x):
    return tf.clip_by_value(x, 1e-3, 1e3)

def make_encoder(activation, num_topics, layer_sizes):

      """Create the encoder function.
    
      Args:
        activation: Activation function to use.
        num_topics: The number of topics.
        layer_sizes: The number of hidden units per layer in the encoder.
        
     Returns:
      encoder: A `callable` mapping a bag-of-words `Tensor` to a
      `tfd.Distribution` instance over topics.
      """

      encoder_net = tf.keras.Sequential()
      for num_hidden_units in layer_sizes:
          encoder_net.add(
                  tf.keras.layers.Dense(
                          num_hidden_units,
                          activation=activation,
                          kernel_initializer=tf.compat.v1.glorot_normal_initializer()))
          encoder_net.add(
                  tf.keras.layers.Dense(
                      num_topics,
                      activation=tf.nn.softplus,
                      kernel_initializer=tf.compat.v1.glorot_normal_initializer()))
    
    
      def encoder(bag_of_words):

          net = _clip_dirichlet_parameters(encoder_net(bag_of_words))
          return tfd.Dirichlet(concentration=net,
                             name="topics_posterior")
      return encoder
#%%
def make_decoder(num_topics, num_words):
    
    

    """Create the decoder function.
    Args:
      num_topics: The number of topics.
      num_words: The number of words.

    Returns:
      decoder: A `callable` mapping a `Tensor` of encodings to a
        `tfd.Distribution` instance over words.
    """
    topics_words_logits = tf.compat.v1.get_variable("topics_words_logits",
                                                    shape=[num_topics, num_words],
                                                    initializer=tf.compat.v1.glorot_normal_initializer())

    topics_words = tf.nn.softmax(topics_words_logits, axis=-1)



    def decoder(topics):

        word_probs = tf.matmul(topics, topics_words)

        # The observations are bag of words and therefore not one-hot. However,

        # log_prob of OneHotCategorical computes the probability correctly in

        # this case.

        return tfd.OneHotCategorical(probs=word_probs,

                                 name="bag_of_words")



    return decoder, topics_words  

#%%%%%%%%%%%%%%%%%%%%%%%%%
def make_prior(num_topics, initial_value):
    
    """Create the prior distribution.

    Args:
      num_topics: Number of topics.
      initial_value: The starting value for the prior parameters.

    Returns:
      prior: A `callable` that returns a `tf.distribution.Distribution`
        instance, the prior distribution.
      prior_variables: A `list` of `Variable` objects, the trainable parameters
        of the prior.
    """
    def _softplus_inverse(x):
        
        return np.log(np.expm1(x))
    
    logit_concentration = tf.compat.v1.get_variable(
        "logit_concentration",
        shape=[1, num_topics],
        initializer=tf.compat.v1.initializers.constant(
            _softplus_inverse(initial_value)))
    concentration = _clip_dirichlet_parameters(
        tf.nn.softplus(logit_concentration))
    
    def prior():
        return tfd.Dirichlet(concentration=concentration,
                             name="topics_prior")
    
    prior_variables = [logit_concentration]
    
    return prior, prior_variables

#%%  
def model_fn(features, mode, params):
    """Build the model function for use in an estimator.

    Arguments:
      features: The input features for the estimator.
      mode: Signifies whether it is train or test or predict.
      params: Some hyperparameters as a dictionary.

    Returns:
      EstimatorSpec: A tf.estimator.EstimatorSpec instance.
    """


    encoder = make_encoder(params["activation"],
                           params["num_topics"],
                           params["layer_sizes"])
    decoder, topics_words = make_decoder(params["num_topics"],
                                         features.shape[1])
    prior, prior_variables = make_prior(params["num_topics"],
                                      params["prior_initial_value"])

    topics_prior = prior()
    alpha = topics_prior.concentration

    topics_posterior = encoder(features)
    topics = topics_posterior.sample()
    random_reconstruction = decoder(topics)

    reconstruction = random_reconstruction.log_prob(features)
    tf.compat.v1.summary.scalar("reconstruction",
                              tf.reduce_mean(input_tensor=reconstruction))

    # Compute the KL-divergence between two Dirichlets analytically.
    # The sampled KL does not work well for "sparse" distributions
    # (see Appendix D of [2]).
    kl = tfd.kl_divergence(topics_posterior, topics_prior)
    tf.compat.v1.summary.scalar("kl", tf.reduce_mean(input_tensor=kl))

    # Ensure that the KL is non-negative (up to a very small slack).
    # Negative KL can happen due to numerical instability.
    with tf.control_dependencies(
        [tf.compat.v1.assert_greater(kl, -1e-3, message="kl")]):
        
        kl = tf.identity(kl)

    elbo = reconstruction - kl
    avg_elbo = tf.reduce_mean(input_tensor=elbo)
    tf.compat.v1.summary.scalar("elbo", avg_elbo)
    loss = -avg_elbo

  # Perform variational inference by minimizing the -ELBO.
    global_step = tf.compat.v1.train.get_or_create_global_step()
    optimizer = tf.compat.v1.train.AdamOptimizer(params["learning_rate"])

    # This implements the "burn-in" for prior parameters (see Appendix D of [2]).
    # For the first prior_burn_in_steps steps they are fixed, and then trained
    # jointly with the other parameters.
    grads_and_vars = optimizer.compute_gradients(loss)
    grads_and_vars_except_prior = [
        x for x in grads_and_vars if x[1] not in prior_variables]

    def train_op_except_prior():
      return optimizer.apply_gradients(
          grads_and_vars_except_prior,
          global_step=global_step)

    def train_op_all():
      return optimizer.apply_gradients(
          grads_and_vars,
          global_step=global_step)

    train_op = tf.cond(
        pred=global_step < params["prior_burn_in_steps"],
        true_fn=train_op_except_prior,
        false_fn=train_op_all)

    # The perplexity is an exponent of the average negative ELBO per word.
    words_per_document = tf.reduce_sum(input_tensor=features, axis=1)
    log_perplexity = -elbo / words_per_document
    tf.compat.v1.summary.scalar(
        "perplexity", tf.exp(tf.reduce_mean(input_tensor=log_perplexity)))
    (log_perplexity_tensor,
    log_perplexity_update) = tf.compat.v1.metrics.mean(log_perplexity)
    perplexity_tensor = tf.exp(log_perplexity_tensor)

  # Obtain the topics summary. Implemented as a py_func for simplicity.
    topics = tf.compat.v1.py_func(
        functools.partial(get_topics_strings, vocabulary=params["vocabulary"]),
        [topics_words, alpha],
        tf.string,
        stateful=False)
    tf.compat.v1.summary.text("topics", topics)

    return tf.estimator.EstimatorSpec(
        mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops={
          "elbo": tf.compat.v1.metrics.mean(elbo),
          "reconstruction": tf.compat.v1.metrics.mean(reconstruction),
          "kl": tf.compat.v1.metrics.mean(kl),
          "perplexity": (perplexity_tensor, log_perplexity_update),
          "topics": (topics, tf.no_op()),
      },
  )    
    
#%%    
def get_topics_strings(topics_words, alpha, vocabulary,
                       topics_to_print=10, words_per_topic=10):
    """Returns the summary of the learned topics.

    Arguments:
      topics_words: KxV tensor with topics as rows and words as columns.
      alpha: 1xK tensor of prior Dirichlet concentrations for the
        topics.
      vocabulary: A mapping of word's integer index to the corresponding string.
      topics_to_print: The number of topics with highest prior weight to
        summarize.
      words_per_topic: Number of wodrs per topic to return.
    Returns:
    summary: A np.array with strings.
    """
    alpha = np.squeeze(alpha, axis=0)
    # Use a stable sorting algorithm so that when alpha is fixed
    # we always get the same topics.
    highest_weight_topics = np.argsort(-alpha, kind="mergesort")
    top_words = np.argsort(-topics_words, axis=1)

    res = []
    for topic_idx in highest_weight_topics[:topics_to_print]:

        l = ["index={} alpha={:.2f}".format(topic_idx, alpha[topic_idx])]
        l += [vocabulary[word] for word in top_words[topic_idx, :words_per_topic]]
        res.append(" ".join(l))

    return np.array(res)    
#%%
import os    
ROOT_PATH = "https://github.com/akashgit/autoencoding_vi_for_topic_models/raw/9db556361409ecb3a732f99b4ef207aeb8516f83/data/20news_clean"
FILE_TEMPLATE = "{split}.txt.npy"


def download(directory, filename):
    filepath = os.path.join(directory, filename)
    if tf.io.gfile.exists(filepath):
      return filepath
    if not tf.io.gfile.exists(directory):
      tf.io.gfile.makedirs(directory)
    url = os.path.join(ROOT_PATH, filename)
    print("Downloading %s to %s" % (url, filepath))
    urllib.request.urlretrieve(url, filepath)
    return filepath

#%%
def newsgroups_dataset(directory, split_name, num_words, shuffle_and_repeat):
    
    """
    Return 20 newsgroups tf.data.Dataset.
    """
    data = np.load(download(directory, FILE_TEMPLATE.format(split=split_name)))
    # The last row is empty in both train and test.
    data = data[:-1]
    
    # Each row is a list of word ids in the document. We first convert this to
    # sparse COO matrix (which automatically sums the repeating words). Then,
    # we convert this COO matrix to CSR format which allows for fast querying of
    # documents.
    num_documents = data.shape[0]
    indices = np.array([(row_idx, column_idx)
                        for row_idx, row in enumerate(data)
                        for column_idx in row])
    sparse_matrix = scipy.sparse.coo_matrix(
        (np.ones(indices.shape[0]), (indices[:, 0], indices[:, 1])),
        shape=(num_documents, num_words),
        dtype=np.float32)
    sparse_matrix = sparse_matrix.tocsr()
    dataset = tf.data.Dataset.range(num_documents)
    
    # For training, we shuffle each epoch and repeat the epochs.
    if shuffle_and_repeat:
        
        dataset = dataset.shuffle(num_documents).repeat()
    
        # Returns a single document as a dense TensorFlow tensor. The dataset is
        # stored as a sparse matrix outside of the graph.
        def get_row_py_func(idx):
          def get_row_python(idx_py):
            return np.squeeze(np.array(sparse_matrix[idx_py].todense()), axis=0)
    
          py_func = tf.compat.v1.py_func(
            get_row_python, [idx], tf.float32, stateful=False)
          py_func.set_shape((num_words,))
          return py_func
    
    dataset = dataset.map(get_row_py_func)
    return dataset

#%%
data = np.load('./data/train.txt.npy',encoding="latin1")
data = data[:-1]
num_documents = data.shape[0]
indices = np.array([(row_idx, column_idx)
                    for row_idx, row in enumerate(data)
                    for column_idx in row])   

#%%   
num_words = 1000
sparse_matrix = scipy.sparse.coo_matrix(
        (np.ones(indices.shape[0]), (indices[:, 0], indices[:, 1])),
        shape=(num_documents, num_words),
        dtype=np.float32)
#%%
sparse_matrix = sparse_matrix.tocsr()
dataset = tf.data.Dataset.range(num_documents)    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    