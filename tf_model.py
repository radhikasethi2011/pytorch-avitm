import numpy as np
import tensorflow as tf
import itertools,time
import sys, os
from collections import OrderedDict
from copy import deepcopy
from time import time
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle

slim = tf.contrib.slim

tf.reset_default_graph()

class VAE(object):
    """
    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """


    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        print('Learning Rate:', self.learning_rate)

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]], name='input')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        

        self.h_dim = (network_architecture["n_z"]) # had a float before
        
        self.test_feature_placeholder = tf.placeholder(
                tf.float32, [self.batch_size, self.h_dim], name = 'test_feature_placeholder')
        self.arr = np.empty((self.batch_size,self.h_dim))

        
        
        self.a = 1*np.ones((1 , self.h_dim)).astype(np.float32)                         # a    = 1
        self.prior_mean = tf.constant((np.log(self.a).T-np.mean(np.log(self.a),1)).T)          # prior_mean  = 0
        self.prior_var = tf.constant(  ( ( (1.0/self.a)*( 1 - (2.0/self.h_dim) ) ).T +       # prior_var = 0.99 + 0.005 = 0.995
                                ( 1.0/(self.h_dim*self.h_dim) )*np.sum(1.0/self.a,1) ).T  )
        self.prior_logvar = tf.log(self.prior_var)

        self._create_network()
        #print('netwrork dimensions')
        #print(self.h_dim) it's right = 50
        
        #self.embeddings = self.embed(self.test_feature_placeholder)
        self.embeddings = self.embed(self.arr)
        print('printing self.embeddings')
        print(self.embeddings)
        print(self.embeddings.shape)
        
        with tf.name_scope('cost'):
            self._create_loss_optimizer()

        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()


        self.sess = tf.InteractiveSession()
        # TF logging to see whether GPU devices are detected
        #self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        self.sess.run(init)

    def _create_network(self):
        """
        steps:
        1. initialize weights
        2. build recognition network
        3. build reconstruction network
        """
        n_z = self.network_architecture['n_z']
        n_hidden_gener_1 = self.network_architecture['n_hidden_gener_1']
        en1 = slim.layers.linear(self.x, self.network_architecture['n_hidden_recog_1'], scope='FC_en1')
        print("printing self x ")
        print(self.x)
        print(self.x.shape)
        en1 = tf.nn.softplus(en1, name='softplus1')
        en2 = slim.layers.linear(en1,    self.network_architecture['n_hidden_recog_2'], scope='FC_en2')
        en2 = tf.nn.softplus(en2, name='softplus2')
        en2_do = slim.layers.dropout(en2, self.keep_prob, scope='en2_dropped')
        self.posterior_mean   = slim.layers.linear(en2_do, self.network_architecture['n_z'], scope='FC_mean')
        self.posterior_logvar = slim.layers.linear(en2_do, self.network_architecture['n_z'], scope='FC_logvar')
        self.posterior_mean   = slim.layers.batch_norm(self.posterior_mean, scope='BN_mean')
        self.posterior_logvar = slim.layers.batch_norm(self.posterior_logvar, scope='BN_logvar')

        with tf.name_scope('z_scope'):
            eps = tf.random_normal((self.batch_size, n_z), 0, 1,                            # take noise
                                   dtype=tf.float32)
            self.z = tf.add(self.posterior_mean,
                            tf.multiply(tf.sqrt(tf.exp(self.posterior_logvar)), eps))         # reparameterization z
            zz = self.z 
            """
            print('printing z')
            print(zz) 
            print(type(zz))
            print(zz.shape)
            print('printing self z')
            print(self.z)
            print(type(self.z))
            print(self.z.shape)
            """
            self.posterior_var = tf.exp(self.posterior_logvar) 

        p = slim.layers.softmax(self.z)
        p_do = slim.layers.dropout(p, self.keep_prob, scope='p_dropped')               # dropout(softmax(z))
        decoded = slim.layers.linear(p_do, n_hidden_gener_1, scope='FC_decoder')

        self.x_reconstr_mean = tf.nn.softmax(slim.layers.batch_norm(decoded, scope='BN_decoder'))                    # softmax(bn(50->1995))

        print(self.x_reconstr_mean)

    def _create_loss_optimizer(self):

        #self.x_reconstr_mean+=1e-10                                                     # prevent log(0)

        NL = -tf.reduce_sum(self.x * tf.log(self.x_reconstr_mean+1e-10), 1)     # cross entropy on categorical
        #reconstr_loss = -tf.reduce_sum(self.x * tf.log(self.x_reconstr_mean), 1)

        var_division    = self.posterior_var  / self.prior_var
        diff            = self.posterior_mean - self.prior_mean
        diff_term       = diff * diff / self.prior_var
        logvar_division = self.prior_logvar - self.posterior_logvar
        KLD = 0.5 * (tf.reduce_sum(var_division + diff_term + logvar_division, 1) - self.h_dim )

        self.cost = tf.reduce_mean(NL + KLD)

        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.99).minimize(self.cost)

    def partial_fit(self, X):

        #if hasattr(self, 'decoder_weight'):
            #decoder_weight = self.decoder_weight
        #else:
        decoder_weight = [v for v in tf.global_variables() if v.name=='FC_decoder/weights:0'][0]
        opt, cost,emb = self.sess.run((self.optimizer, self.cost, decoder_weight),feed_dict={self.x: X,self.keep_prob: .8})
        return cost,emb

    def test(self, X):
        """Test the model and return the lowerbound on the log-likelihood.
        """
        cost = self.sess.run((self.cost),feed_dict={self.x: np.expand_dims(X, axis=0),self.keep_prob: 1.0})
        return cost
    def topic_prop(self, X):
        """heta_ is the topic proportion vector. Apply softmax transformation to it before use.
        """
        theta_ = self.sess.run((self.z),feed_dict={self.x: np.expand_dims(X, axis=0),self.keep_prob: 1.0})
        return theta_
    
    def embed(self,
              X,
              batch_size=64):
        """
        Embed data into the latent space of a trained model
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data to embed.
        batch_size : int, optional (default: 64)
            Batch size for processing input data.
        Returns
        -------
        z : array, shape (n_samples, dim_latent)
            Embedding of the input data in latent space.
        """
        
        X = X.astype(np.float32)

        num_data = X.shape[0]
        num_batches_full = int(num_data / batch_size)
        batch_last = num_data - (num_batches_full * batch_size)
        if batch_last > 0:
            num_batches = num_batches_full + 1
        else:
            num_batches = num_batches_full

        embs = []

        for k in range(num_batches):

            if k == num_batches_full:
                input_batch = X[k * batch_size:]
            else:
                input_batch = X[k * batch_size: (k + 1) * batch_size]

       
            self.sess = tf.InteractiveSession()
            emb = self.sess.run([self.embeddings],
                                feed_dict={self.arr: input_batch})

            embs.append(emb[0])

        # Concatenate
        z = np.concatenate(embs, axis=0)
        
        return z
    
   
        
