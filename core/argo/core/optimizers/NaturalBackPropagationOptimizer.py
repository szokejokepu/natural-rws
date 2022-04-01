import tensorflow as tf

import numpy as np

import pdb

from abc import ABC, abstractmethod

#from .utilsOptimizers import my_loss 

from .DropOneLogitOptimizer import DropOneLogitOptimizer

from tensorflow.python.ops.parallel_for.gradients import batch_jacobian, jacobian
        
def new_get_contraction(i):
    if len(i.get_shape().as_list())==2:
        return "jk,j->k"
    elif len(i.get_shape().as_list())==3:
        return "jkl,j->kl"
    else:
        raise Exception("unexpected dimenstion")

    
def get_contraction(i):
    if len(i.get_shape().as_list())==3:
        return "ijk,ij->ik"
    elif len(i.get_shape().as_list())==4:
        return "ijkl,ij->ikl"
    else:
        raise Exception("unexpected dimenstion")

        
class NaturalBackPropagationOptimizer(tf.train.GradientDescentOptimizer, DropOneLogitOptimizer):

    def __init__(self, learning_rate, *args, **kw):

        self._model = kw["model"]
        self._dumping = kw["dumping"] 
        self._memory_efficient = kw["memory_efficient"]
        
        # remove from args before passing to the constructor of tf.train.GradientDescentOptimizer
        kw.pop("model", None)
        kw.pop("dumping", None)
        kw.pop("memory_efficient", None)

        if "name" not in kw.keys():
            kw["name"] = "NaturalBackPropagation"
            
        super().__init__(learning_rate, *args, **kw)
        
    def compute_gradients(self, loss, *args, **kw):

        grads_and_vars = super().compute_gradients(loss, *args, **kw)
        
        #########################################
        # Natural gradient computed in two steps, through the Jacobian
        #########################################

        logits = self._model.logits
        y = self._model.y
        regularizer = self._model.regularizer
        loss_per_sample = self._model.loss_per_sample
        
        n = logits.get_shape().as_list()[1]

        #################
        
        #logits_sliced = tf.slice(self.logits, [0, 0], [-1, n-1])
        logits_add_last_node = tf.concat([logits,
                                          tf.zeros_like(tf.slice(logits, [0, 0], [-1, 1]))],
                                         1)
        # not used    
        #self.accuracy = tf.reduce_mean(tf.cast(
        #    tf.equal(tf.argmax(logits_add_last_node, axis=1),
        #             tf.cast(y, dtype = tf.int64)),
        #    dtype = tf.float32))
         
        # this is not working
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(self.y, tf.int32), logits = logits_replace_last_node) + self.regularizer

        # TODO change factor in front of the regularizer!
        # previously
        # new_loss = my_loss(y, logits_add_last_node) 
        new_loss = loss_per_sample + regularizer 
        #self._model.loss = new_loss
        #self.likelihood_per_sample = new_loss

        #middle_man = self.logits

        # the [0] is necessary, since it returns a list of tensors
        euclidean_gradient = tf.gradients(new_loss, logits)[0] # remove since it is a list
        #remove 1 from the gradient, since the logits have been reduced by one
        #euclidean_gradient_sliced = tf.slice(euclidean_gradient, [0, 0], [-1, n-1])
        
        probabilities = tf.nn.softmax(logits_add_last_node)
        probabilities_sliced = tf.slice(probabilities, [0, 0], [-1, n])
        #self.prob = probabilities
        #self.prob_sliced = probabilities_sliced
            
        #Fisher = tf.linalg.inv(tf.matrix_diag(probabilities) - tf.einsum('ki,kj->kij', probabilities, probabilities))
        #alpha = 1 #0.9
            
        #FISHER_NUMTOL = 0.1 
        
        ###############################################################################
        # old approach, memory inefficient
        '''
        invA = tf.matrix_diag(1/(probabilities_sliced + FISHER_NUMTOL))
        invFisher =  invA + tf.reshape(1/(1 - (tf.reduce_sum(probabilities_sliced, axis=1) + FISHER_NUMTOL)), [-1,1,1])*tf.ones_like(invA)
        # this can be ignored for the moment
        self.invFisher = invFisher #alpha*invFisher + (1-alpha)*tf.eye(n)
            
        # next two for debugging purposes
        #self.invA = invA
        #self.invB = tf.reshape(1/(1 - (tf.reduce_sum(probabilities_sliced, axis=1) + FISHER_NUMTOL)), [-1,1,1])*tf.ones_like(invA)
            
        #self.gradient = gradient
        old_natural_gradient_loss_theta = tf.einsum('kij,ki->kj', self.invFisher, euclidean_gradient)
        self.old_natural_gradient_loss_theta = old_natural_gradient_loss_theta
        '''
        ###############################################################################
        # new approach, memory efficient, tested to be the same as the old approach
        # also faster

        ############# start debug #################
        # debug of the natural gradient by
        # 1/(p[0]+0.001)*e[0] +(np.sum(e[0])/(1-np.sum(p[0])+0.001))*np.ones(shape=e[0].shape)

        
        #self.euclidean_gradient = euclidean_gradient
        #self.TA = 1/(probabilities_sliced + FISHER_NUMTOL)
        #self.Tup = tf.reduce_sum(euclidean_gradient, axis=1)
        #self.Tdown = 1 - (tf.reduce_sum(probabilities_sliced, axis=1)) + FISHER_NUMTOL
        #self.TB = tf.div( tf.reduce_sum(euclidean_gradient, axis=1) , 1 - (tf.reduce_sum(probabilities_sliced, axis=1)) + FISHER_NUMTOL)

        #self.partA = tf.multiply(1/(probabilities_sliced + FISHER_NUMTOL), euclidean_gradient)
        #self.partB = tf.reshape(tf.div( tf.reduce_sum(euclidean_gradient, axis=1) , 1 - (tf.reduce_sum(probabilities_sliced, axis=1)) + FISHER_NUMTOL),[-1,1])*tf.ones_like(euclidean_gradient)

        ############# end debug #################
        
        natural_gradient_loss_theta = tf.multiply(1/(probabilities_sliced + self._dumping), euclidean_gradient) + tf.reshape(tf.div( tf.reduce_sum(euclidean_gradient, axis=1) , 1 - (tf.reduce_sum(probabilities_sliced, axis=1)) + self._dumping),[-1,1])*tf.ones_like(euclidean_gradient)

        #self.natural_gradient_loss_theta = natural_gradient_loss_theta
        ###############################################################################

        #self.nat_grad_theta = gradient

        
        # how to compute gradient wrt the probabilities?
        #self.natural_loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.cast(self.y, tf.int32), logits = tf.log(self.probabilities)) # not reccomended to pass probabilities, since a softmaxi is appied, see documentation
        #self.nat_grad_eta = tf.gradients(self.natural_loss2, self.probabilities)[0]
        #self.nat_grad_eta = tf.slice(self.nat_grad_theta, [0, 0], [-1, n]) 
        
        
        trainable_vars = tf.trainable_variables()

        if self._memory_efficient:
            
            ####################################################
            # new, memory efficient wat to compute the contraction

            # wrong
            #mean_jacobians = jacobian(tf.reduce_mean(logits, axis=0), trainable_vars)
            #mean_natural_gradient_loss_theta = tf.reduce_mean(natural_gradient_loss_theta, axis=0)
            #self.natural_gradient = [tf.einsum(new_get_contraction(i), i, mean_natural_gradient_loss_theta) for i in mean_jacobians]

            self.natural_gradient = [tf.reduce_sum(i, axis=0) for i in jacobian(tf.reduce_mean(logits*tf.stop_gradient(natural_gradient_loss_theta), axis=0),trainable_vars)]
            
            #self.grads_and_vars = zip(self.natural_gradient, variables)
            #grads_and_vars_not_none = [(g, v) for (g, v) in self.grads_and_vars if g is not None]
            #grads = self.natural_gradient


        else:
            #trainable_vars = flatten_weights([trainable_vars[0]])
            #trainable_vars = flatten_weights(trainable_vars)
            #BATCH_SIZE = 100
            #trainable_vars_tiled = tf.reshape(tf.tile(trainable_vars, [BATCH_SIZE]),
            #                                  [BATCH_SIZE, tf.shape(trainable_vars)[0]])
            #
    
            #logits_replace_last_node_sliced = tf.slice(logits_replace_last_node, [0, 0], [-1, n-1])
            jacobians = jacobian(logits, trainable_vars) # [tf.reduce_sum(i, axis=0) for i in jacobian(self.logits, trainable_vars)]
            
            #self.nat_grad = [tf.reduce_mean(i, axis=0) for i in self.jacobian]
            
            # proper way to compute the contraction
            self.natural_gradient = [tf.reduce_mean(tf.einsum(get_contraction(i), i, natural_gradient_loss_theta), axis=0) for i in jacobians]
            #self.nat_grad = [tf.reduce_mean(tf.tensordot(i, self.nat_grad_theta, [[1], [1]]), axis=[0,  1, -1]) for i in self.jacobian]

        ########################################################################
        # end of experiments with natural gradient
        ########################################################################
        
        variables = [v for (g, v) in grads_and_vars if v is not None]
            
        #self.grads_and_vars = zip(self.natural_gradient, variables)
        #grads_and_vars_not_none = [(g, v) for (g, v) in self.grads_and_vars if g is not None]
        grads = self.natural_gradient

        grads_and_vars = [(g, v) for (g, v) in zip(grads, variables)]
            
        #self.new_logits = logits_replace_last_node


        ############## temp ##############
        '''
        mean_jacobians = jacobian(tf.reduce_mean(logits, axis=0), trainable_vars)
        mean_natural_gradient_loss_theta = tf.reduce_mean(natural_gradient_loss_theta, axis=0)

        ng_a = [tf.einsum(new_get_contraction(i), i, mean_natural_gradient_loss_theta) for i in mean_jacobians]
        
        jacobians = jacobian(logits, trainable_vars) # [tf.reduce_sum(i, axis=0) for i in jacobian(self.logits, trainable_vars)]
            
        # proper way to compute the contraction
        ng_b = [tf.reduce_mean(tf.einsum(get_contraction(i), i, natural_gradient_loss_theta), axis=0) for i in jacobians]


        #ng_c = jacobian(tf.reduce_sum(logits*natural_gradient_loss_theta, axis=1),trainable_vars)
        ng_c = [tf.reduce_sum(i, axis=0) for i in jacobian(tf.reduce_mean(logits*tf.stop_gradient(natural_gradient_loss_theta), axis=0),trainable_vars)]
        
        pdb.set_trace()
        '''
        
        return grads_and_vars


    ##########################################
    # start experiments for natural gradient #
    ##########################################
    '''
    l = []
    for i in range(len(self.grads_and_vars)):
        a = tf.reshape(self.grads_and_vars[i][0],[-1,])
        l.append(a)
        print(a)
        b = tf.reshape(a, self.grads_and_vars[i][0].shape)
        print(b)

    # here I need to comute the Fisher information matrix and invert it
    g = tf.concat(l,axis=0)

    # here I should compute the natural gradient

    start = 0
    for i in range(len(self.grads_and_vars)):
        #pdb.set_trace()
        length = l[i].get_shape().as_list()[0]
        c = tf.slice(g, [start], [length])
        start += length
        print(c)
        d = tf.reshape(c, self.grads_and_vars[i][0].shape)
        print(d)

    # apply the gradients
    '''

    ##########################################
    # end experiments for natural gradient #
    ##########################################
