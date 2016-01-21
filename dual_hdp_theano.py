# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 09:57:06 2015

@author: hutch
"""

import numpy
import theano
from theano import tensor, sparse
from dual_lda_theano import LDA

class HDP(LDA):
    """ with objective function as logsumexp(theta-cumsum(theta)), where
        theta = (wx+b)-0.5*(w**2+b**2)
    """
    def __init__(self, rng, x, topic_num=100):
        
        #input
        L2_input = sparse.csr_matrix("x",dtype=theano.config.floatX)
        #params
        vocab_size = x.shape[1]
        mu, sigma = x.data.mean(), x.data.var()**0.5
        
        rng = numpy.random.RandomState(numpy.random.randint(2**32-1)) if rng is None else rng
        self.L2_w = theano.shared(\
            numpy.asarray(\
                rng.normal(loc=mu,scale=sigma,size=(vocab_size, topic_num)),\
                dtype=theano.config.floatX\
            ),\
            borrow=True\
        )
        self.L2_b = theano.shared(numpy.zeros(topic_num,dtype=theano.config.floatX), borrow=True)
        self.params = [self.L2_w, self.L2_b]
        
        #stick-breaking:sticks->orthgonal sticks
        L2_stick = sparse.dot(L2_input,self.L2_w)+self.L2_b-\
            0.5*(L2_input.size/vocab_size*tensor.sum(self.L2_w**2,0)+self.L2_b**2)  
        zero_space = tensor.zeros((L2_input.shape[0],1),dtype=theano.config.floatX)
        L2_orth_stick = tensor.join(1, L2_stick, zero_space)\
            - tensor.join(1, zero_space, tensor.cumsum(L2_stick,1))
        Pasterik_orth_stick = tensor.log(1 + tensor.exp(L2_orth_stick))      
        #training model definition
        Likelihood = tensor.mean(Pasterik_orth_stick)
        grads = theano.grad(Likelihood, self.params)#gradient w.r.t params
        eta = tensor.scalar("eta")
        updates = [(param, param+eta*grad) for param, grad in zip(self.params, grads)]
        self._fit = theano.function(\
            inputs=[L2_input, eta],\
            outputs=Likelihood,\
            updates=updates\
        )
        #predict model definition
        self._predict = theano.function(\
            inputs=[L2_input],\
            outputs=tensor.argmax(L2_stick,axis=-1)\
        )
        self._codec = theano.function(\
            inputs=[L2_input],\
            outputs=L2_stick>0\
        )
        
    def codec(self, corpus, **kwargs):
        code = (self._codec(corpus)).astype("uint8")
        return code