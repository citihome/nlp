# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:16:07 2015

@author: hutch
"""

import theano, numpy
from theano import tensor, sparse
 
class LDA(object):
    def __init__(self, rng, x, topic_num=100):
        #input
        L2_input = sparse.csr_matrix("x",dtype=theano.config.floatX)
        #params
        vocab_size = x.shape[1]
        mu, sigma = x.data.mean(), 2.56*x.data.var()**0.5
        
        rng = numpy.random.RandomState(numpy.random.randint(2**32-1)) if rng is None else rng
        self.L2_w = theano.shared(\
            numpy.asarray(\
                mu + (mu if mu < sigma else sigma)*rng.uniform(low=-1,high=1,size=(vocab_size, topic_num)),\
                dtype=theano.config.floatX\
            ),\
            borrow=True\
        )
        self.L2_b = theano.shared(numpy.zeros(topic_num, dtype=theano.config.floatX), borrow=True)
        
        self.params = [self.L2_w, self.L2_b]
        #output
        L2_topic = sparse.dot(L2_input,self.L2_w)+self.L2_b
                
        #difference based objective function
        Pasterik_topic = tensor.log(tensor.sum(tensor.exp(L2_topic-L2_topic.max(-1, keepdims=True)),-1))#avoiding overflow
        d_xw_w2 = tensor.mean(Pasterik_topic) -\
            0.5*(L2_input.size*tensor.mean(self.L2_w*self.L2_w)+tensor.dot(self.L2_b,self.L2_b))
        grads = theano.grad(d_xw_w2, self.params)#gradient w.r.t params
        eta = tensor.scalar("eta")
        updates = [(param, param+eta*grad) for param, grad in zip(self.params, grads)]
        #training model definition
        self._fit = theano.function(\
            inputs=[L2_input, eta],\
            outputs=d_xw_w2, \
            updates=updates\
        )
        #predict model definition
        self._predict = theano.function(\
            inputs=[L2_input],\
            outputs=tensor.argmax(L2_topic,axis=-1)\
        )       
        
        
    def fit(self, corpus, n_iter=5, eta=1e-2, timevaring=True, batchsize=10, **kwargs):
        """ online learning solution for the following optimization problem.
            max_w logphi(dot(w, xt)) - eta/2*\|w\|_F^2
            log_phi(.) := (.) - log_sum_exp(.)
        """
        for t in xrange(n_iter):
            L = 0
            for s in xrange(corpus.shape[0]//batchsize):
                alpha = eta/(s+1) if timevaring else eta
                L += self._fit(corpus[s*batchsize:(s+1)*batchsize], alpha)
            s = corpus.shape[0]//batchsize
            if s*batchsize < corpus.shape[0]:
                alpha = eta/(s+1) if timevaring else eta
                L += self._fit(corpus[s*batchsize:], alpha)
        return numpy.ceil(L/batchsize)
            
    def predict(self, x, **kwargs):
        """ prediction for batch data
        """
        y = self._predict(x)
        return y