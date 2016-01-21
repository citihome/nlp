# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 09:30:23 2016

@author: hutch
"""

from dual_lda import LDA
import dual_lda_theano.LDA as LDA_theano
from utils.probspace import euclid2prob, euclid2sign
import theano, numpy
from theano import tensor, sparse
        
class IQLDA(LDA):
    """ Irrelevance Quadratic LDA
        use difference xw - w2 as input of L2_topic, where element of vocabulary is unrelated
    """        
    def fit(self, corpus, n_iter=1, offset=10, eta=1e-2, timevarying=False, **kwargs):
        """ online learning solution for the following optimization problem.
            max_w logphi(dot(xt,w)+b-0.5*(w[I]**2+b**2)) - eta/2*\|w\|_F^2
            log_phi(.) := (.) - log_sum_exp(.)
        """
        vocab_size = corpus.shape[0]
        if self.t == 0:
            mu, sigma = corpus.data.mean(), 2.56*corpus.data.var()**0.5
            self.L2_w = mu + (mu if mu < sigma else sigma)*\
                self.rng.uniform(low=-1,high=1,size=(vocab_size, self.topic_num))
            self.L2_b = numpy.zeros(self.topic_num)
            
        L2_w = numpy.empty((vocab_size, self.topic_num))
        L2_b = numpy.empty(self.topic_num)
            
        P_z = numpy.empty(self.topic_num)      
        for t in xrange(n_iter):
            L2_w[:] = self.L2_w
            L2_b[:] = self.L2_b
            for s in xrange(corpus.shape[0]):
                #update parameter
                #1.learning rate
                alpha = eta/(s+1.) if timevarying else eta
                #2.probability of topic
                idx, xt = corpus[s].nonzero()[1], corpus[s].data
                L2_w_idx = L2_w[idx]
                euclid2prob(xt.dot(L2_w_idx)+L2_b-0.5*((L2_w_idx*L2_w_idx).sum(0)+L2_b*L2_b), P_z, offset)
                #w+= alpha*(xt-w)*P_z-> w-= alpha*w*Pz; w += alpha*xt'*P_z
                L2_w[idx] *= (1-alpha*P_z)
                L2_w[idx] += (alpha*xt)[:,numpy.newaxis]*P_z
                L2_b += alpha*(1-L2_b)*P_z
            ##
            print "."
            self.t += 1; beta = 1./self.t
            self.L2_w *=(1-beta); self.L2_w += beta*L2_w
            self.L2_b *=(1-beta); self.L2_b += beta*L2_b
            self.L2_w2 = self.L2_w*self.L2_w
            self.L2_b_b2 = self.L2_b-0.5*self.L2_b*self.L2_b
            
    def predict(self, x):
        """ prediction for batch data
        """
        y = numpy.array([\
            (xt.dot(self.L2_w)+0.5*numpy.sum(self.L2_w2[xt.nonzero()[1]],0)+self.L2_b_b2).argmax(-1)\
            for xt in x\
            ])
        return y
        
    
class QLDA(LDA_theano):
    """use difference xw - w2 as input of L2_topic
    """
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
        L2_topic = sparse.dot(L2_input,self.L2_w)+self.L2_b-\
            0.5*(L2_input.size*tensor.mean(self.L2_w**2,0)+self.L2_b**2)
                
        #Pasterik based objective function
        Pasterik_topic = tensor.log(tensor.sum(tensor.exp(L2_topic-L2_topic.max(-1, keepdims=True)),-1))#avoiding overflow
        grads = theano.grad(Pasterik_topic, self.params)#gradient w.r.t params
        eta = tensor.scalar("eta")
        updates = [(param, param+eta*grad) for param, grad in zip(self.params, grads)]
        #training model definition
        self._fit = theano.function(\
            inputs=[L2_input, eta],\
            outputs=Pasterik_topic, \
            updates=updates\
        )
        #predict model definition
        self._predict = theano.function(\
            inputs=[L2_input],\
            outputs=tensor.argmax(L2_topic,axis=-1)\
        )
        

class SparseLDA(LDA):
    """Signed(sparsed) LDA, where vocab provide sign/sparse views
    """
    def fit(self, corpus, n_iter=1, offset=10, eta=1e-2, timevarying=False, **kwargs):
        """ online learning solution for the following optimization problem.
            max_w softmax[dot(Pasterik_w,xt)+L2_b)-0.5*(nnz*mean(L2_w**2)+L2_b**2)]+R
        """
        vocab_size = corpus.shape[1]
        if self.t == 0:
            mu, sigma = corpus.data.mean(), 2.56*corpus.data.var()**0.5
            self.L2_w = mu + (mu if mu < sigma else sigma)*\
                self.rng.uniform(low=-1,high=1,size=(vocab_size, self.topic_num))
            self.L2_b = numpy.zeros(self.topic_num)
            
        L2_w = numpy.empty((vocab_size, self.topic_num))
        L2_b = numpy.empty(self.topic_num)
            
        P_topic = numpy.empty(self.topic_num)
        for t in xrange(n_iter):
            L2_w[:] = self.L2_w
            L2_b[:] = self.L2_b
            for s in xrange(corpus.shape[0]):
                #update parameter
                #1.learning rate
                alpha = eta/(s+1.) if timevarying else eta
                #2.difference between xt*prob(topic) - eta*L2_w 
                idx, xt = corpus[s].nonzero()[1], corpus[s].data
                P_w, Pasterik_w = euclid2sign(L2_w[idx])
                euclid2prob(numpy.dot(xt, Pasterik_w)+L2_b, P_topic, offset)
                
                scale = 1-float(xt.size)/L2_w.size*alpha
                L2_w *= scale; L2_b *= scale
                
                L2_w[idx] += (alpha*xt)[:,numpy.newaxis]*P_w*P_topic
                L2_b += alpha*P_topic
            ##
            print "."
            self.t += 1
            beta = 1./self.t
            self.L2_w *= beta; self.L2_w += (1-beta)*L2_w
            self.Pasterik_w = euclid2sign(self.L2_w, offset)[1]
            self.L2_b *= beta; self.L2_b += (1-beta)*L2_b
            
    def predict(self, x):
        """ prediction for batch data
        """
        y = (x.dot(self.Pasterik_w)+self.L2_b).argmax(-1)
        return y
        

        
if __name__ == "__main__":
    import time, os, h5py
    from scipy import io
    # variables for experiment
    rng = numpy.random.RandomState(numpy.random.randint(2**32-1))
    max_iters = 5
    for datasetname in ["20NewsHome","Reuters21578",]:#"20newsgroups","20newsgroups(freq_top100_removed)",]:
        
        print "\n"
        print '='*10, "Experiment LDA on %s"%(datasetname), '='*10
        
        # load data from file
        try:
            fp = io.loadmat("../data/corpus/"+datasetname)
            corpus, gnd = fp['corpus'], fp['gnd'].reshape(-1)
            gnd -= gnd.min(); class_num = gnd.max()+1
        except Exception, excp:
            print excp
            continue
        
        topic_config = [10, 20, 50, 100, 200]#[10,]#
        for k, topic_num in enumerate(topic_config):
            #operand: partition of train data and test data
            shuffle = numpy.random.permutation(corpus.shape[0])
            corpus, gnd = corpus[shuffle], gnd[shuffle]
            
            t0 =time.time()
            iqLda = IQLDA(rng, corpus, n_iter=max_iters, topic_num=topic_num)
            print "elapse time for %10s(%04d)"%("IQLDA", topic_num), time.time() - t0            
            filename = os.getcwd()+"/result/dual_iqlda%d_"%(topic_num)+datasetname
            fp = h5py.File(filename,"w")
            fp["L2_w"]=iqLda.L2_w;fp["L2_b"]=iqLda.L2_b
            fp.close()
            
            t0 =time.time()
            qLda = QLDA(rng, corpus, n_iter=max_iters, topic_num=topic_num)
            print "elapse time for %10s(%04d)"%("QLDA", topic_num), time.time() - t0
            filename = os.getcwd()+"/result/dual_ilda%d_"%(topic_num)+datasetname
            fp = h5py.File(filename,"w")
            fp["L2_w"]=qLda.L2_w;fp["L2_b"]=qLda.L2_b
            fp.close()
            
            t0 =time.time()
            sLda = SparseLDA(rng, corpus, n_iter=max_iters, topic_num=topic_num)
            print "elapse time for %10s(%04d)"%("SparseLDA", topic_num), time.time() - t0
            filename = os.getcwd()+"/result/dual_ilda%d_"%(topic_num)+datasetname
            fp = h5py.File(filename,"w")
            fp["L2_w"]=sLda.L2_w;fp["L2_b"]=sLda.L2_b
            fp.close()
            
            