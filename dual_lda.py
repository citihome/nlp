# -*- coding: utf-8 -*-
"""
LDA revisted

Created on Sun Dec 13 08:21:01 2015

@author: hutch
"""

import numpy    
from utils.probspace import euclid2prob
               
       
class LDA(object):    
    def __init__(self, rng, corpus, topic_num=100):
        self.rng = numpy.random.RandomState(numpy.random.randint(2**32-1)) if rng is None else rng
        self.topic_num=topic_num
        self.t = 0
        
    def fit(self, corpus, n_iter=1, offset=10, eta=1e-2, timevarying=True, **kwargs):
        """ online learning solution for the following optimization problem.
            max_w {softmax(dot(L2_w,xt)+L2_b)-0.5*nnz*mean(\|L2_w\|_F^2+\|L2_b\|^2)} + R
        """        
        vocab_size = corpus.shape[1]
        if self.t == 0:
            mu, sigma = corpus.data.mean(), 2.56*corpus.data.var()**0.5
            self.L2_w = mu + (mu if mu < sigma else sigma)*\
                self.rng.uniform(low=-1,high=1,size=(vocab_size,self.topic_num))
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
                #2.difference between xt*L2_w*P_z - nnz*mean(L2_w**2) 
                euclid2prob(corpus[s].dot(L2_w)+L2_b, P_z, offset)
                
                idx, xt = corpus[s].nonzero()[1], corpus[s].data
                L2_w *= (1-alpha*xt.size/L2_w.size)                
                L2_w[idx] += (alpha*xt)[:,numpy.newaxis]*P_z
                L2_b += alpha*(1-L2_b)*P_z
            ##
            print "."
            self.t += 1; beta = 1./self.t
            self.L2_w *=(1-beta); self.L2_w += beta*L2_w
            self.L2_b *=(1-beta); self.L2_b += beta*L2_b
            
    def predict(self, x,**kwargs):
        """ prediction for batch data
        """
        y = (x.dot(self.L2_w)+self.L2_b).argmax(-1)
        return y
        
if __name__ == "__main__":
    import time, os, h5py
    from scipy import sparse,io
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
            
            #operator
            lda = LDA(rng, corpus, topic_num=topic_num)
            lda_g2p = numpy.zeros((max_iters, class_num, topic_num))
            lda_p2g = numpy.zeros((max_iters, topic_num, class_num))
            for t in xrange(max_iters):
                #####LDA#####
                #learning
                t0 = time.time()
                lda.fit(corpus)
                learning_time = time.time() - t0
                #debug code
                if False:
                    freq = numpy.reshape(numpy.array(corpus.sum(0)),-1)
                    rank = dict(zip(freq.argsort()[::-1],numpy.arange(freq.size)))
                    vocab = fp["vocab"]
                    I = numpy.argsort(lda.L2_w, 0)
                    for k in xrange(I.shape[1]):
                        print [(vocab[idx],rank[idx],freq[idx]) for idx in I[-10:,k]]
                    
                    #prediction
                    t1 = time.time()
                    yhat = lda.predict(corpus)
                    print "prediction_time =", time.time() - t1
                    #map from ground_truth to topic, and reverse
                    cnt = numpy.ones_like(gnd)
                    lda_g2p[t] = sparse.csr_matrix(\
                        (cnt, (gnd,yhat)),\
                        shape=(class_num,topic_num)\
                    ).todense()
                    lda_p2g[t] = sparse.csr_matrix(\
                        (cnt, (yhat,gnd)),\
                        shape=(topic_num,class_num)\
                    ).todense()
                #####
                print "%10s"%("LDA"),\
                    "%11d"%topic_num, "%6d"%(t+1),\
                    "%6.2f"%learning_time
            filename = os.getcwd()+"/result/dual_lda%d(timevarying)_"%(topic_num)+datasetname
            fp = h5py.File(filename,"w")
            fp["L2_w"]=lda.L2_w;fp["L2_b"]=lda.L2_b
            fp.close()