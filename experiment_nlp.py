# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 07:44:20 2015

@author: hutch
"""
import numpy
import time, os
from scipy import sparse, io
import h5py
from sklearn.decomposition import LatentDirichletAllocation
from dual_lda_theano import LDA#from dual_lda import LDA#
from dual_hdp_theano import HDP#from dual_hdp import HDP#


def experimentLDA():
    """Data:Cls = 1:n
    """

    # variables for experiment
    rng = numpy.random.RandomState(numpy.random.randint(2**32-1))
    max_iters = 5#2#
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
                import theano
                if isinstance(lda.L2_w, theano.sandbox.cuda.CudaNdarraySharedVariable):
                    L2_w = lda.L2_w.get_value()
                    L2_b = lda.L2_b.get_value()
                else:
                    L2_w, L2_b = lda.L2_w; lda.L2_b
                #debug code
                if False:
                    freq = numpy.reshape(numpy.array(corpus.sum(0)),-1)
                    rank = dict(zip(freq.argsort()[::-1],numpy.arange(freq.size)))
                    vocab = fp["vocab"]
                    I = numpy.argsort(L2_w, 0)
                    #for k in xrange(I.shape[1]):
                    #    print [(vocab[idx],rank[idx],freq[idx]) for idx in I[-10:,k]]                
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
                print "%10s"%("LDA"), \
                    "%11d"%topic_num, "%6d"%(t+1),\
                   "%6.2f"%learning_time
            filename = os.getcwd()+"/result/dual_lda%d_"%(topic_num)+datasetname
            fp = h5py.File(filename,"w")
            fp["L2_w"]=L2_w;fp["L2_b"]=L2_b
            fp.close()
            
            primal_lda = LatentDirichletAllocation(n_topics=topic_num, max_iter=max_iters)
            primal_lda.fit(corpus)
            import cPickle
            filename = os.getcwd()+"/result/primal_lda%d_"%(topic_num)+datasetname
            cPickle.dump(\
                primal_lda,\
                file(filename,"w")\
            )


def experimentHDP():
    # variables for experiment
    rng = numpy.random.RandomState(numpy.random.randint(2**32-1))
    max_iters = 5#2#
    for datasetname in ["20NewsHome","Reuters21578",]:#"20newsgroups","20newsgroups(freq_top100_removed)",]:
        
        print "\n"
        print '='*10, "Experiment HDP on %s"%(datasetname), '='*10
        
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
            hdp = HDP(rng, corpus, topic_num=topic_num)
            hdp_g2p = numpy.zeros((max_iters, class_num, topic_num+1))
            hdp_p2g = numpy.zeros((max_iters, topic_num+1, class_num))
            for t in xrange(max_iters):
                #learning
                t0 = time.time()
                hdp.fit(corpus)
                learning_time = time.time() - t0
                import theano
                if isinstance(hdp.L2_w, theano.sandbox.cuda.CudaNdarraySharedVariable):
                    L2_w = hdp.L2_w.get_value()
                    L2_b = hdp.L2_b.get_value()
                else:
                    L2_w, L2_b = hdp.L2_w, hdp.L2_b
                #debug code
                if True:
                    freq = numpy.reshape(numpy.array(corpus.sum(0)),-1)
                    rank = dict(zip(freq.argsort()[::-1],numpy.arange(freq.size)))
                    vocab = fp["vocab"]
                    I = numpy.argsort(L2_w, 0)
                    #for k in xrange(I.shape[1]):
                    #    print [(vocab[idx],rank[idx],freq[idx]) for idx in I[-10:,k]]
                
                    #prediction
                    t1 = time.time()
                    yhat = hdp.predict(corpus)
                    print "prediction_time =", time.time() - t1
                    #statistic
                    cnt = numpy.ones_like(gnd)
                    hdp_g2p[t] = sparse.csr_matrix(\
                        (cnt, (gnd,yhat)),\
                        shape=(class_num,topic_num+1)\
                    ).todense()
                    hdp_p2g[t] = sparse.csr_matrix(\
                        (cnt, (yhat,gnd)),\
                        shape=(topic_num+1,class_num)\
                    ).todense()
                #####
                print "%10s"%("HDP"),\
                    "%11d"%topic_num, "%6d"%(t+1),\
                   "%6.2f"%learning_time
            filename = os.getcwd()+"/result/dual_hdp%d_"%(topic_num)+datasetname
            fp = h5py.File(filename,"w")
            fp["L2_w"]=L2_w;fp["L2_b"]=L2_b
            fp.close()
    
if __name__ == "__main__":
#    experimentLDA()
    experimentHDP()