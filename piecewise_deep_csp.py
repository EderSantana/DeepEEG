import numpy as np
import scipy as sp
from scipy.io import loadmat
import theano
import theano.tensor as T
from scipy import signal
from sklearn.linear_model import LogisticRegression as LR
from sklearn.lda import LDA

# Load dataset
data = loadmat('sp1s_aa_2py.mat')
x_train = data['x_train']
y_train = np.array(data['y_train'], dtype=int)
x_test  = data['x_test']
y_test  = np.array(data['y_test'], dtype=int)

# Band-pass filter signal
samp_rate = 100.
(b, a) = signal.butter(5, np.array([8., 30.]) / (samp_rate / 2.), 'band')
x_train_filt = signal.filtfilt(b, a, x_train, axis=0)
x_test_filt  = signal.filtfilt(b, a, x_test, axis=0)

def csp(x_train_filt, y_train):
    """Calculate Common Spatial Patterns Decompostion and Returns 
    spatial filters W"""

    # Calculate correlation matrices
    X0 = x_train_filt[:,:,y_train[:,0]==0]
    X1 = x_train_filt[:,:,y_train[:,0]==1]

    C0 = 0.
    for i in xrange( X0.shape[2] ):
        C0 = C0 + np.dot(X0[:,:,i].transpose() , X0[:,:,i])

    C0 = C0/X0.shape[2]

    C1 = 0.
    for i in xrange( X1.shape[2] ):
        C1 = C1+np.dot(X1[:,:,i].transpose(), X1[:,:,i])

    C1 = C1/X1.shape[2]

    # Calculate CSP
    D, V   = sp.linalg.eig(C1, C1+C0);
    ind = sorted(range(D.size), key=lambda k: D[k])
    V = V[:,ind];
    W = np.hstack([V[:,0:2], V[:,25:]]);

    return W

def classify_csp(W, V, x_train_filt, y_train, x_test_filt, y_test):
    """ Classify data using CSP filter W"""
    # Project data
    proj_train = sp.tensordot(W.transpose(), x_train_filt, axes=[1,1])
    proj_test  = sp.tensordot(W.transpose(), x_test_filt, axes=[1,1])

    # Calculate features
    #ftr = ( np.log(proj_train**2).sum(axis=1) )
    #fte = ( np.log(proj_test**2).sum(axis=1) )
    ftr = np.log( np.tensordot(proj_train**2, V, axes=[1,0]) )[:,:,0]
    fte = np.log( np.tensordot(proj_test **2, V, axes=[1,0]) )[:,:,0]
    # Classify 
    logistic = LR()
    logistic.fit(ftr.transpose(), y_train[:,0])
    sc = logistic.score(fte.transpose(), y_test[:,0])
    
    return sc

W = csp(x_train_filt, y_train)
V = np.ones((50,1))
sc = classify_csp(W, V, x_train_filt, y_train, x_test_filt, y_test) 

# Fine tune CSP pipeline
# Note input data dim: [batches, time, channel]
# Filter dim: [channel_in, channel_out]
from logistic_sgd import LogisticRegression

x_train_filt_T = theano.shared(x_train_filt.transpose(2, 0, 1))
x_test_filt_T  = theano.shared(x_test_filt.transpose(2, 0, 1))
y_train_T      = T.cast( theano.shared(y_train[:,0]), 'int32')
y_test_T       = T.cast( theano.shared(y_test[:,0]) , 'int32')

lr         = .01 # learning rate
batch_size = 316/4
epochs     = 1700
index      = T.lscalar('index')
y          = T.ivector('y')
X          = T.tensor3('X')
csp_w      = theano.shared(W)
avg_v      = theano.shared(V)
proj_csp   = T.tensordot(X,csp_w,axes=[2,0])
layer0_out = T.pow(proj_csp, 2)
variance   = T.tensordot(layer0_out, avg_v, axes=[1,0])
#layer1_out = T.reshape(T.log(T.squeeze(variance)), [316, 5]) 
#layer1_out = T.log(T.squeeze(variance))[:,:,0] 
layer1_out = T.log((variance))[:,:,0] 
layer2     = LogisticRegression(input=layer1_out, n_in=5, n_out=2)
cost       = layer2.negative_log_likelihood(y)+.01*T.sum(T.pow(avg_v,2))

params  = [csp_w, avg_v] + layer2.params
#params = layer2.params
grads   = T.grad(cost,params)
updates = []
for param_i, grad_i in zip(params,grads):
    updates.append((param_i, param_i - lr*grad_i))


train_model = theano.function([index], cost, updates=updates,
      givens={
          X: x_train_filt_T[index * batch_size: (index + 1) * batch_size],
          y: y_train_T[index * batch_size: (index + 1) * batch_size]})

#layer1_out_test = T.reshape(T.log(T.squeeze(variance)), [100, 5]) 
    #layer2_test      = LogisticRegression(input=layer1_out_test, n_in=5, n_out=2)
    #layer2_test.W = theano.shared(layer2.W.get_value(),borrow=True)
    #layer2_test.b = theano.shared(layer2.b.get_value(),borrow=True)
test_model = theano.function([], layer2.errors(y), givens = {
        X: x_test_filt_T, y: y_test_T})


for i in range(epochs):
    for j in range(316/batch_size):
        cost_ij = train_model(j)
    #print 'Cost at epoch %i = %f' % i, cost_ij
    
    er = test_model()
    print 'Epoch = %i' % i
    print 'Cost = %f' % cost_ij
    print 'Test error = % f' % er
