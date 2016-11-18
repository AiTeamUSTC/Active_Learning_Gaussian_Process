# -*- coding: utf-8 -*-


import sys
import numpy as np
from numpy import linalg as LA
import datetime

class ManifoldGP(object):
    
    def __init__(self, n_landmarks=100, batch_size=1000, n_steps=1000,
                 landmarks=None, init_lmk=None, proj=None, rescale=True,
                 random_state=None, verbose=True, **kwargs):
       
        self.n_landmarks = n_landmarks
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.landmarks = landmarks
        self.rescale = rescale
        self.random_state = random_state
        self.verbose = verbose

        if callable(init_lmk):
            self.init_lmk = init_lmk
        else:
            self.init_lmk = _default_init

        if callable(proj):
            self.proj = proj
        else:
            self.proj = _do_nothing

        if type(self.random_state) is int:
            np.random.seed(self.random_state)
        elif self.random_state is not None:
            np.random.setstate(self.random_state)

        self._parse_kwargs(**kwargs)

    def _parse_kwargs(self, **kwargs):
        self.t0 = float(kwargs.get('t0', 0))
        self.gamma = float(kwargs.get('gamma', 0.5))

    def learn_landmarks(self, X, kern_width=None):
       
        if kern_width is None:
            self.kern_width = np.var(X, axis=0).sum()
        else:
            self.kern_width = kern_width

        for l in xrange(self.n_landmarks):
            if self.verbose:
                print("Learning landmark %i:" % l)
                sys.stdout.flush()
            lmk = self._learn_single_landmark(X)
            if self.landmarks is None:
                self.landmarks = lmk[np.newaxis, :]
            else:
                self.landmarks = np.vstack((self.landmarks, lmk))
        return self

    def _learn_single_landmark(self, X):
        n_samples, n_feats = X.shape
        lmk = self.init_lmk(X)#改掉
        for i in xrange(1, self.n_steps + 1):
            idx = np.random.choice(n_samples, size=self.batch_size)
            #lmk = self._grad_step(X[idx], lmk, i)
            lmk = self._var_reduction(X[idx], lmk, i)
            print(i)
            if self.verbose and i % 50 == 0:
                sys.stdout.write('\rProgress: %d/%d' % (i, self.n_steps))
                sys.stdout.flush()
        if self.verbose:
            sys.stdout.write('\n')
        return lmk


    def _grad_step(self, X_batch, lmk, step):
       
                                   
                   
        phi = np.exp(-(2 - 2 * X_batch.dot(lmk)) / self.kern_width)
        if self.landmarks is None:
            M2phi = phi
        else:
            K = np.exp(-(2 - 2 * X_batch.dot(self.landmarks.T))
                       / self.kern_width)
            M2phi = phi - K.dot(LA.lstsq(K.T.dot(K), K.T.dot(phi))[0])
        rho = (self.t0 + step)**(-self.gamma)
        grad_lmk = -4. / self.kern_width * (lmk * phi.dot(M2phi) -
                                            X_batch.T.dot(M2phi * phi))
        if self.rescale:
            grad_lmk /= LA.norm(grad_lmk)
        lmk += rho * grad_lmk
        return self.proj(lmk)
        
        
    def _var_reduction(self, X_batch, lmk, step):
        starttime = datetime.datetime.now()
      
        K = np.exp(-(2 - 2 * X_batch.dot(X_batch.T))
                           / self.kern_width)
        const_s = np.trace(K)
        
        if self.landmarks is None:
            landmarks = lmk.reshape((1, lmk.shape[0]))
        else:
            landmarks =np.vstack([self.landmarks, lmk.reshape((1, lmk.shape[0]))])
            
        size_lmk, dim_t = landmarks.shape
        size_x = X_batch.shape[0]


        Y = np.exp(-(2 - 2 * X_batch.dot(landmarks.T))
                           / self.kern_width)
        X_t = np.exp(-(2 - 2 * X_batch.dot(X_batch.T))
                           / self.kern_width)
                           
        S_inv = LA.lstsq(Y.T.dot(Y), np.eye(size_lmk))[0]
        YS = Y.dot(S_inv)
        YSY = YS.dot(Y.T)
        
        
        const_t = np.trace(X_t.dot(YSY).dot(X_t))
        
        
        phi = np.exp(-(2 - 2 * X_batch.dot(lmk)) / self.kern_width)  #(1000)
        dist = -2*(lmk-X_batch) / self.kern_width # (1000,784)
        phi_dist = dist.T * phi
        
        Y_0 = np.zeros((dim_t, size_x, size_lmk))
        Y_0[:, :, size_lmk-1] = phi_dist
        #print(Y_0.transpose((0, 2, 1)).shape, S_inv.shape)

        Q = np.tensordot(Y_0.transpose((0, 2, 1)), Y, 1)
        Y_0_S_inv = np.tensordot(Y_0, S_inv, 1)
        M = 2*np.tensordot(Y_0_S_inv, Y.T, 1) #- np.tensordot(np.tensordot((Q+Q.transpose((0, 2, 1))).transpose((0, 2, 1)), YS.T, 1).transpose((0, 2, 1)), YS.T, 1)
            

        grad = np.trace(np.tensordot(np.tensordot(M.transpose((0, 2, 1)), X_t.T, 1).transpose((0, 2, 1)), X_t, 1), axis1=1, axis2=2)
              
            
        grad = 2*(const_t-const_s)*grad
        
        rho = (self.t0 + step)**(-self.gamma)
        if self.rescale:
            grad /= LA.norm(grad)
        lmk += rho * grad
        
        endtime = datetime.datetime.now()
        print (endtime - starttime).seconds

        return self.proj(lmk)
        
    def _var_reduction_(self, X_batch, lmk, step):
        starttime = datetime.datetime.now()

        K = np.exp(-(2 - 2 * X_batch.dot(X_batch.T))
                           / self.kern_width)
        const_s = np.trace(K)
        
        if self.landmarks is None:
            landmarks = lmk.reshape((1, lmk.shape[0]))
        else:
            landmarks =np.vstack([self.landmarks, lmk.reshape((1, lmk.shape[0]))])
            
        size_lmk, dim_t = landmarks.shape
        size_x = X_batch.shape[0]


        Y = np.exp(-(2 - 2 * X_batch.dot(landmarks.T))
                           / self.kern_width)
        X_t = np.exp(-(2 - 2 * X_batch.dot(X_batch.T))
                           / self.kern_width)
                           
        S_inv = LA.lstsq(Y.T.dot(Y), np.eye(size_lmk))[0]
        YS = Y.dot(S_inv)
        YSY = YS.dot(Y.T)
        
        const_t = np.trace(X_t.dot(YSY).dot(X_t))
        
        
        phi = np.exp(-(2 - 2 * X_batch.dot(lmk)) / self.kern_width)  #(1000)
        dist = -2*(lmk-X_batch) / self.kern_width # (1000,784)
        
              
        grad = []
        for i in np.arange(dim_t):
            d_t = np.multiply(phi, dist[:,i]).reshape((size_x, 1))
            Y_0 = np.hstack([np.zeros((size_x, size_lmk-1)), d_t])
            #print(Y_0.shape)
            Q = Y_0.T.dot(Y)
            print(Q.shape)
            M = 2*Y_0.dot(S_inv).dot(Y.T) - YS.dot(Q+Q.T).dot(YS.T)
            print(Y_0.dot(S_inv).shape, Y.T.shape)

#            sum_var_X = 0
#            for x in X_t:
#                sum_var_X += x.dot(M).dot(x)
            trace = np.trace(X_t.dot(M).dot(X_t))
            grad.append(trace)
              
            
        grad = np.array(grad) 
        grad = 2*(const_t-const_s)*grad
        
        rho = (self.t0 + step)**(-self.gamma)
        if self.rescale:
            grad /= LA.norm(grad)
        lmk += rho * grad
        
        endtime = datetime.datetime.now()
        print (endtime - starttime).seconds

        return self.proj(lmk)
# helper functions to initialize landmarks and projection #
def _do_nothing(x):
    return x


def proj_pos(x):
    x[x < 0] = 0
    return x

def proj_sph_pos(x):
    x = proj_pos(x)
    return x/LA.norm(x)

def _default_init(n_feats):
    lmk = proj_sph_pos(np.ones(n_feats))
    return lmk
    
    
#添加的gauss 初始化landmarks  
def gaussian_init(X):
    cov_X = np.cov(X, rowvar=False)
    diag_cov_X = np.diag(np.diag(cov_X))
    mean_X = np.mean(X, axis=0)
    lmk = np.random.multivariate_normal(mean_X, diag_cov_X)
    return proj_sph_pos(lmk)
    
    
    
    

