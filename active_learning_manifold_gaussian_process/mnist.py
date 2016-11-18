# -*- coding: utf-8 -*-
import pkl, cPickle, gzip, numpy as np
from landmarks import ManifoldGP
from landmarks import proj_pos
from landmarks import gaussian_init
from numpy import linalg as LA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def normalize(Xs):
    """
    normalize the data
    """
    rs = []
    for X in Xs:
        rs.append(X / LA.norm(X, axis=1)[:, None])
    return tuple(rs)
  
def compute_dist(Xs, landmarks, kern_width):
    """
    compute the distance
    """
    rs = []
    for X in Xs:
        X = np.exp(-(2 - 2 * X.dot(landmarks.T))
                       / kern_width) 
        rs.append(X)
    return tuple(rs)
    

def tune_c(X_train, Y_train, X_valid, Y_valid):
    """
    tuning the parameter C
    """
    Cs = np.logspace(-3, 5, 9)
    rs = []
    best_lr = {'lr':None, 'acc': 0, 'C':0.001}
    
    for C in Cs:
        print(C)        
        
        lr = LogisticRegression(solver='lbfgs', penalty='l2', C=C, multi_class="multinomial", n_jobs=20, max_iter=100)
        lr.fit(X_train, Y_train)
        s = lr.score(X_valid, Y_valid)
        
        rs.append(s)
        
        if s>best_lr['acc']:
            best_lr['lr'] = lr
            best_lr['acc'] = s
            best_lr['C'] = C
        
    #plt.semilogx(Cs, rs,'-o')
    
    return best_lr['lr'], best_lr['acc'], best_lr['C']   
    
    
def generate_mgp(X_train, nums, name):
    """
    generate mgp
    """
    #specify the params of ManifoldGP 
    mgp = ManifoldGP(n_landmarks=nums, batch_size=1000, n_steps=1000,
                     landmarks=None, init_lmk=gaussian_init, proj=proj_pos, rescale=True,
                     random_state=1024, verbose=True, t0=10, gama=0.51)
                     
    
    mgp.learn_landmarks(X_train)
    pkl._save(data_dir+name, mgp)
    
    
def idx(num):
    """
    get the index of landmarks
    """
    count = num / 10
    idx = []
    for i in np.arange(0, 100, 10):
        idx.append(np.random.choice(np.arange(i, i+10), count, replace=False))
    idx = np.hstack(idx)
    return idx
    
    
# Load the dataset
data_dir = "data/minst/"
f = gzip.open(data_dir+'mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

X_train, Y_train = train_set
X_valid, Y_valid = valid_set
X_test, Y_test = test_set


#normalize the data
X_train, X_test, X_valid= normalize([X_train, X_test, X_valid])
 
 
#generate mgp
#1 original [0.91139999999999999, 0.91910000000000003, 0.9204, 0.92400000000000004, 0.92520000000000002, 0.92649999999999999]
generate_mgp(X_train, 100, "mgp_var_reduction.pkl")
#2 advanced [0.9173, 0.91779999999999995, 0.92410000000000003, 0.92720000000000002, 0.93520000000000003, 0.93630000000000002]
#for i in np.arange(10):
#    generate_mgp(X_train[Y_train==i], cc10, "mgp_class_"+str(i)+".pkl")
    

##load mgp
##1
#mgp = pkl._load(data_dir+"mgp_1.pkl")
#landmarks = mgp.landmarks
##2 
##ls = []
##for i in np.arange(10):
##    mgp = pkl._load(data_dir+"mgp_class_"+str(i)+".pkl")
##    ls.append(mgp.landmarks)
##landmarks =  np.vstack(ls)
#kern_width = mgp.kern_width
#
#
##get the test score
#nums = np.arange(50, 110, 10)
#test_scores = []
#for num in nums:
#    print(num)
#    #2
#    #get the index of landmarks
#    
#    #index = idx(num)
#    
#    #compute the distance
#    X_tr, X_va, X_te= compute_dist([X_train, X_valid,  X_test], landmarks[:num, :], kern_width)
#    lr, acc, C = tune_c(X_tr, Y_train, X_va, Y_valid)
#    print("acc={acc} C={C}".format(acc=acc, C=C))
#    test_score = lr.score(X_te, Y_test)
#    test_scores.append(test_score)
#    
#plt.plot(nums, test_scores)
#print(test_scores)







