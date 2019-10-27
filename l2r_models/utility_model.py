import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import KFold, ShuffleSplit
import scipy
from scipy.optimize import minimize

def run_utility_model(data, loss_fun, nsplits, test_size, train_size):
    print("[TRAIN]Running Utility Model")
    d = len(data[0][0])
    k = 1
    lambda_reg = 1

    print("[TRAIN]Number of Features="+str(d))

    feat_trans = set([])
    feat_trans_dup = list(itertools.product(range(d + 1), repeat=k))
    for t in feat_trans_dup:
        feat_trans.add(tuple(sorted(t)))
    feat_trans = list(feat_trans)
    feat_trans.remove(tuple([d] * k))

    d_ext = len(feat_trans)
    learnt_beta = np.zeros((1, d_ext))

    compars = []
    for i in range(len(data)):
        x1 = np.array(data[i][0])
        x2 = np.array(data[i][1])
        label = np.array(data[i][2])

        if (label == 1):
            compars.append(np.array([x1, x2]))
        else:
            print(np.array([x2, x1]).shape)
            compars.append(np.array([x2, x1]))

    compars = np.array(compars)

    kf = ShuffleSplit(n_splits=nsplits, test_size=test_size, train_size=train_size)

    accuracy = []
    for train_index, test_index in kf.split(compars):
        train_comps = compars[train_index,]
        test_comps = compars[test_index,]

        print("[TRAIN] Length of Train="+str(len(train_comps)))
        print("[TRAIN] Length of Test=" + str(len(test_comps)))

        k_train_comps = []
        k_test_comps = []

        n_train = len(train_comps)
        n_test = len(test_comps)

        for j in range(n_train):
            altA = train_comps[j, 0, :]
            altA = list(altA)
            altA.append(1)
            k_altA = []
            for t in feat_trans:
                this_prod = 1
                for index in t:
                    this_prod *= altA[index]
                k_altA.append(this_prod)

            altB = train_comps[j, 1, :]
            altB = list(altB)
            altB.append(1)
            k_altB = []
            for t in feat_trans:
                this_prod = 1
                for index in t:
                    this_prod *= altB[index]
                k_altB.append(this_prod)

            k_train_comps.append([k_altA, k_altB])
        k_train_comps = np.array(k_train_comps)

        for j in range(n_test):
            altA = test_comps[j, 0, :]
            altA = list(altA)
            altA.append(1)
            k_altA = []
            for t in feat_trans:
                this_prod = 1
                for index in t:
                    this_prod *= altA[index]
                k_altA.append(this_prod)

            altB = test_comps[j, 1, :]
            altB = list(altB)
            altB.append(1)
            k_altB = []
            for t in feat_trans:
                this_prod = 1
                for index in t:
                    this_prod *= altB[index]
                k_altB.append(this_prod)

            k_test_comps.append([k_altA, k_altB])
        k_test_comps = np.array(k_test_comps)

        THRES = 0.00001
        this_diff = np.zeros((n_train, d_ext))

        for j in range(n_train):  # Pre-compute X_j - Z_j and store
            this_diff[j, :] = k_train_comps[j, 0, :] - k_train_comps[j, 1, :]

        this_beta = np.random.uniform(-0.001, 0.001,
                                      d_ext)  # Initialize parameter randomly	# Could instead use np.zeros(d), since it's a convex program

        '''
        REGULARIZATION
        '''

        def normal_likeli(beta):  # negative of the log-likelihood
            # print(f'normal likeli: {-np.sum(scipy.stats.norm.logcdf(np.dot(this_diff, beta)))}')

            # L2 regularization
            reg_penalty = lambda_reg * np.dot(beta, beta)

            return -np.sum(scipy.stats.norm.logcdf(np.dot(this_diff, beta))) - reg_penalty

        def der_normal_likeli(beta):
            dot_prod = np.dot(this_diff, beta)
            pdfs = scipy.stats.norm.pdf(dot_prod)
            cdfs = scipy.stats.norm.cdf(dot_prod)

            grad = np.zeros(d_ext)
            for j in range(n_train):
                grad += this_diff[j] * pdfs[j] / (max(cdfs[j], THRES))

            # print(f'normal grad: {grad}')
            reg_penalty = 2 * np.sum(beta)

            return -grad - reg_penalty

        def logistic_likeli(beta):  # negative of the log-likelihood
            return np.sum(np.log(1 + np.exp(-np.dot(this_diff, beta))))

        def der_logistic_likeli(beta):
            dot_prod = np.dot(this_diff, beta)

            grad = np.zeros(d_ext)
            for j in range(n_train):
                grad += this_diff[j] * np.exp(-dot_prod[j]) / (1 + np.exp(-dot_prod[j]))

            return -grad

        if (loss_fun == 'normal'):
            likeli = normal_likeli
            der_likeli = der_normal_likeli
        elif (loss_fun == 'logistic'):
            likeli = logistic_likeli
            der_likeli = der_logistic_likeli

        res = minimize(likeli, this_beta, method='BFGS', jac=der_likeli, options={'gtol': 1e-10, 'disp': False})
        this_beta = res.x
        learnt_beta[0, :] = this_beta

        num_correct = 0

        TRUE_pos = {}
        TRUE_neg = {}
        FALSE_pos = {}
        FALSE_neg = {}

        for j in range(n_test):
            mean_util_0 = np.dot(this_beta, k_test_comps[j, 0, :])
            mean_util_1 = np.dot(this_beta, k_test_comps[j, 1, :])

            prob_0_beats_1 = scipy.stats.norm.cdf(mean_util_0 - mean_util_1, scale=2)
            if (mean_util_0 > mean_util_1):
                num_correct += 1

            else:
                diff = k_test_comps[j, 0, :] -  k_test_comps[j, 1, :]
                print(mean_util_0, mean_util_1)
                print("---"*30)

            diff = k_test_comps[j, 0, :] - k_test_comps[j, 1, :]
            pos_symbols = np.where(diff>0)[0]
            neg_symbols = np.where(diff<0)[0]

            if(mean_util_0>mean_util_1):
                for p in pos_symbols:
                    try:
                        TRUE_pos[p] +=1
                    except KeyError as e:
                        TRUE_pos[p] = 1

                for n in neg_symbols:
                    try:
                        TRUE_neg[n] +=1
                    except KeyError as e:
                        TRUE_neg[n] = 1
            else:
                for p in pos_symbols:
                    try:
                        FALSE_pos[p]+=1
                    except KeyError as e:
                        FALSE_pos[p] = 1

                for n in neg_symbols:
                    try:
                        FALSE_neg[n]+=1
                    except KeyError as e:
                        FALSE_neg[n] = 1
                        
        print("="*50)
        print("[TRAIN]Weight Feature Learnt=" + str(len(this_beta)))
        print("="*50)

        accuracy.append(float(num_correct)/n_test)
        print(accuracy)
        print("Number Correct="+str(num_correct)+ " OUT OF "+str(n_test))

    print("[TRAIN] This Beta="+str(len(this_beta)))
    return accuracy, this_beta