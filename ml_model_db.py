from argparse import ArgumentParser
import numpy as np
from sys import exit

import sys
sys.path.append('./l2r_models')
import svm_dtree, svm_learn2rank, utility_model

import scipy
import scipy.stats
import itertools
from math import sqrt
from scipy.optimize import minimize
import json
import pickle
import os
import psycopg2
import pandas.io.sql as sqlio

from models import ModelWeights

get_local_path = lambda s: os.path.join(os.path.dirname(os.path.realpath(__file__)), s)

def make_result_dirs():
    dirs = ['RESULT', 'RESULT/betas', 'RESULT/correct_comps', 'RESULT/incorrect_comps']
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)

def convert_categorical_features(feature):
    value = feature['feat_value']
    possible_values = feature['possible_values']

    if (("High" in possible_values) and ("Med" in possible_values) and ("Low" in possible_values)):
        if value == "High":
            mod_value = 1.0
        elif value == "Med":
            mod_value = 0.5
        elif value == "Low":
            mod_value = 0.0

        return mod_value

    elif (("Yes" in possible_values) and ("No" in possible_values)):
        if value == "Yes":
            mod_value = 1.0
        elif value == "No":
            mod_value = 0.0

        return mod_value

    else:
        arr = [0.0] * len(possible_values)
        arr[possible_values.index(value)] = 1.0

    return arr

def scale(feature, is_scale):
    '''
	:param feature: JSON object for the feature
	:param is_scale: Whether to scale OR not.
	:return: scaled OR unscaled value of the feature
	'''
    if (feature['feat_type'] == 'categorical'):
        mod_feature = convert_categorical_features(feature)
        return mod_feature
    else:
        value = float(feature['feat_value'])
        if (is_scale == False):
            mod_value = float(value)
        elif ((is_scale == True) and (feature['feat_type'] == 'continuous')):
            mod_value = float(value)
            mod_value = (value - float(feature['feat_min'])) / (float(feature['feat_max']) - float(feature['feat_min']))
        else:
            print("Not Implemented Yet- Scale")
            exit(0)

    return mod_value

def split_train_test(compars, test_frac, feat_trans):
    n = len(compars)
    np.random.shuffle(compars)

    n_test = int(test_frac * n)
    n_train = n - n_test

    train_comps = compars[:n_train, :, :]
    test_comps = compars[n_train:, :, :]

    k_train_comps = []
    #print("Train="+str(train_comps.shape))
    #print("Test="+str(test_comps.shape))

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

    k_test_comps = []
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

    return k_train_comps, k_test_comps, n_train, n_test

def get_scenarios_json(data, is_scale=True):
    pid = int(data['participant_id'])
    pairwise_type = data['request_type']

    all_samples = []
    imp_features = set()

    comparisons = data['comparisons']

    feature_array1 = []  # Values for feature array1
    feature_array2 = []  # Values for feature array2

    scenario_counter = 0
    for instance in comparisons:
        scenario_1 = instance['scenario_1']
        scenario_2 = instance['scenario_2']
        choice = instance['choice']

        scenario_1 = sorted(scenario_1, key=lambda elem: elem["feat_id"])
        scenario_2 = sorted(scenario_2, key=lambda elem: elem["feat_id"])

        feat_ids1 = []  # Feature ids - to check if we are comparing apples and apples
        feat_ids2 = []

        array_1 = []  # Actual Values
        array_2 = []

        for f1 in scenario_1:
            feat_ids1.append(f1['feat_id'])  # Makes sense.

            modified_features = scale(f1, is_scale)

            if (type(modified_features) is list):
                for mf in modified_features:
                    array_1.append(mf)
            else:
                array_1.append(modified_features)

            imp_features.add(f1['feat_id'])
        feature_array1.append(feat_ids1)

        for f2 in scenario_2:
            feat_ids2.append(f2['feat_id'])

            modified_features = scale(f2, is_scale)

            if(type(modified_features) is list):
                for mf in modified_features:
                    array_2.append(mf)
            else:
                array_2.append(modified_features)

            imp_features.add(f2['feat_id'])

        feature_array2.append(feat_ids2)

        all_samples.append([array_1, array_2, choice])

    print("Number of Scenarios=" + str(len(all_samples)))
    return all_samples, imp_features

def run_model(data, db):
    print("="*10)
    print(data)
    print("="*10)

    make_result_dirs()

    pid = int(data['participant_id'])
    pairwise_type = data['request_type']
    fid = int(data['feedback_round'])
    data_type = 'D'
    normalize = 1
    k = 1
    loss_fun = 'normal'
    num_iters = 100
    size_type = 'cardinalsizes'
    test_frac = 0.5

    data, imp_features = get_scenarios_json(data, is_scale=True)
    print("[TRAIN] IMP FEATURES="+str(len(imp_features)))
    um_accuracy, um_model = utility_model.run_utility_model(data, loss_fun, nsplits=10, test_size=0.15, train_size=0.85)

    '''
    lambda_reg = 1
    d = len(imp_features)

    print("Dimension d="+str(d))

    feat_trans = set([])
    feat_trans_dup = list(itertools.product(range(d + 1), repeat=k))
    for t in feat_trans_dup:
        feat_trans.add(tuple(sorted(t)))

    feat_trans = list(feat_trans)
    feat_trans.remove(tuple([d] * k))

    N = 1
    d_ext = len(feat_trans)
    learnt_beta = np.zeros((N, d_ext))

    compars = []
    for i in range(len(data)):
        altA = data[i][0]
        altB = data[i][1]
        choice = data[i][2]

        if (choice == 'A'):
            compars.append(np.array([altA, altB]))
        else:
            compars.append(np.array([altB, altA]))

    print("PRINTING SHAPE OF EACH ROW")
    for x in compars:
        print(x.shape)

    compars = np.array(compars)
    print("Length of comparisons=" + str(len(compars)))

    for iter in range(num_iters):
        total_num_correct = 0
        total_test_compars = 0
        total_soft_loss = 0

        # compars[j,0,:] is first alternative of comparison j, and compars[j,1,:] is the other.
        # Storing such that the 0 alternative is chosen over 1 alternative.

        k_train_comps, k_test_comps, n_train, n_test = split_train_test(compars, test_frac, feat_trans)

        # ETA = 0.00000001	# Learning rate
        # EPS = 0.00005	# Stopping criteria parameter
        THRES = 0.00001  # Threshold for cdf value. If smaller than this, then just use this

        # Learn parameters of voter i using his comparisons, by gradient descent
        this_diff = np.zeros((n_train, d_ext))

        for j in range(n_train):  # Pre-compute X_j - Z_j and store
            this_diff[j, :] = k_train_comps[j, 0, :] - k_train_comps[j, 1, :]

        this_beta = np.random.uniform(-0.001, 0.001,
                                      d_ext)  # Initialize parameter randomly	# Could instead use np.zeros(d), since it's a convex program


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

        # print("Learnt parameters for " + data_files[i] + ":", this_beta, '\n')
        learnt_beta[0, :] = this_beta

    TEST_SET = np.vstack((k_train_comps, k_test_comps))
    print(TEST_SET.shape)

    soft_loss, num_correct, n_test = test(this_beta, np.vstack((k_train_comps, k_test_comps)), n_test, pid, loss_fun,
                                          size_type, test_frac)
    print("FINAL LEARNT BETA")
    print(this_beta)

    print("Soft LOSS=" + str(soft_loss))
    print("Accuracy=" + str(float(num_correct) / n_test))
    '''

    weights_json = {'weights': list(um_model)}
    print("[TRAIN] Weights Model="+str(len(um_model)))

    indata = ModelWeights(participant_id=pid, feedback_round=fid, category=pairwise_type, weights=json.dumps(weights_json))
    print(indata)
    try:
        db.session.add(indata)
        db.session.commit()
    except Exception as e:
        print("\n FAILED entry: {}\n".format(data))
        print(e)

    return um_model

def test(this_beta, k_test_comps, n_test, pid, loss_fun, size_type, test_frac):
    num_correct = 0
    soft_loss = 0
    incorrect_comps = []
    correct_comps = []

    for j in range(n_test):
        mean_util_0 = np.dot(this_beta, k_test_comps[j, 0, :])
        mean_util_1 = np.dot(this_beta, k_test_comps[j, 1, :])

        prob_0_beats_1 = scipy.stats.norm.cdf(mean_util_0 - mean_util_1, scale=2)

        if (mean_util_0 > mean_util_1):
            num_correct += 1
            correct_comps.append([list(k_test_comps[j, 0, :]), list(k_test_comps[j, 1, :])])
        else:
            incorrect_comps.append([list(k_test_comps[j, 0, :]), list(k_test_comps[j, 1, :])])

        soft_loss += (1 - prob_0_beats_1) ** 2

    if (len(incorrect_comps) > 0):
        incorrect_comps_filename = get_local_path(
            "RESULT/incorrect_comps/Participant_" + str(pid) + "_" + str(loss_fun) + "_" + str(size_type) + "_" + str(
                int(test_frac) * 100) + "_errors.txt")
        with open(incorrect_comps_filename, 'w') as incorrect_outfile:
            for incorrect_comp in incorrect_comps:
                incorrect_outfile.write(str(incorrect_comp))
                incorrect_outfile.write('\n')

    # write correct comparisons out to file
    if len(correct_comps) > 0:
        correct_comps_filename = get_local_path(
            "RESULT/correct_comps/Participant_" + str(pid) + "_" + str(loss_fun) + "_" + str(size_type) + "_" + str(
                int(test_frac) * 100) + "_correct.txt")
        with open(correct_comps_filename, 'w') as correct_outfile:
            for correct_comp in correct_comps:
                correct_outfile.write(str(correct_comp))
                correct_outfile.write('\n')

    print("soft loss=" + str(soft_loss) + " accuracy=" + str(float(num_correct) / float(n_test)))
    return soft_loss, num_correct, n_test