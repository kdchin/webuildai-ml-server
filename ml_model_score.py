from argparse import ArgumentParser
import numpy as np
from sys import exit
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
import datetime

get_local_path = lambda s: os.path.join(os.path.dirname(os.path.realpath(__file__)), s)


def connect():
    connection = None
    try:
        connection = psycopg2.connect(user="mwbecker", host="127.0.0.1", database="WeBuildAi_development")
        print("Connection Successful")
    except Exception:
        print("Connection ERROR")
        exit(0)

    return connection


def get_possible_feature_values(connection):
    cursor = connection.cursor()
    query = "SELECT * from categorical_data_options"
    df = sqlio.read_sql_query(query, connection)

    possible_values = {}

    data_range_id_values = df['data_range_id'].values
    option_values = df['option_value'].values

    for i in range(len(option_values)):
        try:
            possible_values[int(data_range_id_values[i])].append(option_values[i])
        except KeyError as e:
            possible_values[int(data_range_id_values[i])] = [option_values[i]]

    return possible_values


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


def get_feature_info(connection):
    query = '''
    select a.id,name,description, category, b.is_categorical, b.lower_bound, b.upper_bound FROM features a, data_ranges b WHERE a.id=b.id;
    '''

    df = sqlio.read_sql_query(query, connection)

    feature_ids = df['id'].values
    feature_names = df['name'].values
    feature_desc = df['description'].values
    feature_categ = df['category'].values
    feature_type = df['is_categorical'].values
    feature_lb = df['lower_bound'].values
    feature_ub = df['upper_bound'].values

    feature_info = {}
    for i in range(len(feature_ids)):
        feature_info[feature_ids[i]] = [feature_names[i], feature_desc[i], feature_categ[i], feature_type[i],
                                        feature_lb[i], feature_ub[i]]

    return feature_info


def get_candidates(connection, pid, category, feature_info, possible_values, is_scale):
    cursor = connection.cursor()
    query = """
    SELECT id, participant_id, features FROM individual_scenarios WHERE participant_id=%s AND category='%s';         
    """ % (str(pid), str(category))

    all_samples = []
    candidate_ids = []
    df = sqlio.read_sql_query(query, connection)

    imp_features = set()

    scenarios = df['features'].values
    scenario_ids = df['id'].values

    for i in range(len(scenario_ids)):
        candidate_ids.append(scenario_ids[i])
        scenario = scenarios[i]
        feature_arr = []
        for f_key in scenario.keys():
            feature_obj = {}
            feature_id = int(f_key)
            imp_features.add(feature_id)
            feature_value = scenario[f_key]
            feature_name = feature_info[feature_id][0]
            feature_category = feature_info[feature_id][2]
            feature_type = feature_info[feature_id][3]
            if (feature_type == True):
                feature_type = "categorical"
            elif (feature_type == False):
                feature_type = "continuous"

            feature_min = feature_info[feature_id][4]
            feature_max = feature_info[feature_id][5]

            feature_obj['feat_id'] = feature_id
            feature_obj['feat_name'] = feature_name
            feature_obj['feat_category'] = feature_category
            feature_obj['feat_value'] = feature_value
            feature_obj['feat_type'] = feature_type
            if (feature_type == "categorical"):
                f_poss_values = possible_values[feature_id]
            else:
                f_poss_values = []
            feature_obj['possible_values'] = f_poss_values
            feature_obj['feat_min'] = feature_min
            feature_obj['feat_max'] = feature_max

            feature_arr.append(feature_obj)

        feature_arr = sorted(feature_arr, key=lambda elem: elem["feat_id"])
        num_features = len(feature_arr)

        array_f = []  # Actual Values
        for f1 in feature_arr:
            modified_features = scale(f1, is_scale)

            if (type(modified_features) is list):
                for mf in modified_features:
                    array_f.append(mf)
            else:
                array_f.append(modified_features)

        all_samples.append(array_f)

    return all_samples, imp_features, candidate_ids


def score_model(args):
    connection = connect()
    pid = args.pid
    fid = args.fid
    pairwise_type = args.type
    data_type = args.d
    normalize = args.n
    k = args.k
    loss_fun = args.l
    num_iters = args.num
    size_type = args.sz
    test_frac = args.tf

    # Load Model
    model = pickle.load(
        open('./RESULT/betas/Participant_' + str(pid) + '_' + str(pairwise_type) + '_BETA_Round' + str(fid) + '.pkl',
             'rb'))
    # Score using this model
    possible_values = get_possible_feature_values(connection)
    feature_info = get_feature_info(connection)
    candidates, imp_features, candidate_ids = get_candidates(connection, pid, pairwise_type, feature_info,
                                                             possible_values, True)
    # candidates, imp_features = get_candidates(pid, fid, connection)

    lambda_reg = 1
    d = len(imp_features)

    feat_trans = set([])
    feat_trans_dup = list(itertools.product(range(d + 1), repeat=k))
    for t in feat_trans_dup:
        feat_trans.add(tuple(sorted(t)))

    feat_trans = list(feat_trans)
    feat_trans.remove(tuple([d] * k))

    N = 1
    d_ext = len(feat_trans)
    learnt_beta = np.zeros((N, d_ext))

    score_arr = []
    for i in range(len(candidates)):
        altA = candidates[i]
        altA = list(altA)
        altA.append(1)
        k_altA = []
        for t in feat_trans:
            this_prod = 1
            for index in t:
                this_prod *= altA[index]
            k_altA.append(this_prod)

        k_altA = np.array(k_altA)
        score = np.dot(model, k_altA)
        score_arr.append(score)

    return score_arr, candidate_ids

def insert_score_arr(connection, candidate_ids, score_arr, participant_id, round):
    cursor = connection.cursor()

    # Get ranklist id i.e.
    # Select id from ranklists where participant_id = 18 and round=1;
    # Hopefully, get the most recent ranklist
    # Insert into that ranklist id based ranklist_element

    query = """
    SELECT id from ranklists WHERE participant_id=%s AND round=%s
    ORDER BY created_at DESC
    """ % (str(participant_id), str(round))

    df = sqlio.read_sql_query(query, connection)
    ranklist_id = df['id'].values[0]

    ranks = np.argsort(-np.array(score_arr))
    insert_query = """
        INSERT INTO ranklist_element(ranklist_id, individual_scenario_id, model_rank, human_rank, created_at, updated_at)
        VALUES(%s, %s, %s, %s, %s, %s)
    """
    sql_args = []
    for i in range(len(candidate_ids)):
        sql_args.append(
            [ranklist_id, candidate_ids[i], ranks[i] + 1.0, 0, datetime.datetime.now(), datetime.datetime.now()])

    print(sql_args)
    cursor.executemany(insert_query, sql_args)
    cursor.close()
    connection.commit()

def get_scenarios_json(data, is_scale):
    all_samples = []
    imp_features = set()

    individual_scenarios = data['scenarios']
    candidate_ids = []

    for i in range(len(individual_scenarios)):
        candidate_ids.append(int(individual_scenarios[i]['id']))
        feature_arr = individual_scenarios[i]['features']
        feature_arr = sorted(feature_arr, key=lambda elem: elem["feat_id"])
        num_features = len(feature_arr)

        array_f = []  # Actual Values
        for f1 in feature_arr:
            imp_features.add(f1['feat_id'])
            modified_features = scale(f1, is_scale)

            if (type(modified_features) is list):
                for mf in modified_features:
                    array_f.append(mf)
            else:
                array_f.append(modified_features)

        all_samples.append(array_f)

    return all_samples, imp_features, candidate_ids

def score_instances(data):
    pid = int(data['participant_id'])
    pairwise_type = data['request_type']
    fid = int(data['feedback_round'])
    data_type = 'D'
    normalize = 1
    k = 1
    loss_fun = 'normal'
    num_iters = 100
    size_type = 'cardinalsizes'
    test_frac = 0.15

    filename = str(pid) + "_" + str(pairwise_type)

    model = pickle.load(open(get_local_path(
        'RESULT/betas/Participant_' + str(filename) + '_BETA_Round' + str(fid) + '.pkl'),'rb'))

    candidates, imp_features, candidate_ids = get_scenarios_json(data, is_scale=True)

    lambda_reg = 1
    d = len(imp_features)

    feat_trans = set([])
    feat_trans_dup = list(itertools.product(range(d + 1), repeat=k))
    for t in feat_trans_dup:
        feat_trans.add(tuple(sorted(t)))

    feat_trans = list(feat_trans)
    feat_trans.remove(tuple([d] * k))

    N = 1
    d_ext = len(feat_trans)
    learnt_beta = np.zeros((N, d_ext))

    score_arr = []
    for i in range(len(candidates)):
        altA = candidates[i]
        altA = list(altA)
        altA.append(1)
        k_altA = []
        for t in feat_trans:
            this_prod = 1
            for index in t:
                this_prod *= altA[index]
            k_altA.append(this_prod)

        k_altA = np.array(k_altA)
        score = np.dot(model, k_altA)
        score_arr.append(score)

    return score_arr, candidate_ids


if __name__ == '__main__':
    parser = ArgumentParser(description="Inputs.")
    parser.add_argument('-pid', type=str, default='None', help='participant_id')
    parser.add_argument('-fid', type=str, default=0, help='feedback round number')
    parser.add_argument('-type', type=str, default='request', help='Running model for social/individual preferences')
    parser.add_argument('-d', type=str, default='D', help='data type (str)')
    parser.add_argument('-n', type=int, default=1, help='normalize features (int)')
    parser.add_argument('-k', type=int, default=1, help='degree of polynomial (int)')
    parser.add_argument('-l', type=str, default='normal', help='loss function(str')
    parser.add_argument('-num', type=int, default=100, help='value of num iters>0 (int)')
    parser.add_argument('-sz', type=str, default='cardinalsizes', help='size type (str)')
    parser.add_argument('-tf', type=float, default=0.5, help='test frac (float)')
    args = parser.parse_args()

    score_arr, candidate_ids = score_model(args)
    connection = connect()
    insert_score_arr(connection, candidate_ids, score_arr, args.pid, args.fid)