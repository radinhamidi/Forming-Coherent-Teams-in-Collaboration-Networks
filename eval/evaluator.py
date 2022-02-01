from collections import Counter

import numpy as np
import copy
from dal.graph_util import *
import ml_metrics as metrics
from keras_metrics.metrics import true_negative


def r_at_k(prediction, true, k=10):
    all_recall = []
    for pred_indices, t_indices in zip(prediction, true):
        recall = 0
        for t_index in t_indices:
            if t_index in pred_indices[:k]:
                recall += 1
        all_recall.append(recall / len(t_indices))
    return np.mean(all_recall), all_recall


def sensitivity(prediction, true, k=10):
    all_recall = []
    for pred_indices, t_indices in zip(prediction, true):
        recall = 0
        t_indices_temp = copy.deepcopy(t_indices)
        for t_index in t_indices_temp:
            if t_index in pred_indices[:k]:
                t_indices_temp.remove(t_index)
        if len(t_indices_temp) == 0:
            recall = 1
        all_recall.append(recall)
    return np.mean(all_recall), all_recall


def p_at_k(prediction, true, k=10):
    all_precision = []
    for pred, t in zip(prediction, true):
        t = np.asarray(t)
        pred = np.asarray(pred)

        t_indices = np.argwhere(t)
        if t_indices.__len__() == 0:
            continue
        pred_indices = pred.argsort()[-k:][::-1]

        precision = 0
        for pred_index in pred_indices:
            if pred_index in t_indices:
                precision += 1
        all_precision.append(precision / pred_indices.__len__())
    return np.mean(all_precision), all_precision


def find_indices(prediction, true, min_true=1):
    preds = []
    trues = []
    for pred, t in zip(prediction, true):
        t = np.asarray(t)
        pred = np.asarray(pred)
        t_indices = np.argwhere(t)
        if t_indices.__len__() == 0:
            continue
        pred_indices = pred.argsort()[:][::-1]  # sorting checkup
        pred_indices = list(pred_indices)
        pred_indices = [i for i in pred_indices if i in np.argwhere(pred)]
        if len(pred_indices) == 0:
            pred_indices.append(-1)
        if len(t_indices) >= min_true:
            preds.append(pred_indices)
            trues.append([int(t) for t in t_indices])
    return preds, trues


def find_indices_t2v(prediction, true, min_true=1):
    preds = []
    trues = []
    for pred, t in zip(prediction, true):
        t_indices = np.nonzero(t[0])[1]
        pred_indices = list(np.asarray(pred))
        if t_indices.__len__() == 0:
            continue
        if len(pred_indices) == 0:
            pred_indices.append(-1)
        if len(t_indices) >= min_true:
            preds.append(pred_indices)
            trues.append([int(t) for t in t_indices])
    return preds, trues


def init_eval_holder(evaluation_k_set=None):
    if evaluation_k_set is None:
        evaluation_k_set = [10]

    dict = {}
    for k in evaluation_k_set:
        dict[k] = []
    return dict


def cal_relevance_score(prediction, truth, k=50):
    rs = []
    for p, t in zip(prediction, truth):
        r = []
        for p_record in p[:k]:
            if p_record in t:
                r.append(1)
            else:
                r.append(0)
        rs.append(r)
    return rs


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def coverage(pred, truth, k=10):
    pass


def help_hurt(pred_1, pred_2):
    len1 = len(pred_1)
    len2 = len(pred_2)
    min_len = min(len1, len2)
    if len1 != len2:
        print("Two given predictions have not same size.")

    diff = []
    for i, j in zip(pred_1[:min_len], pred_2[:min_len]):
        if (i - j) != 0:
            diff.append(i - j)
    diff = np.sort(diff)
    return diff


# mode can be 'feasibility', 'hindex', 'communication', 'skill_cover', or 'personal'
def team_formation_feasibility(predictions, truth, user_x_dict, k=10, mode='feasibility', hindex_mode='avg',
                               graph=None):
    if mode.lower() not in ['feasibility', 'hindex', 'communication', 'skill_cover', 'personal']:
        raise ValueError(
            'Wrong mode selected. It should be either "communication", "skill_cover", "personal", "feasibility" or "hindex".')
    score = []
    for p, t in zip(predictions, truth):
        if mode.lower() == 'communication':
            if len(t) > 1:
                score.append(team_communication_cost(graph, user_x_dict, p[:k], t))
        elif mode.lower() == 'skill_cover':
            score.append(team_skill_coverage(p[:k], t, user_x_dict))
        elif mode.lower() == 'personal':
            score.append(team_personal_cost(p[:k], t, user_x_dict))
        elif mode.lower() == 'feasibility':
            score.append(team_validtor(p, t, user_x_dict, k=k))
        elif mode.lower() == 'hindex':
            score.append(team_hindex(p, t, user_x_dict, hindex_mode, k=k))
        else:
            print('Mode is not defined. Please stop the evaluation and fix the mode typo.')
    return np.mean(score)


def team_validtor(p_users, t_users, user_skill_dict, k=10):
    having_skills = Counter()
    required_skills = Counter()

    for t_user in t_users:
        if t_user in user_skill_dict.keys():
            required_skills.update(user_skill_dict[t_user])

    for p_user in p_users[:k]:
        if p_user in user_skill_dict.keys():
            having_skills.update(user_skill_dict[p_user])

    for skill in required_skills.keys():
        if skill not in having_skills.keys():
            return 0
    return 1


def team_hindex(p_users, t_users, user_hindex_dict, hindex_mode, k=10):
    having_hindex = []
    required_hindex = []

    for t_user in t_users:
        if t_user in user_hindex_dict.keys():
            required_hindex.append(user_hindex_dict[t_user])
        else:
            print('target user not in hindex dictionary.')

    for p_user in p_users[:k]:
        if p_user in user_hindex_dict.keys():
            having_hindex.append(user_hindex_dict[p_user])
        else:
            print('predicted user not in hindex dictionary.')

    if len(having_hindex) == 0:
        having_hindex.append(0)
    if len(required_hindex) == 0:
        required_hindex.append(0)

    if hindex_mode.lower() == 'min':
        if np.min(having_hindex) < np.min(required_hindex): return 0
    elif hindex_mode.lower() == 'avg':
        if np.average(having_hindex) < np.average(required_hindex): return 0
    elif hindex_mode.lower() == 'max':
        if np.max(having_hindex) < np.max(required_hindex): return 0
    elif hindex_mode.lower() == 'diff':
        return np.abs(np.average(having_hindex) - np.average(required_hindex))
    else:
        print('H-Index mode is not in the list. Please stop the evaluation to fix the typo.')
    return 1


def team_communication_cost(graph, authorNameId_dict, p_team, t_team):
    t_cost = []
    p_cost = []

    for member in t_team:
        for neighbor in t_team:
            if neighbor != member:
                t_cost.append(graph.shortest_path_name(authorNameId_dict.get(member), authorNameId_dict.get(neighbor)))

    for member in p_team:
        for neighbor in p_team:
            if neighbor != member:
                p_cost.append(graph.shortest_path_name(authorNameId_dict.get(member), authorNameId_dict.get(neighbor)))

    if len(t_team) > 1:
        # t_cost = np.sum(t_cost) / (len(t_team) * (len(t_team) - 1))
        t_cost = np.sum(t_cost) / len(t_team)
    else:
        t_cost = 0
    if len(p_team) > 1:
        # p_cost = np.sum(p_cost) / (len(p_team) * (len(p_team) - 1))
        p_cost = np.sum(p_cost) / len(t_team)
    else:
        p_cost = 0

    return np.abs((t_cost - p_cost) / t_cost)


def team_skill_coverage(p_team, t_team, user_skill_dict):
    having_skills = Counter()
    required_skills = Counter()

    for member in t_team:
        if member in user_skill_dict.keys():
            required_skills.update(user_skill_dict[member])

    for member in p_team:
        if member in user_skill_dict.keys():
            having_skills.update(user_skill_dict[member])

    skill_counter = 0
    for skill in required_skills.keys():
        if skill in having_skills.keys():
            skill_counter += 1
    if len(t_team) == 0:
        return len(p_team)
    return skill_counter / len(required_skills)


def team_personal_cost(p_team, t_team, user_hindex_dict):
    having_hindex = []
    required_hindex = []

    for member in t_team:
        if member in user_hindex_dict.keys():
            required_hindex.append(user_hindex_dict[member])

    for member in p_team:
        if member in user_hindex_dict.keys():
            having_hindex.append(user_hindex_dict[member])

    if len(having_hindex) == 0:
        having_hindex.append(1)
    if len(required_hindex) == 0:
        required_hindex.append(1)

    return np.abs(1 - (np.mean(having_hindex) / np.mean(required_hindex)))


def load_output_file(file_path, foldIDsampleID_strata_dict):
    pred_indices = {}
    true_indices = {}
    calc_time_userStrata = {}
    calc_time_skillStrata = {}
    foldIDsampleID_strata_dict_temp = copy.deepcopy(foldIDsampleID_strata_dict)

    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            results = line.split(sep=',')
            method_name = results[0]
            kfold = int(results[1])
            fold_number = int(results[2])
            prediction_number = int(results[3])
            true_number = int(results[4])
            # skill_strata = foldIDsampleID_strata_dict_temp[fold_number].pop(0)
            elps_time = float(results[5])
            prediction_index = [int(i) for i in results[6:6 + prediction_number]]
            true_index = [int(i) for i in results[6 + prediction_number:6 + prediction_number + true_number]]

            if fold_number not in pred_indices.keys():
                pred_indices[fold_number] = []
            if fold_number not in true_indices.keys():
                true_indices[fold_number] = []
            if true_number not in calc_time_userStrata.keys():
                calc_time_userStrata[true_number] = []
            # if skill_strata not in calc_time_skillStrata.keys():
            #     calc_time_skillStrata[skill_strata] = []
            pred_indices[fold_number].append(prediction_index)
            true_indices[fold_number].append(true_index)
            calc_time_userStrata[true_number].append(elps_time)
            # calc_time_skillStrata[skill_strata].append(elps_time)

        f.close()
    return method_name, pred_indices, true_indices, \
           calc_time_userStrata, calc_time_skillStrata, kfold, prediction_number
