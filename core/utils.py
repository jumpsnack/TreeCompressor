import numpy as np


def travel_recur(ruleset, idx):
    if idx not in ruleset.keys():
        return 0
    node = ruleset[idx]  # idx: l_c, r_c, feat, th, classes
    (l_c, r_c,_, imp, feat, th, classes) = node

    if feat == -2:
        nrof_params = len(classes) +1
    else:
        nrof_params = 5

    if l_c != -1:
        n_p = travel_recur(ruleset, l_c)
        nrof_params += n_p

    if r_c != -1:
        n_p = travel_recur(ruleset, r_c)
        nrof_params += n_p

    return nrof_params


def extract_rules_dt_cls(dt):
    len_rule_set, len_new_rule_set = 0, 0
    nrof_op, nrof_params = 0, 0
    feat_contribs = []

    n_nodes = dt.tree_.node_count
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    value = dt.tree_.value.squeeze(axis=1)
    contribs_by_feat = [[0.] * dt.n_classes_] * dt.n_features_

    normalizer = dt.tree_.value.squeeze(axis=1).sum(axis=1)[:, np.newaxis]
    normalizer[normalizer == 0.] = 1.
    value /= normalizer

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    rule_set = []
    path = []

    for i in range(n_nodes):
        del path[node_depth[i]:]
        path.append((i, children_left[i], children_right[i], feature[i], threshold[i], value[i]))
        if is_leaves[i]:
            rule_set.append(path.copy())

    max_contrib = -float('inf')
    sum_contrib = 0
    conts = []
    for rule in rule_set:
        contrib = 0
        for i in range(len(rule) - 1):
            contrib += (rule[i + 1][-1] - rule[i][-1])
            contribs_by_feat[rule[i][-3]] += (rule[i + 1][-1] - rule[i][-1])
        bias = rule[0][-1]
        pred = rule[-1][-1]
        contrib = (np.sum(contrib))
        rule.append((pred, bias, contrib))
        if max_contrib < contrib: max_contrib = contrib
        sum_contrib += abs(contrib)
        conts.append(contrib)

    new_rule_set = sorted(rule_set, key=lambda rule: rule[-1][-1])[int(len(rule_set) * 0):]

    len_rule_set += len(rule_set)
    len_new_rule_set += len(new_rule_set)
    rule_tree = {}

    for rule in new_rule_set:
        for node_idx in range(len(rule) - 1):
            idx = rule[node_idx][0]
            if idx not in rule_tree:
                rule_tree[idx] = rule[node_idx][1:]

    tmp_op = 0
    for _r in new_rule_set:
        tmp_op += len(_r[:-1]) - 1
    nrof_op += tmp_op / len(new_rule_set)
    nrof_params += travel_recur(rule_tree, 0)

    feat_contribs.append(contribs_by_feat)

    return rule_tree, len_rule_set, len_new_rule_set, nrof_op, nrof_params, feat_contribs


def extract_rules_dt_cls_v2(dt):
    len_rule_set, len_new_rule_set = 0, 0
    nrof_op, nrof_params = 0, 0
    feat_contribs = []

    n_nodes = dt.tree_.node_count
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    value = dt.tree_.value.squeeze(axis=1)
    contribs_by_feat = [[0.] * dt.n_classes_] * dt.n_features_

    normalizer = dt.tree_.value.squeeze(axis=1).sum(axis=1)[:, np.newaxis]
    normalizer[normalizer == 0.] = 1.
    value /= normalizer

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    rule_set = []
    path = []

    for i in range(n_nodes):
        del path[node_depth[i]:]
        path.append((i, children_left[i], children_right[i], feature[i], threshold[i], value[i]))
        if is_leaves[i]:
            rule_set.append(path.copy())

    max_contrib = -float('inf')
    sum_contrib = 0
    conts = []
    for rule in rule_set:
        contrib = 0
        for i in range(len(rule) - 1):
            contrib += (rule[i + 1][-1] - rule[i][-1])
            contribs_by_feat[rule[i][-3]] += (rule[i + 1][-1] - rule[i][-1])
        bias = rule[0][-1]
        pred = rule[-1][-1]
        contrib = (np.sum(contrib))
        rule.append((pred, bias, contrib))
        if max_contrib < contrib: max_contrib = contrib
        sum_contrib += abs(contrib)
        conts.append(contrib)

    new_rule_set = sorted(rule_set, key=lambda rule: rule[-1][-1])[int(len(rule_set) * 0):]

    len_rule_set += len(rule_set)
    len_new_rule_set += len(new_rule_set)
    # rule_tree = {}
    #
    # for rule in new_rule_set:
    #     for node_idx in range(len(rule) - 1):
    #         idx = rule[node_idx][0]
    #         if idx not in rule_tree:
    #             rule_tree[idx] = rule[node_idx][1:]

    # tmp_op = 0
    # for _r in new_rule_set:
    #     tmp_op += len(_r[:-1]) - 1
    # nrof_op += tmp_op / len(new_rule_set)
    # nrof_params += travel_recur(rule_tree, 0)

    feat_contribs.append(contribs_by_feat)

    return new_rule_set, len_rule_set, len_new_rule_set, feat_contribs


def extract_rules_rf_cls(rf):
    rule_forests = []
    len_rule_set, len_new_rule_set = 0, 0
    nrof_op, nrof_params = 0, 0
    feat_contribs = []
    for estimator in rf.estimators_:
        rule_tree, len_r_s, len_n_r_s, n_op, n_params, f_contribs = extract_rules_dt_cls(estimator)
        rule_forests.append(rule_tree)
        len_rule_set += len_r_s
        len_new_rule_set += len_n_r_s
        nrof_op += n_op
        nrof_params += n_params
        feat_contribs.append(f_contribs)

    return rule_forests, len_rule_set, len_new_rule_set, nrof_op, nrof_params, feat_contribs


def extract_rules_rf_cls_v2(rf):
    dt_rules = []
    len_rule_set, len_new_rule_set = 0, 0
    feat_contribs = []

    for estimator in rf.estimators_:
        dt_rule, len_r_s, len_n_r_s, f_contribs = extract_rules_dt_cls_v2(estimator)
        dt_rules.append(dt_rule)
        len_rule_set += len_r_s
        len_new_rule_set += len_n_r_s
        feat_contribs.append(f_contribs)

    return dt_rules, len_rule_set, len_new_rule_set, feat_contribs


def pred_prob_single_rules_cls(rule, X):
    result = {}

    for i in range(len(X)):
        _x = X[i]
        pred = []

        idx = 0
        while True:
            node = rule[idx]
            (l_idx, r_idx, _, imp, feat_idx, th, value) = node
            if feat_idx == -2:
                pred.append(value)
                break
            if _x[feat_idx] <= th:
                idx = l_idx
                if idx not in rule:
                    break
            else:
                idx = r_idx
                if idx not in rule:
                    break
        if len(pred) == 0:
            result[i] = np.zeros_like(value)
        else:
            result[i] = np.mean(pred, axis=0)

    return np.array(list(result.values()))


def is_reachable_rule(rule, X):
    result = {}

    for i in range(len(X)):
        _x = X[i]
        pred = []

        idx = 0
        while True:
            node = rule[idx]
            (l_idx, r_idx, feat_idx, th, value) = node
            if feat_idx == -2:
                pred.append(value)
                break
            if _x[feat_idx] <= th:
                idx = l_idx
                if idx not in rule:
                    break
            else:
                idx = r_idx
                if idx not in rule:
                    break
        if len(pred) == 0:
            result[i] = np.zeros_like(value)
        else:
            result[i] = np.mean(pred, axis=0)

    return np.array(list(result.values()))


def pred_prob_single_rules_cls_threaded(rule, X):
    def _job(i, X, result):
        _x = X[i]
        pred = []

        idx = 0
        while True:
            node = rule[idx]
            (l_idx, r_idx, feat_idx, th, value) = node
            if feat_idx == -2:
                pred.append(value)
                break
            if _x[feat_idx] <= th:
                idx = l_idx
                if idx not in rule:
                    break
            else:
                idx = r_idx
                if idx not in rule:
                    break
        if len(pred) == 0:
            result[i] = np.zeros_like(value)
        else:
            result[i] = np.mean(pred, axis=0)

    result = {}
    Parallel(n_jobs=-1, verbose=0)(
        delayed(_job)(i, X, result) for i in range(len(X)))

    # for i in range(len(X)):
    #     _x = X[i]
    #     pred = []
    #
    #     idx = 0
    #     while True:
    #         node = rule[idx]
    #         (l_idx, r_idx, feat_idx, th, value) = node
    #         if feat_idx == -2:
    #             pred.append(value)
    #             break
    #         if _x[feat_idx] <= th:
    #             idx = l_idx
    #             if idx not in rule:
    #                 break
    #         else:
    #             idx = r_idx
    #             if idx not in rule:
    #                 break
    #     if len(pred) == 0:
    #         result[i] = np.zeros_like(value)
    #     else:
    #         result[i] = np.mean(pred, axis=0)

    return np.array(list(result.values()))


def pred_prob_multi_rule_cls_threaded(rules, X):
    probs = []
    for rule in rules:
        prob = pred_prob_single_rules_cls_threaded(rule, X)
        probs.append(prob)

    return np.mean(probs, axis=0)


def pred_prob_multi_rule_cls(rules, X):
    probs = []
    for rule in rules:
        prob = pred_prob_single_rules_cls(rule, X)
        probs.append(prob)

    return np.mean(probs, axis=0)


import tqdm


def determine_shapley_value(rules, X, rate):
    target_probs = pred_prob_multi_rule_cls(rules, X)
    shapley_values = []

    len_rules = len(rules)
    for r_idx in tqdm.tqdm(range(len_rules)):
        target_player = rules[r_idx]
        remaining_player = rules[:r_idx] + rules[r_idx + 1:]

        remaining_probs = pred_prob_multi_rule_cls(remaining_player, X)

        shapely_value = np.sum((target_probs - remaining_probs) ** 2)
        shapley_values.append((r_idx, shapely_value))

    shapley_values = sorted(shapley_values, key=lambda v: v[1])
    len_shapley = len(shapley_values)
    new_shapley_values = shapley_values[:int(len_shapley * rate)]
    new_rules = [rules[i] for i, v in new_shapley_values]

    return new_rules


from joblib import Parallel, delayed
import random


def chunk_list_with_best_effort(input_list, max_chunk_size, shuffle=True):
    if shuffle:
        random.shuffle(input_list)

    chunk_tokens = []
    while len(input_list) > 0:
        if len(input_list) > max_chunk_size:
            chunk = input_list[:max_chunk_size]
            input_list = input_list[max_chunk_size:]
        else:
            chunk = input_list[:len(input_list)]
            input_list = input_list[len(input_list) + 1:]
        chunk_tokens.append(chunk)

    return chunk_tokens


def generate_random_chunk_list(input_list, max_chunk_size, iter=4):
    chunk_tokens = []

    for _ in range(iter):
        target_list = np.copy(input_list)

        random.shuffle(target_list)

        while len(target_list) > 0:
            if len(target_list) > max_chunk_size:
                chunk = target_list[:max_chunk_size]
                target_list = target_list[max_chunk_size:]
            else:
                chunk = target_list[:len(target_list)]
                target_list = target_list[len(target_list) + 1:]
            chunk_tokens.append(chunk)

    return chunk_tokens


def reduce_redundancy(shapley_values):
    shapley_set = {}
    shapley_acc_set = {}

    for v in shapley_values:
        r_idx, value, acc = v

        if r_idx in shapley_set.keys():
            prev_v = shapley_set[r_idx]
            # shapley_set[r_idx] = (prev_v + value)/2
            shapley_set[r_idx].append(value)
            shapley_acc_set[r_idx].append(acc)
        else:
            shapley_set[r_idx] = [value]
            shapley_acc_set[r_idx] = [acc]

    return [(r_idx, np.sum(shapley_set[r_idx]) / np.exp(np.var(shapley_set[r_idx])), np.sum(shapley_acc_set[r_idx])) for r_idx in shapley_set.keys()]


def determine_shapley_value_threaded(rules, X, y, target_probs, group, sharpely_by_contrib=False, multi_sampling=True):
    def _job(footages, r_idx, rules, target_probs, y):
        target_player = rules[r_idx]
        remaining_player = rules[:r_idx] + rules[r_idx + 1:]

        remaining_player_rule_dt, _, _ = cvt_dt_rule_to_dt_form(remaining_player)

        remaining_probs = pred_prob_single_rules_cls(remaining_player_rule_dt, X)

        # shapely_value = np.sum((target_probs - remaining_probs) ** 2)

        pred_acc = np.mean(np.argmax(remaining_probs, axis=-1) == y)
        target_acc = np.mean(np.argmax(target_probs, axis=-1) == y)
        shapely_value = pred_acc - target_acc

        footages.append((r_idx, shapely_value))

    def _job_2(chunk, shapley_values, dt_rule, target_probs, y):
        target_players = []
        remaining_players = []
        for r_idx in range(len(dt_rule)):
            if r_idx in chunk:
                target_players.append(dt_rule[r_idx])
            else:
                remaining_players.append(dt_rule[r_idx])

        remaining_player_rule_dt, _, _ = cvt_dt_rule_to_dt_form(remaining_players)
        remaining_probs = pred_prob_single_rules_cls(remaining_player_rule_dt, X)

        pred_acc = np.mean(np.argmax(remaining_probs, axis=-1) == y)
        target_acc = np.mean(np.argmax(target_probs, axis=-1) == y)
        shapely_value = pred_acc - target_acc

        for r_idx in chunk:
            shapley_values.append((r_idx, shapely_value))

    def _job_3(dt_rule, dt_shapley_values):
        shapley_values = []

        len_rules = len(dt_rule)

        assert len_rules > 0
        assert len_rules > group

        if multi_sampling:
            chunked_candidates = generate_random_chunk_list(list(range(len_rules)), group, iter=4)
        else:
            chunked_candidates = chunk_list_with_best_effort(list(range(len_rules)), group)

        if sharpely_by_contrib:
            dt_contribs = [rule[-1][-1] for rule in dt_rule]
            min = np.min(dt_contribs)
            max = np.max(dt_contribs)

        for chunk in tqdm.tqdm(chunked_candidates):
            target_players = []
            remaining_players = []
            for r_idx in range(len_rules):
                if r_idx in chunk:
                    target_players.append(dt_rule[r_idx])
                else:
                    remaining_players.append(dt_rule[r_idx])

            remaining_player_rule_dt, _, _ = cvt_dt_rule_to_dt_form(remaining_players)
            remaining_probs = pred_prob_single_rules_cls(remaining_player_rule_dt, X)

            pred_acc = np.mean(np.argmax(remaining_probs, axis=-1) == y)
            target_acc = np.mean(np.argmax(target_probs, axis=-1) == y)
            shapely_value = pred_acc - target_acc

            for r_idx in chunk:
                if sharpely_by_contrib:
                    contrib_value = dt_rule[r_idx][-1][-1]
                    norm_contrib = contrib_value - min
                    norm_contrib = norm_contrib / (max - min)
                    shapely_value = shapely_value * norm_contrib
                shapley_values.append((r_idx, shapely_value,shapely_value))

        if multi_sampling: shapley_values = reduce_redundancy(shapley_values)
        shapley_values = sorted(shapley_values, key=lambda v: v[1])
        return shapley_values
        # dt_shapley_values.append(shapley_values)

    # = pred_prob_multi_rule_cls(rules, X)
    dt_shapley_values = []

    dt_shapley_values = Parallel(n_jobs=-1, verbose=0)(
        delayed(_job_3)(dt_rule, dt_shapley_values) for dt_rule in rules)

    # for dt_rule in rules:
    #     shapley_values = []
    #
    #     len_rules = len(dt_rule)
    #
    #     chunked_candidates = chunk_list_with_best_effort(list(range(len_rules)), 10)
    #
    #     for chunk in tqdm.tqdm(chunked_candidates):
    #         target_players = []
    #         remaining_players = []
    #         for r_idx in range(len_rules):
    #             if r_idx in chunk:
    #                 target_players.append(dt_rule[r_idx])
    #             else:
    #                 remaining_players.append(dt_rule[r_idx])
    #
    #         remaining_player_rule_dt, _, _ = cvt_dt_rule_to_dt_form(remaining_players)
    #         remaining_probs = pred_prob_single_rules_cls(remaining_player_rule_dt, X)
    #
    #         pred_acc = np.mean(np.argmax(remaining_probs, axis=-1) == y)
    #         target_acc = np.mean(np.argmax(target_probs, axis=-1) == y)
    #         shapely_value = pred_acc - target_acc
    #
    #         for r_idx in chunk:
    #             shapley_values.append((r_idx, shapely_value))
    #     # Parallel(n_jobs=-1, verbose=0)(
    #     #         delayed(_job_2)(chunk, shapley_values,dt_rule ,target_probs, y) for chunk in tqdm.tqdm(chunked_candidates))
    #
    #     # for r_idx in tqdm.tqdm(range(len_rules)):
    #     #     target_player = dt_rule[r_idx]
    #     #     remaining_player = dt_rule[:r_idx] + dt_rule[r_idx + 1:]
    #     #
    #     #     remaining_player_rule_dt, _, _ = cvt_dt_rule_to_dt_form(remaining_player)
    #     #
    #     #     # remaining_probs = pred_prob_single_rules_cls_threaded(remaining_player_rule_dt, X)
    #     #     remaining_probs = pred_prob_single_rules_cls(remaining_player_rule_dt, X)
    #     #
    #     #     pred_acc = np.mean(np.argmax(remaining_probs, axis=-1) == y)
    #     #     target_acc = np.mean(np.argmax(target_probs, axis=-1) == y)
    #     #     shapely_value = pred_acc - target_acc
    #     #
    #     #     shapley_values.append((r_idx, shapely_value))
    #
    #     # Parallel(n_jobs=400, backend='threading', verbose=0)(
    #     #     delayed(_job)(shapley_values, r_idx, dt_rule, target_probs, y) for r_idx in range(len_rules))
    #
    #     shapley_values = sorted(shapley_values, key=lambda v: v[1])
    #     dt_shapley_values.append(shapley_values)

    return dt_shapley_values


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    # p = softmax(X) + 3.4
    p = X + 3.4
    # We use multidimensional array indexing to extract
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m),np.argmax(y, -1)])
    loss = np.sum(log_likelihood)
    loss /= float(m)
    return loss


def determine_entropy_shapley_value_threaded_v2(rules, X, y, target_probs, group, sharpely_by_contrib=False, multi_sampling=True, only_contrib=False, iter=None):

    def _job_3(dt_rule):
        shapley_values = []

        len_rules = len(dt_rule)

        assert len_rules > 0
        assert len_rules > group

        if multi_sampling:
            assert iter != None, 'iter is None'
            chunked_candidates = generate_random_chunk_list(list(range(len_rules)), group, iter=iter)
        else:
            chunked_candidates = chunk_list_with_best_effort(list(range(len_rules)), group)

        dt_contribs = [rule[-1][-1] for rule in dt_rule]
        min = np.min(dt_contribs)
        max = np.max(dt_contribs)

        for chunk in tqdm.tqdm(chunked_candidates):
            target_players = []
            remaining_players = []
            for r_idx in range(len_rules):
                if r_idx in chunk:
                    target_players.append(dt_rule[r_idx])
                else:
                    remaining_players.append(dt_rule[r_idx])

            remaining_player_rule_dt, _, _ = cvt_dt_rule_to_dt_form(remaining_players)
            remaining_probs = pred_prob_single_rules_cls(remaining_player_rule_dt, X)

            shapely_value = -cross_entropy(remaining_probs, y)

            pred_acc = np.mean(np.argmax(remaining_probs, axis=-1) == np.argmax(y, axis=-1))
            target_acc = np.mean(np.argmax(target_probs, axis=-1) == np.argmax(y, axis=-1))
            acc_term = pred_acc - target_acc

            for r_idx in chunk:
                contrib_value = dt_rule[r_idx][-1][-1]
                norm_contrib = contrib_value - min
                norm_contrib = norm_contrib / (max - min)

                if only_contrib:
                    shapley_values.append((r_idx, norm_contrib,acc_term))
                else:
                    if sharpely_by_contrib:
                        shapely_value = shapely_value * np.sum(norm_contrib)
                    shapley_values.append((r_idx, shapely_value, acc_term))

        if multi_sampling: shapley_values = reduce_redundancy(shapley_values)
        shapley_values = sorted(shapley_values, key=lambda v: v[1])
        return shapley_values

    # = pred_prob_multi_rule_cls(rules, X)

    # dt_shapley_values = []
    # for dt_rule in rules:
    #     shapley_values = []
    #
    #     len_rules = len(dt_rule)
    #
    #     assert len_rules > 0
    #     assert len_rules > group
    #
    #     if multi_sampling:
    #         chunked_candidates = generate_random_chunk_list(list(range(len_rules)), group, iter=4)
    #     else:
    #         chunked_candidates = chunk_list_with_best_effort(list(range(len_rules)), group)
    #
    #     if sharpely_by_contrib:
    #         dt_contribs = [rule[-1][-1] for rule in dt_rule]
    #         min = np.min(dt_contribs)
    #         max = np.max(dt_contribs)
    #
    #     for chunk in tqdm.tqdm(chunked_candidates):
    #         target_players = []
    #         remaining_players = []
    #         for r_idx in range(len_rules):
    #             if r_idx in chunk:
    #                 target_players.append(dt_rule[r_idx])
    #             else:
    #                 remaining_players.append(dt_rule[r_idx])
    #
    #         remaining_player_rule_dt, _, _ = cvt_dt_rule_to_dt_form(remaining_players)
    #         remaining_probs = pred_prob_single_rules_cls(remaining_player_rule_dt, X)
    #
    #         shapely_value = -cross_entropy(target_probs, remaining_probs)
    #
    #         # pred_acc = np.mean(np.argmax(remaining_probs, axis=-1) == target_probs)
    #         # target_acc = np.mean(np.argmax(target_probs, axis=-1) == target_probs)
    #         # shapely_value = pred_acc - target_acc
    #
    #         for r_idx in chunk:
    #             if sharpely_by_contrib:
    #                 contrib_value = dt_rule[r_idx][-1][-1]
    #                 norm_contrib = contrib_value - min
    #                 norm_contrib = norm_contrib / (max - min)
    #                 shapely_value = shapely_value * norm_contrib
    #             shapley_values.append((r_idx, shapely_value))
    #
    #     if multi_sampling: shapley_values = reduce_redundancy(shapley_values)
    #     shapley_values = sorted(shapley_values, key=lambda v: v[1])
    #     dt_shapley_values.append(shapley_values)

    dt_shapley_values = Parallel(n_jobs=-1, verbose=0)(
        delayed(_job_3)(dt_rule) for dt_rule in rules)

    # dt_shapley_values = [_job_3(dt_rule) for dt_rule in rules]

    # for dt_rule in rules:
    #     shapley_values = []
    #
    #     len_rules = len(dt_rule)
    #
    #     chunked_candidates = chunk_list_with_best_effort(list(range(len_rules)), 10)
    #
    #     for chunk in tqdm.tqdm(chunked_candidates):
    #         target_players = []
    #         remaining_players = []
    #         for r_idx in range(len_rules):
    #             if r_idx in chunk:
    #                 target_players.append(dt_rule[r_idx])
    #             else:
    #                 remaining_players.append(dt_rule[r_idx])
    #
    #         remaining_player_rule_dt, _, _ = cvt_dt_rule_to_dt_form(remaining_players)
    #         remaining_probs = pred_prob_single_rules_cls(remaining_player_rule_dt, X)
    #
    #         pred_acc = np.mean(np.argmax(remaining_probs, axis=-1) == y)
    #         target_acc = np.mean(np.argmax(target_probs, axis=-1) == y)
    #         shapely_value = pred_acc - target_acc
    #
    #         for r_idx in chunk:
    #             shapley_values.append((r_idx, shapely_value))
    #     # Parallel(n_jobs=-1, verbose=0)(
    #     #         delayed(_job_2)(chunk, shapley_values,dt_rule ,target_probs, y) for chunk in tqdm.tqdm(chunked_candidates))
    #
    #     # for r_idx in tqdm.tqdm(range(len_rules)):
    #     #     target_player = dt_rule[r_idx]
    #     #     remaining_player = dt_rule[:r_idx] + dt_rule[r_idx + 1:]
    #     #
    #     #     remaining_player_rule_dt, _, _ = cvt_dt_rule_to_dt_form(remaining_player)
    #     #
    #     #     # remaining_probs = pred_prob_single_rules_cls_threaded(remaining_player_rule_dt, X)
    #     #     remaining_probs = pred_prob_single_rules_cls(remaining_player_rule_dt, X)
    #     #
    #     #     pred_acc = np.mean(np.argmax(remaining_probs, axis=-1) == y)
    #     #     target_acc = np.mean(np.argmax(target_probs, axis=-1) == y)
    #     #     shapely_value = pred_acc - target_acc
    #     #
    #     #     shapley_values.append((r_idx, shapely_value))
    #
    #     # Parallel(n_jobs=400, backend='threading', verbose=0)(
    #     #     delayed(_job)(shapley_values, r_idx, dt_rule, target_probs, y) for r_idx in range(len_rules))
    #
    #     shapley_values = sorted(shapley_values, key=lambda v: v[1])
    #     dt_shapley_values.append(shapley_values)

    return dt_shapley_values



def get_node_value_by_rule(rule, X, target_probs):
    result = []
    for i in range(len(X)):
        _x = X[i]
        pred = []
        idx = 0
        for node in rule[:-1]:
            (n_dix, l_idx, r_idx,_, _, feat_idx, th, value) = node
            if n_dix != idx:
                break
            if feat_idx == -2:
                pred.append(value)
                break
            if _x[feat_idx] <= th:
                idx = l_idx
            else:
                idx = r_idx
        if len(pred) > 0:
            result.append((softmax(np.mean(pred, axis=0)), softmax(target_probs[i])))
    return result


def determine_entropy_shapley_value_threaded(rf_rules, X, y, target_probs, group, sharpely_by_contrib=False,
                                             multi_sampling=True):
    # def _job(dt_rule, dt_shapley_values):
    #     shapley_values = []
    #
    #     len_rules = len(dt_rule)
    #
    #     assert len_rules > 0
    #     assert len_rules > group
    #
    #     if multi_sampling:
    #         chunked_candidates = generate_random_chunk_list(list(range(len_rules)), group, iter=4)
    #     else:
    #         chunked_candidates = chunk_list_with_best_effort(list(range(len_rules)), group)
    #
    #     if sharpely_by_contrib:
    #         dt_contribs = [rule[-1][-1] for rule in dt_rule]
    #         min = np.min(dt_contribs)
    #         max = np.max(dt_contribs)
    #
    #     for chunk in tqdm.tqdm(chunked_candidates):
    #         target_players = []
    #         remaining_players = []
    #         for r_idx in range(len_rules):
    #             if r_idx in chunk:
    #                 target_players.append(dt_rule[r_idx])
    #             else:
    #                 remaining_players.append(dt_rule[r_idx])
    #
    #         remaining_player_rule_dt, _, _ = cvt_dt_rule_to_dt_form(remaining_players)
    #         remaining_probs = pred_prob_single_rules_cls(remaining_player_rule_dt, X)
    #
    #         pred_acc = np.mean(np.argmax(remaining_probs, axis=-1) == y)
    #         target_acc = np.mean(np.argmax(target_probs, axis=-1) == y)
    #         shapely_value = pred_acc - target_acc
    #
    #         for r_idx in chunk:
    #             if sharpely_by_contrib:
    #                 contrib_value = dt_rule[r_idx][-1][-1]
    #                 norm_contrib = contrib_value - min
    #                 norm_contrib = norm_contrib / (max - min)
    #                 shapely_value = shapely_value * norm_contrib
    #             shapley_values.append((r_idx, shapely_value))
    #
    #     if multi_sampling: shapley_values = reduce_redundancy(shapley_values)
    #     shapley_values = sorted(shapley_values, key=lambda v: v[1])
    #     return shapley_values
    #     # dt_shapley_values.append(shapley_values)
    #
    # # = pred_prob_multi_rule_cls(rules, X)
    # dt_shapley_values = []
    #
    # dt_shapley_values = Parallel(n_jobs=-1, verbose=0)(
    #     delayed(_job)(dt_rule, dt_shapley_values) for dt_rule in rules)
    #
    # # for dt_rule in rules:
    # #     shapley_values = []
    # #
    # #     len_rules = len(dt_rule)
    # #
    # #     chunked_candidates = chunk_list_with_best_effort(list(range(len_rules)), 10)
    # #
    # #     for chunk in tqdm.tqdm(chunked_candidates):
    # #         target_players = []
    # #         remaining_players = []
    # #         for r_idx in range(len_rules):
    # #             if r_idx in chunk:
    # #                 target_players.append(dt_rule[r_idx])
    # #             else:
    # #                 remaining_players.append(dt_rule[r_idx])
    # #
    # #         remaining_player_rule_dt, _, _ = cvt_dt_rule_to_dt_form(remaining_players)
    # #         remaining_probs = pred_prob_single_rules_cls(remaining_player_rule_dt, X)
    # #
    # #         pred_acc = np.mean(np.argmax(remaining_probs, axis=-1) == y)
    # #         target_acc = np.mean(np.argmax(target_probs, axis=-1) == y)
    # #         shapely_value = pred_acc - target_acc
    # #
    # #         for r_idx in chunk:
    # #             shapley_values.append((r_idx, shapely_value))
    # #     # Parallel(n_jobs=-1, verbose=0)(
    # #     #         delayed(_job_2)(chunk, shapley_values,dt_rule ,target_probs, y) for chunk in tqdm.tqdm(chunked_candidates))
    # #
    # #     # for r_idx in tqdm.tqdm(range(len_rules)):
    # #     #     target_player = dt_rule[r_idx]
    # #     #     remaining_player = dt_rule[:r_idx] + dt_rule[r_idx + 1:]
    # #     #
    # #     #     remaining_player_rule_dt, _, _ = cvt_dt_rule_to_dt_form(remaining_player)
    # #     #
    # #     #     # remaining_probs = pred_prob_single_rules_cls_threaded(remaining_player_rule_dt, X)
    # #     #     remaining_probs = pred_prob_single_rules_cls(remaining_player_rule_dt, X)
    # #     #
    # #     #     pred_acc = np.mean(np.argmax(remaining_probs, axis=-1) == y)
    # #     #     target_acc = np.mean(np.argmax(target_probs, axis=-1) == y)
    # #     #     shapely_value = pred_acc - target_acc
    # #     #
    # #     #     shapley_values.append((r_idx, shapely_value))
    # #
    # #     # Parallel(n_jobs=400, backend='threading', verbose=0)(
    # #     #     delayed(_job)(shapley_values, r_idx, dt_rule, target_probs, y) for r_idx in range(len_rules))
    # #
    # #     shapley_values = sorted(shapley_values, key=lambda v: v[1])
    # #     dt_shapley_values.append(shapley_values)

    def _job(rules):
        result = {}
        for r in tqdm.tqdm(range(len(rules))):
            rule = rules[r]
            result[r] = get_node_value_by_rule(rule, X, target_probs)
        return result

    rf_entropy_shapley = []

    # for rules in rf_rules:
    #     result = {}
    #     for r in tqdm.tqdm(range(len(rules))):
    #         footage = Parallel(n_jobs=-1, verbose=0)(
    #             delayed(_job)(i, rules[r]) for i in range(len(X)))
    #         result[r] = footage
    #
    #     # for r in range(len(rules)):
    #     #     rule = rules[r]
    #     #     result[r] = []
    #     #
    #     #     for i in range(len(X)):
    #     #         _x = X[i]
    #     #         pred = []
    #     #
    #     #         idx = 0
    #     #         for node in rule[:-1]:
    #     #             (n_dix, l_idx, r_idx, feat_idx, th, value) = node
    #     #             if n_dix != idx:
    #     #                 break
    #     #             if feat_idx == -2:
    #     #                 pred.append(value)
    #     #                 break
    #     #             if _x[feat_idx] <= th:
    #     #                 idx = l_idx
    #     #             else:
    #     #                 idx = r_idx
    #     #         if len(pred) > 0:
    #     #             result[r].append((softmax(np.mean(pred, axis=0)), softmax(y[i])) )
    #
    #     rf_entropy_shapley.append(result)

    rf_entropy_shapley = Parallel(n_jobs=-1, verbose=0)(
                delayed(_job)(dt_rule[:50]) for dt_rule in tqdm.tqdm(rf_rules))

    quit()
    return rf_entropy_shapley


def target_replacement(Y):
    uniq_Y = sorted(np.unique(Y))
    nrof_classes = len(uniq_Y)
    for i, u_y in enumerate(uniq_Y):
        Y[Y == u_y] = i
    return Y.astype(np.int)


def cut_off_rf_by_shapley(shapley_rf_set, rf_rules, r):
    # distilated_rule_rf = {}
    #
    # target_flatten_rules = []
    # for dt_i, shapley_dt_set in enumerate(shapley_rf_set):
    #     for r_idx, s_v in shapley_dt_set:
    #         target_flatten_rules.append((dt_i, r_idx, s_v))
    # target_flatten_rules = sorted(target_flatten_rules, key=lambda v: v[2])
    #
    # new_flatten_rules = target_flatten_rules[:int(len(target_flatten_rules) * r)]
    #
    # for virtual_rule in new_flatten_rules:
    #     rf_idx, r_idx, s_v = virtual_rule
    #     if rf_idx not in distilated_rule_rf.keys():
    #         distilated_rule_rf[rf_idx] = []
    #     distilated_rule_rf[rf_idx].append(rf_rules[rf_idx][r_idx])
    # return list(distilated_rule_rf.values()), len(target_flatten_rules), len(new_flatten_rules)

    distilated_rule_rf = []
    len_prev_r, len_new_r = 0, 0
    for i, shapley_dt_set in enumerate(shapley_rf_set):
        len_shapley = len(shapley_dt_set)
        len_prev_r += len_shapley
        new_shapley_values = shapley_dt_set[:int(len_shapley * r)]
        len_new_r += len(new_shapley_values)
        distilated_rule_dt = [rf_rules[i][j] for j, v, _ in new_shapley_values]
        distilated_rule_rf.append(distilated_rule_dt)
    return distilated_rule_rf, len_prev_r, len_new_r


import matplotlib.pyplot as plt


def balanced_hist_thresholding(b):
    # Starting point of histogram
    i_s = np.min(np.where(b[0] > 0))
    # End point of histogram
    i_e = np.max(np.where(b[0] > 0))
    # Center of histogram
    i_m = (i_s + i_e) // 2
    # Left side weight
    w_l = np.sum(b[0][0:i_m + 1])
    # Right side weight
    w_r = np.sum(b[0][i_m + 1:i_e + 1])
    # Until starting point not equal to endpoint
    while (i_s != i_e):
        # If right side is heavier
        if (w_r > w_l):
            # Remove the end weight
            w_r -= b[0][i_e]
            i_e -= 1
            # Adjust the center position and recompute the weights
            if ((i_s + i_e) // 2) < i_m:
                w_l -= b[0][i_m]
                w_r += b[0][i_m]
                i_m -= 1
        else:
            # If left side is heavier, remove the starting weight
            w_l -= b[0][i_s]
            i_s += 1
            # Adjust the center position and recompute the weights
            if ((i_s + i_e) // 2) > i_m:
                w_l += b[0][i_m + 1]
                w_r -= b[0][i_m + 1]
                i_m += 1
    return i_m


def cut_off_rf_by_contrib(rf_rules, rate=0.0):
    # rf_contrib = []
    # for dt_rule in rf_rules:
    #     dt_contrib = []
    #     for rule in dt_rule:
    #         dt_contrib.append(rule[-1][-1])
    #     rf_contrib.append(dt_contrib)
    #     b1 = plt.hist(dt_contrib, bins=len(set(dt_contrib)))
    #     plt.show()
    #     balanced_hist_thresholding(b1)
    # rf_contrib = np.hstack(rf_contrib)
    # plt.hist(rf_contrib, bins=len(set(rf_contrib)))
    # plt.show()
    rf_rule_set = []
    for dt_rule in rf_rules:
        new_rule_set = sorted(dt_rule, key=lambda rule: rule[-1][-1])[int(len(dt_rule) * rate):]
        rf_rule_set.append(new_rule_set)
    return rf_rule_set


def cvt_rf_rule_to_rf_form(shapley_rf_set):
    rule_forests = []
    nrof_op, nrof_params = 0, 0
    len_rule_set = 0

    for shapley_dt_set in shapley_rf_set:
        rule_tree, n_op, n_param = cvt_dt_rule_to_dt_form(shapley_dt_set)
        nrof_op += n_op
        nrof_params += n_param

        rule_forests.append(rule_tree)
        len_rule_set += len(shapley_dt_set)
    return rule_forests, len_rule_set, nrof_op, nrof_params


def cvt_dt_rule_to_dt_form(dt_rules):
    rule_tree = {}
    nrof_op, nrof_params = 0, 0

    for rule in dt_rules:
        for node_idx in range(len(rule) - 1):
            idx = rule[node_idx][0]
            if idx not in rule_tree:
                rule_tree[idx] = rule[node_idx][1:]

    tmp_op = 0
    for _r in dt_rules:
        tmp_op += len(_r[:-1]) - 1
    if len(dt_rules) > 0:
        nrof_op = tmp_op / len(dt_rules)
        nrof_params = travel_recur(rule_tree, 0)

    return rule_tree, nrof_op, nrof_params