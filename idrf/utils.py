import numpy as np


def travel_recur(ruleset, idx):
    if idx not in ruleset.keys():
        return 0
    node = ruleset[idx]  # idx: l_c, r_c, feat, th, classes
    (l_c, r_c, feat, th, classes) = node

    nrof_params = 4 + len(classes)

    if l_c != -1:
        n_p = travel_recur(ruleset, l_c)
        nrof_params += n_p

    if r_c != -1:
        n_p = travel_recur(ruleset, r_c)
        nrof_params += n_p

    return nrof_params

def eliminate_rules_dt_cls(dt, rate):
    rule_forests = []
    len_rule_set, len_new_rule_set = 0,0
    nrof_op, nrof_params = 0,0
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

    new_rule_set = sorted(rule_set, key=lambda rule: rule[-1][-1])[int(len(rule_set) * rate):]

    len_rule_set += len(rule_set)
    len_new_rule_set += len(new_rule_set)
    rule_tree = {}

    for rule in new_rule_set:
        for node_idx in range(len(rule) - 1):
            idx = rule[node_idx][0]
            if idx not in rule_tree:
                rule_tree[idx] = rule[node_idx][1:]
    if len(rule_tree) > 0:
        rule_forests.append(rule_tree)

    tmp_op = 0
    for _r in new_rule_set:
        tmp_op += len(_r[:-1]) - 1
    nrof_op += tmp_op / len(new_rule_set)
    nrof_params += travel_recur(rule_tree, 0)

    feat_contribs.append(contribs_by_feat)

    return rule_forests, len_rule_set, len_new_rule_set, nrof_op, nrof_params, feat_contribs


def eliminate_rules_rfs_cls(rfs, rate):
    rule_forests = []
    len_rule_set, len_new_rule_set = 0, 0
    nrof_op, nrof_params = 0, 0
    feat_contribs = []
    for estimator in rfs.estimators_:
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        value = estimator.tree_.value.squeeze(axis=1)
        contribs_by_feat = [[0.] * rfs.n_classes_] * rfs.n_features_

        normalizer = estimator.tree_.value.squeeze(axis=1).sum(axis=1)[:, np.newaxis]
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

        new_rule_set = sorted(rule_set, key=lambda rule: rule[-1][-1])[int(len(rule_set) * rate):]

        len_rule_set += len(rule_set)
        len_new_rule_set += len(new_rule_set)
        rule_tree = {}

        for rule in new_rule_set:
            for node_idx in range(len(rule) - 1):
                idx = rule[node_idx][0]
                if idx not in rule_tree:
                    rule_tree[idx] = rule[node_idx][1:]
        if len(rule_tree) > 0:
            rule_forests.append(rule_tree)

        tmp_op = 0
        for _r in new_rule_set:
            tmp_op += len(_r[:-1]) - 1
        nrof_op += tmp_op / len(new_rule_set)
        nrof_params += travel_recur(rule_tree, 0)

        feat_contribs.append(contribs_by_feat)

    return rule_forests, len_rule_set, len_new_rule_set, nrof_op, nrof_params, feat_contribs

def eliminate_rules_rfs_rgs(rfs, rate):
    rule_forests = []
    len_rule_set, len_new_rule_set = 0, 0
    nrof_op, nrof_params = 0, 0
    feat_contribs = []
    for estimator in rfs.estimators_:
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        value = estimator.tree_.value.squeeze()
        # contribs_by_feat = [[0.] * rfs.n_classes_] * rfs.n_features_

        # normalizer = estimator.tree_.value.squeeze(axis=1).sum(axis=1)[:, np.newaxis]
        # normalizer[normalizer == 0.] = 1.
        # value /= normalizer

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
                contrib += (np.abs(rule[i + 1][-1]) - np.abs(rule[i][-1]))
                # contribs_by_feat[rule[i][-3]] += (rule[i + 1][-1] - rule[i][-1])
            bias = rule[0][-1]
            pred = rule[-1][-1]
            contrib = (np.sum(contrib))
            rule.append((pred, bias, contrib))
            if max_contrib < contrib: max_contrib = contrib
            sum_contrib += abs(contrib)
            conts.append(contrib)

        new_rule_set = sorted(rule_set, key=lambda rule: rule[-1][-1])[int(len(rule_set) * rate):]

        len_rule_set += len(rule_set)
        len_new_rule_set += len(new_rule_set)
        rule_tree = {}

        for rule in new_rule_set:
            for node_idx in range(len(rule) - 1):
                idx = rule[node_idx][0]
                if idx not in rule_tree:
                    rule_tree[idx] = rule[node_idx][1:]
        if len(rule_tree) > 0:
            rule_forests.append(rule_tree)

        tmp_op = 0
        for _r in new_rule_set:
            tmp_op += len(_r[:-1]) - 1
        nrof_op += tmp_op / len(new_rule_set)
        nrof_params += travel_recur(rule_tree, 0)

        # feat_contribs.append(contribs_by_feat)

    return rule_forests, len_rule_set, len_new_rule_set, nrof_op, nrof_params, feat_contribs

def pred_prob_single_rules_cls(rule, X):
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
from scipy.stats import entropy
def extract_rules_dt_rgs_v2(dt):
    def softmax_log(a):
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a

        return np.log(y+1)

    def calc_feat_corr(c_node, p_node):
        c_feat, p_feat = c_node[-3], p_node[-3]
        c_dist, p_dist = c_node[-4], p_node[-4]
        c_prob, p_prob = c_node[-1], p_node[-1]

        entropy_parent = entropy(p_prob, base=2)

        entropy_p_c = np.sum(c_prob) / np.sum(p_prob) * entropy(c_prob, base=2)

        gain = entropy_parent - entropy_p_c


        return p_feat, c_feat, gain
        # return c_feat, p_feat, gain

    len_rule_set, len_new_rule_set = 0, 0
    nrof_op, nrof_params = 0, 0
    feat_contribs = {}
    feat_correlations = np.zeros([dt.n_features_,dt.n_features_])
    for i in range(dt.n_features_):
        feat_contribs[i] = []


    n_nodes = dt.tree_.node_count
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    impurity = dt.tree_.impurity
    value = dt.tree_.value.squeeze()
    weighted_n_node_samples = dt.tree_.weighted_n_node_samples
    # dt.tree_.n_node_samples
    n_node_samples = dt.tree_.n_node_samples
    # contribs_by_feat = [[0.] * dt.n_classes_] * dt.n_features_

    # normalizer = dt.tree_.value.squeeze(axis=1).sum(axis=1)[:, np.newaxis]
    # normalizer[normalizer == 0.] = 1.
    # value /= normalizer

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
        path.append((i, children_left[i], children_right[i], weighted_n_node_samples[i], (n_node_samples[i] * value[i]).astype(int), feature[i], threshold[i], value[i]))
        if is_leaves[i]:
            rule_set.append(path.copy())

    max_contrib = -float('inf')
    sum_contrib = 0
    conts = []
    for rule in rule_set:
        contrib = 0
        for i in range(len(rule) - 1):
            _c = (np.abs(rule[i + 1][-1]) - np.abs(rule[i][-1]))
            _c = softmax_log(_c)
            contrib += _c
            _f, _th = rule[i][-3], rule[i][-2]
            # feat_contribs[_f].append(np.sum(np.abs(_c))*10)
            feat_contribs[_f].append(_c)
            # contribs_by_feat[rule[i][-3]] += (np.abs(rule[i + 1][-1]) - np.abs(rule[i][-1]))
            _from, _to, _f_corr = calc_feat_corr(rule[i + 1], rule[i])
            feat_correlations[_from][_to] += _f_corr
        bias = rule[0][-1]
        pred = rule[-1][-1]
        # contrib = (np.sum(contrib))
        rule.append((pred, bias, contrib))
        # if max_contrib < contrib: max_contrib = contrib
        # sum_contrib += (contrib)
        # conts.append(contrib)

    new_rule_set = sorted(rule_set, key=lambda rule: np.sum(rule[-1][-1]))[int(len(rule_set) * 0):]

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

    # feat_contribs.append(contribs_by_feat)

    return new_rule_set, len_rule_set, len_new_rule_set, feat_contribs, feat_correlations



def extract_rules_rf_rgs_v2(rf):
    dt_rules = []
    len_rule_set, len_new_rule_set = 0, 0
    feat_contribs = {}
    feat_correlations = np.zeros([rf.n_features_, rf.n_features_])
    for i in range(rf.n_features_):
        feat_contribs[i] = np.zeros(rf.n_outputs_)

    for estimator in rf.estimators_:
        dt_rule, len_r_s, len_n_r_s, f_contribs, f_corrs = extract_rules_dt_rgs_v2(estimator)
        dt_rules.append(dt_rule)
        len_rule_set += len_r_s
        len_new_rule_set += len_n_r_s
        # feat_contribs.append(f_contribs)
        for i in range(rf.n_features_):
            # feat_contribs[i] += f_contribs[i]
            feat_contribs[i] += np.sum(f_contribs[i], axis=0)
        # feat_contribs.append(f_contribs)
        feat_correlations += f_corrs

    for i in range(rf.n_features_):
        # feat_contribs[i] += f_contribs[i]
        feat_contribs[i] = np.log(feat_contribs[i])
    return dt_rules, len_rule_set, len_new_rule_set, feat_contribs, feat_correlations