import autogluon as ag
from autogluon import TabularPrediction as task

from sklearn.datasets import load_iris, fetch_olivetti_faces,load_breast_cancer,fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import copy
from joblib import dump, load

import numpy as np
import matplotlib.pyplot as plt

from core.utils import *
from idrf.utils import *

from joblib import dump, load

# (sharpley_rule_forests, dt_rules) = load("SHAP_entropy_multi_rf_group{}".format(20))

dir = 'nn_search_Census-Income' # specifies folder where to store trained models

column = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]



# column = [
#     "Class", "a1", "a2", "a3", "a4", "a5", "a6"
# ]
label_column = 'class'
def adult_process(path):
    import pandas as pd
    global label_column
    df = pd.read_csv(path, header=None, delimiter=r",")
    df.columns = column
    df["Income"] = df["Income"].map({ ' <=50K': 0, ' >50K': 1 })
    df["WorkClass"] = df["WorkClass"].map({b: a for a, b in enumerate(set(sorted(df.WorkClass.unique())))})
    df["Education"] = df["Education"].map({b: a for a, b in enumerate(set(sorted(df.Education.unique())))})
    df["MaritalStatus"] = df["MaritalStatus"].map({b: a for a, b in enumerate(set(sorted(df.MaritalStatus.unique())))})
    df["Occupation"] = df["Occupation"].map({b: a for a, b in enumerate(set(sorted(df.Occupation.unique())))})
    df["Relationship"] = df["Relationship"].map({b: a for a, b in enumerate(set(sorted(df.Relationship.unique())))})
    df["Race"] = df["Race"].map({b: a for a, b in enumerate(set(sorted(df.Race.unique())))})
    df["Gender"] = df["Gender"].map({b: a for a, b in enumerate(set(sorted(df.Gender.unique())))})
    df["NativeCountry"] = df["NativeCountry"].map({b: a for a, b in enumerate(set(sorted(df.NativeCountry.unique())))})
    df.Age = df.Age.astype(float)
    df.fnlwgt = df.fnlwgt.astype(float)
    df.EducationNum = df.EducationNum.astype(float)
    df.HoursPerWeek = df.HoursPerWeek.astype(float)

    y = df.Income.to_numpy()
    df.drop(columns=["Income"], inplace=True)
    X = df.to_numpy()

    label_column = 'Income'

    return X, y

def monk_process(path):
    import pandas as pd
    global label_column
    df = pd.read_csv(path, header=None, delimiter=r",")
    df.columns = column
    df.Class = df.Class.astype(int)
    df.a1 = df.a1.astype(float)
    df.a2 = df.a2.astype(float)
    df.a3 = df.a3.astype(float)
    df.a4 = df.a4.astype(float)
    df.a5 = df.a5.astype(float)
    df.a6 = df.a6.astype(float)

    y = df.Class.to_numpy()
    df.drop(columns=["Class"], inplace=True)
    X = df.to_numpy()

    label_column = 'Class'

    return X, y

# # ddata = load_iris()
# # ddata = load_breast_cancer()
# # ddata = fetch_olivetti_faces()
# # ddata = fetch_openml('mnist_784', data_home='../scikit_learn_data')
# # ddata = fetch_openml('monks-problems-1', data_home='../scikit_learn_data')
# # ddata = fetch_openml('heart-h',version=3, data_home='../scikit_learn_data')
# ddata = fetch_openml('adult',data_home='../scikit_learn_data')
# D_X = ddata.data
# D_y = ddata.target.astype(int)
# new_D_y = np.zeros_like(D_y)
# for n, tup in {b: a for a, b in enumerate(set(sorted(np.unique(D_y))))}.items():
#     new_D_y[D_y == n] = tup
# D_y = new_D_y
# # if np.min(D_y) == 1:
# #     D_y -= 1
# if hasattr(ddata, 'feature_names'):
#     column = np.asarray(ddata.feature_names)
#     column = np.hstack([column, label_column])
#
# else:
#     column = np.asarray([i for i in range(D_X.shape[1])])
#     column = np.hstack([column, label_column])

X_train, y_train = adult_process('adult.data')
X_test, y_test = adult_process('adult.test')

# X_train = np.delete(X_train, 5,1)
# X_test = np.delete(X_test, 5,1)
#
# column = [
#     "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
#     "Occupation", "Relationship", "Race", "Gender",
#     "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
# ]

# X_train, y_train = monk_process('monks.data')
# X_test, y_test = monk_process('monks.test')

# X_train, X_test, y_train, y_test = train_test_split(D_X, D_y, test_size=0.20, random_state=42)


# test_idx = []
# for i in range(40):
#     targets = random.sample(range(10), 5)
#     test_idx.append([e+i*10 for e in targets ])
# test_idx = np.hstack(test_idx)
# train_idx = [i for i in range(400) if i not in test_idx]
# X_train, X_test = ddata.data[train_idx], ddata.data[test_idx]
# y_train, y_test = ddata.target[train_idx], ddata.target[test_idx]

ag_train = np.hstack([X_train.astype(str), np.expand_dims(y_train.astype(int).astype(str), axis=-1)])
ag_test = np.hstack([X_test.astype(str), np.expand_dims(y_test.astype(int).astype(str), axis=-1)])


ag_train = np.vstack([column, ag_train])
ag_test = np.vstack([column, ag_test])

np.savetxt("dataset/tmp_train.csv", ag_train.astype('object'), fmt='%s',delimiter=",")
np.savetxt("dataset/tmp_test.csv", ag_test.astype('object'), fmt='%s',delimiter=",")

####################################

train_data = task.Dataset(file_path='dataset/tmp_train.csv')
# train_data = train_data.head(500) # subsample 500 data points for faster demo
print(train_data.head())

test_data = task.Dataset(file_path="dataset/tmp_test.csv")
# y_test = test_data[label_column]  # values to predict


print("Summary of class variable: \n", train_data[label_column].describe())

#
# nn_options = { # specifies non-default hyperparameter values for neural network models
#     'num_epochs': 3, # number of training epochs (controls training time of NN models)
#     'learning_rate': ag.space.Real(1e-6, 1e-2, default=5e-4, log=True), # learning rate used in training (real-valued hyperparameter searched on log-scale)
#     'activation': ag.space.Categorical('relu'), # activation function used in NN (categorical hyperparameter, default = first entry)
#     'layers': ag.space.Categorical([1000],[200,100], [500,1000,100]),
#     # 'layers': ag.space.Categorical([1000],[200,100],[300,200,100]),
#       # Each choice for categorical hyperparameter 'layers' corresponds to list of sizes for each NN layer to use
#     'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1), # dropout probability (real-valued hyperparameter)
# }
#
# hyperparameters = {'NN': nn_options}
#
# time_limits = 2*60  # train various models for ~2 min
# num_trials = 1 # try at most 3 different hyperparameter configurations for each type of model
# search_strategy = 'skopt'  # to tune hyperparameters using SKopt Bayesian optimization routine
#
#
# # predictor = task.fit(train_data=train_data,tuning_data=test_data, label=label_column, output_directory=dir,
# #                      time_limits=time_limits, num_trials=num_trials, hyperparameter_tune=True, hyperparameters=hyperparameters,
# #                      search_strategy=search_strategy, ngpus_per_trial=1, problem_type='multiclass')
#
# test_data_nolab = test_data.drop(labels=[label_column],axis=1) # delete label column to prove we're not cheating
# train_data_nolab = train_data.drop(labels=[label_column],axis=1) # delete label column to prove we're not cheating
# print(test_data_nolab.head())
#
# predictor = task.load(dir) # unnecessary, just demonstrates how to load previously-trained predictor from file
#
# # y_pred = predictor.predict(test_data_nolab)
# # print('Test Accuracy', np.mean(y_pred == y_test))
# y_te_pred_prob = predictor.predict_proba(test_data_nolab)
# y_tr_pred_prob = predictor.predict_proba(train_data_nolab)

# np.save('y_tr_pred_prob', y_tr_pred_prob)
y_tr_pred_prob = np.load('y_tr_pred_prob.npy')

tau = 0.75

new_y = np.exp(y_tr_pred_prob/tau)/np.sum(np.exp(y_tr_pred_prob/tau), axis=-1)[:,np.newaxis]
# new_y = y_tr_pred_prob

# tree = RandomForestRegressor(n_estimators=20, criterion='mse', n_jobs=-1, max_features="auto", min_samples_leaf=3, max_depth=13)
# # tree = RandomForestRegressor(n_estimators=100, criterion='mse', n_jobs=-1, max_features=None, min_samples_leaf=2, max_depth=8, oob_score=False)
# # tree_vanilla = copy.deepcopy(tree)
# tree.fit(X_train, new_y)

# dump(tree, 'trained_tree')
tree = load('trained_tree')

y_pred = tree.predict(X_test)
print('Surrogate RgRF', np.mean(np.argmax(y_pred, -1) == y_test))




target_probs = tree.predict(X_train)
dt_rules, len_rule_set, len_new_rule_set, feat_contribs, feat_correlations = extract_rules_rf_rgs_v2(tree)
# len_rule_set = len(len_rule_set)
print('Total number of rules', len_rule_set)

sharpley_rule_forests = determine_entropy_shapley_value_threaded_v2(dt_rules, X_train, new_y, target_probs,
                                                                        20)

dump((sharpley_rule_forests, dt_rules), "SHAP_entropy_multi_rf_group{}".format(20))

exit()


for i in range(len(feat_contribs)):
    row = feat_contribs[i]
    print(f'{column[i]} {np.sum(row)}')

for i in range(len(feat_contribs)):
    row = feat_contribs[i]
    print(f'{column[i]} {np.mean(row)}')

plt.imshow(feat_correlations)
plt.xticks(list(range(0, 14)), column)
plt.yticks(list(range(0, 14)), column)
plt.colorbar()
plt.show()
plt.savefig('feat_corr.png')

exit()

group_sizes = [20]
for group_size in group_sizes:
    print('Entropy Rule group ', group_size, ' multisampling')
    sharpley_rule_forests = determine_entropy_shapley_value_threaded_v2(dt_rules, X_train, new_y, target_probs,
                                                                        group_size)

    dump((sharpley_rule_forests, dt_rules), "entropy_multi_rf_group{}".format(group_size))

    # for i, rf_value in enumerate(sharpley_rule_forests):
    #     csv_file = open("shap_values_entropy_multi_{}_rf{}.csv".format(group_size, i), 'w', newline='')
    #     for value in rf_value:
    #         a,b,c = value
    #         csv_file.write('{},{},{}\n'.format(a,b,c))
    #     csv_file.close()

    for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        # len_shapley = len(sharpley_rule_forests)
        # new_shapley_values = sharpley_rule_forests[:int(len_shapley * r)]
        # distilated_rule_forests = [rule_forests[i] for i, v in new_shapley_values]

        distilated_rule_rf, len_prev_r_s, len_new_r_s = cut_off_rf_by_shapley(sharpley_rule_forests, dt_rules, r)
        distilated_rule_rf, len_distillated_rule_set, nrof_op, nrof_params = cvt_rf_rule_to_rf_form(distilated_rule_rf)

        y_pred = pred_prob_multi_rule_cls(distilated_rule_rf, X_test)
        print('Surrogate RgRF shapley', np.mean(np.argmax(y_pred, axis=-1) == y_test), 'remaining',
              len_distillated_rule_set / len_rule_set, 'op', nrof_op, 'param', nrof_params)

    print('Entropy Rule group ', group_size, ' multisampling, new contrib only')
    sharpley_rule_forests = determine_entropy_shapley_value_threaded_v2(dt_rules, X_train, new_y, target_probs,
                                                                        group_size, only_contrib=True)

    # dump((sharpley_rule_forests, dt_rules), "entropy_multi_rf_group{}".format(group_size))

    # for i, rf_value in enumerate(sharpley_rule_forests):
    #     csv_file = open("shap_values_entropy_multi_{}_rf{}.csv".format(group_size, i), 'w', newline='')
    #     for value in rf_value:
    #         a,b,c = value
    #         csv_file.write('{},{},{}\n'.format(a,b,c))
    #     csv_file.close()

    for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        # len_shapley = len(sharpley_rule_forests)
        # new_shapley_values = sharpley_rule_forests[:int(len_shapley * r)]
        # distilated_rule_forests = [rule_forests[i] for i, v in new_shapley_values]

        distilated_rule_rf, len_prev_r_s, len_new_r_s = cut_off_rf_by_shapley(sharpley_rule_forests, dt_rules, r)
        distilated_rule_rf, len_distillated_rule_set, nrof_op, nrof_params = cvt_rf_rule_to_rf_form(distilated_rule_rf)

        y_pred = pred_prob_multi_rule_cls(distilated_rule_rf, X_test)
        print('Surrogate RgRF shapley', np.mean(np.argmax(y_pred, axis=-1) == y_test), 'remaining',
              len_distillated_rule_set / len_rule_set, 'op', nrof_op, 'param', nrof_params)

    print('Entropy Rule group ', group_size)
    sharpley_rule_forests = determine_entropy_shapley_value_threaded_v2(dt_rules, X_train, new_y, target_probs,
                                                                        group_size, multi_sampling=False)
    dump((sharpley_rule_forests, dt_rules), "entropy_rf_group{}".format(group_size))
    # for i, rf_value in enumerate(sharpley_rule_forests):
    #     csv_file = open("shap_values_entropy_{}_rf{}.csv".format(group_size, i), 'w', newline='')
    #     for value in rf_value:
    #         a,b,c = value
    #         csv_file.write('{},{},{}\n'.format(a,b,c))
    #     csv_file.close()

    for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        # len_shapley = len(sharpley_rule_forests)
        # new_shapley_values = sharpley_rule_forests[:int(len_shapley * r)]
        # distilated_rule_forests = [rule_forests[i] for i, v in new_shapley_values]

        distilated_rule_rf, len_prev_r_s, len_new_r_s = cut_off_rf_by_shapley(sharpley_rule_forests, dt_rules, r)
        distilated_rule_rf, len_distillated_rule_set, nrof_op, nrof_params = cvt_rf_rule_to_rf_form(distilated_rule_rf)

        y_pred = pred_prob_multi_rule_cls(distilated_rule_rf, X_test)
        print('Surrogate RgRF shapley', np.mean(np.argmax(y_pred, axis=-1) == y_test), 'remaining',
              len_distillated_rule_set / len_rule_set, 'op', nrof_op, 'param', nrof_params)

    print('Entropy Rule group ', group_size, ' multisampling, new contrib shapley')
    sharpley_rule_forests = determine_entropy_shapley_value_threaded_v2(dt_rules, X_train, new_y, target_probs,
                                                                        group_size, sharpely_by_contrib=True)

    # dump((sharpley_rule_forests, dt_rules), "entropy_multi_rf_group{}".format(group_size))

    # for i, rf_value in enumerate(sharpley_rule_forests):
    #     csv_file = open("shap_values_entropy_multi_{}_rf{}.csv".format(group_size, i), 'w', newline='')
    #     for value in rf_value:
    #         a,b,c = value
    #         csv_file.write('{},{},{}\n'.format(a,b,c))
    #     csv_file.close()

    for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        # len_shapley = len(sharpley_rule_forests)
        # new_shapley_values = sharpley_rule_forests[:int(len_shapley * r)]
        # distilated_rule_forests = [rule_forests[i] for i, v in new_shapley_values]

        distilated_rule_rf, len_prev_r_s, len_new_r_s = cut_off_rf_by_shapley(sharpley_rule_forests, dt_rules, r)
        distilated_rule_rf, len_distillated_rule_set, nrof_op, nrof_params = cvt_rf_rule_to_rf_form(distilated_rule_rf)

        y_pred = pred_prob_multi_rule_cls(distilated_rule_rf, X_test)
        print('Surrogate RgRF shapley', np.mean(np.argmax(y_pred, axis=-1) == y_test), 'remaining',
              len_distillated_rule_set / len_rule_set, 'op', nrof_op, 'param', nrof_params)

    print('Rule group ', group_size, ' multisampling')
    sharpley_rule_forests = determine_shapley_value_threaded(dt_rules, X_train, y_train, target_probs,
                                                             group_size)
    for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        # len_shapley = len(sharpley_rule_forests)
        # new_shapley_values = sharpley_rule_forests[:int(len_shapley * r)]
        # distilated_rule_forests = [rule_forests[i] for i, v in new_shapley_values]

        distilated_rule_rf, len_prev_r_s, len_new_r_s = cut_off_rf_by_shapley(sharpley_rule_forests, dt_rules, r)
        distilated_rule_rf, len_distillated_rule_set, nrof_op, nrof_params = cvt_rf_rule_to_rf_form(distilated_rule_rf)

        y_pred = pred_prob_multi_rule_cls(distilated_rule_rf, X_test)
        print('Surrogate RgRF shapley', np.mean(np.argmax(y_pred, axis=-1) == y_test), 'remaining',
              len_distillated_rule_set / len_rule_set, 'op', nrof_op, 'param', nrof_params)

    print('Rule group', group_size)
    sharpley_rule_forests = determine_shapley_value_threaded(dt_rules, X_train, y_train, target_probs,
                                                             group_size, multi_sampling=False)
    for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        # len_shapley = len(sharpley_rule_forests)
        # new_shapley_values = sharpley_rule_forests[:int(len_shapley * r)]
        # distilated_rule_forests = [rule_forests[i] for i, v in new_shapley_values]

        distilated_rule_rf, len_prev_r_s, len_new_r_s = cut_off_rf_by_shapley(sharpley_rule_forests, dt_rules, r)
        distilated_rule_rf, len_distillated_rule_set, nrof_op, nrof_params = cvt_rf_rule_to_rf_form(distilated_rule_rf)

        y_pred = pred_prob_multi_rule_cls(distilated_rule_rf, X_test)
        print('Surrogate RgRF shapley', np.mean(np.argmax(y_pred, axis=-1) == y_test), 'remaining',
              len_distillated_rule_set / len_rule_set, 'op', nrof_op, 'param', nrof_params)

quit()
group_size = 8

print('Entropy Rule group ',group_size,' multisampling')
sharpley_rule_forests = determine_entropy_shapley_value_threaded_v2(dt_rules, X_train, new_y, target_probs,
                                                         group_size)
for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    # len_shapley = len(sharpley_rule_forests)
    # new_shapley_values = sharpley_rule_forests[:int(len_shapley * r)]
    # distilated_rule_forests = [rule_forests[i] for i, v in new_shapley_values]

    distilated_rule_rf, len_prev_r_s, len_new_r_s = cut_off_rf_by_shapley(sharpley_rule_forests, dt_rules, r)
    distilated_rule_rf, len_distillated_rule_set, nrof_op, nrof_params = cvt_rf_rule_to_rf_form(distilated_rule_rf)

    y_pred = pred_prob_multi_rule_cls(distilated_rule_rf, X_test)
    print('Surrogate RgRF shapley', np.mean(np.argmax(y_pred, axis=-1) == y_test), 'remaining',
          len_distillated_rule_set / len_rule_set, 'op', nrof_op, 'param', nrof_params)

print('Entropy Rule group ',group_size)
sharpley_rule_forests = determine_entropy_shapley_value_threaded_v2(dt_rules, X_train, new_y, target_probs,
                                                         group_size, multi_sampling=False)
for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    # len_shapley = len(sharpley_rule_forests)
    # new_shapley_values = sharpley_rule_forests[:int(len_shapley * r)]
    # distilated_rule_forests = [rule_forests[i] for i, v in new_shapley_values]

    distilated_rule_rf, len_prev_r_s, len_new_r_s = cut_off_rf_by_shapley(sharpley_rule_forests, dt_rules, r)
    distilated_rule_rf, len_distillated_rule_set, nrof_op, nrof_params = cvt_rf_rule_to_rf_form(distilated_rule_rf)

    y_pred = pred_prob_multi_rule_cls(distilated_rule_rf, X_test)
    print('Surrogate RgRF shapley', np.mean(np.argmax(y_pred, axis=-1) == y_test), 'remaining',
          len_distillated_rule_set / len_rule_set, 'op', nrof_op, 'param', nrof_params)


print('Rule group ',group_size,' multisampling')
sharpley_rule_forests = determine_shapley_value_threaded(dt_rules, X_train, y_train, target_probs,
                                                         group_size)
for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    # len_shapley = len(sharpley_rule_forests)
    # new_shapley_values = sharpley_rule_forests[:int(len_shapley * r)]
    # distilated_rule_forests = [rule_forests[i] for i, v in new_shapley_values]

    distilated_rule_rf, len_prev_r_s, len_new_r_s = cut_off_rf_by_shapley(sharpley_rule_forests, dt_rules, r)
    distilated_rule_rf, len_distillated_rule_set, nrof_op, nrof_params = cvt_rf_rule_to_rf_form(distilated_rule_rf)

    y_pred = pred_prob_multi_rule_cls(distilated_rule_rf, X_test)
    print('Surrogate RgRF shapley', np.mean(np.argmax(y_pred, axis=-1) == y_test), 'remaining',
          len_distillated_rule_set / len_rule_set, 'op', nrof_op, 'param', nrof_params)

print('Rule group',group_size)
sharpley_rule_forests = determine_shapley_value_threaded(dt_rules, X_train, y_train, target_probs,
                                                         group_size, multi_sampling=False)
for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    # len_shapley = len(sharpley_rule_forests)
    # new_shapley_values = sharpley_rule_forests[:int(len_shapley * r)]
    # distilated_rule_forests = [rule_forests[i] for i, v in new_shapley_values]

    distilated_rule_rf, len_prev_r_s, len_new_r_s = cut_off_rf_by_shapley(sharpley_rule_forests, dt_rules, r)
    distilated_rule_rf, len_distillated_rule_set, nrof_op, nrof_params = cvt_rf_rule_to_rf_form(distilated_rule_rf)

    y_pred = pred_prob_multi_rule_cls(distilated_rule_rf, X_test)
    print('Surrogate RgRF shapley', np.mean(np.argmax(y_pred, axis=-1) == y_test), 'remaining',
          len_distillated_rule_set / len_rule_set, 'op', nrof_op, 'param', nrof_params)

group_size = 5
print('Rule group ',group_size,' multisampling')
sharpley_rule_forests = determine_shapley_value_threaded(dt_rules, X_train, y_train, target_probs,
                                                         group_size)
for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    # len_shapley = len(sharpley_rule_forests)
    # new_shapley_values = sharpley_rule_forests[:int(len_shapley * r)]
    # distilated_rule_forests = [rule_forests[i] for i, v in new_shapley_values]

    distilated_rule_rf, len_prev_r_s, len_new_r_s = cut_off_rf_by_shapley(sharpley_rule_forests, dt_rules, r)
    distilated_rule_rf, len_distillated_rule_set, nrof_op, nrof_params = cvt_rf_rule_to_rf_form(distilated_rule_rf)

    y_pred = pred_prob_multi_rule_cls(distilated_rule_rf, X_test)
    print('Surrogate RgRF shapley', np.mean(np.argmax(y_pred, axis=-1) == y_test), 'remaining',
          len_distillated_rule_set / len_rule_set, 'op', nrof_op, 'param', nrof_params)

print('Rule group', group_size)
sharpley_rule_forests = determine_shapley_value_threaded(dt_rules, X_train,y_train, target_probs,
                                                         group_size, multi_sampling=False)
for r in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
    # len_shapley = len(sharpley_rule_forests)
    # new_shapley_values = sharpley_rule_forests[:int(len_shapley * r)]
    # distilated_rule_forests = [rule_forests[i] for i, v in new_shapley_values]

    distilated_rule_rf, len_prev_r_s, len_new_r_s = cut_off_rf_by_shapley(sharpley_rule_forests, dt_rules, r)
    distilated_rule_rf, len_distillated_rule_set, nrof_op, nrof_params = cvt_rf_rule_to_rf_form(distilated_rule_rf)

    y_pred = pred_prob_multi_rule_cls(distilated_rule_rf, X_test)
    print('Surrogate RgRF shapley', np.mean(np.argmax(y_pred, axis=-1) == y_test), 'remaining',
          len_distillated_rule_set / len_rule_set, 'op', nrof_op, 'param', nrof_params)
