import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from xgboost.sklearn import XGBClassifier
from importlib import reload
from matplotlib import pyplot as plt
reload(sys)
sys.setdefaultencoding( "utf-8")
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from random import shuffle



folderOfData = '/Users/Code/Data Collections/bank default/'
mydata = pd.read_csv(folderOfData+'allData_3.csv', header= 0,encoding = 'gbk')
size = mydata.shape[0]

######### 集成方式1：Bagging ###########
#待选模型：LR， XGBOOST， DNN
#数据集准备：从原数据集中进行有放回地抽样形成某一个训练集，袋外数据用来进行参数选择
trainData = mydata.sample(n=size,replace=True)
trainDataIndex = set(list(trainData.Idx))   #18910个样本
OOB = mydata.loc[~mydata['Idx'].isin(list(trainDataIndex))]

all_features = [feature for feature in list(trainData.columns) if feature.find('WOE')>=0]

C_list = np.arange(0.01,1,0.01)
auc = []
for c in C_list:
    LR = LogisticRegression(C=c).fit(trainData[all_features], trainData['target'])
    pred = LR.predict_proba(OOB[all_features])[:,1]
    test_auc = roc_auc_score(OOB['target'], pred)
    auc.append(test_auc)

position = auc.index(max(auc))
C_best = C_list[position]
print(max(auc))
LR = LogisticRegression(C=C_best).fit(trainData[all_features], trainData['target'])
lr_pred = LR.predict_proba(mydata[all_features])[:,1]
lr_auc = roc_auc_score(mydata['target'], lr_pred)    #0.762



#XGBOOST
trainData = mydata.sample(n=size,replace=True)
trainDataIndex = set(list(trainData.Idx))   #18910个样本
OOB = mydata.loc[~mydata['Idx'].isin(list(trainDataIndex))]
best_max_depth, best_min_child_weight = 9,3
best_gamma = 0
best_colsample_bytree, best_subsample  = 0.8, 0.6
best_reg_alpha = 50
best_n_estimators = 390

best_xgb = XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth= best_max_depth, gamma=best_gamma,
                         colsample_bytree = best_colsample_bytree, subsample = best_subsample, reg_alpha=best_reg_alpha,
                         min_child_weight=best_min_child_weight, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)
best_xgb.fit(trainData[all_features], trainData['target'])
xgb_pred = best_xgb.predict_proba(trainData[all_features])[:,1]
xgb_auc = roc_auc_score(trainData['target'], xgb_pred)   #0.795


#ANN
trainData = mydata.sample(n=size,replace=True)
trainDataIndex = set(list(trainData.Idx))   #18910个样本
OOB = mydata.loc[~mydata['Idx'].isin(list(trainDataIndex))]
ann_clf = MLPClassifier(hidden_layer_sizes=(80,70,60))
ann_clf.fit(trainData[all_features], trainData['target'])
ann_pred = ann_clf.predict_proba(mydata[all_features])[:,1]
ann_auc = roc_auc_score(mydata['target'], ann_pred)    #0.879


###### 模型的融合 ########
bagging_pred = pd.DataFrame({'LR':lr_pred, 'XGB':xgb_pred, 'ANN':ann_pred})
bagging_pred['avg_prob'] = bagging_pred.apply(np.mean, axis=1)
avg_auc = roc_auc_score(mydata['target'], bagging_pred['avg_prob']) #0.8925


######### 集成方式2：Boosting ###########
#Adaboost 模型
bad = mydata[mydata.target == 1]
good0 = mydata[mydata.target == 0]
good = good0.iloc[:bad.shape[0]*10]
train_data_ada = pd.concat([bad, good])
param_test = {'n_estimators':range(5,101,5),'learning_rate':np.arange(0.1,1.1,0.1)}
gsearch = GridSearchCV(estimator = AdaBoostClassifier(),
                        param_grid = param_test,iid=False,cv=5)
gsearch.fit(train_data_ada[all_features], train_data_ada['target'])
best_n_estimators, best_learning_rate= gsearch.best_params_['n_estimators'],gsearch.best_params_['learning_rate']  #5,0.05

ada_clf = AdaBoostClassifier(n_estimators=best_n_estimators,learning_rate=best_learning_rate)
ada_clf.fit(train_data_ada[all_features], train_data_ada['target'])
ada_pred = ada_clf.predict_proba(mydata[all_features])[:,1]
ada_auc = roc_auc_score(mydata['target'], ada_pred)   #0.773


######### 集成方式3：Stacking ###########
# Tier1 采用xgboost和ANN， Tier2 采用LR
idx = list(mydata.Idx)
K = 5
num_of_models = 2
xbg_result = pd.DataFrame()
ann_result = pd.DataFrame()


#采用xgboost做Tier1中的第一个模型
#为了避免Tier1的子模型产生一定的相关性，构建交叉验证时将样本随机排列，使得Tier1的子模型的验证集拥有多样性
idx_shuffle = idx.copy()
shuffle(idx)
#使用K折交叉法，将数据集等分成K等分。又由于数据集的样本量未必是K的整倍数，因此第1~K-1份子集的大小是一致的，
# 第K份子集的大小不低于第1~K-1份子集的大小
sub_n = int(np.floor(len(idx_shuffle)/K))
interval_starts = [i*sub_n for i in range(K)]
interval_ends = [(i+1)*sub_n-1 for i in range(K-1)] + [len(idx_shuffle)-1]
for j in range(K):
    #拿出第j份子集作为验证集，其余的子集合并称为小训练集
    start_pot, end_pot = interval_starts[j],interval_ends[j]
    validation_idx = [idx_shuffle[m] for m in range(start_pot,end_pot+1)]
    validation_set = mydata.loc[mydata['Idx'].isin(validation_idx)]
    train_set = mydata.loc[~mydata['Idx'].isin(validation_idx)]
    #训练xgboost
    best_xgb = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=best_max_depth, gamma=best_gamma,
                             colsample_bytree=best_colsample_bytree, subsample=best_subsample,
                             reg_alpha=best_reg_alpha,
                             min_child_weight=best_min_child_weight, objective='binary:logistic', nthread=4,
                             scale_pos_weight=1, seed=27)
    best_xgb.fit(train_set[all_features], train_set['target'])
    xgb_pred = best_xgb.predict_proba(validation_set[all_features])[:, 1]
    auc = roc_auc_score(validation_set['target'], xgb_pred)
    print("AUC is {}".format(auc))
    result_validation = pd.DataFrame({'Idx':validation_idx, 'xgb_pred':xgb_pred})
    xbg_result = pd.concat([xbg_result, result_validation])
'''
AUC is 0.6928400298699413
AUC is 0.7018707487623171
AUC is 0.7258867969723856
AUC is 0.7572941130555957
AUC is 0.7134017338057725
'''



#采用ANN做Tier1中的第二个模型。对数据集的预处理与XGBoost时一样的。
shuffle(idx_shuffle)
sub_n = int(np.floor(len(idx_shuffle)/K))
interval_starts = [i*sub_n for i in range(K)]
interval_ends = [(i+1)*sub_n-1 for i in range(K-1)] + [len(idx_shuffle)-1]
for j in range(K):
    start_pot, end_pot = interval_starts[j],interval_ends[j]
    validation_idx = [idx_shuffle[m] for m in range(start_pot,end_pot+1)]
    validation_set = mydata.loc[mydata['Idx'].isin(validation_idx)]
    train_set = mydata.loc[~mydata['Idx'].isin(validation_idx)]
    #训练ann
    ann_clf = MLPClassifier(hidden_layer_sizes=(80, 70, 60))
    ann_clf.fit(train_set[all_features], train_set['target'])
    ann_pred = ann_clf.predict_proba(validation_set[all_features])[:, 1]
    auc = roc_auc_score(validation_set['target'], ann_pred)
    print("AUC is {}".format(auc))
    result_validation = pd.DataFrame({'Idx':validation_idx, 'ann_pred':ann_pred})
    ann_result = pd.concat([ann_result, result_validation])
'''
AUC is 0.6486770578384644
AUC is 0.6674674305446555
AUC is 0.6105685164636292
AUC is 0.6161161272102443
AUC is 0.633197559547805
'''




tier1_pred = pd.merge(left=xbg_result,right=ann_result, on='Idx', how='inner')
tier1_target = mydata[['Idx','target']]
tier1_train = pd.merge(left=tier1_pred,right=tier1_target, on='Idx', how='inner')

LR = LogisticRegression().fit(tier1_train[['ann_pred','xgb_pred']], tier1_train['target'])
tier2_pred = LR.predict_proba(tier1_train[['ann_pred','xgb_pred']])[:,1]
roc_auc_score(tier1_train['target'], tier2_pred)    #0.718
roc_auc_score(tier1_train['target'], tier1_train['xgb_pred'])
roc_auc_score(tier1_train['target'], tier1_train['ann_pred'])
