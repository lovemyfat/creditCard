import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
import datetime
import collections
import numpy as np
import numbers
import random
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from importlib import reload
from matplotlib import pyplot as plt
reload(sys)
sys.setdefaultencoding( "utf-8")
# -*- coding: utf-8 -*-

### 对时间窗口，计算累计产比 ###
def TimeWindowSelection(df, daysCol, time_windows):
    '''
    :param df: the dataset containg variabel of days
    :param daysCol: the column of days
    :param time_windows: the list of time window
    :return:
    '''
    freq_tw = {}
    for tw in time_windows:
        freq = sum(df[daysCol].apply(lambda x: int(x<=tw)))
        freq_tw[tw] = freq
    return freq_tw


def DeivdedByZero(nominator, denominator):
    '''
    当分母为0时，返回0；否则返回正常值
    '''
    if denominator == 0:
        return 0
    else:
        return nominator*1.0/denominator


#对某些统一的字段进行统一
def ChangeContent(x):
    y = x.upper()
    if y == '_MOBILEPHONE':
        y = '_PHONE'
    return y

def MissingCategorial(df,x):
    missing_vals = df[x].map(lambda x: int(x!=x))
    return sum(missing_vals)*1.0/df.shape[0]

def MissingContinuous(df,x):
    missing_vals = df[x].map(lambda x: int(np.isnan(x)))
    return sum(missing_vals) * 1.0 / df.shape[0]

def MakeupRandom(x, sampledList):
    if x==x:
        return x
    else:
        randIndex = random.randint(0, len(sampledList)-1)
        return sampledList[randIndex]


def Outlier_Dectection(df,x):
    '''
    :param df:
    :param x:
    :return:
    '''
    p25, p75 = np.percentile(df[x], 25),np.percentile(df[x], 75)
    d = p75 - p25
    upper, lower =  p75 + 1.5*d, p25-1.5*d
    truncation = df[x].map(lambda x: max(min(upper, x), lower))
    return truncation

############################################################
#Step 0: 数据分析的初始工作, 包括读取数据文件、检查用户Id的一致性等#
############################################################

folderOfData = '/Users/Code/Data Collections/bank default/'
data1 = pd.read_csv(folderOfData+'PPD_LogInfo_3_1_Training_Set.csv', header = 0)
data2 = pd.read_csv(folderOfData+'PPD_Training_Master_GBK_3_1_Training_Set.csv', header = 0,encoding = 'gbk')
data3 = pd.read_csv(folderOfData+'PPD_Userupdate_Info_3_1_Training_Set.csv', header = 0)

#将数据集分为训练集与测试集
all_ids = data2['Idx']
train_ids, test_ids = train_test_split(all_ids, test_size=0.3)
train_ids = pd.DataFrame(train_ids)
test_ids = pd.DataFrame(test_ids)


data1_train = pd.merge(left=train_ids,right = data1, on='Idx', how='inner')
data2_train = pd.merge(left=train_ids,right = data2, on='Idx', how='inner')
data3_train = pd.merge(left=train_ids,right = data3, on='Idx', how='inner')

data1_test = pd.merge(left=test_ids,right = data1, on='Idx', how='inner')
data2_test = pd.merge(left=test_ids,right = data2, on='Idx', how='inner')
data3_test = pd.merge(left=test_ids,right = data3, on='Idx', how='inner')



#############################################################################################
# Step 1: 从PPD_LogInfo_3_1_Training_Set &  PPD_Userupdate_Info_3_1_Training_Set数据中衍生特征#
#############################################################################################
# compare whether the four city variables match
data2_train['city_match'] = data2_train.apply(lambda x: int(x.UserInfo_2 == x.UserInfo_4 == x.UserInfo_8 == x.UserInfo_20),axis = 1)
del data2_train['UserInfo_2']
del data2_train['UserInfo_4']
del data2_train['UserInfo_8']
del data2_train['UserInfo_20']

### 提取申请日期，计算日期差，查看日期差的分布
data1_train['logInfo'] = data1_train['LogInfo3'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1_train['Listinginfo'] = data1_train['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1_train['ListingGap'] = data1_train[['logInfo','Listinginfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)

### 提取申请日期，计算日期差，查看日期差的分布
'''
使用180天作为最大的时间窗口计算新特征
所有可以使用的时间窗口可以有7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days.
在每个时间窗口内，计算总的登录次数，不同的登录方式，以及每种登录方式的平均次数
'''
time_window = [7, 30, 60, 90, 120, 150, 180]
var_list = ['LogInfo1','LogInfo2']
data1GroupbyIdx = pd.DataFrame({'Idx':data1_train['Idx'].drop_duplicates()})

for tw in time_window:
    data1_train['TruncatedLogInfo'] = data1_train['Listinginfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data1_train.loc[data1_train['logInfo'] >= data1_train['TruncatedLogInfo']]
    for var in var_list:
        #count the frequences of LogInfo1 and LogInfo2
        count_stats = temp.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var)+'_'+str(tw)+'_count'] = data1GroupbyIdx['Idx'].map(lambda x: count_stats.get(x,0))

        # count the distinct value of LogInfo1 and LogInfo2
        Idx_UserupdateInfo1 = temp[['Idx', var]].drop_duplicates()
        uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_unique'] = data1GroupbyIdx['Idx'].map(lambda x: uniq_stats.get(x,0))

        # calculate the average count of each value in LogInfo1 and LogInfo2
        data1GroupbyIdx[str(var) + '_' + str(tw) + '_avg_count'] = data1GroupbyIdx[[str(var)+'_'+str(tw)+'_count',str(var) + '_' + str(tw) + '_unique']].\
            apply(lambda x: DeivdedByZero(x[0],x[1]), axis=1)


data3_train['ListingInfo'] = data3_train['ListingInfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3_train['UserupdateInfo'] = data3_train['UserupdateInfo2'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3_train['ListingGap'] = data3_train[['UserupdateInfo','ListingInfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)
collections.Counter(data3_train['ListingGap'])
hist_ListingGap = np.histogram(data3_train['ListingGap'])
hist_ListingGap = pd.DataFrame({'Freq':hist_ListingGap[0],'gap':hist_ListingGap[1][1:]})
hist_ListingGap['CumFreq'] = hist_ListingGap['Freq'].cumsum()
hist_ListingGap['CumPercent'] = hist_ListingGap['CumFreq'].map(lambda x: x*1.0/hist_ListingGap.iloc[-1]['CumFreq'])

'''
对 QQ和qQ, Idnumber和idNumber,MOBILEPHONE和PHONE 进行统一
在时间切片内，计算
 (1) 更新的频率
 (2) 每种更新对象的种类个数
 (3) 对重要信息如IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE的更新
'''
data3_train['UserupdateInfo1'] = data3_train['UserupdateInfo1'].map(ChangeContent)
data3GroupbyIdx = pd.DataFrame({'Idx':data3_train['Idx'].drop_duplicates()})

time_window = [7, 30, 60, 90, 120, 150, 180]
for tw in time_window:
    data3_train['TruncatedLogInfo'] = data3_train['ListingInfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data3_train.loc[data3_train['UserupdateInfo'] >= data3_train['TruncatedLogInfo']]

    #frequency of updating
    freq_stats = temp.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_'+str(tw)+'_freq'] = data3GroupbyIdx['Idx'].map(lambda x: freq_stats.get(x,0))

    # number of updated types
    Idx_UserupdateInfo1 = temp[['Idx','UserupdateInfo1']].drop_duplicates()
    uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_unique'] = data3GroupbyIdx['Idx'].map(lambda x: uniq_stats.get(x, x))

    #average count of each type
    data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_avg_count'] = data3GroupbyIdx[['UserupdateInfo_'+str(tw)+'_freq', 'UserupdateInfo_' + str(tw) + '_unique']]. \
        apply(lambda x: x[0] * 1.0 / x[1], axis=1)

    #whether the applicant changed items like IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE
    Idx_UserupdateInfo1['UserupdateInfo1'] = Idx_UserupdateInfo1['UserupdateInfo1'].map(lambda x: [x])
    Idx_UserupdateInfo1_V2 = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].sum()
    for item in ['_IDNUMBER','_HASBUYCAR','_MARRIAGESTATUSID','_PHONE']:
        item_dict = Idx_UserupdateInfo1_V2.map(lambda x: int(item in x)).to_dict()
        data3GroupbyIdx['UserupdateInfo_' + str(tw) + str(item)] = data3GroupbyIdx['Idx'].map(lambda x: item_dict.get(x, x))

# Combine the above features with raw features in PPD_Training_Master_GBK_3_1_Training_Set
allData = pd.concat([data2_train.set_index('Idx'), data3GroupbyIdx.set_index('Idx'), data1GroupbyIdx.set_index('Idx')],axis= 1)
allData.to_csv(folderOfData+'allData_0.csv',encoding = 'gbk')




########################################
# Step 2: 对类别型变量和数值型变量进行预处理#
########################################
allData = pd.read_csv(folderOfData+'allData_0.csv',header = 0,encoding = 'gbk')
allFeatures = list(allData.columns)
allFeatures.remove('target')
if 'Idx' in allFeatures:
    allFeatures.remove('Idx')
allFeatures.remove('ListingInfo')

#检查是否有常数型变量，并且检查是类别型还是数值型变量
numerical_var = []
for col in allFeatures:
    if len(set(allData[col])) == 1:
        print('delete {} from the dataset because it is a constant'.format(col))
        del allData[col]
        allFeatures.remove(col)
    else:
        uniq_valid_vals = [i for i in allData[col] if i == i]
        uniq_valid_vals = list(set(uniq_valid_vals))
        if len(uniq_valid_vals) >= 10 and isinstance(uniq_valid_vals[0], numbers.Real):
            numerical_var.append(col)

categorical_var = [i for i in allFeatures if i not in numerical_var]


#检查变量的最多值的占比情况,以及每个变量中占比最大的值
records_count = allData.shape[0]
col_most_values,col_large_value = {},{}
for col in allFeatures:
    value_count = allData[col].groupby(allData[col]).count()
    col_most_values[col] = max(value_count)/records_count
    large_value = value_count[value_count== max(value_count)].index[0]
    col_large_value[col] = large_value
col_most_values_df = pd.DataFrame.from_dict(col_most_values, orient = 'index')
col_most_values_df.columns = ['max percent']
col_most_values_df = col_most_values_df.sort_values(by = 'max percent', ascending = False)
pcnt = list(col_most_values_df[:500]['max percent'])
vars = list(col_most_values_df[:500].index)
plt.bar(range(len(pcnt)), height = pcnt)
plt.title('Largest Percentage of Single Value in Each Variable')

#计算多数值占比超过90%的字段中，少数值的坏样本率是否会显著高于多数值
large_percent_cols = list(col_most_values_df[col_most_values_df['max percent']>=0.9].index)
bad_rate_diff = {}
for col in large_percent_cols:
    large_value = col_large_value[col]
    temp = allData[[col,'target']]
    temp[col] = temp.apply(lambda x: int(x[col]==large_value),axis=1)
    bad_rate = temp.groupby(col).mean()
    if bad_rate.iloc[0]['target'] == 0:
        bad_rate_diff[col] = 0
        continue
    bad_rate_diff[col] = np.log(bad_rate.iloc[0]['target']/bad_rate.iloc[1]['target'])
bad_rate_diff_sorted = sorted(bad_rate_diff.items(),key=lambda x: x[1], reverse=True)
bad_rate_diff_sorted_values = [x[1] for x in bad_rate_diff_sorted]
plt.bar(x = range(len(bad_rate_diff_sorted_values)), height = bad_rate_diff_sorted_values)

#由于所有的少数值的坏样本率并没有显著高于多数值，意味着这些变量可以直接剔除
for col in large_percent_cols:
    if col in numerical_var:
        numerical_var.remove(col)
    else:
        categorical_var.remove(col)
    del allData[col]

'''
对类别型变量，如果缺失超过80%, 就删除，否则保留。
'''
missing_pcnt_threshould_1 = 0.8
for col in categorical_var:
    missingRate = MissingCategorial(allData,col)
    print('{0} has missing rate as {1}'.format(col,missingRate))
    if missingRate > missing_pcnt_threshould_1:
        categorical_var.remove(col)
        del allData[col]
allData_bk = allData.copy()

'''
用one-hot对类别型变量进行编码
'''
dummy_map = {}
dummy_columns = []
for raw_col in categorical_var:
    dummies = pd.get_dummies(allData.loc[:, raw_col], prefix=raw_col)
    col_onehot = pd.concat([allData[raw_col], dummies], axis=1)
    col_onehot = col_onehot.drop_duplicates()
    allData = pd.concat([allData, dummies], axis=1)
    del allData[raw_col]
    dummy_map[raw_col] = col_onehot
    dummy_columns = dummy_columns + list(dummies)




with open(folderOfData+'dummy_map.pkl',"wb") as f:
    f.write(pickle.dumps(dummy_map))

with open(folderOfData+'dummy_columns.pkl',"wb") as f:
    f.write(pickle.dumps(dummy_columns))


'''
检查数值型变量
'''
missing_pcnt_threshould_2 = 0.8
deleted_var = []
for col in numerical_var:
    missingRate = MissingContinuous(allData, col)
    print('{0} has missing rate as {1}'.format(col, missingRate))
    if missingRate > missing_pcnt_threshould_2:
        deleted_var.append(col)
        print('we delete variable {} because of its high missing rate'.format(col))
    else:
        if missingRate > 0:
            not_missing = allData.loc[allData[col] == allData[col]][col]
            #makeuped = allData[col].map(lambda x: MakeupRandom(x, list(not_missing)))
            missing_position = allData.loc[allData[col] != allData[col]][col].index
            not_missing_sample = random.sample(list(not_missing), len(missing_position))
            allData.loc[missing_position,col] = not_missing_sample
            #del allData[col]
            #allData[col] = makeuped
            missingRate2 = MissingContinuous(allData, col)
            print('missing rate after making up is:{}'.format(str(missingRate2)))

if deleted_var != []:
    for col in deleted_var:
        numerical_var.remove(col)
        del allData[col]

'''
对极端值变量做处理。
'''
max_min_standardized = {}
for col in numerical_var:
    truncation = Outlier_Dectection(allData, col)
    upper, lower = max(truncation), min(truncation)
    d = upper - lower
    if d == 0:
        print("{} is almost a constant".format(col))
        numerical_var.remove(col)
        continue
    allData[col] = truncation.map(lambda x: (upper - x)/d)
    max_min_standardized[col] = [lower, upper]



with open(folderOfData+'max_min_standardized.pkl',"wb") as f:
    f.write(pickle.dumps(max_min_standardized))


allData.to_csv(folderOfData+'allData_1_DNN.csv', header=True,encoding='gbk', columns = allData.columns, index=False)

allData = pd.read_csv(folderOfData+'allData_1_DNN.csv', header=0,encoding='gbk')


####################################
# Step 3: 构建XGBoost模型用于违约预测 #
####################################
allData = pd.read_csv(folderOfData+'allData_1_DNN.csv', header=0,encoding='gbk')
all_features = list(allData.columns)
all_features.remove('target')
#all_features.remove('ListingInfo')
X_train, y_train = allData[all_features], allData['target']

param_test1 = {'max_depth':range(3,10,2), 'min_child_weight':range(1,6,2)}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=100, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                  objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27),
                        param_grid = param_test1,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch1.fit(X_train,y_train)
best_max_depth, best_min_child_weight = gsearch1.best_params_['max_depth'],gsearch1.best_params_['min_child_weight']   #9,3

param_test2 = {'gamma':[i/10.0 for i in range(0,5)]}
gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.8, max_depth= best_max_depth,
                                                  min_child_weight=best_min_child_weight, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27),
                        param_grid = param_test2,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch2.fit(X_train,y_train)
best_gamma = gsearch2.best_params_['gamma']  #0


param_test3 = {'subsample':[i/10.0 for i in range(6,10)],'colsample_bytree':[i/10.0 for i in range(6,10)]}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth= best_max_depth, gamma=best_gamma,
                                                  min_child_weight=best_min_child_weight, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27),
                        param_grid = param_test3,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch3.fit(X_train,y_train)
best_colsample_bytree, best_subsample = gsearch3.best_params_['colsample_bytree'], gsearch3.best_params_['subsample']  #0.8, 0.6


param_test4 = {'reg_alpha':[0.01,0.1,1,10,50,100,200,500]}
gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth= best_max_depth, gamma=best_gamma,
                                                  colsample_bytree = best_colsample_bytree, subsample = best_subsample,
                                                  min_child_weight=best_min_child_weight, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27),
                        param_grid = param_test4,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch4.fit(X_train,y_train)
best_reg_alpha = gsearch4.best_params_['reg_alpha']   #50


param_test5 = {'n_estimators':range(100,401,10)}
gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1,  max_depth= best_max_depth, gamma=best_gamma,
                                                  colsample_bytree = best_colsample_bytree, subsample = best_subsample,reg_alpha=best_reg_alpha,
                                                  min_child_weight=best_min_child_weight, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27),
                        param_grid = param_test5,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gsearch5.fit(X_train,y_train)
best_n_estimators = gsearch5.best_params_     #390


#用获取得到的最优参数再次训练模型
best_xgb = XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth= best_max_depth, gamma=best_gamma,
                         colsample_bytree = best_colsample_bytree, subsample = best_subsample, reg_alpha=best_reg_alpha,
                         min_child_weight=best_min_child_weight, objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27)
best_xgb.fit(X_train,y_train)
y_pred = best_xgb.predict_proba(X_train)[:,1]
roc_auc_score(y_train, y_pred)   #0.8
feature_importance = best_xgb.feature_importances_


#利用特征重要性筛去一部分无用的变量
X_train_temp = X_train.copy()
features_in_model = all_features
while(min(feature_importance)<0.00001):
    features_in_model = [features_in_model[i] for i in range(len(feature_importance)) if feature_importance[i] > 0.00001]
    X_train_temp= X_train_temp[features_in_model]
    best_xgb.fit(X_train_temp, y_train)
    feature_importance = best_xgb.feature_importances_

y_pred = best_xgb.predict_proba(X_train_temp)[:,1]
roc_auc_score(y_train, y_pred)  # 0.8
print('There are {} features in the raw data'.format(X_train.shape[1]))  #400
print('There are {} features in the reduced data'.format(X_train_temp.shape[1]))  #158
