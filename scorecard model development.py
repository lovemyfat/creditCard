import pandas as pd
import datetime
import collections
import numpy as np
import numbers
import random
import sys
import pickle
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from importlib import reload
from matplotlib import pyplot as plt
reload(sys)
sys.setdefaultencoding( "utf-8")
from scorecard_functions import *
from sklearn.linear_model import LogisticRegressionCV
# -*- coding: utf-8 -*-

################################
######## UDF: 自定义函数 ########
################################
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



############################################################
#Step 0: 数据分析的初始工作, 包括读取数据文件、检查用户Id的一致性等#
############################################################

folderOfData = '/Users/Code/Data Collections/bank default/'
data1 = pd.read_csv(folderOfData+'PPD_LogInfo_3_1_Training_Set.csv', header = 0)
data2 = pd.read_csv(folderOfData+'PPD_Training_Master_GBK_3_1_Training_Set.csv', header = 0,encoding = 'gbk')
data3 = pd.read_csv(folderOfData+'PPD_Userupdate_Info_3_1_Training_Set.csv', header = 0)

#############################################################################################
# Step 1: 从PPD_LogInfo_3_1_Training_Set &  PPD_Userupdate_Info_3_1_Training_Set数据中衍生特征#
#############################################################################################
# compare whether the four city variables match
data2['city_match'] = data2.apply(lambda x: int(x.UserInfo_2 == x.UserInfo_4 == x.UserInfo_8 == x.UserInfo_20),axis = 1)
del data2['UserInfo_2']
del data2['UserInfo_4']
del data2['UserInfo_8']
del data2['UserInfo_20']

### 提取申请日期，计算日期差，查看日期差的分布
data1['logInfo'] = data1['LogInfo3'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1['Listinginfo'] = data1['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
data1['ListingGap'] = data1[['logInfo','Listinginfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)
plt.hist(data1['ListingGap'],bins=200)
plt.title('Days between login date and listing date')
ListingGap2 = data1['ListingGap'].map(lambda x: min(x,365))
plt.hist(ListingGap2,bins=200)

timeWindows = TimeWindowSelection(data1, 'ListingGap', range(30,361,30))

'''
使用180天作为最大的时间窗口计算新特征
所有可以使用的时间窗口可以有7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days.
在每个时间窗口内，计算总的登录次数，不同的登录方式，以及每种登录方式的平均次数
'''
time_window = [7, 30, 60, 90, 120, 150, 180]
var_list = ['LogInfo1','LogInfo2']
data1GroupbyIdx = pd.DataFrame({'Idx':data1['Idx'].drop_duplicates()})

for tw in time_window:
    data1['TruncatedLogInfo'] = data1['Listinginfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data1.loc[data1['logInfo'] >= data1['TruncatedLogInfo']]
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


data3['ListingInfo'] = data3['ListingInfo1'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3['UserupdateInfo'] = data3['UserupdateInfo2'].map(lambda x: datetime.datetime.strptime(x,'%Y/%m/%d'))
data3['ListingGap'] = data3[['UserupdateInfo','ListingInfo']].apply(lambda x: (x[1]-x[0]).days,axis = 1)
collections.Counter(data3['ListingGap'])
hist_ListingGap = np.histogram(data3['ListingGap'])
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
data3['UserupdateInfo1'] = data3['UserupdateInfo1'].map(ChangeContent)
data3GroupbyIdx = pd.DataFrame({'Idx':data3['Idx'].drop_duplicates()})

time_window = [7, 30, 60, 90, 120, 150, 180]
for tw in time_window:
    data3['TruncatedLogInfo'] = data3['ListingInfo'].map(lambda x: x + datetime.timedelta(-tw))
    temp = data3.loc[data3['UserupdateInfo'] >= data3['TruncatedLogInfo']]

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
allData = pd.concat([data2.set_index('Idx'), data3GroupbyIdx.set_index('Idx'), data1GroupbyIdx.set_index('Idx')],axis= 1)
allData.to_csv(folderOfData+'allData_0.csv',encoding = 'gbk')




#######################################
# Step 2: 对类别型变量和数值型变量进行补缺#
######################################
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
对类别型变量，如果缺失超过80%, 就删除，否则当成特殊的状态
'''
missing_pcnt_threshould_1 = 0.8
for col in categorical_var:
    missingRate = MissingCategorial(allData,col)
    print('{0} has missing rate as {1}'.format(col,missingRate))
    if missingRate > missing_pcnt_threshould_1:
        categorical_var.remove(col)
        del allData[col]
    if 0 < missingRate < missing_pcnt_threshould_1:
        uniq_valid_vals = [i for i in allData[col] if i == i]
        uniq_valid_vals = list(set(uniq_valid_vals))
        if isinstance(uniq_valid_vals[0], numbers.Real):
            missing_position = allData.loc[allData[col] != allData[col]][col].index
            not_missing_sample = [-1]*len(missing_position)
            allData.loc[missing_position, col] = not_missing_sample
        else:
            # In this way we convert NaN to NAN, which is a string instead of np.nan
            allData[col] = allData[col].map(lambda x: str(x).upper())

allData_bk = allData.copy()
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


allData.to_csv(folderOfData+'allData_1.csv', header=True,encoding='gbk', columns = allData.columns, index=False)

allData = pd.read_csv(folderOfData+'allData_1.csv', header=0,encoding='gbk')




###################################
# Step 3: 基于卡方分箱法对变量进行分箱#
###################################
'''
对不同类型的变量，分箱的处理是不同的：
（1）数值型变量可直接分箱
（2）取值个数较多的类别型变量，需要用bad rate做编码转换成数值型变量，再分箱
（3）取值个数较少的类别型变量不需要分箱，但是要检查是否每个类别都有好坏样本。如果有类别只有好或坏，需要合并
'''

#for each categorical variable, if it has distinct values more than 5, we use the ChiMerge to merge it

trainData = pd.read_csv(folderOfData+'allData_1.csv',header = 0, encoding='gbk',dtype=object)
allFeatures = list(trainData.columns)
allFeatures.remove('ListingInfo')
allFeatures.remove('target')
#allFeatures.remove('Idx')

#将特征区分为数值型和类别型
numerical_var = []
for var in allFeatures:
    uniq_vals = list(set(trainData[var]))
    if np.nan in uniq_vals:
        uniq_vals.remove( np.nan)
    if len(uniq_vals) >= 10 and isinstance(uniq_vals[0],numbers.Real):
        numerical_var.append(var)

categorical_var = [i for i in allFeatures if i not in numerical_var]

for col in categorical_var:
    #for Chinese character, upper() is not valid
    if col not in ['UserInfo_7','UserInfo_9','UserInfo_19']:
        trainData[col] = trainData[col].map(lambda x: str(x).upper())


'''
对于类别型变量，按照以下方式处理
1，如果变量的取值个数超过5，计算bad rate进行编码
2，除此之外，其他任何类别型变量如果有某个取值中，对应的样本全部是坏样本或者是好样本，进行合并。
'''
deleted_features = []   #将处理过的变量删除，防止对后面建模的干扰
encoded_features = {}   #将bad rate编码方式保存下来，在以后的测试和生产环境中需要使用
merged_features = {}    #将类别型变量合并方案保留下来
var_IV = {}  #save the IV values for binned features       #将IV值保留和WOE值
var_WOE = {}
for col in categorical_var:
    print('we are processing {}'.format(col))
    if len(set(trainData[col]))>5:
        print('{} is encoded with bad rate'.format(col))
        col0 = str(col)+'_encoding'

        #(1), 计算坏样本率并进行编码
        encoding_result = BadRateEncoding(trainData, col, 'target')
        trainData[col0], br_encoding = encoding_result['encoding'],encoding_result['bad_rate']

        #(2), 将（1）中的编码后的变量也加入数值型变量列表中，为后面的卡方分箱做准备
        numerical_var.append(col0)

        #(3), 保存编码结果
        encoded_features[col] = [col0, br_encoding]

        #(4), 删除原始值

        deleted_features.append(col)
    else:
        bad_bin = trainData.groupby([col])['target'].sum()
        #对于类别数少于5个，但是出现0坏样本的特征需要做处理
        if min(bad_bin) == 0:
            print('{} has 0 bad sample!'.format(col))
            col1 = str(col) + '_mergeByBadRate'
            #(1), 找出最优合并方式，使得每一箱同时包含好坏样本
            mergeBin = MergeBad0(trainData, col, 'target')
            #(2), 依照（1）的结果对值进行合并
            trainData[col1] = trainData[col].map(mergeBin)
            maxPcnt = MaximumBinPcnt(trainData, col1)
            #如果合并后导致有箱占比超过90%，就删除。
            if maxPcnt > 0.9:
                print('{} is deleted because of large percentage of single bin'.format(col))
                deleted_features.append(col)
                categorical_var.remove(col)
                del trainData[col]
                continue
            #(3) 如果合并后的新的变量满足要求，就保留下来
            merged_features[col] = [col1, mergeBin]
            WOE_IV = CalcWOE(trainData, col1, 'target')
            var_WOE[col1] = WOE_IV['WOE']
            var_IV[col1] = WOE_IV['IV']
            #del trainData[col]
            deleted_features.append(col)
        else:
            WOE_IV = CalcWOE(trainData, col, 'target')
            var_WOE[col] = WOE_IV['WOE']
            var_IV[col] = WOE_IV['IV']


'''
对于连续型变量，处理方式如下：
1，利用卡方分箱法将变量分成5个箱
2，检查坏样本率的单带性，如果发现单调性不满足，就进行合并，直到满足单调性
'''
var_cutoff = {}
for col in numerical_var:
    print("{} is in processing".format(col))
    col1 = str(col) + '_Bin'

    #(1),用卡方分箱法进行分箱，并且保存每一个分割的端点。例如端点=[10,20,30]表示将变量分为x<10,10<x<20,20<x<30和x>30.
    #特别地，缺失值-1不参与分箱
    if -1 in set(trainData[col]):
        special_attribute = [-1]
    else:
        special_attribute = []
    cutOffPoints = ChiMerge(trainData, col, 'target',special_attribute=special_attribute)
    var_cutoff[col] = cutOffPoints
    trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))

    #(2), check whether the bad rate is monotone
    BRM = BadRateMonotone(trainData, col1, 'target',special_attribute=special_attribute)
    if not BRM:
        if special_attribute == []:
            bin_merged = Monotone_Merge(trainData, 'target', col1)
            removed_index = []
            for bin in bin_merged:
                if len(bin)>1:
                    indices = [int(b.replace('Bin ','')) for b in bin]
                    removed_index = removed_index+indices[0:-1]
            removed_point = [cutOffPoints[k] for k in removed_index]
            for p in removed_point:
                cutOffPoints.remove(p)
            var_cutoff[col] = cutOffPoints
            trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
        else:
            cutOffPoints2 = [i for i in cutOffPoints if i not in special_attribute]
            temp = trainData.loc[~trainData[col].isin(special_attribute)]
            bin_merged = Monotone_Merge(temp, 'target', col1)
            removed_index = []
            for bin in bin_merged:
                if len(bin) > 1:
                    indices = [int(b.replace('Bin ', '')) for b in bin]
                    removed_index = removed_index + indices[0:-1]
            removed_point = [cutOffPoints2[k] for k in removed_index]
            for p in removed_point:
                cutOffPoints2.remove(p)
            cutOffPoints2 = cutOffPoints2 + special_attribute
            var_cutoff[col] = cutOffPoints2
            trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints2, special_attribute=special_attribute))

    #(3), 分箱后再次检查是否有单一的值占比超过90%。如果有，删除该变量
    maxPcnt = MaximumBinPcnt(trainData, col1)
    if maxPcnt > 0.9:
        # del trainData[col1]
        deleted_features.append(col)
        numerical_var.remove(col)
        print('we delete {} because the maximum bin occupies more than 90%'.format(col))
        continue

    WOE_IV = CalcWOE(trainData, col1, 'target')
    var_IV[col] = WOE_IV['IV']
    var_WOE[col] = WOE_IV['WOE']
    #del trainData[col]



trainData.to_csv(folderOfData+'allData_2.csv', header=True,encoding='gbk', columns = trainData.columns, index=False)



with open(folderOfData+'var_WOE.pkl',"wb") as f:
    f.write(pickle.dumps(var_WOE))

with open(folderOfData+'var_IV.pkl',"wb") as f:
    f.write(pickle.dumps(var_IV))


with open(folderOfData+'var_cutoff.pkl',"wb") as f:
    f.write(pickle.dumps(var_cutoff))


with open(folderOfData+'merged_features.pkl',"wb") as f:
    f.write(pickle.dumps(merged_features))

########################################
# Step 4: WOE编码后的单变量分析与多变量分析#
########################################
trainData = pd.read_csv(folderOfData+'allData_2.csv', header=0, encoding='gbk')


with open(folderOfData+'var_WOE.pkl',"rb") as f:
    var_WOE = pickle.load(f)

with open(folderOfData+'var_IV.pkl',"rb") as f:
    var_IV = pickle.load(f)


with open(folderOfData+'var_cutoff.pkl',"rb") as f:
    var_cutoff = pickle.load(f)


with open(folderOfData+'merged_features.pkl',"rb") as f:
    merged_features = pickle.load(f)

#将一些看起来像数值变量实际上是类别变量的字段转换成字符
num2str = ['SocialNetwork_13','SocialNetwork_12','UserInfo_6','UserInfo_5','UserInfo_10','UserInfo_17']
for col in num2str:
    trainData[col] = trainData[col].map(lambda x: str(x))


for col in var_WOE.keys():
    print(col)
    col2 = str(col)+"_WOE"
    if col in var_cutoff.keys():
        cutOffPoints = var_cutoff[col]
        special_attribute = []
        if - 1 in cutOffPoints:
            special_attribute = [-1]
        binValue = trainData[col].map(lambda x: AssignBin(x, cutOffPoints,special_attribute=special_attribute))
        trainData[col2] = binValue.map(lambda x: var_WOE[col][x])
    else:
        trainData[col2] = trainData[col].map(lambda x: var_WOE[col][x])

trainData.to_csv(folderOfData+'allData_3.csv', header=True,encoding='gbk', columns = trainData.columns, index=False)



### (i) 选择IV高于阈值的变量
trainData = pd.read_csv(folderOfData+'allData_3.csv', header=0,encoding='gbk')
all_IV = list(var_IV.values())
all_IV = sorted(all_IV, reverse=True)
plt.bar(x=range(len(all_IV)), height = all_IV)
iv_threshould = 0.02
varByIV = [k for k, v in var_IV.items() if v > iv_threshould]



### (ii) 检查WOE编码后的变量的两两线性相关性

var_IV_selected = {k:var_IV[k] for k in varByIV}
var_IV_sorted = sorted(var_IV_selected.items(), key=lambda d:d[1], reverse = True)
var_IV_sorted = [i[0] for i in var_IV_sorted]

removed_var  = []
roh_thresould = 0.6
for i in range(len(var_IV_sorted)-1):
    if var_IV_sorted[i] not in removed_var:
        x1 = var_IV_sorted[i]+"_WOE"
        for j in range(i+1,len(var_IV_sorted)):
            if var_IV_sorted[j] not in removed_var:
                x2 = var_IV_sorted[j] + "_WOE"
                roh = np.corrcoef([trainData[x1], trainData[x2]])[0, 1]
                if abs(roh) >= roh_thresould:
                    print('the correlation coeffient between {0} and {1} is {2}'.format(x1, x2, str(roh)))
                    if var_IV[var_IV_sorted[i]] > var_IV[var_IV_sorted[j]]:
                        removed_var.append(var_IV_sorted[j])
                    else:
                        removed_var.append(var_IV_sorted[i])

var_IV_sortet_2 = [i for i in var_IV_sorted if i not in removed_var]

### (iii）检查是否有变量与其他所有变量的VIF > 10
for i in range(len(var_IV_sortet_2)):
    x0 = trainData[var_IV_sortet_2[i]+'_WOE']
    x0 = np.array(x0)
    X_Col = [k+'_WOE' for k in var_IV_sortet_2 if k != var_IV_sortet_2[i]]
    X = trainData[X_Col]
    X = np.matrix(X)
    regr = LinearRegression()
    clr= regr.fit(X, x0)
    x_pred = clr.predict(X)
    R2 = 1 - ((x_pred - x0) ** 2).sum() / ((x0 - x0.mean()) ** 2).sum()
    vif = 1/(1-R2)
    if vif > 10:
        print("Warning: the vif for {0} is {1}".format(var_IV_sortet_2[i], vif))



#########################
# Step 5: 应用逻辑回归模型#
#########################
multi_analysis = [i+'_WOE' for i in var_IV_sortet_2]
y = trainData['target']
X = trainData[multi_analysis].copy()
X['intercept'] = [1]*X.shape[0]


LR = sm.Logit(y, X).fit()
summary = LR.summary2()
pvals = LR.pvalues.to_dict()
params = LR.params.to_dict()

#发现有变量不显著，因此需要单独检验显著性
varLargeP = {k: v for k,v in pvals.items() if v >= 0.1}
varLargeP = sorted(varLargeP.items(), key=lambda d:d[1], reverse = True)
varLargeP = [i[0] for i in varLargeP]
p_value_list = {}
for var in varLargeP:
    X_temp = trainData[var].copy().to_frame()
    X_temp['intercept'] = [1] * X_temp.shape[0]
    LR = sm.Logit(y, X_temp).fit()
    p_value_list[var] = LR.pvalues[var]
for k,v in p_value_list.items():
    print("{0} has p-value of {1} in univariate regression".format(k,v))


#发现有变量的系数为正，因此需要单独检验正确性
varPositive = [k for k,v in params.items() if v >= 0]
coef_list = {}
for var in varPositive:
    X_temp = trainData[var].copy().to_frame()
    X_temp['intercept'] = [1] * X_temp.shape[0]
    LR = sm.Logit(y, X_temp).fit()
    coef_list[var] = LR.params[var]
for k,v in coef_list.items():
    print("{0} has coefficient of {1} in univariate regression".format(k,v))


selected_var = [multi_analysis[0]]
for var in multi_analysis[1:]:
    try_vars = selected_var+[var]
    X_temp = trainData[try_vars].copy()
    X_temp['intercept'] = [1] * X_temp.shape[0]
    LR = sm.Logit(y, X_temp).fit()
    #summary = LR.summary2()
    pvals, params = LR.pvalues, LR.params
    del params['intercept']
    if max(pvals)<0.1 and max(params)<0:
        selected_var.append(var)

LR.summary2()

y_pred = LR.predict(X_temp)
roc_auc_score(trainData['target'], y_pred)



#########################
# Step 6: 尺度化与性能检验#
#########################
scores = Prob2Score(y_pred, 200, 100)
plt.hist(score,bins=100)

scorecard = pd.DataFrame({'y_pred':y_pred, 'y_real':list(trainData['target']),'score':scores})
KS(scorecard,'score','y_real')
ROC_AUC(df, score, target)

# 也可用sklearn带的函数
roc_auc_score(trainData['target'], y_pred)