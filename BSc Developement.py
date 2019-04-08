import pandas as pd
import numpy as np
import pickle
import random
import datetime
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import seaborn as sns

def DelqFeatures(event,window,type):
    current = 12
    start = 12 - window + 1
    delq1 = [event[a] for a in ['Delq1_' + str(t) for t in range(current, start - 1, -1)]]
    delq2 = [event[a] for a in ['Delq2_' + str(t) for t in range(current, start - 1, -1)]]
    delq3 = [event[a] for a in ['Delq3_' + str(t) for t in range(current, start - 1, -1)]]
    if type == 'max delq':
        if max(delq3) == 1:
            return 3
        elif max(delq2) == 1:
            return 2
        elif max(delq1) == 1:
            return 1
        else:
            return 0
    if type in ['M0 times','M1 times', 'M2 times']:
        if type.find('M0')>-1:
            return sum(delq1)
        elif type.find('M1')>-1:
            return sum(delq2)
        else:
            return sum(delq3)

def UrateFeatures(event, window, type):
    current = 12
    start = 12 - window + 1
    monthlySpend = [event[a] for a in ['Spend_' + str(t) for t in range(current, start - 1, -1)]]
    limit = event['Loan_Amount']
    monthlyUrate = [x / limit for x in monthlySpend]
    if type == 'mean utilization rate':
        return np.mean(monthlyUrate)
    if type == 'max utilization rate':
        return max(monthlyUrate)
    if type == 'increase utilization rate':
        currentUrate = monthlyUrate[0:-1]
        previousUrate = monthlyUrate[1:]
        compareUrate = [int(x[0]>x[1]) for x in zip(currentUrate,previousUrate)]
        return sum(compareUrate)

def PaymentFeatures(event, window, type):
    current = 12
    start = 12 - window + 1
    currentPayment = [event[a] for a in ['Payment_' + str(t) for t in range(current, start - 1, -1)]]
    previousOS = [event[a] for a in ['OS_' + str(t) for t in range(current-1, start - 2, -1)]]
    monthlyPayRatio = []
    for Pay_OS in zip(currentPayment,previousOS):
        if Pay_OS[1]>0:
            payRatio = Pay_OS[0]*1.0 / Pay_OS[1]
            monthlyPayRatio.append(payRatio)
        else:
            monthlyPayRatio.append(1)
    if type == 'min payment ratio':
        return min(monthlyPayRatio)
    if type == 'max payment ratio':
        return max(monthlyPayRatio)
    if type == 'mean payment ratio':
        total_payment = sum(currentPayment)
        total_OS = sum(previousOS)
        if total_OS > 0:
            return total_payment / total_OS
        else:
            return 1

#################################
#   1, 读取数据，衍生初始变量   #
#################################
folderOfData = '/Users/Code/Data Collections/behavioural data/'
creditData = pd.read_csv(folderOfData+'behavioural data.csv',header=0)

trainData, testData = train_test_split(creditData, train_size=0.7)
allFeatures = []
'''
逾期类型的特征在行为评分卡（预测违约行为）中，一般是非常显著的变量。
通过设定时间窗口，可以衍生以下类型的逾期变量：
'''
# 考虑过去1个月，3个月，6个月，12个月
for t in [1,3,6,12]:
    # 1，过去t时间窗口内的最大逾期状态
    allFeatures.append('maxDelqL'+str(t)+"M")
    trainData['maxDelqL'+str(t)+"M"] = trainData.apply(lambda x: DelqFeatures(x,t,'max delq'),axis=1)

    # 2，过去t时间窗口内的，M0,M1,M2的次数
    allFeatures.append('M0FreqL' + str(t) + "M")
    trainData['M0FreqL' + str(t) + "M"] = trainData.apply(lambda x: DelqFeatures(x,t,'M0 times'),axis=1)

    allFeatures.append('M1FreqL' + str(t) + "M")
    trainData['M1FreqL' + str(t) + "M"] = trainData.apply(lambda x: DelqFeatures(x, t, 'M1 times'), axis=1)

    allFeatures.append('M2FreqL' + str(t) + "M")
    trainData['M2FreqL' + str(t) + "M"] = trainData.apply(lambda x: DelqFeatures(x, t, 'M2 times'), axis=1)



'''
额度使用率类型特征在行为评分卡模型中，通常是与违约高度相关的
'''
# 考虑过去1个月，3个月，6个月，12个月
for t in [1,3,6,12]:
    # 1，过去t时间窗口内的最大月额度使用率
    allFeatures.append('maxUrateL' + str(t) + "M")
    trainData['maxUrateL' + str(t) + "M"] = trainData.apply(lambda x: UrateFeatures(x,t,'max utilization rate'),axis = 1)

    # 2，过去t时间窗口内的平均月额度使用率
    allFeatures.append('avgUrateL' + str(t) + "M")
    trainData['avgUrateL' + str(t) + "M"] = trainData.apply(lambda x: UrateFeatures(x, t, 'mean utilization rate'),
                                                            axis=1)

    # 3，过去t时间窗口内，月额度使用率增加的月份。该变量要求t>1
    if t > 1:
        allFeatures.append('increaseUrateL' + str(t) + "M")
        trainData['increaseUrateL' + str(t) + "M"] = trainData.apply(lambda x: UrateFeatures(x, t, 'increase utilization rate'),
                                                                axis=1)

'''
还款类型特征也是行为评分卡模型中常用的特征
'''
# 考虑过去1个月，3个月，6个月，12个月
for t in [1,3,6,12]:
    # 1，过去t时间窗口内的最大月还款率
    allFeatures.append('maxPayL' + str(t) + "M")
    trainData['maxPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'max payment ratio'),
                                                            axis=1)

    # 2，过去t时间窗口内的最小月还款率
    allFeatures.append('minPayL' + str(t) + "M")
    trainData['minPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'min payment ratio'),
                                                          axis=1)

    # 3，过去t时间窗口内的平均月还款率
    allFeatures.append('avgPayL' + str(t) + "M")
    trainData['avgPayL' + str(t) + "M"] = trainData.apply(lambda x: PaymentFeatures(x, t, 'mean payment ratio'),
                                                          axis=1)


'''
类别型变量：过去t时间内最大的逾期状态
需要检查与bad的相关度
'''
trainData.groupby(['maxDelqL1M'])['label'].mean()
trainData.groupby(['maxDelqL3M'])['label'].mean()
trainData.groupby(['maxDelqL6M'])['label'].mean()
trainData.groupby(['maxDelqL12M'])['label'].mean()

for x in allFeatures:
    for y in allFeatures:
        if x!=y:
            print(x,'   ',y,'   ',np.corrcoef(trainData[x],trainData[y])[0,1])

trainData_bk = trainData.copy()
############################
#   2, 分箱，计算WOE并编码   #
############################
#金额型的变量已经转换成比率，故可以将其移除出待选变量
numericalFeatures = trainData.columns.tolist()
removed_features_1 = ['OS_'+str(i) for i in range(13)]
removed_features_2 = ['Payment_'+str(i) for i in range(1,13)]
removed_features_3 = ['Spend_'+str(i) for i in range(1,13)]
removed_features = ['Loan_Amount','CUST_ID', 'label']+removed_features_1+removed_features_2+removed_features_3

for fea in removed_features:
    numericalFeatures.remove(fea)


#对于取值较少的数值型变量，无需分箱。但是要保证bad rate单调。对于不单调的情形需要合并
#取固定值的变量，需要移除
removed_features_4 = []
short_features = []
long_features = []
for fea in numericalFeatures:
    if len(set(trainData[fea])) == 1:
        removed_features_4.append(fea)
    elif len(set(trainData[fea])) <=5:
        short_features.append(fea)
    else:
        long_features.append(fea)

for fea in removed_features_4:
    numericalFeatures.remove(fea)

#检查取值少的变量的bad rate单调性
var_cutoff = {}
unchanged_features = []
bin_features = []
for col in short_features:
    BRM = BadRateMonotone(trainData, col, 'label')
    #如果col对应的badrate已经单调，则col不做任何处理，直接带入下一阶段的WOE编码
    if BRM:
        unchanged_features.append(col)
        continue
    min_val, max_val = min(trainData[col]),max(trainData[col])
    bin_merged = Monotone_Merge(trainData, 'label', col)
    for bin in bin_merged:
        if (min_val in bin or max_val in bin) and len(bin)>=2:
            bin_merged.remove(bin)
    bin_merged = [sorted(i) for i in bin_merged]
    cutOffPoints = [i[0] for i in bin_merged]
    col1 = str(col) + '_Bin'
    trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints))
    var_cutoff[col] = cutOffPoints
    bin_features.append(col1)


'''
对于数值型变量，需要先分箱，再计算WOE、IV
分箱的结果需要满足：
1，箱数不超过5
2，bad rate单调
3，每箱占比不低于5%
'''

deleted_features = []   #将处理过的变量删除，防止对后面建模的干扰
for col in long_features:
    print("{} is in processing".format(col))
    col1 = str(col) + '_Bin'

    #(1),用卡方分箱法进行分箱，并且保存每一个分割的端点。例如端点=[10,20,30]表示将变量分为x<10,10<x<20,20<x<30和x>30.
    cutOffPoints = ChiMerge(trainData, col, 'label')
    var_cutoff[col] = cutOffPoints
    trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints))

    #(2), check whether the bad rate is monotone
    BRM = BadRateMonotone(trainData, col1, 'label')
    if BRM: continue
    bin_merged = Monotone_Merge(trainData, 'label', col1)
    removed_index = []
    for bin in bin_merged:
        if len(bin)>1:
            indices = [int(b.replace('Bin ','')) for b in bin]
            removed_index = removed_index+indices[0:-1]
    removed_point = [cutOffPoints[k] for k in removed_index]
    for p in removed_point:
        cutOffPoints.remove(p)

    trainData[col1] = trainData[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
    #(3), 分箱后再次检查是否有单一的值占比超过90%。如果有，删除该变量
    maxPcnt = MaximumBinPcnt(trainData, col1)
    if maxPcnt > 0.9:
        del trainData[col1]
        print('we delete {} because the maximum bin occupies more than 90%'.format(col))
    else:
        var_cutoff[col] = cutOffPoints
        bin_features.append(col1)

var_IV = {}  # save the IV values for binned features       #将IV值保留和WOE值
var_WOE = {}
for col in bin_features+unchanged_features:
    WOE_IV = CalcWOE(trainData, col, 'label')
    var_IV[col] = WOE_IV['IV']
    var_WOE[col] = WOE_IV['WOE']



#  选取IV高于0.1的变量
high_IV = [(k,v) for k,v in var_IV.items() if v >= 0.1]
high_IV_sorted = sorted(high_IV, key=lambda k: k[1],reverse=True)
high_IV_features = [i[0] for i in high_IV_sorted]
high_IV_values = [i[1] for i in high_IV_sorted]
for var in high_IV_features:
    newVar = var+"_WOE"
    trainData[newVar] = trainData[var].map(lambda x: var_WOE[var][x])

plt.bar(x=range(len(high_IV_values)), height = high_IV_values)



##############################
#   3, 单变量分析和多变量分析   #
##############################
'''
单变量分析：比较两两线性相关性。如果相关系数的绝对值高于阈值，剔除IV较低的一个
'''

removed_var  = []
roh_thresould = 0.6
for i in range(len(high_IV_features)-1):
    if high_IV_features[i] not in removed_var:
        x1 = high_IV_features[i]+"_WOE"
        for j in range(i+1,len(high_IV_features)):
            if high_IV_features[j] not in removed_var:
                x2 = high_IV_features[j] + "_WOE"
                roh = np.corrcoef([trainData[x1], trainData[x2]])[0, 1]
                if abs(roh) >= roh_thresould:
                    print('the correlation coeffient between {0} and {1} is {2}'.format(x1, x2, str(roh)))
                    if var_IV[high_IV_features[i]] > var_IV[high_IV_features[j]]:
                        removed_var.append(high_IV_features[j])
                    else:
                        removed_var.append(high_IV_features[i])

multivariates = [i+"_WOE" for i in high_IV_features if i not in removed_var]


dfData = trainData[multivariates].corr()
plt.subplots(figsize=(9, 9)) # 设置画面大小
sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")

### (iii）检查是否有变量与其他所有变量的VIF > 10
X = np.mat(trainData[multivariates])
vif_list = []
for i in range(len(multivariates)):
    vif = variance_inflation_factor(X, i)
    vif_list.append(vif)
    if vif > 10:
        print("Warning: the vif for {0} is {1}".format(high_IV_features_2[i], vif))

plt.bar(x=range(len(vif_list)), height = sorted(vif_list,reverse=True))
'''
这一步没有发现有多重共线性
'''

################################
#   4, 建立逻辑回归模型预测违约   #
################################
X = trainData[multivariates]
X['intercept'] = [1] * X.shape[0]
y = trainData['label']
logit = sm.Logit(y, X)
logit_result = logit.fit()
pvalues = logit_result.pvalues
params = logit_result.params
fit_result = pd.concat([params,pvalues],axis=1)
fit_result.columns = ['coef','p-value']
'''
                         coef       p-value
M1FreqL3M_Bin_WOE   -0.830779  0.000000e+00
M2FreqL3M_Bin_WOE   -0.348182  1.091914e-04
M1FreqL1M_WOE       -0.099797  1.509626e-03
Delq1_11_WOE        -0.219532  2.382291e-09
Delq1_10_WOE        -0.260345  2.095687e-09
minPayL3M_Bin_WOE    0.260876  1.689237e-06
Delq3_10_WOE        -0.285445  9.591112e-03
M2FreqL1M_WOE        0.283429  9.376783e-03
maxDelqL12M_Bin_WOE -0.059895  2.667239e-01
avgPayL1M_Bin_WOE   -0.474006  9.747042e-18
intercept           -1.778571  0.000000e+00

发现变量minPayL3M_Bin_WOE和M2FreqL1M_WOE的系数为正；maxDelqL12M_Bin_WOE系数不显著。要单独检验这三个变量。
'''

sm.Logit(y, trainData['minPayL3M_Bin_WOE']).fit().params  # -0.825956
sm.Logit(y, trainData['M2FreqL1M_WOE']).fit().params  # -29.060739
sm.Logit(y, trainData['maxDelqL12M_Bin_WOE']).fit().pvalues  #  0.00
'''
三个变量分别做单变量回归时，系数的符号和p值都是符合要求的。说明依然有多重共线性的存在。
'''
#向前筛选法，要求每一步选进来的变量需要使得所有变量的系数的符号和p值同时符合要求
selected_var = [multivariates[0]]
for var in multivariates[1:]:
    try_vars = selected_var+[var]
    X_temp = trainData[try_vars].copy()
    X_temp['intercept'] = [1] * X_temp.shape[0]
    LR = sm.Logit(y, X_temp).fit()
    pvals, params = LR.pvalues, LR.params
    del params['intercept']
    if max(pvals)<0.1 and max(params)<0:
        selected_var.append(var)

LR.summary2()
'''
------------------------------------------------------------------
                   Coef.  Std.Err.    z     P>|z|   [0.025  0.975]
------------------------------------------------------------------
M1FreqL3M_Bin_WOE -0.8372   0.0207 -40.3954 0.0000 -0.8778 -0.7966
M2FreqL3M_Bin_WOE -0.2194   0.0492  -4.4635 0.0000 -0.3158 -0.1231
M1FreqL1M_WOE     -0.0902   0.0310  -2.9124 0.0036 -0.1509 -0.0295
Delq1_11_WOE      -0.1556   0.0337  -4.6240 0.0000 -0.2216 -0.0897
Delq1_10_WOE      -0.1666   0.0391  -4.2645 0.0000 -0.2431 -0.0900
Delq3_10_WOE      -0.4313   0.0766  -5.6326 0.0000 -0.5813 -0.2812
avgPayL1M_Bin_WOE -0.3650   0.0498  -7.3360 0.0000 -0.4625 -0.2675
intercept         -1.7851   0.0206 -86.5733 0.0000 -1.8255 -1.7446
'''

X_final = trainData[selected_var]
X_final['intercept'] = [1] * X_final.shape[0]
y_pred = LR.predict(X_final)
scores = Prob2Score(y_pred, 400, 50)
pred_result = pd.DataFrame({'label':trainData['label'], 'scores':scores})
KS(pred_result, 'scores', 'label',plot=False)  #0.600
roc_auc_score(trainData['label'], y_pred)   #0.823
ROC_AUC(pred_result, 'scores', 'label')

Kendal's Tau