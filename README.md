[https://www.kaggle.com/c/titanic/overview](https://www.kaggle.com/c/titanic/overview)

### 0 模型准备

```
#Essentials
import pandas as pd
import numpy as np

#Plots
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

#Models
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
import xgboost
import lightgbm
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from mlxtend.regressor import StackingCVRegressor
from mlxtend.classifier import StackingCVClassifier

#Misc
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn import neighbors
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

#ignore warnings
import warnings
warnings.filterwarnings("ignore")


# path='C:\\Users\\sunsharp\\Desktop\\kaggle\\Titanic\\'
path=r'/Users/ranmo/Desktop/kaggle/Titanic/'
```
```
#===========
# 函数定义
#===========

#1、训练模型
def model_eval(model,X_train,y_train):
    acc=[]
    kf=model_selection.KFold(10,random_state=10)
    for train,test in kf.split(X_train):
        X_train1 = X_train.iloc[train]
        y_train1 = y_train.iloc[train]
        X_test1 = X_train.iloc[test]
        y_test1 = y_train.iloc[test]

        y_pred1=model.fit(X_train1,y_train1).predict(X_test1)
        acc.append(metrics.accuracy_score(y_pred1,y_test1))  #准确性作为指标
    print(acc)
    print('acc_mean:',np.mean(acc))
    print('acc_std:',np.std(acc))
    print('mean-1.5*std:',np.mean(acc)-1.5*(np.std(acc)))            #可以调整，这里是发现测试集和实际成绩的差异为1.5个标准差
    print()
    print()
    return ('mean-1.5*std:',np.mean(acc)-1.5*(np.std(acc)))

#2、模型预测
def model_predict(model,X_test,outpath):
    Survived=model.predict(X_test)
    df_pre=pd.DataFrame({'PassengerId':X_test.index,'Survived':Survived.astype('int')}).set_index('PassengerId')  #转化为整数
    df_pre.to_csv('%stest_pred.csv'%outpath)
```
- exploration
```
#看一下特征情况
df_train=pd.read_csv('%sdata/train.csv'%path)
df_train=df_train.set_index('PassengerId')
df_train.columns

#看一下非数值类的数据，需要进行独热编码
df_train.dtypes[df_train.dtypes==object]
df_train.Ticket.value_counts()
df_train.Name.value_counts()
df_train.Cabin.value_counts()

#sex和embark肯定要独热编码，ticket、name、cabin也直接去掉？因为分类数目太多。。

#=============
#联合处理训练集和测试集
#=============

df_test=pd.read_csv('%sdata//test.csv'%path)
df_test=df_test.set_index('PassengerId')
temp=pd.concat([df_train,df_test],axis=0).drop(['Cabin','Name','Ticket'],axis=1)
temp=pd.get_dummies(temp)

#缺失值处理
for i in temp.columns:
    if temp[i].value_counts().sum()!=len(temp):
        print(i)
#age 用train的mean填充
temp['Age'].fillna(df_train['Age'].mean(),inplace=True)
#fare 用train的mean填充
temp['Fare'].fillna(df_train['Fare'].mean(),inplace=True)


#分离训练集和测试集
df_train_modified=temp.loc[df_train.index.to_list()]
df_test_modified=temp.loc[df_test.index.to_list()]

X_train=df_train_modified.drop('Survived',axis=1)
y_train=df_train_modified['Survived']
X_test=df_test_modified.drop('Survived',axis=1)
```
```
#=============
#跑一下基本模型
#=============

clf=xgboost.XGBClassifier(objective='binary:logistic',n_jobs=-1,eval_metric='error',random_state=10)
model_eval(clf,X_train,y_train)

#模型预测
outpath='%s//clf//0103//base//'%path
clf.fit(X_train,y_train)
model_predict(clf,X_test,outpath)  

#成绩0.77033
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-b69a4ba276116e7d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 一、EDA

##### 1.1 因子EDA

```
f,ax=plt.subplots(1,2,figsize=(18,8))  #分图要在分的时候就设置图片大小
df_train.Survived.value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
sns.countplot(x='Survived',data=df_train,ax=ax[1])
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-014e3d6f99fae0c4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 等级
```
- 阶层等级
df_train[['Pclass','Survived']].pivot_table(aggfunc=['count','sum'],index='Pclass')


f,ax=plt.subplots(1,2,figsize=(18,8))
df_train['Pclass'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title('Number Of Passengers By Pclass')
ax[0].set_ylabel('Count')

sns.countplot('Pclass',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('Pclass:Survived vs Dead')
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-29a41e46b81f5f12.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 性别
```
##性别
df_train.groupby(['Sex','Survived'])['Survived'].count()
f,ax=plt.subplots(1,2,figsize=(18,8))
df_train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=df_train,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-a419bf15b3d2c780.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
#结合等级看一下

sns.factorplot('Pclass','Survived',hue='Sex',data=df_train)
# 上层阶层的女性更容易存活，显而易见
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-46084df6696074a6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 年龄
```
#年龄
f,ax=plt.subplots(1,2,figsize=(20,10))
df_train[df_train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
df_train[df_train['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
#其实这里仍然应该画提琴图，看比例的
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-aec00e7ce6e893cf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
#年龄
#结合阶层和性别看一下
print(df_train.Age.describe())

f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=df_train,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
sns.violinplot("Sex","Age", hue="Survived", data=df_train,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-4f4604eea7397aed.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 港口
```
# Embarked登机口

pd.crosstab([df_train.Embarked,df_train.Pclass],[df_train.Sex,df_train.Survived],margins=True).style.background_gradient(cmap='summer_r')
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-39698d2c02e599d8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
# Embarked港口
sns.factorplot('Embarked','Survived',data=df_train)
fig=plt.gcf()
fig.set_size_inches(5,3)
plt.show()
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-0c4bc778549e69a3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 平级亲属
```
#SibSip 平级亲属
f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('SibSp','Survived',data=df_train,ax=ax[0])
ax[0].set_title('SibSp vs Survived')
sns.factorplot('SibSp','Survived',data=df_train,ax=ax[1])
ax[1].set_title('SibSp vs Survived')
plt.close(2)
plt.show()
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-e2441640b76f9d51.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```
#SibSip 平级亲属
pd.crosstab(df_train.SibSp,df_train.Pclass).style.background_gradient(cmap='summer_r')
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-bbda818cb7c446c9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 直系亲属
```
#Parch 直系亲属
f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('Parch','Survived',data=df_train,ax=ax[0])
ax[0].set_title('Parch vs Survived')
sns.factorplot('Parch','Survived',data=df_train,ax=ax[1])
ax[1].set_title('Parch vs Survived')
plt.close(2)
plt.show()
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-7206dec19fd0036f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
#Parch 直系亲属
pd.crosstab(df_train.Parch,df_train.Pclass).style.background_gradient(cmap='summer_r')
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-ac756afe79e91c5a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 票价
```
#票价

f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(df_train[df_train['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(df_train[df_train['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(df_train[df_train['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')

```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-f7e3a0f6c4abbdc3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 1.2 相关性

- 相关性
```
#相关性

sns.heatmap(df_train.corr(),annot=True,linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(10,8)
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-74fdad6eec54014f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 1.3 总结
![image.png](https://upload-images.jianshu.io/upload_images/18032205-ddea8e62a3f72cb5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 二、特征工程

```
# 合并train和test，但同时要防止leak

df_train=pd.read_csv('%sdata//train.csv'%path).set_index('PassengerId')
df_test=pd.read_csv('%sdata//test.csv'%path).set_index('PassengerId')
df_feature=pd.concat([df_train,df_test],axis=0)
df_feature
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-0fc4c754fea914a3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
#提取name中的关键性别字，并进行性别转化

main_title_map = {'Lady': 'Mrs', 'Mme': 'Mrs', 'Dona': 'Mrs', 'the Countess': 'Mrs',
         'Ms': 'Miss', 'Mlle': 'Miss',
         'Sir': 'Mr', 'Major': 'Mr', 'Capt': 'Mr', 'Jonkheer': 'Mr', 'Don': 'Mr', 'Col': 'Mr', 'Rev': 'Mr', 'Dr': 'Mr'}

def get_title(full_name):
    return full_name.split(',')[1].split('.')[0].strip()

def set_title_mr(data):
    titles = data['Name'].apply(get_title).replace(main_title_map)
    data['Title_Mr'] = titles.apply(lambda title: 1 if title == 'Mr' else 0)
    
set_title_mr(df_feature)   #为mr则为1，否则为0


male_map = {'male': 1, 'female': 0}
df_feature['Male']=df_feature['Sex'].map(male_map) #为male则为1，否则为0

df_feature
#所以有的虽然是male，但不是mr，主要是因为是孩子，这里也是做了一个简单的保留（妇幼优先政策）
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-5f5773d9644c3a86.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
# Placss_3
class3_map = {1: 0, 2: 0, 3: 1}
df_feature['Pclass_3'] = df_feature['Pclass'].map(class3_map)

df_feature
#三个阶层只分为两类

```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-c1f5cd8b7afef3d0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
# 组队的存活率，eda中发现有家人或伴侣的，会影响其存活率

# 第一步、family name
def extract_lastname(full_name):
    return full_name.split(',')[0]

df_feature['Last name'] = df_feature['Name'].apply(extract_lastname)


#第二步，组队的存活率，eda中发现有家人或伴侣的，会影响其存活率。所以利用LAST NAME 或者是票号来判别是否是组队，优先用票号来判定，比LAST NAME要更可行
print(df_feature['Last name'].value_counts())
print(df_feature['Ticket'].value_counts())

def prepare_family_ticket_frequencies_actual(data, is_train, train, last_names_survival, tickets_survival):
    data['Known family/ticket survived %'] = np.NaN

    mean_train_survive = train['Survived'].mean()

    
    for i in data.index:
        did_survive = 1 if (is_train == 1) and (train.loc[i, 'Survived'] == 1) else 0
        last_name = data.loc[i, 'Last name']
        ticket = data.loc[i, 'Ticket']
        family_survived = np.NaN
        ticket_survived = np.NaN

        if last_name in last_names_survival:
            last_name_count, last_name_sum = last_names_survival[last_name]
            if last_name_count > is_train:
                family_survived = (last_name_sum - did_survive) / (last_name_count - is_train)

        if ticket in tickets_survival:
            ticket_count, ticket_sum = tickets_survival[ticket]
            if ticket_count > is_train:
                ticket_survived = (ticket_sum - did_survive) / (ticket_count - is_train)
        if np.isnan(family_survived) == False:
            if np.isnan(ticket_survived) == False:
                data.loc[i, 'Known family/ticket survived %'] = (family_survived + ticket_survived) / 2
            else:
                data.loc[i, 'Known family/ticket survived %'] = family_survived
        elif np.isnan(ticket_survived) == False:
            data.loc[i, 'Known family/ticket survived %'] = ticket_survived
        else:
            data.loc[i, 'Known family/ticket survived %'] = mean_train_survive
            
    
def prepare_family_ticket_frequencies(train, test):
    last_names_survival = {}
    for last_name in (set(train['Last name'].unique()) | set(test['Last name'].unique())):
        last_name_survived = train[train['Last name'] == last_name]['Survived']
        if last_name_survived.shape[0] > 0:
            last_names_survival[last_name] = (last_name_survived.count(), last_name_survived.sum())

    tickets_survival = {}
    for ticket in (set(train['Ticket'].unique()) | set(test['Ticket'].unique())):
        ticket_survived = train[train['Ticket'] == ticket]['Survived']
        if ticket_survived.shape[0] > 0:
            tickets_survival[ticket] = (ticket_survived.count(), ticket_survived.sum())

    prepare_family_ticket_frequencies_actual(train, True, train, last_names_survival, tickets_survival)
    prepare_family_ticket_frequencies_actual(test, False, train, last_names_survival, tickets_survival)
    
df_train1=df_feature.loc[df_train.index]
df_test1=df_feature.loc[df_test.index]
prepare_family_ticket_frequencies(df_train1,df_test1)
df_feature=pd.concat([df_train1,df_test1],axis=0)

df_feature
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-489744d77cbb05f8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
#保留有效特征

feat_to_train_on = ['Title_Mr', 'Male', 'Pclass_3', 'Known family/ticket survived %']   #如果要进一步的话可以把港口一起考虑进去，但实际港口反映的是阶层差异
X_train=df_feature[feat_to_train_on].loc[df_train.index]
y_train=df_feature['Survived'].loc[df_train.index]
X_test=df_feature[feat_to_train_on].loc[df_test.index]

#里面的年龄特征其实不是很重要，主要是孩童的存活率比较高，处理在了Title_Mr里边。
```

```
#检查是否有缺失值
print(X_train.isnull().any())
print('---')
print(y_train.isnull().any())
print('---')
print(X_test.isnull().any())
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-59fe836d4431d168.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
X_train.to_csv('%sdata//X_train.csv'%path)
y_train.to_csv('%sdata//y_train.csv'%path)
X_test.to_csv('%sdata//X_test.csv'%path)
```

### 三、训练模型

```
X_train=pd.read_csv('%sdata//X_train.csv'%path).set_index('PassengerId')
y_train=pd.read_csv('%sdata//y_train.csv'%path).set_index('PassengerId')
X_test=pd.read_csv('%sdata//X_test.csv'%path).set_index('PassengerId')
```

```
#=============
#跑一下基本模型
#=============

clf=xgboost.XGBClassifier(objective='binary:logistic',n_jobs=-1,eval_metric='error',random_state=10)
model_eval(clf,X_train,y_train)

#模型预测
outpath='%s//clf//0108//base//'%path
clf.fit(X_train,y_train)
model_predict(clf,X_test,outpath)  

#成绩0.81339 
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-785355230018b57b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

```
# 函数定义：
def find_cv(model,X_train,y_train,param_test):
    kflod = model_selection.KFold(10,random_state=10)
    model_cv=model_selection.GridSearchCV(model,param_test,cv=kflod,n_jobs=-1,scoring='accuracy')  
    #理论上是应该会保持和model_eval中的k折结果一样才对，但又不一样，不知道问题在哪里。。。
    model_cv.fit(X_train,y_train)

    print("mean-1.5*std:",model_cv.cv_results_['mean_test_score']-1.5*model_cv.cv_results_['std_test_score'])  #结果是开根号值

    print()
    temp=model_cv.cv_results_['mean_test_score']-1.5*model_cv.cv_results_['std_test_score']
    print(temp.max())
    print(model_cv.cv_results_['params'][temp.argmax()])  #本来有一个best_params，但是这里没有采用
```

##### 3.1 single model ：lr\svm\rf\gdbt\lightgbm\xgb
- lr
```
#模型预测
clf_lr=linear_model.LogisticRegression (random_state=10,C=0.1,penalty='l1').fit(X_train,y_train) 
model_eval(clf_lr,X_train,y_train)
#out
outpath='%s//clf//0108//lr_l1l2//'%path
model_predict(clf_lr,X_test,outpath)  

#实际成绩0.81818
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-17438bb18f100fe9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- svm
```
#模型预测
clf_svm=svm.SVC(C=0.5,gamma=0.8)
model_eval(clf_svm,X_train,y_train)
#out
clf_svm.fit(X_train,y_train)
outpath='%s//clf//0108//svm//'%path
model_predict(clf_svm,X_test,outpath)  

#实际成绩0.81818
```
- rf
```
#模型预测
clf_rf=ensemble.RandomForestClassifier(n_estimators=10,n_jobs=-1,random_state=10,max_depth=5,max_features=0.5).fit(X_train,y_train) 
model_eval(clf_rf,X_train,y_train)
#out
outpath='%s//clf//0108//rf//'%path
model_predict(clf_rf,X_test,outpath)  

#实际成绩0.80861
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-d476768a7344e1f8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- gdbt
```
#模型预测
clf_gdbt=ensemble.GradientBoostingClassifier(random_state=10,
                                             max_depth=5,
                                             min_weight_fraction_leaf=5e-05,
                                             subsample=1,
                                             max_features=0.8,
                                             n_estimators=500,
                                             learning_rate=0.1)
model_eval(clf_gdbt,X_train,y_train)
#out
clf_gdbt.fit(X_train,y_train)
outpath='%s//clf//0108//gdbt//'%path
model_predict(clf_gdbt,X_test,outpath)  

#实际成绩0.80861
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-50ecdd74a4439a2d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- lgbm
```
#模型预测
clf_lgbm=lightgbm.LGBMClassifier(random_state=10,
                                 max_depth=2,
                                 min_child_samples=20,
                                 min_child_weight=1e-05,
                                 min_split_gain=0,
                                 subsample=0.7,
                                 colsample_bytree=0.8,
                                 subsample_freq=0,
                                 reg_alpha=0,
                                 reg_lambda=0,
                                 learning_rate=0.1,
                                 n_estimators=100
                                )
model_eval(clf_lgbm,X_train,y_train)
#out
clf_lgbm.fit(X_train,y_train)
outpath='%s//clf//0108//lgbm//'%path
model_predict(clf_lgbm,X_test,outpath)  

#实际成绩0.81818
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-46795f0fd198a5d8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- xgbt
```
#模型预测
clf_xgbt=xgboost.XGBClassifier(random_state=10,
                                 max_depth=3,
                                 min_child_weight=1,
                                 min_split_gain=0,
                                 subsample=1,
                                 colsample_bytree=1,
                                 reg_alpha=1,
                                 reg_lambda=0,
                                 learning_rate=0.1,
                                 n_estimators=100
                                )
model_eval(clf_xgbt,X_train,y_train)
# out
clf_xgbt.fit(X_train,y_train)
outpath='%s//clf//0108//xgbt//'%path
model_predict(clf_xgbt,X_test,outpath)  

#实际成绩0.81339
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-d974128e54a6c919.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

##### 3.2 stacking and blending

```
clf_lr=linear_model.LogisticRegression (random_state=10,C=0.1,penalty='l1').fit(X_train,y_train) 
clf_svm=svm.SVC(C=0.5,gamma=0.8)
clf_rf=ensemble.RandomForestClassifier(n_estimators=10,n_jobs=-1,random_state=10,max_depth=5,max_features=0.5).fit(X_train,y_train) 
clf_gdbt=ensemble.GradientBoostingClassifier(random_state=10,
                                             max_depth=5,
                                             min_weight_fraction_leaf=5e-05,
                                             subsample=1,
                                             max_features=0.8,
                                             n_estimators=500,
                                             learning_rate=0.1)
clf_lgbm=lightgbm.LGBMClassifier(random_state=10,
                                 max_depth=2,
                                 min_child_samples=20,
                                 min_child_weight=1e-05,
                                 min_split_gain=0,
                                 subsample=0.7,
                                 colsample_bytree=0.8,
                                 subsample_freq=0,
                                 reg_alpha=0,
                                 reg_lambda=0,
                                 learning_rate=0.1,
                                 n_estimators=100
                                )
clf_xgbt=xgboost.XGBClassifier(random_state=10,
                                 max_depth=3,
                                 min_child_weight=1,
                                 min_split_gain=0,
                                 subsample=1,
                                 colsample_bytree=1,
                                 reg_alpha=1,
                                 reg_lambda=0,
                                 learning_rate=0.1,
                                 n_estimators=100
                                )
```

```
#模型预测
clf_stack=StackingCVClassifier(classifiers=(clf_lr,clf_svm,clf_rf,clf_gdbt,clf_lgbm),
                               meta_classifier=xgboost.XGBClassifier(random_state=10),
                               random_state=10)  #必须给一个初始值，但是不影响网格寻优 
model_eval(clf_stack,X_train,y_train)
# out
clf_stack.fit(X_train,y_train)
outpath='%s//clf//0108//stacking//'%path
model_predict(clf_stack,X_test,outpath)  

#实际成绩0.80861
```
![image.png](https://upload-images.jianshu.io/upload_images/18032205-93c8adeac079616e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 四、 submission mix

**分数：**
- lr,0.81818
- svm,0.81818
- rf,0.80861
- gdbt,0.80861
- lgbm, 0.81818
- xgbt,.81339
- stacking, 0.80861
```
sub1=pd.read_csv('%s//clf//0108//lr_l1l2//test_pred.csv'%path).set_index('PassengerId')
sub2=pd.read_csv('%s//clf//0108//svm//test_pred.csv'%path).set_index('PassengerId')
sub3=pd.read_csv('%s//clf//0108//rf//test_pred.csv'%path).set_index('PassengerId')
sub4=pd.read_csv('%s//clf//0108//gdbt//test_pred.csv'%path).set_index('PassengerId')
sub5=pd.read_csv('%s//clf//0108//lgbm//test_pred.csv'%path).set_index('PassengerId')
sub6=pd.read_csv('%s//clf//0108//xgbt//test_pred.csv'%path).set_index('PassengerId')
sub7=pd.read_csv('%s//clf//0108//stacking//test_pred.csv'%path).set_index('PassengerId')
```
```
outpath='%s//clf//0108//mix1//'%path 
Survived=(sub1.Survived+sub2.Survived+sub3.Survived+sub4.Survived+sub5.Survived+sub6.Survived+sub7.Survived)/7
df_pre=pd.DataFrame({'PassengerId':X_test.index,'Survived':Survived.astype('int')}).set_index('PassengerId')  #转化为整数
df_pre.to_csv('%stest_pred.csv'%outpath)
#实际成绩0.81339
```
```
outpath='%s//clf//0108//mix2//'%path 
Survived=(sub1.Survived+sub3.Survived+sub6.Survived)/3
df_pre=pd.DataFrame({'PassengerId':X_test.index,'Survived':Survived.astype('int')}).set_index('PassengerId')  #转化为整数
df_pre.to_csv('%stest_pred.csv'%outpath)
#实际成绩0.81339
```
**最优成绩0.81818。**

