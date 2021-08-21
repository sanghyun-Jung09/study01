# 문자와 숫자컬럼구분
import pandas as pd
import numpy as np

df=pd.read_csv('C:/Users/sanghyun/Desktop/dacon parking/train.csv')
test= pd.read_csv('C:/Users/sanghyun/Desktop/dacon parking/test.csv')
age_info=pd.read_csv('C:/Users/sanghyun/Desktop/dacon parking/age_gender_info.csv')

df.info()


df['임대보증금'][df['임대보증금']=='-']=np.nan
df['임대보증금']=df['임대보증금'].astype('float')
df['임대료'][df['임대료']=='-']=np.nan
df['임대료']=df['임대료'].astype('float')

test['임대보증금'][test['임대보증금']=='-']=np.nan
test['임대보증금']=test['임대보증금'].astype('float')
test['임대료'][test['임대료']=='-']=np.nan
test['임대료']=test['임대료'].astype('float')

df['전용면적'] = df['전용면적']//5*5
test['전용면적'] = test['전용면적']//5*5

#전용면적 상하한적용(15~100)
idx = df[df['전용면적']>100].index
df.loc[idx, '전용면적'] = 100
idx = test[test['전용면적']>100].index
test.loc[idx, '전용면적'] = 100

idx = df[df['전용면적']<15].index
df.loc[idx, '전용면적'] = 15
idx = test[test['전용면적']<15].index
test.loc[idx, '전용면적'] = 15

# 결국 최종 제출을 위해서는 단지코드별 grouping이 필요함. 예제코드에서는 그러한 이유로 차이가
# 발생하는 변수는 모두 버리고 모델에 학습을 시킴
# 단지코드별 중복없는게 전용면적, 전용면적별세대수, 임대보증금, 임대료가 있음. 
# 이경우 한 단지내 개별 집집마다 사용가능한 평수가 다르고 임대료 임대보증금이 다르다는 이야기이기
# 때문에 이경우에는 최빈값을 따라가는게 상식적으로 옳다고 봄

# 결측치 대치
df.info()

df=df.fillna(0)


df1=df[df.columns.difference(['단지코드','등록차량수'])]

r_code=df.단지코드
df1_target=df['등록차량수']

test1=test[test.columns.difference(['단지코드'])]
r_code_t=test.단지코드


df1.info()

df1=pd.merge(df1,age_info,on='지역')
test=pd.merge(test,age_info,on='지역')


from sklearn.preprocessing import StandardScaler

cat_col=[ i for i in df1.columns.values if df1[i].dtype == 'object']

num_col=[ j for j in df1.columns.values if df1[j].dtype in ['int64','float64'] ]

test_cat_col=[ i for i in test1.columns.values if test1[i].dtype == 'object']

test_num_col=[ j for j in test1.columns.values if test1[j].dtype in ['int64','float64'] ]




scaler=StandardScaler()

df1_cat=pd.get_dummies(df1.loc[:,cat_col])
df1_num=pd.DataFrame(scaler.fit_transform(df1.loc[:,num_col]),columns=num_col)

df1_sol=pd.concat([df1_cat,df1_num],axis=1)


test_cat=pd.get_dummies(test.loc[:,test_cat_col])
test_num=pd.DataFrame(scaler.fit_transform(test.loc[:,test_num_col]),columns=test_num_col)

test_sol=pd.concat([test_cat,test_num],axis=1)


len(df1_sol.columns)
len(test_sol.columns)

# test의 범주형 변수의 수가 train데이터보다 작음 변수축소를 해야 모델에 적용가능

df1_sol=df1_sol.reindex(test_sol.columns.values,axis=1)
len(df1_sol.columns)

df1_sol=pd.concat([r_code,df1_sol,df1_target],axis=1)
test_sol=pd.concat([r_code_t,test_sol],axis=1)
#df1_target 이 차량등록수임


df1_sol.dtypes
f1 = lambda x : x.mode().median()

df1_sol=df1_sol.groupby('단지코드').agg(f1).reset_index()

df1_target=df1_sol.등록차량수

df1_sol=df1_sol.iloc[:,1:-1]

test_sol=test_sol.groupby('단지코드').agg(f1).reset_index()
test_sol=test_sol.iloc[:,1:]


from xgboost import XGBRegressor

xgbreg=XGBRegressor()

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

params= {'max_depth':[2,8,15,20,40],'n_estimators':[10,50,100],'learning_rate':[0.1,0.2,0.3]}
gridcv=GridSearchCV(xgbreg,params,scoring='neg_mean_squared_error',cv=5)

gridcv.fit(df1_sol,df1_target)

gridcv.best_params_
# {'learning_rate': 0.2, 'max_depth': 8, 'n_estimators': 50}

xgbreg=XGBRegressor(learning_rate=0.1,max_depth=2,n_estimators=10)

scores=cross_val_score(xgbreg,df1_sol,df1_target,scoring='neg_mean_squared_error',cv=10)
scores=-scores
scores.mean() #-37483.654683348665

df1_target.max()

xgbreg.fit(df1_sol,df1_target)
y_pred=xgbreg.predict(test_sol)

submission=pd.read_csv('C:/Users/sanghyun/Desktop/dacon parking/sample_submission.csv')


# 제출용
submission['num']=y_pred


submission.to_csv('baseline.csv', index=False)



# 연습 #################################################################################################
#full_pipeline= ColumnTransformer([
#        ('std', StandardScaler(), num_col ),
#        ('cat', OneHotEncoder(handle_unknown='ignore',sparse=False), cat_col),
#        ])
#
#full_pipeline_1= ColumnTransformer([
#        ('std', StandardScaler(), test_num_col ),
#        ('cat', OneHotEncoder(handle_unknown='ignore',sparse=False), test_cat_col),
#        ])
#
#
#
#df1_ary=full_pipeline.fit_transform(df1)
#test1_ary=full_pipeline.fit_transform(test1)
#
#
#df1_ary
#test1_ary
#
#df1=pd.concat([r_code,pd.DataFrame(df1_ary),df1_target],axis=1,ignore_index=True)
#test1=pd.concat([r_code_t,pd.DataFrame(test1_ary)],axis=1,ignore_index=True)
#
#f1 = lambda x: x.mode().median()
#
## 각 단지코드별 변수당 최빈값 추출
#df1_1=df1.groupby(0).agg(f1).reset_index()
#test1_1=test1.groupby(0).agg(f1).reset_index()
#
#
#df1_1.info()
#
#df1_1_target=df1_1.iloc[:,53]
#df1_1=df1_1.iloc[:,1:53]
#
#test1_1=test1_1.iloc[:,1:47]
#
#from xgboost import XGBRegressor
#
#xgbreg=XGBRegressor()
#
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import cross_val_score
#
#params={'max_depth':[2,4,6,8,10],'learning_rate':[0.1,0.2,0.3]}
#
#gridcv=GridSearchCV(xgbreg,params,scoring='neg_mean_squared_error',cv=5)
#
#gridcv.fit(np.matrix(df1_1),df1_1_target)
#
#gridcv.best_estimator_
#
#xgbreg.fit(np.matrix(df1_1),df1_1_target)
#
#xgbreg=XGBRegressor(learning_rate=0.2,max_depth=8,n_estimators=100, n_jobs=8)
#
#xgbreg.predict(test1_1)
#
#
#
#
#
#
#df.columns
#
#col_a=['단지코드', '총세대수', '임대건물구분', '공급유형', '전용면적', '전용면적별세대수', '공가수',
#       '자격유형', '임대보증금', '임대료', '도보 10분거리 내 지하철역 수(환승노선 수 반영)',
#       '도보 10분거리 내 버스정류장 수', '단지내주차면수', '등록차량수']
#
#df.info()
#
#age_info=pd.read_csv('C:/Users/sanghyun/Desktop/dacon parking/age_gender_info.csv')
#
#
#df1=pd.merge(df,age_info,on='지역')
#test=pd.merge(test,age_info,on='지역')
#
#
#df1.info()
#
#df.loc[df.지역=='경상북도','지역'].mode()
#
#import matplotlib.pyplot as plt
#
#plt.rc('font',family='Malgun Gothic')
#
#age_info.hist()
#
#
#from sklearn.preprocessing import LabelEncoder
#from sklearn.pipeline import Pipeline
#from sklearn.impute import SimpleImputer
#
#from sklearn.compose import ColumnTransformer
#    
#lb_encoder=LabelEncoder()
#eh_encoder=OneHotEncoder()
#
#
#
#train_target=df1['등록차량수']
#
#
#df1=df1[df1.columns.difference(['등록차량수'])]
#
#
#
#
#
## 연습
##a1=eh_encoder.fit_transform(df1.loc[:,cat_col]).toarray()
##
##a1.shape
##
##pd.DataFrame(a1)
##
##pd.get_dummies(df1.loc[:,cat_col])
#
#
#
##여기까지 공부해봤을 때, 생각하는 것은 “굳이 OneHotEncoder를 쓸 필요가 있는건가?” 였다.
## 왜냐하면, OneHotEncoder는 모든 특성을 범주형이라고 가정할 뿐더러 get_dummies는 문자형
## 특성만을 기본값으로 가변수로 바꿀 수 있었으며, 필요에 따라 colums매개변수에 열을 추가할 수
## 있었다. 하지만, ColumnTransformer때문에 사용할 가치가 충분하다고 생각이 바뀌었다.
#df1.info()
#
#df1.loc[df1['임대료'].isnull(),:]['자격유형'].unique()
#
#
## 값없음
#df1.loc[df1['자격유형']=='D',:]['임대료'].sum() # 0
#df1.loc[df1['자격유형']=='D','임대료']=0.0
#
## 값있음
#df1.loc[df1['자격유형']=='K',:]['임대료'].sum() 
#df1.loc[df1['자격유형']=='K',:]['임대료'].hist()
#
#df1.loc[df1['자격유형']=='H',:]['임대료'].sum() 
#df1.loc[df1['자격유형']=='H',:]['임대료'].hist() 
#
#df1.loc[df1['자격유형']=='A',:]['임대료'].sum() 
#df1.loc[df1['자격유형']=='A',:]['임대료'].hist()
#
#
#    
#    
#df1_prepared=full_pipeline.fit_transform(df1)
#
#
#
#
#df1.columns
#test.columns
#
#a2=df1.columns.values.tolist()
#
#test=test.reindex(a2,axis=1)
#test_prepared=full_pipeline.transform(test)
#
#from xgboost import XGBRegressor
#
#xgbreg=XGBRegressor()
#
#xgbreg.fit(df1_prepared,train_target)
#
#
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import cross_val_score
#
#
## 하이퍼파라미터탐색
#params = {'max_depth':[2,4,6,8,10],'learning_rate':[0.1,0.2,0.3]}
#grid_search= GridSearchCV(xgbreg,params,cv=5,scoring='neg_mean_squared_error')
#
#grid_search.fit(df1_prepared,train_target)
#
#
#grid_search.best_params_
#
#xgbreg1=XGBRegressor(learning_rate=0.2,max_depth=4)
#
## 교차검증을 이용한 평가
#scores= cross_val_score(xgbreg1,df1_prepared,train_target,scoring='neg_mean_squared_error',cv=10)
#
#sol= -scores
#
#sol.mean()
#
#xgbreg1.fit(df1_prepared,train_target)
#pred=xgbreg1.predict(test_prepared)
#
#submission=pd.read_csv('C:/Users/sanghyun/Desktop/dacon parking/sample_submission.csv')
#
#submission['num']=pred
#
#len(df1['단지코드'].unique())
#
#
#len(test.단지코드.unique())
#
#
#len(submission.code.unique())
#
#
#df.loc[df.단지코드=='C2483','임대료']
#df.전용면적.describe()
#df.columns
#
#
#
#
#
#
#c1=df.columns.values













