"""Exploratory held-out fold permutation sensitivity for tabular XGBoost.
No micrograph pairing is assumed. Permutation is within held-out folds.
"""
from pathlib import Path
import numpy as np,pandas as pd
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
P=Path('/mnt/data'); a=pd.read_csv(P/'steel_alloy_grouped_cv_assignments.csv'); o=pd.read_csv(P/'tensile-properties-from-jeffs-ml-sheet-20220208(1).csv')
keys=['Alloy','Temperature, [C]','Yield Stress, [MPa]']
for df in (a,o):
 for k in keys[1:]:df[k]=pd.to_numeric(df[k],errors='coerce').round(8)
 df['_occurrence']=df.groupby(keys,dropna=False).cumcount()
o['_row']=np.arange(len(o)); m=a.merge(o[keys+['_occurrence','_row']],on=keys+['_occurrence'],how='left',validate='one_to_one',sort=False)
assert m['_row'].notna().all()
d=o.iloc[m['_row'].astype(int).to_numpy()].reset_index(drop=True)
features=['Fe','C','Cr','Mn','Si','Ni','Co','Mo','W','Nb','Al','P','Cu','Ti','V','Temperature, [C]']
X=d[features].apply(pd.to_numeric,errors='coerce'); y=d['Yield Stress, [MPa]'].to_numpy(float); folds=a.fold.to_numpy(); groups=a.group_id.to_numpy()
rng=np.random.default_rng(20261117); result=[]; pred=np.full(len(y),np.nan)
for fold in sorted(np.unique(folds)):
 tr=np.where(folds!=fold)[0];te=np.where(folds==fold)[0]
 assert not set(groups[tr])&set(groups[te])
 imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(X.iloc[tr]); sc=StandardScaler().fit(imp.transform(X.iloc[tr]))
 xtr=sc.transform(imp.transform(X.iloc[tr]));xte=sc.transform(imp.transform(X.iloc[te]))
 model=XGBRegressor(n_estimators=180,max_depth=4,learning_rate=.06,subsample=.8,colsample_bytree=.8,random_state=42,n_jobs=2,objective='reg:squarederror')
 model.fit(xtr,y[tr]);pred[te]=model.predict(xte)
 base=mean_squared_error(y[te],pred[te])
 for j,feature in enumerate(features):
  increases=[]
  for repeat in range(10):
   z=xte.copy();z[:,j]=z[rng.permutation(len(te)),j]
   increases.append(mean_squared_error(y[te],model.predict(z))-base)
  result.append({'fold':int(fold),'feature':feature,'delta_mse_mean':np.mean(increases),'delta_mse_sd':np.std(increases,ddof=1)})
assert np.isfinite(pred).all()
per=pd.DataFrame(result);per.to_csv(P/'steel_xgboost_permutation_by_fold.csv',index=False)
summary=per.groupby('feature').agg(delta_mse_mean=('delta_mse_mean','mean'),delta_mse_sd_across_folds=('delta_mse_mean','std'),positive_folds=('delta_mse_mean',lambda x:int((x>0).sum()))).sort_values('delta_mse_mean',ascending=False)
summary.to_csv(P/'steel_xgboost_permutation_summary.csv')
alloys=pd.DataFrame({'alloy':a.Alloy,'fold':folds,'measured_yield_mpa':y,'predicted_yield_mpa':pred})
alloys['absolute_error_mpa']=np.abs(y-pred)
alloys.groupby('alloy').agg(n=('absolute_error_mpa','size'),mae_mpa=('absolute_error_mpa','mean'),rmse_mpa=('absolute_error_mpa',lambda x:float(np.sqrt(np.mean(x*x)))),mean_measured_mpa=('measured_yield_mpa','mean')).sort_values('rmse_mpa',ascending=False).to_csv(P/'steel_xgboost_errors_by_alloy.csv')
print('Pooled RMSE',np.sqrt(mean_squared_error(y,pred)))
print(summary.head(10).to_string())
print('Largest alloy errors:')
print(pd.read_csv(P/'steel_xgboost_errors_by_alloy.csv').head(8).to_string(index=False))
