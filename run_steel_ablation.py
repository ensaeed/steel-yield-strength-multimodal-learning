import numpy as np,pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pathlib import Path
P=Path('/mnt/data')
a=pd.read_csv(P/'steel_alloy_grouped_cv_assignments.csv');o=pd.read_csv(P/'tensile-properties-from-jeffs-ml-sheet-20220208(1).csv')
keys=['Alloy','Temperature, [C]','Yield Stress, [MPa]']
for k in keys[1:]:
 a[k]=pd.to_numeric(a[k],errors='coerce').round(8);o[k]=pd.to_numeric(o[k],errors='coerce').round(8)
a['_occurrence']=a.groupby(keys,dropna=False).cumcount();o['_occurrence']=o.groupby(keys,dropna=False).cumcount();o['_original_row']=np.arange(len(o))
m=a.merge(o[keys+['_occurrence','_original_row']],on=keys+['_occurrence'],how='left',validate='one_to_one',sort=False)
assert m['_original_row'].notna().all()
d=o.iloc[m['_original_row'].astype(int).to_numpy()].reset_index(drop=True)
y=d['Yield Stress, [MPa]'].to_numpy(float);folds=a.fold.to_numpy();groups=a.group_id.to_numpy()
features=['Fe','C','Cr','Mn','Si','Ni','Co','Mo','W','Nb','Al','P','Cu','Ti','V']
results=[];predictions=pd.DataFrame({'alloy':a['Alloy'],'fold':folds,'measured_yield_mpa':y})
for name,cols in [('composition_only',features),('temperature_only',['Temperature, [C]'])]:
 X=d[cols].apply(pd.to_numeric,errors='coerce');p=np.full(len(y),np.nan)
 for f in sorted(np.unique(folds)):
  tr=np.where(folds!=f)[0];te=np.where(folds==f)[0]
  assert not set(groups[tr])&set(groups[te]);assert len(te)>0
  imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(X.iloc[tr]);sc=StandardScaler().fit(imp.transform(X.iloc[tr]))
  xtr=sc.transform(imp.transform(X.iloc[tr]));xte=sc.transform(imp.transform(X.iloc[te]))
  model=XGBRegressor(n_estimators=180,max_depth=4,learning_rate=.06,subsample=.8,colsample_bytree=.8,random_state=42,n_jobs=2,objective='reg:squarederror')
  model.fit(xtr,y[tr]);p[te]=model.predict(xte)
  results.append({'model':name,'fold':int(f),'n_validation':len(te),'rmse_mpa':np.sqrt(mean_squared_error(y[te],p[te])),'mae_mpa':mean_absolute_error(y[te],p[te]),'r2':r2_score(y[te],p[te])})
 predictions[name+'_predicted_mpa']=p
 results.append({'model':name,'fold':'POOLED','n_validation':len(y),'rmse_mpa':np.sqrt(mean_squared_error(y,p)),'mae_mpa':mean_absolute_error(y,p),'r2':r2_score(y,p)})
res=pd.DataFrame(results);res.to_csv(P/'steel_xgboost_ablation_metrics.csv',index=False);predictions.to_csv(P/'steel_xgboost_ablation_predictions.csv',index=False)
print(res.to_string(index=False))
