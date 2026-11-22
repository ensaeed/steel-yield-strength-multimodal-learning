import pandas as pd,numpy as np
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from pathlib import Path
P=Path('/mnt/data');a=pd.read_csv(P/'steel_alloy_grouped_cv_assignments.csv');o=pd.read_csv(P/'tensile-properties-from-jeffs-ml-sheet-20220208(1).csv')
k=['Alloy','Temperature, [C]','Yield Stress, [MPa]']
for d in [a,o]:
 for c in k[1:]:d[c]=pd.to_numeric(d[c],errors='coerce').round(8)
 d['_occurrence']=d.groupby(k,dropna=False).cumcount()
o['_original_row']=np.arange(len(o));m=a.merge(o[k+['_occurrence','_original_row']],on=k+['_occurrence'],how='left',validate='one_to_one',sort=False)
assert m['_original_row'].notna().all()
d=o.iloc[m['_original_row'].astype(int).to_numpy()].reset_index(drop=True)
features=['Fe','C','Cr','Mn','Si','Ni','Co','Mo','W','Nb','Al','P','Cu','Ti','V','Temperature, [C]'];X=d[features].apply(pd.to_numeric,errors='coerce');y=d['Yield Stress, [MPa]'].to_numpy(float);folds=a.fold.to_numpy();groups=a.group_id.to_numpy()
old=pd.read_csv(P/'steel_tabular_xgboost_oof_predictions.csv');assert np.allclose(y,old['Yield Stress, [MPa]'])
rows=[];pred=np.full(len(y),np.nan)
for f in sorted(np.unique(folds)):
 tr=np.flatnonzero(folds!=f);te=np.flatnonzero(folds==f);assert not set(groups[tr])&set(groups[te]);imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(X.iloc[tr]);sc=StandardScaler().fit(imp.transform(X.iloc[tr]));xt=sc.transform(imp.transform(X.iloc[tr]));xv=sc.transform(imp.transform(X.iloc[te]));model=XGBRegressor(n_estimators=180,max_depth=4,learning_rate=.06,subsample=.8,colsample_bytree=.8,random_state=42,n_jobs=2,objective='reg:squarederror');model.fit(xt,y[tr]);pred[te]=model.predict(xv);rows.append({'fold':int(f),'n_validation':len(te),'rmse':np.sqrt(mean_squared_error(y[te],pred[te])),'mae':mean_absolute_error(y[te],pred[te]),'r2':r2_score(y[te],pred[te]),'max_abs_prediction_difference_vs_previous':np.max(np.abs(pred[te]-old.predicted_yield_mpa.to_numpy()[te]))})
rows.append({'fold':'POOLED','n_validation':len(y),'rmse':np.sqrt(mean_squared_error(y,pred)),'mae':mean_absolute_error(y,pred),'r2':r2_score(y,pred),'max_abs_prediction_difference_vs_previous':np.max(np.abs(pred-old.predicted_yield_mpa.to_numpy()))})
pd.DataFrame(rows).to_csv(P/'steel_xgboost_reconciliation_metrics.csv',index=False)
pd.DataFrame({'alloy':a.Alloy,'fold':folds,'source_csv_row':m['_original_row'],'measured_yield_mpa':y,'reproduced_prediction_mpa':pred,'previous_prediction_mpa':old.predicted_yield_mpa,'prediction_difference_mpa':pred-old.predicted_yield_mpa}).to_csv(P/'steel_xgboost_reconciliation_predictions.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
