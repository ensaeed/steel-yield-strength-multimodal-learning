from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
P=Path(__file__).parent
rows=[]
def add(name,source,y,p,fold,protocol):
 y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);f=np.asarray(fold)
 if len(y)!=213 or not (np.isfinite(y).all() and np.isfinite(p).all()):raise ValueError((name,'invalid predictions'))
 for label,sel in [('POOLED',np.ones(len(y),dtype=bool))]+[(str(k),f==k) for k in sorted(set(f))]:
  rows.append(dict(model=name,source=source,protocol=protocol,fold=label,n=int(sel.sum()),rmse_mpa=float(np.sqrt(mean_squared_error(y[sel],p[sel]))),mae_mpa=float(mean_absolute_error(y[sel],p[sel])),r2=float(r2_score(y[sel],p[sel]))))
a=pd.read_csv(P/'steel_alloy_grouped_cv_assignments.csv')
old=pd.read_csv(P/'steel_tabular_xgboost_oof_predictions.csv')
assert (old.fold.values==a.fold.values).all()
add('XGBoost initial (disputed)','steel_tabular_xgboost_oof_predictions.csv',old['Yield Stress, [MPa]'],old.predicted_yield_mpa,old.fold,'tabular, original pipeline unverified')
r=pd.read_csv(P/'steel_xgboost_reconciliation_predictions.csv')
add('XGBoost reproducible','steel_xgboost_reconciliation_predictions.csv',r.measured_yield_mpa,r.reproduced_prediction_mpa,r.fold,'tabular, median imputation and scaling')
b=pd.read_csv(P/'steel_additional_tabular_baseline_oof_predictions.csv')
for model,g in b.groupby('model',sort=False):add(str(model),'steel_additional_tabular_baseline_oof_predictions.csv',g.measured_yield_mpa,g.predicted_yield_mpa,g.fold,'tabular, separate training implementation')
c=pd.read_csv(P/'steel_xgboost_ablation_predictions.csv')
for col in [x for x in c if x.endswith('_predicted_mpa')]:add(col.replace('_predicted_mpa',''),'steel_xgboost_ablation_predictions.csv',c.measured_yield_mpa,c[col],c.fold,'tabular feature ablation')
t=pd.read_csv(P/'steel_texture_sensitivity_predictions.csv')
for model,g in t.groupby('model',sort=False):add(str(model),'steel_texture_sensitivity_predictions.csv',g.measured,g.predicted,g.fold,'handcrafted image texture; unverified image pairing')
mean=pd.read_csv(P/'steel_train_mean_grouped_baseline_predictions.csv');print('Mean columns:',mean.columns.tolist())
for col in ['predicted_yield_mpa','prediction_mpa','predicted']:
 if col in mean.columns:
  ycol=next(x for x in ['measured_yield_mpa','measured','Yield Stress, [MPa]'] if x in mean.columns)
  add('Training-mean baseline','steel_train_mean_grouped_baseline_predictions.csv',mean[ycol],mean[col],mean.fold,'training-fold target mean');break
out=pd.DataFrame(rows);out.to_csv(P/'steel_saved_prediction_metrics_audit.csv',index=False)
print(out[out.fold=='POOLED'][['model','n','rmse_mpa','mae_mpa','r2']].to_string(index=False))
