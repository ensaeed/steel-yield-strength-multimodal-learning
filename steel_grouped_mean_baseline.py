import pandas as pd,numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from pathlib import Path
P=Path('/mnt/data');a=pd.read_csv(P/'steel_alloy_grouped_cv_assignments.csv');y=a['Yield Stress, [MPa]'].to_numpy(float);f=a.fold.to_numpy();pred=np.empty(len(y));rows=[]
for k in sorted(np.unique(f)):
 tr=f!=k;te=f==k;pred[te]=np.mean(y[tr]);rows.append(dict(fold=int(k),n=int(te.sum()),train_mean_mpa=float(np.mean(y[tr])),validation_mean_mpa=float(np.mean(y[te])),rmse_mpa=float(np.sqrt(mean_squared_error(y[te],pred[te]))),mae_mpa=float(mean_absolute_error(y[te],pred[te])),r2=float(r2_score(y[te],pred[te]))))
rows.append(dict(fold='POOLED',n=len(y),train_mean_mpa=np.nan,validation_mean_mpa=float(np.mean(y)),rmse_mpa=float(np.sqrt(mean_squared_error(y,pred))),mae_mpa=float(mean_absolute_error(y,pred)),r2=float(r2_score(y,pred))))
pd.DataFrame(rows).to_csv(P/'steel_train_mean_grouped_baseline_metrics.csv',index=False)
pd.DataFrame(dict(alloy=a.Alloy,fold=f,measured_yield_mpa=y,predicted_yield_mpa=pred)).to_csv(P/'steel_train_mean_grouped_baseline_predictions.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
