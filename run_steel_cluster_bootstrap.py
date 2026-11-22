import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
root=Path('/mnt/data'); out=root/'steel_cluster_uncertainty';out.mkdir(exist_ok=True)
base=pd.read_csv(root/'steel_xgboost_reconciliation_predictions.csv')
mm=pd.read_csv(root/'steel_multimodal_results/multimodal_xgboost_oof_predictions.csv')
nn=pd.read_csv(root/'steel_neural_fusion_results/neural_fusion_oof_predictions.csv')
img=pd.read_csv(root/'steel_image_only_results/image_only_xgboost_oof_predictions.csv')
for name,d in [('multimodal',mm),('neural',nn),('image_only',img)]:
 assert len(d)==len(base)==213
 assert np.allclose(d.measured_yield_mpa,base.measured_yield_mpa)
 assert np.array_equal(d.fold.to_numpy(),base.fold.to_numpy())
 assert np.array_equal(d.group_id.astype(str).to_numpy(),base.alloy.astype(str).to_numpy())
y=base.measured_yield_mpa.to_numpy();group=base.alloy.to_numpy()
models={'tabular_xgboost':base.reproduced_prediction_mpa.to_numpy(),'multimodal_xgboost':mm.predicted_yield_mpa.to_numpy(),'neural_fusion':nn.predicted_yield_mpa.to_numpy(),'image_only_xgboost':img.predicted_yield_mpa.to_numpy()}
unique=np.unique(group); indices=[np.where(group==g)[0] for g in unique]
rng=np.random.default_rng(20261118); B=10000
# Resample alloys, not rows. Preserve all rows within each selected alloy.
boot=np.empty((B,len(models)))
for b in range(B):
 sample=np.concatenate([indices[i] for i in rng.integers(0,len(unique),len(unique))])
 for j,pred in enumerate(models.values()):
  boot[b,j]=np.sqrt(np.mean((y[sample]-pred[sample])**2))
summary=[]
for j,(name,pred) in enumerate(models.items()):
 summary.append(dict(model=name,rmse=np.sqrt(np.mean((y-pred)**2)),bootstrap_ci_low=np.percentile(boot[:,j],2.5),bootstrap_ci_high=np.percentile(boot[:,j],97.5),n_alloys=len(unique),bootstrap_replicates=B))
pd.DataFrame(summary).to_csv(out/'cluster_bootstrap_rmse_ci.csv',index=False)
comparisons=[]
for j,name in enumerate(list(models)[1:],1):
 diffs=boot[:,j]-boot[:,0]
 observed=summary[j]['rmse']-summary[0]['rmse']
 comparisons.append(dict(comparison=f'{name} minus tabular_xgboost',observed_rmse_difference_mpa=observed,ci_low=np.percentile(diffs,2.5),ci_high=np.percentile(diffs,97.5),bootstrap_fraction_difference_positive=np.mean(diffs>0),n_alloys=len(unique)))
pd.DataFrame(comparisons).to_csv(out/'paired_cluster_bootstrap_differences.csv',index=False)
alloy=[]
for g,ix in zip(unique,indices):
 row={'alloy':g,'n':len(ix)}
 for name,pred in models.items():row[name+'_rmse']=np.sqrt(np.mean((y[ix]-pred[ix])**2))
 alloy.append(row)
pd.DataFrame(alloy).to_csv(out/'alloy_level_errors.csv',index=False)
print(pd.DataFrame(summary).round(3).to_string(index=False))
print(pd.DataFrame(comparisons).round(3).to_string(index=False))
