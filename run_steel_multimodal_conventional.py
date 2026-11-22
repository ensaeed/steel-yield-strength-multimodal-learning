"""Exploratory grouped-CV conventional models on frozen ResNet18 plus 16 tabular features.
Not specimen-validated; image pairing is alloy-folder representative only.
"""
import argparse,io,zipfile,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from PIL import Image
from torchvision import models
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
FEATURES=['Fe','C','Cr','Mn','Si','Ni','Co','Mo','W','Nb','Al','P','Cu','Ti','V','Temperature, [C]']
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--assignments',required=True);p.add_argument('--original-csv',required=True);p.add_argument('--weights',required=True);p.add_argument('--output-dir',default='/mnt/data/steel_conventional_multimodal_results');args=p.parse_args()
 out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 a=pd.read_csv(args.assignments);o=pd.read_csv(args.original_csv)
 keys=['Alloy','Temperature, [C]','Yield Stress, [MPa]']
 for k in keys[1:]:a[k]=pd.to_numeric(a[k],errors='coerce').round(8);o[k]=pd.to_numeric(o[k],errors='coerce').round(8)
 a['_occ']=a.groupby(keys,dropna=False).cumcount();o['_occ']=o.groupby(keys,dropna=False).cumcount();o['_row']=np.arange(len(o))
 m=a.merge(o[keys+['_occ','_row']],on=keys+['_occ'],how='left',validate='one_to_one',sort=False)
 if m['_row'].isna().any():raise ValueError('Unmatched records')
 df=o.iloc[m['_row'].astype(int).to_numpy()].reset_index(drop=True)
 paths=a['Matched_Image_Archive_Path'].tolist();folds=a.fold.astype(int).to_numpy();groups=a.group_id.astype(str).to_numpy();y=df['Yield Stress, [MPa]'].to_numpy(float)
 state=torch.load(args.weights,map_location='cpu',weights_only=True);backbone=models.resnet18(weights=None);backbone.load_state_dict(state);backbone.fc=torch.nn.Identity();backbone.eval();torch.set_num_threads(2)
 tf=models.ResNet18_Weights.IMAGENET1K_V1.transforms(); embs={}
 with zipfile.ZipFile(args.archive) as z,torch.inference_mode():
  for path in dict.fromkeys(paths):
   im=Image.open(io.BytesIO(z.read(path))).convert('RGB');embs[path]=backbone(tf(im).unsqueeze(0)).numpy().ravel()
 image_x=np.stack([embs[p] for p in paths]);tab=df[FEATURES].apply(pd.to_numeric,errors='coerce')
 metrics=[];preds=[]
 for name in ['random_forest','linear_regression']:
  outpred=np.full(len(df),np.nan)
  for fold in sorted(set(folds)):
   tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
   assert not (set(groups[tr]) & set(groups[te]));assert not (set(np.array(paths)[tr]) & set(np.array(paths)[te]))
   median=tab.iloc[tr].median().fillna(0);train=tab.iloc[tr].fillna(median).fillna(0);test=tab.iloc[te].fillna(median).fillna(0)
   scaler=StandardScaler().fit(train);xtr=np.column_stack((image_x[tr],scaler.transform(train)));xte=np.column_stack((image_x[te],scaler.transform(test)))
   if name=='random_forest':model=RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=2)
   else:model=LinearRegression()
   model.fit(xtr,y[tr]);pr=model.predict(xte);outpred[te]=pr
   metrics.append(dict(model=name,fold=int(fold),n_validation=len(te),rmse_mpa=np.sqrt(mean_squared_error(y[te],pr)),mae_mpa=mean_absolute_error(y[te],pr),r2=r2_score(y[te],pr)))
  assert np.isfinite(outpred).all()
  preds.extend(dict(model=name,fold=int(folds[i]),group_id=groups[i],measured_yield_mpa=y[i],predicted_yield_mpa=outpred[i]) for i in range(len(df)))
  print(name,dict(rmse=np.sqrt(mean_squared_error(y,outpred)),mae=mean_absolute_error(y,outpred),r2=r2_score(y,outpred)),flush=True)
 pd.DataFrame(metrics).to_csv(out/'conventional_multimodal_fold_metrics.csv',index=False)
 pd.DataFrame(preds).to_csv(out/'conventional_multimodal_oof_predictions.csv',index=False)
 summary=[]
 for name in ['random_forest','linear_regression']:
  subset=pd.DataFrame(preds);subset=subset[subset.model==name];actual=subset.measured_yield_mpa;pred=subset.predicted_yield_mpa
  summary.append(dict(model=name,rmse_mpa=np.sqrt(mean_squared_error(actual,pred)),mae_mpa=mean_absolute_error(actual,pred),r2=r2_score(actual,pred)))
 pd.DataFrame(summary).to_csv(out/'conventional_multimodal_pooled_metrics.csv',index=False)
 print('Saved:',out)
if __name__=='__main__':main()
