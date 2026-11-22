"""Reviewer 1 sensitivity experiment: frozen ImageNet ResNet18 + tabular XGBoost.
Run in Google Colab with uploaded ZIP and alloy-grouped assignments CSV.
Important: alloy-level image pairing is unverified; results are exploratory only.
"""
import argparse, io, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from xgboost import XGBRegressor

FEATURES=['Fe','C','Cr','Mn','Si','Ni','Co','Mo','W','Nb','Al','P','Cu','Ti','V','Temperature, [C]']

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--archive',required=True)
 ap.add_argument('--assignments',required=True)
 ap.add_argument('--original-csv',required=True)
 ap.add_argument('--output-dir',default='steel_multimodal_results')
 args=ap.parse_args()
 out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 assigned=pd.read_csv(args.assignments)
 original=pd.read_csv(args.original_csv)
 # Resolve original CSV rows by identifying keys, including repeated identical records.
 # Earlier alignment source_record_index is a reset position, NOT original CSV index.
 keys=['Alloy','Temperature, [C]','Yield Stress, [MPa]']
 a=assigned.copy(); o=original.copy()
 for key in keys[1:]:
  a[key]=pd.to_numeric(a[key],errors='coerce').round(8)
  o[key]=pd.to_numeric(o[key],errors='coerce').round(8)
 a['_occurrence']=a.groupby(keys,dropna=False).cumcount()
 o['_occurrence']=o.groupby(keys,dropna=False).cumcount()
 o['_original_row']=np.arange(len(o))
 merged=a.merge(o[keys+['_occurrence','_original_row']],on=keys+['_occurrence'],how='left',validate='one_to_one',sort=False)
 if merged['_original_row'].isna().any():
  raise ValueError('Could not map all assignment records to original CSV. Check identifiers and duplicate records.')
 df=original.iloc[merged['_original_row'].astype(int).to_numpy()].reset_index(drop=True)
 paths=assigned['Matched_Image_Archive_Path'].tolist()
 folds=assigned['fold'].astype(int).to_numpy()
 groups=assigned['group_id'].astype(str).to_numpy()
 y=df['Yield Stress, [MPa]'].to_numpy(dtype=float)
 assert np.isfinite(y).all()
 # The weights are mandatory. Do not silently fall back to random weights.
 weights=models.ResNet18_Weights.IMAGENET1K_V1
 backbone=models.resnet18(weights=weights)
 backbone.fc=torch.nn.Identity(); backbone.eval()
 for p in backbone.parameters():p.requires_grad_(False)
 tf=weights.transforms()
 unique_paths=list(dict.fromkeys(paths))
 embeddings={}
 with zipfile.ZipFile(args.archive) as z, torch.inference_mode():
  for path in unique_paths:
   with z.open(path) as f: im=Image.open(io.BytesIO(f.read())).convert('RGB')
   embeddings[path]=backbone(tf(im).unsqueeze(0)).cpu().numpy().ravel()
 image_X=np.stack([embeddings[p] for p in paths]).astype(np.float32)
 assert image_X.shape==(len(df),512)
 tab=df[FEATURES].apply(pd.to_numeric,errors='coerce').copy()
 rows=[]; predictions=np.full(len(df),np.nan)
 for fold in sorted(np.unique(folds)):
  train=np.where(folds!=fold)[0]; test=np.where(folds==fold)[0]
  assert not set(groups[train]).intersection(groups[test])
  assert not set(np.array(paths)[train]).intersection(np.array(paths)[test])
  # Fit all preprocessing only on training partition.
  median=tab.iloc[train].median().fillna(0)
  tr=tab.iloc[train].fillna(median).fillna(0)
  te=tab.iloc[test].fillna(median).fillna(0)
  scaler=StandardScaler().fit(tr)
  xtr=np.hstack([image_X[train],scaler.transform(tr)])
  xte=np.hstack([image_X[test],scaler.transform(te)])
  model=XGBRegressor(n_estimators=180,max_depth=4,learning_rate=.06,subsample=.8,
    colsample_bytree=.8,random_state=42,n_jobs=2,objective='reg:squarederror')
  model.fit(xtr,y[train]); pred=model.predict(xte)
  predictions[test]=pred
  rows.append({'fold':int(fold),'n_validation':len(test),'rmse_mpa':np.sqrt(mean_squared_error(y[test],pred)),
   'mae_mpa':mean_absolute_error(y[test],pred),'r2':r2_score(y[test],pred)})
 assert np.isfinite(predictions).all()
 metrics=pd.DataFrame(rows)
 metrics.to_csv(out/'multimodal_xgboost_fold_metrics.csv',index=False)
 result=assigned.copy();result['measured_yield_mpa']=y;result['predicted_yield_mpa']=predictions
 result.to_csv(out/'multimodal_xgboost_oof_predictions.csv',index=False)
 pooled={'rmse_mpa':np.sqrt(mean_squared_error(y,predictions)),
 'mae_mpa':mean_absolute_error(y,predictions),'r2':r2_score(y,predictions)}
 pd.DataFrame([pooled]).to_csv(out/'multimodal_xgboost_pooled_metrics.csv',index=False)
 print('Fold metrics:\n',metrics.to_string(index=False))
 print('Pooled:',pooled)
 print('Mean ± sample SD:\n',metrics[['rmse_mpa','mae_mpa','r2']].agg(['mean','std']).to_string())
 print('Results saved to',out.resolve())
 print('CAUTION: Images are alloy-folder representatives, not verified specimen-level pairs.')

if __name__=='__main__':main()
