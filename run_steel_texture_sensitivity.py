"""Exploratory alloy-level texture sensitivity; NOT a replacement for pretrained ResNet18."""
import zipfile, io, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
P=Path('/mnt/data'); A=pd.read_csv(P/'steel_alloy_grouped_cv_assignments.csv'); O=pd.read_csv(P/'tensile-properties-from-jeffs-ml-sheet-20220208(1).csv')
keys=['Alloy','Temperature, [C]','Yield Stress, [MPa]']
for d in (A,O):
 for c in keys[1:]: d[c]=pd.to_numeric(d[c],errors='coerce').round(8)
 d['_occurrence']=d.groupby(keys,dropna=False).cumcount()
O['_original_row']=np.arange(len(O)); M=A.merge(O[keys+['_occurrence','_original_row']],on=keys+['_occurrence'],how='left',validate='one_to_one',sort=False)
assert M._original_row.notna().all();D=O.iloc[M._original_row.astype(int).to_numpy()].reset_index(drop=True)
cols=['Fe','C','Cr','Mn','Si','Ni','Co','Mo','W','Nb','Al','P','Cu','Ti','V','Temperature, [C]']
T=D[cols].apply(pd.to_numeric,errors='coerce');y=D['Yield Stress, [MPa]'].to_numpy(float);fold=A.fold.to_numpy();paths=A.Matched_Image_Archive_Path.tolist()
features={}; hashes={}
with zipfile.ZipFile(P/'Prediction of mechanical properties-20261116T230131Z-1-001.zip') as z:
 for path in dict.fromkeys(paths):
  raw=z.read(path); hashes[path]=hashlib.sha256(raw).hexdigest(); im=Image.open(io.BytesIO(raw)).convert('L').resize((128,128)); arr=np.asarray(im,dtype=np.uint8)
  lbp=local_binary_pattern(arr,8,1,method='uniform'); h=np.histogram(lbp,bins=np.arange(11),density=True)[0]
  glcm=graycomatrix(arr//8,distances=[1,3],angles=[0,np.pi/2],levels=32,symmetric=True,normed=True)
  g=np.concatenate([graycoprops(glcm,p).ravel() for p in ('contrast','dissimilarity','homogeneity','energy','correlation')])
  hist=np.histogram(arr,bins=16,range=(0,256),density=True)[0]
  features[path]=np.r_[arr.mean(),arr.std(),np.percentile(arr,[10,25,50,75,90]),hist,h,g]
I=np.array([features[p] for p in paths]); assert len(set(hashes.values()))==len(hashes)
rows=[]; preds=[]
for label in ['texture_only','tabular_plus_texture','tabular_only_recheck']:
 pred=np.full(len(y),np.nan)
 for f in sorted(set(fold)):
  tr=np.where(fold!=f)[0];te=np.where(fold==f)[0]
  assert not set(A.group_id.iloc[tr]) & set(A.group_id.iloc[te]); assert not set(np.array(paths)[tr]) & set(np.array(paths)[te])
  imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(T.iloc[tr]); sc=StandardScaler().fit(imp.transform(T.iloc[tr])); ttr=sc.transform(imp.transform(T.iloc[tr]));tte=sc.transform(imp.transform(T.iloc[te]))
  if label=='texture_only': xtr,xte=I[tr],I[te]
  elif label=='tabular_plus_texture': xtr,xte=np.c_[ttr,I[tr]],np.c_[tte,I[te]]
  else:xtr,xte=ttr,tte
  model=XGBRegressor(n_estimators=180,max_depth=4,learning_rate=.06,subsample=.8,colsample_bytree=.8,random_state=42,n_jobs=2,objective='reg:squarederror')
  model.fit(xtr,y[tr]);pred[te]=model.predict(xte)
  rows.append(dict(model=label,fold=int(f),n=len(te),rmse=np.sqrt(mean_squared_error(y[te],pred[te])),mae=mean_absolute_error(y[te],pred[te]),r2=r2_score(y[te],pred[te])))
 rows.append(dict(model=label,fold='POOLED',n=len(y),rmse=np.sqrt(mean_squared_error(y,pred)),mae=mean_absolute_error(y,pred),r2=r2_score(y,pred)))
 preds.append(pd.DataFrame({'model':label,'fold':fold,'alloy':A.Alloy,'measured':y,'predicted':pred}))
pd.DataFrame(rows).to_csv(P/'steel_texture_sensitivity_metrics.csv',index=False)
pd.concat(preds,ignore_index=True).to_csv(P/'steel_texture_sensitivity_predictions.csv',index=False)
print(pd.DataFrame(rows).query("fold == 'POOLED'").to_string(index=False))
print('Unique image files',len(hashes),'texture features',I.shape[1]);print('CAVEAT: arbitrary alloy-level representative images; not specimen-level mappings.')
