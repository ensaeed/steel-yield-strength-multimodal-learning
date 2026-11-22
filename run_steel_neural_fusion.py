"""Exploratory frozen ResNet18 + tabular MLP late-fusion, alloy grouped 5-fold CV.
Images are unverified alloy-level representatives, NOT specimen-matched data.
"""
import io,zipfile,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from PIL import Image
from torchvision import models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from run_multimodal_xgboost_local import FEATURES

ROOT=Path('/mnt/data'); OUT=ROOT/'steel_neural_fusion_results';OUT.mkdir(exist_ok=True)
a=pd.read_csv(ROOT/'steel_alloy_grouped_cv_assignments.csv');o=pd.read_csv(ROOT/'steel_original_for_run.csv')
keys=['Alloy','Temperature, [C]','Yield Stress, [MPa]']
for d in [a,o]:
 for k in keys[1:]:d[k]=pd.to_numeric(d[k],errors='coerce').round(8)
 d['_occ']=d.groupby(keys,dropna=False).cumcount()
o['_row']=np.arange(len(o));m=a.merge(o[keys+['_occ','_row']],on=keys+['_occ'],validate='one_to_one',sort=False)
assert m['_row'].notna().all()
d=o.iloc[m['_row'].astype(int).to_numpy()].reset_index(drop=True)
paths=a['Matched_Image_Archive_Path'].tolist(); folds=a.fold.to_numpy(); groups=a.group_id.astype(str).to_numpy(); y=d['Yield Stress, [MPa]'].to_numpy(float)
assert np.isfinite(y).all()
weights=models.ResNet18_Weights.IMAGENET1K_V1; net=models.resnet18(weights=None)
net.load_state_dict(torch.load(ROOT/'resnet18-f37072fd.pth',map_location='cpu',weights_only=True));net.fc=torch.nn.Identity();net.eval(); tf=weights.transforms()
emb={}
with zipfile.ZipFile(ROOT/'Prediction of mechanical properties-20261116T230131Z-1-001.zip') as z,torch.inference_mode():
 for p in dict.fromkeys(paths):
  im=Image.open(io.BytesIO(z.read(p))).convert('RGB');emb[p]=net(tf(im).unsqueeze(0)).numpy().ravel()
image=np.stack([emb[p] for p in paths]).astype('float32'); tab=d[FEATURES].apply(pd.to_numeric,errors='coerce')
class Fusion(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.tab=torch.nn.Sequential(torch.nn.Linear(16,32),torch.nn.ReLU(),torch.nn.Linear(32,16),torch.nn.ReLU())
  self.head=torch.nn.Sequential(torch.nn.Linear(528,128),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(128,1))
 def forward(self,im,t):return self.head(torch.cat([im,self.tab(t)],1)).squeeze(1)

preds=np.full(len(d),np.nan);rows=[]
torch.set_num_threads(2)
for f in sorted(set(folds)):
 tr=np.where(folds!=f)[0];te=np.where(folds==f)[0]
 assert not set(groups[tr])&set(groups[te]);assert not set(np.array(paths)[tr])&set(np.array(paths)[te])
 med=tab.iloc[tr].median().fillna(0);xt=tab.fillna(med).fillna(0)
 scaler=StandardScaler().fit(xt.iloc[tr]);x=scaler.transform(xt).astype('float32')
 ym=y[tr].mean();ys=y[tr].std();ys=max(ys,1.)
 torch.manual_seed(42+int(f));np.random.seed(42+int(f));random.seed(42+int(f))
 model=Fusion();opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
 it=torch.tensor(image[tr]);tt=torch.tensor(x[tr]);yt=torch.tensor(((y[tr]-ym)/ys).astype('float32'))
 rng=np.random.default_rng(42+int(f))
 for epoch in range(120):
  model.train()
  for ix in np.array_split(rng.permutation(len(tr)),max(1,int(np.ceil(len(tr)/32)))):
   opt.zero_grad();loss=torch.nn.functional.mse_loss(model(it[ix],tt[ix]),yt[ix]);loss.backward();opt.step()
 model.eval()
 with torch.no_grad(): p=model(torch.tensor(image[te]),torch.tensor(x[te])).numpy()*ys+ym
 preds[te]=p
 rows.append({'fold':int(f),'n_validation':len(te),'rmse_mpa':np.sqrt(mean_squared_error(y[te],p)),'mae_mpa':mean_absolute_error(y[te],p),'r2':r2_score(y[te],p)})
 print('fold',f,rows[-1],flush=True)
assert np.isfinite(preds).all()
pd.DataFrame(rows).to_csv(OUT/'neural_fusion_fold_metrics.csv',index=False)
a.assign(measured_yield_mpa=y,predicted_yield_mpa=preds).to_csv(OUT/'neural_fusion_oof_predictions.csv',index=False)
pooled={'rmse_mpa':np.sqrt(mean_squared_error(y,preds)),'mae_mpa':mean_absolute_error(y,preds),'r2':r2_score(y,preds)}
pd.DataFrame([pooled]).to_csv(OUT/'neural_fusion_pooled_metrics.csv',index=False)
print('POOLED',pooled)
