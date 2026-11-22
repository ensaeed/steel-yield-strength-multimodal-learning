"""Leakage-aware steel yield strength benchmarks; see README for limitations."""
import argparse, io, json, random, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

FEATURES=['Fe','C','Cr','Mn','Si','Ni','Co','Mo','W','Nb','Al','P','Cu','Ti','V','Temperature, [C]']
TARGET='Yield Stress, [MPa]'
KEYS=['Alloy','Temperature, [C]',TARGET]

def load_records(csv_path, assignments_path):
    original=pd.read_csv(csv_path); assignments=pd.read_csv(assignments_path)
    required=KEYS+['Matched_Image_Archive_Path','group_id','fold']
    missing=[x for x in required if x not in assignments]
    if missing: raise ValueError(f'Missing assignment columns: {missing}')
    for col in FEATURES+[TARGET]:
        if col not in original: raise ValueError(f'Missing source column: {col}')
    a=assignments.copy(); b=original.copy()
    for k in KEYS[1:]:
        a[k]=pd.to_numeric(a[k],errors='coerce').round(8)
        b[k]=pd.to_numeric(b[k],errors='coerce').round(8)
    a['_occ']=a.groupby(KEYS,dropna=False).cumcount()
    b['_occ']=b.groupby(KEYS,dropna=False).cumcount()
    b['_row']=np.arange(len(b))
    match=a.merge(b[KEYS+['_occ','_row']],on=KEYS+['_occ'],how='left',sort=False,validate='one_to_one')
    if match['_row'].isna().any(): raise ValueError('Assignment-to-original matching failed; verify duplicate record keys')
    records=original.iloc[match['_row'].astype(int).to_numpy()].reset_index(drop=True)
    assignments=assignments.reset_index(drop=True)
    if records[TARGET].isna().any(): raise ValueError('Missing targets: no target imputation permitted')
    for g, subset in assignments.groupby('group_id'):
        if subset['fold'].nunique()!=1: raise ValueError(f'Group leakage: {g}')
    if len(assignments['fold'].unique())!=5: raise ValueError('Expected exactly five folds')
    return records, assignments

def embeddings_from_zip(paths, archive, device):
    weights=ResNet18_Weights.IMAGENET1K_V1
    model=resnet18(weights=weights); model.fc=nn.Identity(); model.eval().to(device)
    for p in model.parameters(): p.requires_grad_(False)
    transform=weights.transforms(); cache={}
    with zipfile.ZipFile(archive) as z, torch.inference_mode():
        available=set(z.namelist())
        for p in dict.fromkeys(paths):
            if p not in available: raise FileNotFoundError(f'Image missing from ZIP: {p}')
            with z.open(p) as f: image=Image.open(io.BytesIO(f.read())).convert('RGB')
            cache[p]=model(transform(image).unsqueeze(0).to(device)).cpu().numpy().reshape(-1)
    return np.stack([cache[p] for p in paths]).astype('float32')

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.tab=nn.Sequential(nn.Linear(16,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(528,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,1))
    def forward(self,tab,img):
        return self.head(torch.cat([img,self.tab(tab)],dim=1)).squeeze(-1)

def neural_predict(xtr, itr, ytr, xte, ite, seed, device, epochs=120):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    model=Fusion().to(device); optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    mean=float(np.mean(ytr)); std=float(np.std(ytr)); std=std if std>0 else 1.
    xt=torch.as_tensor(xtr,dtype=torch.float32,device=device); it=torch.as_tensor(itr,dtype=torch.float32,device=device)
    yt=torch.as_tensor((ytr-mean)/std,dtype=torch.float32,device=device)
    generator=torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        for ids in torch.randperm(len(yt),generator=generator).split(32):
            ids=ids.to(device); optimizer.zero_grad(); loss=nn.functional.mse_loss(model(xt[ids],it[ids]),yt[ids]); loss.backward(); optimizer.step()
    model.eval()
    with torch.inference_mode():
        pred=model(torch.as_tensor(xte,dtype=torch.float32,device=device),torch.as_tensor(ite,dtype=torch.float32,device=device))
    return pred.cpu().numpy()*std+mean

def score(y,p):
    return {'rmse':float(np.sqrt(mean_squared_error(y,p))),'mae':float(mean_absolute_error(y,p)),'r2':float(r2_score(y,p))}

def bootstrap(y, base, other, groups, iterations=10000, seed=42):
    rng=np.random.default_rng(seed); unique=np.unique(groups); idx={g:np.flatnonzero(groups==g) for g in unique}
    diffs=[]
    for _ in range(iterations):
        sampled=rng.choice(unique,len(unique),replace=True)
        rows=np.concatenate([idx[g] for g in sampled]); diffs.append(score(y[rows],other[rows])['rmse']-score(y[rows],base[rows])['rmse'])
    return [float(v) for v in np.percentile(diffs,[2.5,97.5])]

def run(args):
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    records,a=load_records(args.csv,args.assignments)
    y=pd.to_numeric(records[TARGET],errors='raise').to_numpy(dtype=float)
    tab=records[FEATURES].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    groups=a['group_id'].astype(str).to_numpy(); folds=a['fold'].to_numpy()
    device='cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    images=embeddings_from_zip(a['Matched_Image_Archive_Path'].astype(str).tolist(),args.images_zip,device)
    models=['tabular_xgboost','image_xgboost','multimodal_xgboost','multimodal_random_forest','multimodal_linear','neural_fusion']
    preds={name:np.full(len(y),np.nan) for name in models}; fold_rows=[]
    for i,fold in enumerate(sorted(np.unique(folds))):
        tr=np.flatnonzero(folds!=fold); te=np.flatnonzero(folds==fold)
        if set(groups[tr])&set(groups[te]): raise ValueError('Group leakage')
        # Entirely missing training columns receive a zero fallback.
        med=np.nanmedian(tab[tr],axis=0)
        med=np.where(np.isfinite(med),med,0.0)
        train_imp=np.where(np.isfinite(tab[tr]),tab[tr],med)
        test_imp=np.where(np.isfinite(tab[te]),tab[te],med)
        scaler=StandardScaler().fit(train_imp)
        xtr=scaler.transform(train_imp); xte=scaler.transform(test_imp)
        combined_tr=np.column_stack([xtr,images[tr]]); combined_te=np.column_stack([xte,images[te]])
        specs={
          'tabular_xgboost':(XGBRegressor(n_estimators=300,max_depth=3,learning_rate=.05,subsample=.9,colsample_bytree=.9,random_state=args.seed,n_jobs=1),xtr,xte),
          'image_xgboost':(XGBRegressor(n_estimators=300,max_depth=3,learning_rate=.05,subsample=.9,colsample_bytree=.9,random_state=args.seed,n_jobs=1),images[tr],images[te]),
          'multimodal_xgboost':(XGBRegressor(n_estimators=300,max_depth=3,learning_rate=.05,subsample=.9,colsample_bytree=.9,random_state=args.seed,n_jobs=1),combined_tr,combined_te),
          'multimodal_random_forest':(RandomForestRegressor(n_estimators=300,random_state=args.seed,n_jobs=1),combined_tr,combined_te),
          'multimodal_linear':(LinearRegression(),combined_tr,combined_te)}
        for name,(model,x_train,x_test) in specs.items():
            model.fit(x_train,y[tr]); preds[name][te]=model.predict(x_test)
        preds['neural_fusion'][te]=neural_predict(xtr,images[tr],y[tr],xte,images[te],args.seed+i,device,args.epochs)
        for name in models: fold_rows.append({'model':name,'fold':int(fold),'n':len(te),**score(y[te],preds[name][te])})
    if any(not np.isfinite(p).all() for p in preds.values()): raise ValueError('Incomplete out-of-fold predictions')
    pd.DataFrame({'alloy':groups,'fold':folds,'y_true':y,**{f'pred_{k}':v for k,v in preds.items()}}).to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv',index=False)
    metrics={k:score(y,v) for k,v in preds.items()}
    pd.DataFrame([{'model':k,**v} for k,v in metrics.items()]).to_csv(out/'pooled_metrics.csv',index=False)
    ci={name:{'rmse_difference':metrics[name]['rmse']-metrics['tabular_xgboost']['rmse'],'ci95':bootstrap(y,preds['tabular_xgboost'],preds[name],groups,args.bootstrap,args.seed)} for name in ['multimodal_xgboost','image_xgboost','neural_fusion']}
    (out/'bootstrap_comparisons.json').write_text(json.dumps(ci,indent=2))
    print(json.dumps({'metrics':metrics,'bootstrap':ci},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',required=True,help='Original mechanical properties CSV')
    parser.add_argument('--assignments',required=True,help='Alloy-grouped fold assignment CSV')
    parser.add_argument('--images-zip',required=True,help='Original image archive ZIP with matching paths')
    parser.add_argument('--output',default='results')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=120)
    parser.add_argument('--bootstrap',type=int,default=10000)
    parser.add_argument('--cpu',action='store_true')
    run(parser.parse_args())
