
import io,zipfile,time
from pathlib import Path, PurePosixPath
import numpy as np,pandas as pd,torch
from PIL import Image
from torchvision import models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from xgboost import XGBRegressor

ROOT=Path("/mnt/data")
OUT=ROOT/"steel_image_selection_sensitivity"
OUT.mkdir(exist_ok=True)
a=pd.read_csv(ROOT/"steel_alloy_grouped_cv_assignments.csv")
orig=pd.read_csv(ROOT/"steel_original_for_run.csv")
features=['Fe','C','Cr','Mn','Si','Ni','Co','Mo','W','Nb','Al','P','Cu','Ti','V','Temperature, [C]']
keys=['Alloy','Temperature, [C]','Yield Stress, [MPa]']
for k in keys[1:]:
    a[k]=pd.to_numeric(a[k],errors='coerce').round(8)
    orig[k]=pd.to_numeric(orig[k],errors='coerce').round(8)
a["_occ"]=a.groupby(keys,dropna=False).cumcount()
orig["_occ"]=orig.groupby(keys,dropna=False).cumcount()
orig["_row"]=np.arange(len(orig))
m=a.merge(orig[keys+["_occ","_row"]],on=keys+["_occ"],how="left",validate="one_to_one",sort=False)
assert m["_row"].notna().all()
df=orig.iloc[m["_row"].astype(int)].reset_index(drop=True)
y=df["Yield Stress, [MPa]"].to_numpy(float)
tab=df[features].apply(pd.to_numeric,errors="coerce")
folds=a["fold"].to_numpy()
groups=a["group_id"].astype(str).to_numpy()
folders=a["Matched_Image_Archive_Path"].map(lambda p:str(PurePosixPath(p).parent)).to_numpy()
with zipfile.ZipFile(ROOT/"Prediction of mechanical properties-20261116T230131Z-1-001.zip") as z:
    folders_all={}
    for p in z.namelist():
        if p.lower().endswith(".bmp"):
            folders_all.setdefault(str(PurePosixPath(p).parent),[]).append(p)
    for k in folders_all: folders_all[k].sort()
    unique_folders=sorted(set(folders))
    selected={}
    for folder in unique_folders:
        options=folders_all[folder]
        selected[folder]={"first":options[0],"middle":options[len(options)//2],"last":options[-1]}
    paths=sorted({p for d in selected.values() for p in d.values()})
    net=models.resnet18(weights=None)
    net.load_state_dict(torch.load(ROOT/"resnet18-f37072fd.pth",map_location="cpu",weights_only=True))
    net.fc=torch.nn.Identity()
    net.eval()
    for p in net.parameters():p.requires_grad_(False)
    tf=models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    emb={}
    torch.set_num_threads(2)
    with torch.inference_mode():
        for j,path in enumerate(paths):
            im=Image.open(io.BytesIO(z.read(path))).convert("RGB")
            emb[path]=net(tf(im).unsqueeze(0)).numpy().reshape(-1)
            if (j+1)%20==0:print("Embedded",j+1,"/",len(paths),flush=True)
rows=[]; predictions=[]
for selection in ["first","middle","last"]:
    ximage=np.stack([emb[selected[f][selection]] for f in folders])
    oof=np.full(len(a),np.nan)
    for fold in sorted(set(folds)):
        tr=np.where(folds!=fold)[0];te=np.where(folds==fold)[0]
        assert not set(groups[tr]) & set(groups[te])
        median=tab.iloc[tr].median().fillna(0)
        tab_tr=tab.iloc[tr].fillna(median).fillna(0)
        tab_te=tab.iloc[te].fillna(median).fillna(0)
        scaler=StandardScaler().fit(tab_tr)
        xtr=np.column_stack([ximage[tr],scaler.transform(tab_tr)])
        xte=np.column_stack([ximage[te],scaler.transform(tab_te)])
        model=XGBRegressor(n_estimators=180,max_depth=4,learning_rate=.06,subsample=.8,colsample_bytree=.8,random_state=42,n_jobs=2,objective="reg:squarederror")
        model.fit(xtr,y[tr])
        pred=model.predict(xte)
        oof[te]=pred
        rows.append(dict(selection=selection,fold=int(fold),n=len(te),rmse=np.sqrt(mean_squared_error(y[te],pred)),mae=mean_absolute_error(y[te],pred),r2=r2_score(y[te],pred)))
    assert np.isfinite(oof).all()
    predictions.append(pd.DataFrame(dict(selection=selection,fold=folds,alloy=groups,measured_yield_mpa=y,predicted_yield_mpa=oof)))
    print(selection,"pooled RMSE",np.sqrt(mean_squared_error(y,oof)),flush=True)
pd.DataFrame(rows).to_csv(OUT/"fold_metrics.csv",index=False)
pd.concat(predictions).to_csv(OUT/"oof_predictions.csv",index=False)
summary=[]
for sel,part in pd.concat(predictions).groupby("selection"):
    summary.append(dict(selection=sel,n=len(part),pooled_rmse=np.sqrt(mean_squared_error(part.measured_yield_mpa,part.predicted_yield_mpa)),pooled_mae=mean_absolute_error(part.measured_yield_mpa,part.predicted_yield_mpa),pooled_r2=r2_score(part.measured_yield_mpa,part.predicted_yield_mpa)))
pd.DataFrame(summary).to_csv(OUT/"pooled_metrics.csv",index=False)
pd.DataFrame([{"folder":f,"selection":s,"path":p} for f,vals in selected.items() for s,p in vals.items()]).to_csv(OUT/"selected_images.csv",index=False)
print(pd.DataFrame(summary).to_string(index=False))
