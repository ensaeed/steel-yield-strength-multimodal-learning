import pandas as pd,numpy as np,hashlib,json
from pathlib import Path
p=Path('/mnt/data'); a=pd.read_csv(p/'steel_alloy_grouped_cv_assignments.csv');o=pd.read_csv(p/'tensile-properties-from-jeffs-ml-sheet-20220208(1).csv')
k=['Alloy','Temperature, [C]','Yield Stress, [MPa]']
for d in (a,o):
 for c in k[1:]:d[c]=pd.to_numeric(d[c],errors='coerce').round(8)
 d['_occ']=d.groupby(k,dropna=False).cumcount()
o['csv_row']=np.arange(len(o));m=a.merge(o[k+['_occ','csv_row']],on=k+['_occ'],how='left',validate='one_to_one',sort=False)
assert m.csv_row.notna().all();d=o.iloc[m.csv_row.astype(int).values].reset_index(drop=True)
f=['Fe','C','Cr','Mn','Si','Ni','Co','Mo','W','Nb','Al','P','Cu','Ti','V','Temperature, [C]']
x=d[f].apply(pd.to_numeric,errors='coerce');y=d['Yield Stress, [MPa]'].to_numpy(float)
rows=[]
for fold in sorted(a.fold.unique()):
 tr=a.fold!=fold;te=~tr
 rows.append(dict(fold=int(fold),train_records=int(tr.sum()),test_records=int(te.sum()),train_alloys=int(a.loc[tr,'group_id'].nunique()),test_alloys=int(a.loc[te,'group_id'].nunique()),alloy_overlap=len(set(a.loc[tr,'group_id'])&set(a.loc[te,'group_id'])),image_overlap=len(set(a.loc[tr,'Matched_Image_Archive_Path'])&set(a.loc[te,'Matched_Image_Archive_Path'])),test_y_mean=float(y[te].mean()),test_y_std=float(y[te].std()),test_y_min=float(y[te].min()),test_y_max=float(y[te].max())))
report={'n_records':len(a),'n_alloys':int(a.group_id.nunique()),'n_selected_images':int(a.Matched_Image_Archive_Path.nunique()),'missing_features':x.isna().sum().to_dict(),'exact_duplicate_feature_vectors':int(x.duplicated().sum()),'exact_duplicate_feature_target_pairs':int(pd.concat([x,pd.Series(y,name='y')],axis=1).duplicated().sum()),'input_sha256':{z:hashlib.sha256((p/z).read_bytes()).hexdigest() for z in ['steel_alloy_grouped_cv_assignments.csv','tensile-properties-from-jeffs-ml-sheet-20220208(1).csv']}}
pd.DataFrame(rows).to_csv(p/'steel_locked_fold_integrity.csv',index=False)
(p/'steel_locked_data_manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2));print(pd.DataFrame(rows).to_string(index=False))
