import pandas as pd,numpy as np
from pathlib import Path
P=Path('/mnt/data');a=pd.read_csv(P/'steel_alloy_grouped_cv_assignments.csv');o=pd.read_csv(P/'tensile-properties-from-jeffs-ml-sheet-20220208(1).csv')
k=['Alloy','Temperature, [C]','Yield Stress, [MPa]']
for d in [a,o]:
 for c in k[1:]:d[c]=pd.to_numeric(d[c],errors='coerce').round(8)
 d['_occurrence']=d.groupby(k,dropna=False).cumcount()
o['_original_row']=np.arange(len(o));m=a.merge(o[k+['_occurrence','_original_row']],on=k+['_occurrence'],how='left',validate='one_to_one',sort=False)
assert m._original_row.notna().all()
d=o.iloc[m._original_row.astype(int).to_numpy()].reset_index(drop=True)
features=['Fe','C','Cr','Mn','Si','Ni','Co','Mo','W','Nb','Al','P','Cu','Ti','V','Temperature, [C]']
X=d[features].apply(pd.to_numeric,errors='coerce');y=d['Yield Stress, [MPa]'].astype(float)
# Match identical inputs, treating missing values as a consistent missing token.
key=X.fillna(987654321.123456)
ids=key.groupby(features,dropna=False,sort=False).ngroup()
records=[];sse=0;duplicate_records=0
for gid,ix in pd.Series(range(len(ids))).groupby(ids):
 idx=ix.to_numpy();vals=y.iloc[idx].to_numpy();
 if len(idx)>1:
  duplicate_records+=len(idx);sse+=np.sum((vals-vals.mean())**2)
  records.append({'input_group':int(gid),'n_records':len(idx),'n_distinct_targets':len(np.unique(vals)),'target_min_mpa':vals.min(),'target_max_mpa':vals.max(),'target_range_mpa':np.ptp(vals),'within_group_sse_mpa2':np.sum((vals-vals.mean())**2),'alloys':'; '.join(sorted(set(d.iloc[idx]['Alloy'].astype(str)))),'record_indices':';'.join(map(str,idx))})
report=pd.DataFrame(records).sort_values(['target_range_mpa','n_records'],ascending=False)
report.to_csv(P/'steel_duplicate_input_target_groups.csv',index=False)
summary=pd.DataFrame([{'n_records':len(d),'unique_feature_vectors':len(np.unique(ids)),'records_in_repeated_groups':duplicate_records,'extra_records_beyond_unique_vectors':len(d)-len(np.unique(ids)),'groups_with_repeated_vectors':len(report),'groups_with_discordant_targets':int((report.n_distinct_targets>1).sum()),'max_target_range_mpa':float(report.target_range_mpa.max()) if len(report) else 0,'within_identical_input_rmse_lower_bound_mpa':float(np.sqrt(sse/len(d)))}]);summary.to_csv(P/'steel_duplicate_input_target_summary.csv',index=False)
print(summary.to_string(index=False));print(report.head(8).to_string(index=False))
