import zipfile,io,hashlib,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from PIL import Image
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path('/mnt/data'); archive=ROOT/'Prediction of mechanical properties-20261116T230131Z-1-001.zip'
checkpoint=ROOT/'resnet18-f37072fd.pth'; out=ROOT/'steel_embedding_retrieval';out.mkdir(exist_ok=True)
state=torch.load(checkpoint,map_location='cpu',weights_only=True)
model=models.resnet18(weights=None);model.load_state_dict(state);model.fc=torch.nn.Identity();model.eval()
tf=models.ResNet18_Weights.IMAGENET1K_V1.transforms()
with zipfile.ZipFile(archive) as z:
 paths=sorted([n for n in z.namelist() if n.lower().endswith('.bmp') and any('/'+s+'/' in n for s in ['cpj_alloys','hr_alloys20220208','p92_alloys'])]); folders={}
 for p in paths:folders.setdefault(p.rsplit('/',1)[0],[]).append(p)
 # Deterministic sample of at most 4 images per alloy folder, using natural sorted filenames
 sample=[]
 for folder,items in sorted(folders.items()):
  items=sorted(items,key=lambda s:(int(Path(s).stem) if Path(s).stem.isdigit() else 999999,s))
  ix=np.linspace(0,len(items)-1,min(4,len(items)),dtype=int)
  sample.extend([items[i] for i in ix])
 vectors=[]; metadata=[]
 torch.set_num_threads(2)
 with torch.inference_mode():
  for j,p in enumerate(sample):
   with z.open(p) as f: data=f.read()
   im=Image.open(io.BytesIO(data)).convert('RGB');v=model(tf(im).unsqueeze(0)).numpy().ravel()
   vectors.append(v);metadata.append({'path':p,'folder':p.rsplit('/',1)[0], 'family': 'CPJ' if '/cpj_alloys/' in p else 'HR' if '/hr_alloys20220208/' in p else 'P92','sha256':hashlib.sha256(data).hexdigest()})
   if (j+1)%40==0:print('embedded',j+1,flush=True)
X=np.stack(vectors);meta=pd.DataFrame(metadata);s=cosine_similarity(X);np.fill_diagonal(s,-np.inf)
records=[]
for i in range(len(meta)):
 ix=np.argsort(-s[i])[:5]
 for rank,k in enumerate(ix,1):records.append({'query_path':meta.path[i],'query_folder':meta.folder[i],'query_family':meta.family[i],'rank':rank,'neighbor_path':meta.path[k],'neighbor_folder':meta.folder[k],'neighbor_family':meta.family[k],'cosine_similarity':float(s[i,k]),'same_folder':bool(meta.folder[i]==meta.folder[k]),'same_family':bool(meta.family[i]==meta.family[k])})
r=pd.DataFrame(records);r.to_csv(out/'nearest_neighbors_top5.csv',index=False);meta.to_csv(out/'image_sample_manifest.csv',index=False)
summary=[]
for family,g in r.groupby('query_family'):
 one=g[g['rank']==1];five=g.groupby('query_path')
 summary.append({'family':family,'n_queries':len(one),'top1_same_folder_fraction':one.same_folder.mean(),'top1_same_family_fraction':one.same_family.mean(),'top5_any_same_folder_fraction':five.same_folder.any().mean(),'top5_any_same_family_fraction':five.same_family.any().mean()})
one=r[r['rank']==1];summary.append({'family':'ALL','n_queries':len(one),'top1_same_folder_fraction':one.same_folder.mean(),'top1_same_family_fraction':one.same_family.mean(),'top5_any_same_folder_fraction':r.groupby('query_path').same_folder.any().mean(),'top5_any_same_family_fraction':r.groupby('query_path').same_family.any().mean()})
pd.DataFrame(summary).to_csv(out/'retrieval_summary.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False));print('files:',out)
