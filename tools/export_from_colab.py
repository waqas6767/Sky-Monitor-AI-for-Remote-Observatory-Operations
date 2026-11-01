# =====================================================================================
#  Sky Monitor AI - export everything the web app needs  ->  skyapp_assets.zip
#  Paste this whole file into ONE new cell at the end of your Colab notebook and run it,
#  in the same session where the models were trained (needs: cnn, res_v2, res_v3,
#  backbone, emb, clu2, clu5, Z, u_idx, y_u, bright, results, moon_alt, illum, te, ...).
# =====================================================================================
!pip -q install onnx onnxruntime onnxscript
import json, shutil, copy
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

OUT = Path('/content/skyapp_assets'); shutil.rmtree(OUT, ignore_errors=True)
for d in ['models', 'data', 'thumbs/test', 'thumbs/tsne', 'samples']:
    (OUT / d).mkdir(parents=True, exist_ok=True)
FINE = CLASSES.index('Data_fine'); CLOUDY = 1 - FINE
S = IMG_SIZE

# ---------- 1. Models with two outputs: logits + last feature maps (for exact CAM maps) ----------
class CNNFeat(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        f = self.m.features(x)
        return self.m.head(f), f

class ResFeat(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        f = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        return m.fc(torch.flatten(m.avgpool(f), 1)), f

nets = {'smallcnn':  CNNFeat(copy.deepcopy(cnn)).cpu().eval(),
        'resnet_v2': ResFeat(copy.deepcopy(res_v2)).cpu().eval(),
        'resnet_v3': ResFeat(copy.deepcopy(res_v3)).cpu().eval()}
embedder = copy.deepcopy(backbone).cpu().eval()            # ImageNet ResNet-18, fc = Identity

def export_onnx(model, path, out_names):
    x = torch.randn(2, 3, S, S)
    dyn = {'input': {0: 'batch'}, **{n: {0: 'batch'} for n in out_names}}
    try:
        torch.onnx.export(model, (x,), str(path), input_names=['input'], output_names=out_names,
                          dynamic_axes=dyn, opset_version=17, dynamo=False)
    except TypeError:                                        # older torch without the dynamo argument
        torch.onnx.export(model, (x,), str(path), input_names=['input'], output_names=out_names,
                          dynamic_axes=dyn, opset_version=17)
    except Exception as e:                                   # newest torch: classic exporter removed
        print('classic exporter failed:', e, '-> using the dynamo exporter')
        torch.onnx.export(model, (x,), str(path), input_names=['input'], output_names=out_names,
                          dynamic_shapes=({0: torch.export.Dim('batch', min=1, max=1024)},), dynamo=True)

for k, m in nets.items():
    export_onnx(m, OUT / 'models' / f'{k}.onnx', ['logits', 'features'])
export_onnx(embedder, OUT / 'models' / 'embedder.onnx', ['embedding'])

# check: ONNX gives the same numbers as PyTorch (batch of 4 test images)
xb = torch.stack([eval_tf(load_img(paths[k])) for k in te[:4]])
onnx_check = {}
for k, m in list(nets.items()) + [('embedder', embedder)]:
    s = ort.InferenceSession(str(OUT / 'models' / f'{k}.onnx'), providers=['CPUExecutionProvider'])
    o = s.run(None, {'input': xb.numpy()})[0]
    with torch.no_grad():
        t = m(xb); t = (t[0] if isinstance(t, tuple) else t).numpy()
    onnx_check[k] = float(np.abs(o - t).max())
print('max |ONNX - PyTorch| :', onnx_check)
assert max(onnx_check.values()) < 1e-2, 'ONNX export does not match PyTorch'

# ---------- 2. Classifier weights (for CAM maps) + unsupervised cluster model ----------
P = {'smallcnn_w': cnn.head[3].weight, 'smallcnn_b': cnn.head[3].bias,
     'resnet_v2_w': res_v2.fc.weight, 'resnet_v2_b': res_v2.fc.bias,
     'resnet_v3_w': res_v3.fc.weight, 'resnet_v3_b': res_v3.fc.bias}
P = {k: v.detach().cpu().numpy() for k, v in P.items()}

E = np.concatenate(emb)                                      # embeddings of the 4000 clustering images
scaler = StandardScaler().fit(E)
pca = PCA(50, random_state=SEED).fit(scaler.transform(E))
Xe = pca.transform(scaler.transform(E))
km = KMeans(5, n_init=10, random_state=SEED).fit(Xe)
mapping = {k: int(np.bincount(clu5[km.labels_ == k], minlength=5).argmax()) for k in range(5)}
assert len(set(mapping.values())) == 5, f'cluster mapping not one-to-one: {mapping}'
centers = np.zeros_like(km.cluster_centers_)
for k, c in mapping.items(): centers[c] = km.cluster_centers_[k]   # same numbering as the analysis
d_tr = np.sqrt(((Xe[:, None] - centers[None]) ** 2).sum(-1)); lab_tr = d_tr.argmin(1)
print('agreement with the clusters analysed earlier:', round(float((lab_tr == clu5).mean()), 4))
cluster_p99 = [float(np.percentile(d_tr[lab_tr == c, c], 99)) for c in range(5)]
P.update(scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, pca_mean=pca.mean_,
         pca_components=pca.components_, km_centers=centers)
np.savez(OUT / 'models' / 'params.npz', **{k: np.asarray(v, np.float32) for k, v in P.items()})

print(pd.crosstab(pd.Series([CLASSES[i] for i in y_u], name='true'), pd.Series(clu5, name='cluster')))
CLUSTER_NAMES = ['Dark clear sky', 'Broken / textured cloud', 'Moonlit clear sky',
                 'Wet dome (droplets / rain)', 'Bright moonlit overcast']
WET = 3
gmm_fine = int(pd.Series(clu2[y_u == FINE]).mode()[0])       # which GMM cluster holds the fine frames

# ---------- 3. Test-set predictions, cloud-cover estimate and sky type for every test frame ----------
m224 = torch.tensor(sky_mask((S, S))[..., 0])
gpu = {k: copy.deepcopy(m).to(device).eval() for k, m in nets.items()}
emb_gpu = copy.deepcopy(embedder).to(device).eval()
dW = {k: (torch.tensor(P[f'{k}_w'][FINE] - P[f'{k}_w'][CLOUDY], device=device),
          float(P[f'{k}_b'][FINE] - P[f'{k}_b'][CLOUDY])) for k in nets}
pf = {k: [] for k in nets}; cover, stype, sdist = [], [], []
with torch.no_grad():
    for x, _ in dl_te:                                        # dl_te keeps the order of te
        x = x.to(device); maps = []
        for k, net in gpu.items():
            lo, f = net(x)
            pf[k].append(F.softmax(lo, 1)[:, FINE].cpu().numpy())
            w, b = dW[k]
            loc = torch.einsum('c,nchw->nhw', w, f) + b       # local logit: evidence of each sky cell
            maps.append(F.interpolate(torch.sigmoid(loc)[:, None], size=(S, S), mode='bilinear',
                                      align_corners=False)[:, 0])
        fm = torch.stack(maps).mean(0).cpu()
        cover.append((((fm < 0.5) & m224).sum((1, 2)).float() / m224.sum()).numpy())
        z = pca.transform(scaler.transform(emb_gpu(x).cpu().numpy()))
        d = np.sqrt(((z[:, None] - centers[None]) ** 2).sum(-1))
        stype.append(d.argmin(1)); sdist.append(d.min(1))

t_te = pd.to_datetime([Path(p).stem[:12] for p in paths[te]], format='%y%m%d%H%M%S')
df_test = pd.DataFrame({
    'stem': [Path(p).stem for p in paths[te]], 'night': te_nights,
    'time': t_te.strftime('%Y-%m-%d %H:%M:%S'), 'label': labels[te],
    **{f'p_{k}': np.concatenate(v) for k, v in pf.items()},
    'p_handcrafted': results['Handcrafted_GB'][0][:, FINE],
    'moon_alt': np.round(moon_alt, 2), 'moon_illum': np.round(illum, 3),
    'cloud_cover': np.concatenate(cover), 'sky_type': np.concatenate(stype),
    'sky_type_dist': np.concatenate(sdist)})
df_test.to_csv(OUT / 'data' / 'test_predictions.csv', index=False)
print('max |P(fine) export - earlier results| :',
      float(np.abs(df_test.p_smallcnn - results['SmallCNN'][0][:, FINE]).max()))

df_tsne = pd.DataFrame({'stem': [Path(paths[i]).stem for i in u_idx],
                        'night': [night_key(paths[i]) for i in u_idx], 'label': y_u,
                        'x': Z[:, 0], 'y': Z[:, 1], 'gmm2': clu2, 'sky_type': clu5, 'brightness': bright})
df_tsne.to_csv(OUT / 'data' / 'tsne.csv', index=False)

# ---------- 4. Thumbnails (masked, cropped to the sky circle) and full-size sample frames ----------
def save_thumb(path, out, size):
    im = load_img(path); w, h = im.size; r = MASK_RADIUS_FRAC * min(w, h) * 1.01
    im.crop((int(w/2 - r), int(h/2 - r), int(w/2 + r), int(h/2 + r))).resize((size, size), Image.BILINEAR).save(out, quality=82)
for p in paths[te]: save_thumb(p, OUT / 'thumbs' / 'test' / f'{Path(p).stem}.jpg', 192)
for i in u_idx:     save_thumb(paths[i], OUT / 'thumbs' / 'tsne' / f'{Path(paths[i]).stem}.jpg', 96)

samples = []
for c in range(5):                                            # 2 typical frames per sky type
    idx = np.where(lab_tr == c)[0]; idx = idx[np.argsort(d_tr[idx, c])][[0, len(idx) // 3]]
    for j in idx:
        p = paths[u_idx[j]]; shutil.copy(p, OUT / 'samples' / Path(p).name)
        samples.append({'file': Path(p).name, 'caption': f'{CLUSTER_NAMES[c]} ({night_key(p)})',
                        'label': CLASSES[labels[u_idx[j]]]})
for stem, cap in [('211219215701', 'Hard case: full moon, partly cloudy (labelled cloudy)'),
                  ('211124194628', 'Hard case: overcast frame labelled fine')]:
    hit = [p for p in paths if Path(p).stem == stem]
    if hit:
        shutil.copy(hit[0], OUT / 'samples' / Path(hit[0]).name)
        samples.append({'file': Path(hit[0]).name, 'caption': cap, 'label': CLASSES[labels[list(paths).index(hit[0])]]})

# ---------- 5. Config + training histories, then zip ----------
to_list = lambda h: {k: [float(v) for v in vals] for k, vals in h.items()}
cfg = dict(classes=CLASSES, fine_index=FINE, img_size=S, mean=MEAN, std=STD,
           mask_radius_frac=MASK_RADIUS_FRAC, models=list(nets), cluster_names=CLUSTER_NAMES,
           cluster_p99=cluster_p99, wet_cluster=WET, gmm_fine_cluster=gmm_fine,
           site=dict(lat=34.3776, lon=132.7769, height=500, utc_offset_hours=9,
                     name='Higashi-Hiroshima Observatory'),
           samples=samples, onnx_check=onnx_check,
           history={'smallcnn': to_list(h_cnn), 'resnet_v3': to_list({k: h1[k] + h2[k] for k in h1})})
(OUT / 'config.json').write_text(json.dumps(cfg, indent=1))

zip_path = shutil.make_archive('/content/skyapp_assets', 'zip', root_dir=OUT)
print(f'\nModels + data: {zip_path}  ({os.path.getsize(zip_path) / 1e6:.0f} MB)  <- for the online (GitHub Release) version')

# ---------- 6. Standalone package: application + trained models + data in ONE zip ----------
import zipfile
APP_ZIP = next((p for p in ['/content/sky-monitor-ai.zip', '/content/drive/MyDrive/sky-monitor-ai.zip']
                if os.path.exists(p)), None)
if APP_ZIP is None:
    from google.colab import files
    print('\nUpload sky-monitor-ai.zip (the application code, about 100 KB):')
    up = files.upload(); APP_ZIP = '/content/' + next(iter(up))
shutil.rmtree('/content/_app', ignore_errors=True); shutil.rmtree('/content/SkyMonitorAI', ignore_errors=True)
with zipfile.ZipFile(APP_ZIP) as z:
    z.extractall('/content/_app')
app_dir = next(Path('/content/_app').rglob('streamlit_app.py')).parent
PKG = Path('/content/SkyMonitorAI')
shutil.copytree(app_dir, PKG, ignore=shutil.ignore_patterns('assets', '.venv', '__pycache__'))
shutil.copytree(OUT, PKG / 'assets')
for sh in ['run_mac.command', 'run_linux.sh']:
    if (PKG / sh).exists(): os.chmod(PKG / sh, 0o755)
std = shutil.make_archive('/content/SkyMonitorAI_standalone', 'zip', root_dir='/content', base_dir='SkyMonitorAI')
print(f'Standalone app: {std}  ({os.path.getsize(std) / 1e6:.0f} MB)  <- unzip anywhere, double-click the launcher')

for f in [zip_path, std]:
    try:
        shutil.copy(f, f'/content/drive/MyDrive/{Path(f).name}'); print('Copied to My Drive/' + Path(f).name)
    except Exception as e:
        print('Could not copy to Drive:', e)
