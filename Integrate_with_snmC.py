selslc = ['3C', '4B', '6B', '7B', '4A', '5A', '3D']
meta = pd.read_msgpack(indir + '../../RS1/matrix/TotalClusteringResults.msg')
metars1 = meta[['SubRegion', 'Region', 'CellClass', 'Replicate', 'MajorType']]
metars1 = metars1.loc[np.array([x in selslc for x in metars1['Region']])]

mcdslist = np.loadtxt(indir + 'list.txt', dtype=np.str)
cellname, mcrs1, tcrs1 = [], [], []
for xx in mcdslist:
	f = h5py.File(indir + xx, 'r')
	cellname.append(f['cell'][()])
	mcidx = np.where(f['mc_type'][()]=='CHN')[0][0]
	mcrs1.append(f['chrom100k_da'][:,:,mcidx,0,0])
	tcrs1.append(f['chrom100k_da'][:,:,mcidx,0,1])
	binrs1 = np.array([f['chrom100k_chrom'][()],f['chrom100k_bin_start'][()]]).T.astype(np.str)
	f.close()
	print(xx)

cellname = np.concatenate(cellname)
mcrs1 = np.concatenate(mcrs1, axis=0)
tcrs1 = np.concatenate(tcrs1, axis=0)
metars1 = metars1.loc[cellname]
cellfilter = ((~(pd.isnull(metars1['Region']))) & (metars1['CellClass']!='NonN'))
metars1 = metars1[cellfilter]
mcrs1 = mcrs1[cellfilter]
tcrs1 = tcrs1[cellfilter]

bindict = {'-'.join(x):i for i,x in enumerate(binrs1)}
bin_all = np.loadtxt('/gale/netapp/home/zhoujt/genome/mm10/mm10.100kb.bin.bed', dtype=np.str)
binfilter = np.load('/gale/netapp/home/tanpengcheng/projects/CEMBA_RS2/matrix/cell_16971_binfilter90_autosomal.npy')
orderbin = np.array([bindict['-'.join(x[:2])] for x in bin_all[binfilter]])
mcrs1 = mcrs1[:,orderbin]
tcrs1 = tcrs1[:,orderbin]

mcrs1 = np.load('/gale/netapp/home/tanpengcheng/projects/CEMBA_RS2/matrix/RS1_cell_15782_MOp.SSp.ACA.AI_100kb.bin_mCH.npy')
tcrs1 = np.load('/gale/netapp/home/tanpengcheng/projects/CEMBA_RS2/matrix/RS1_cell_15782_MOp.SSp.ACA.AI_100kb.bin_tCH.npy')
ratebrs1 = calculate_posterior_mc_rate(mcrs1[:,binfilter], tcrs1[:,binfilter])

cellname = np.load('/gale/netapp/home/tanpengcheng/projects/CEMBA_RS2/matrix/RS1_cell_15782_MOp.SSp.ACA.AI_meta.npy', allow_pickle=True)
metars1 = metars1.loc[cellname[:,0]]

metars2 = metaall[:, 6:]
metars2[~facsfilter, 2] = 'Unknown'
cellfilter = np.array([x in selslc for x in metars2[:,1]])
meta = np.concatenate((metars1.values, metars2[cellfilter]), axis=0)
meta[np.logical_or(meta[:,2]=='Exc', meta[:,2]=='Inh'), 2] = 'Unknown'

ratemerge = np.concatenate((ratebrs1, rateb[cellfilter]), axis=0)

readb = np.load('/gale/netapp/home/tanpengcheng/projects/CEMBA_RS2/matrix/cell_11827_read_bin.mCH.npy')
readb = readb[:,binfilter]
readb = readb[cellfilter]

disp = highly_variable_methylation_feature(ratemerge, (np.sum(tcrs1, axis=0) + np.sum(readb, axis=0)) / (len(tcrs1) + len(readb)), np.arange(np.sum(binfilter)))
idx = np.argsort(disp)[::-1]
data = ratemerge[:, idx[:2000]]
pca = PCA(n_components=50, random_state=0)
rateb_reduce = pca.fit_transform(data)
np.save(outdir + 'matrix/cell_22144_RS1_RS2_posterior_disp2k_pca50.npy', rateb_reduce)

tsne = MulticoreTSNE(perplexity=50, verbose=3, random_state=0, n_jobs=10)
y = tsne.fit_transform(rateb_reduce[:, :50])
np.savetxt(outdir + 'matrix/cell_22144_RS1_RS2_posterior_disp2k_pca50_p50_rs0_tsne.txt', y, delimiter='\t', fmt='%s')

clusterdict = {'CGE-Lamp5':'Inh', 'PAL-Inh':'Others', 'Unc5c':'Inh', 'L6-IT':'L6-IT', 'MGE-Pvalb':'Inh', 'IT-L23':'L2/3', 'IT-L4':'L4', 'L5-ET':'L5-ET', 'PT-L5':'L5-ET', 'NP':'NP', 'L6-CT':'L6-CT', 'IT-L6':'L6-IT', 'EP':'Others', 'L2/3':'L2/3', 'OLF-Exc':'Others', 'MGE-Sst':'Inh', 'MSN-D2':'Others', 'Inh':'Inh', 'CT-L6':'L6-CT', 'CGE-Vip':'Inh', 'CLA':'CLA', 'L6b':'L6b', 'L5-IT':'L5-IT', 'L4':'L4', 'MSN-D1':'Others', 'IT-L5':'L5-IT', 'NP-L6':'NP', 'IG-CA2':'Others', 'OLF':'Others'}
meta[:,4] = np.array([clusterdict[x] for x in meta[:,4]])

leg = np.array(['L2/3', 'L4', 'L5-IT', 'L6-IT', 'L5-ET', 'L6-CT', 'L6b', 'NP', 'CLA', 'Inh', 'Others'])
tar = np.array(['STR', 'Pons', 'SC', 'TH', 'VTA', 'MOp', 'SSp', 'ACA', 'VISp', 'MY'])
reg = np.array(['MOp', 'SSp', 'ACA', 'AI'])

fig, axes = plt.subplots(1, 3, figsize=(18, 4))
ax = axes[0]
for i,x in enumerate(leg):
	cell = (meta[:, 4]==x)
	ax.scatter(y[cell, 0], y[cell, 1], c=legcolor[i], s=5, edgecolors='none', alpha=0.8, label=x, rasterized=True)

ax = axes[1]
for i,x in enumerate(reg):
	cell = (meta[:, 0]==x)
	ax.scatter(y[cell, 0], y[cell, 1], c=color[i], s=5, edgecolors='none', alpha=0.8, label=x, rasterized=True)

ax = axes[2]
cell = (meta[:, 2]=='Unknown')
ax.scatter(y[cell, 0], y[cell, 1], c='grey', s=5, edgecolors='none', alpha=0.1, label='Unknown', rasterized=True)
for i,x in enumerate(tar):
	cell = (meta[:, 2]==x)
	ax.scatter(y[cell, 0], y[cell, 1], c=tarcolor[x], s=5, edgecolors='none', alpha=0.8, label='->'+x, rasterized=True)

for ax in axes:
	ax.set_xlabel('t-SNE-1', fontsize=20)
	ax.set_ylabel('t-SNE-2', fontsize=20)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(axis='both', which='both', length=0)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.legend(markerscale=5, prop={'size': 10}, bbox_to_anchor=(1,1), loc='upper left')

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_22144_RS1_RS2_posterior_disp2k_pca50_p50_rs0.meta.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.close()

