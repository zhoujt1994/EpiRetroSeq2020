y = np.loadtxt(indir + 'cell_4541_integrated_cell_4176_L5ET_cell_365_Tasic_hvg10k_Scanoroma_integrated_sigma100_p50_y.txt')
metarna = pd.read_csv('/gale/netapp/home/zhoujt/project/CEMBA/Tasic2018/GSE115746_complete_metadata_28706-cells.csv', header = 0)
expr = pd.read_csv('/gale/netapp/home/zhoujt/project/CEMBA/Tasic2018/GSE115746_cells_exon_counts.csv', header = 0, index_col = 0)

marker = ['L5 PT ALM Npsr1','L5 PT ALM Slco2a1','L5 PT ALM Hpgd']
metarna = metarna[(metarna['cell_cluster']=='L5 PT ALM Npsr1')| (metarna['cell_cluster']=='L5 PT ALM Slco2a1') | (metarna['cell_cluster']=='L5 PT ALM Hpgd')]
expr = expr.loc[:,metarna['sample_name']]
genename = expr.index.to_numpy()
cellname = metarna['sample_name'].to_numpy()
expr = expr.to_numpy().T
metarna = metarna.to_numpy()[:,1:]

f = h5py.File('/gale/netapp/home/zhoujt/project/CEMBA/Tasic2018/L5-PT_ALM.h5py', 'w')
tmp = f.create_dataset('genes', genename.shape, dtype=h5py.string_dtype(encoding='utf-8'), compression='gzip')
tmp[()] = genename
tmp = f.create_dataset('meta', metarna.shape, dtype=h5py.string_dtype(encoding='utf-8'), compression='gzip')
tmp[()] = metarna
tmp = f.create_dataset('expr', expr.shape, dtype=float, compression='gzip')
tmp[()] = expr
f.close()


f = h5py.File('/gale/netapp/home/zhoujt/project/CEMBA/Tasic2018/L5-PT_ALM.h5py', 'r')
metarna = f['meta'][()]
expr = f['expr'][()]
genename = f['genes'][()]
f.close()

import anndata
import scanpy as sc

exprad = anndata.AnnData(X = pd.DataFrame(expr, columns = genename, index = metarna[:,0]), var = pd.DataFrame(index = genename), obs = pd.DataFrame(metarna[:,1:], index = metarna[:,0]))
sc.pp.filter_genes(exprad, min_cells=10)
sc.pp.normalize_per_cell(exprad)
sc.pp.log1p(exprad)
sc.pp.highly_variable_genes(exprad, n_top_genes=10000)
print(np.sum(exprad.var['highly_variable']))
expr0 = exprad[:, exprad.var['highly_variable']]
metarna = expr0.obs.values

data = [-zscore(rate, axis=0), zscore(expr0.X, axis=0)]
genelist = [gene[:,-1], expr0.var.index]
integrated, genes = scanorama.integrate(data, genelist, sigma=100)

integrated_reduce = np.concatenate(integrated, axis=0)
y = tsne.fit_transform(integrated_reduce[:,:30])

np.save(outdir + 'matrix/cell_4541_L5ET_methyl_rateg_rna_hvg10k_zscore_sigma100_pc100.npy', integrated_reduce)
np.savetxt(outdir + 'matrix/cell_4541_L5ET_methyl_rateg_rna_hvg10k_zscore_sigma100_pc30_p50.txt',y, fmt='%s', delimiter='\t')


tarrnadict = {'No Injection':'No Injection', 'RSP':'CTX', 'BLA':'CTX', 'PF':'TH', 'VAL':'TH', 'ZI':'HY', 'SC':'SC', 'PRNc':'Pons', 'PG':'Pons', 'GRN':'MY', 'IRN':'MY', 'PARN':'MY'}
tarrna = np.array([tarrnadict[x] for x in metarna[:,6]])
tar = np.array(['Unknown', 'STR', 'MOp', 'SSp', 'ACA', 'VISp', 'Pons', 'SC', 'VTA', 'TH', 'MY'])
assay = np.array(['snmC-Seq' for i in range(len(meta))] + ['SMART-Seq' for i in range(len(metarna))])
fig, axes = plt.subplots(3, 3, figsize=(14,9), sharex='all', sharey='all')

ax = axes[0,0]
cell = (assay=='snmC-Seq')
ax.scatter(y[cell, 0], y[cell, 1], c='grey', s=5, edgecolors='none', alpha=0.3, rasterized=True)
for i,yy in enumerate(marker):
	cell = (assay=='SMART-Seq')
	cell[cell] = (metarna[:,18]==yy)
	ax.scatter(y[cell, 0], y[cell, 1], c=legcolor[i], s=10, edgecolors='none', alpha=0.8, rasterized=True)
	ax.text(np.median(y[cell,0]), np.median(y[cell,1]), yy, fontsize=15, horizontalalignment='center', verticalalignment='center')

for k,tmp in enumerate([tarrna, metarna[:,6]]):
	ax = axes[0,k+1]
	cell = (assay=='snmC-Seq')
	ax.scatter(y[cell, 0], y[cell, 1], c='grey', s=5, edgecolors='none', alpha=0.3, rasterized=True)
	for i,yy in enumerate([['No Injection', 'CTX', 'TH', 'HY', 'SC', 'Pons', 'MY'], tarrnadict.keys()][k]):
		cell = (assay=='SMART-Seq')
		cell[cell] = (tmp==yy)
		ax.scatter(y[cell, 0], y[cell, 1], c=(['grey']+legcolor.tolist())[i], s=10, edgecolors='none', alpha=0.8, rasterized=True, label=yy)
	ax.legend(bbox_to_anchor=(1,1.1), markerscale=3)

ax = axes[1,1]
cell = (assay=='SMART-Seq')
ax.scatter(y[cell, 0], y[cell, 1], c='grey', s=5, edgecolors='none', alpha=0.3, rasterized=True)
for i,yy in enumerate(reg):
	cell = (assay=='snmC-Seq')
	cell[cell] = (meta[:,0]==yy)
	ax.scatter(y[cell, 0], y[cell, 1], c=color[i], s=5, edgecolors='none', alpha=0.8, rasterized=True, label=yy)

ax.legend(bbox_to_anchor=(1,1), markerscale=3)

ax = axes[1,2]
cell = (assay=='SMART-Seq')
ax.scatter(y[cell, 0], y[cell, 1], c='grey', s=5, edgecolors='none', alpha=0.3, rasterized=True)
for i,yy in enumerate(tar):
	cell = (assay=='snmC-Seq')
	cell[cell] = (meta[:,2]==yy)
	ax.scatter(y[cell, 0], y[cell, 1], c=tarcolor[yy], s=5, edgecolors='none', alpha=0.8, rasterized=True, label=yy)

ax.legend(bbox_to_anchor=(1,1), markerscale=3)

for k,xx in enumerate(['MOp','SSp']):
	ax = axes[k+1,0]
	cell = (assay=='snmC-Seq')
	cell[cell] = (meta[:,0]==xx)
	ax.scatter(y[~cell, 0], y[~cell, 1], c='grey', s=5, edgecolors='none', alpha=0.3, rasterized=True)
	studytmp = study[region==xx]
	label = regcluster[k][studytmp=='RS2']
	nc = len(set(label))
	for i in range(nc):
		celltmp = cell.copy()
		celltmp[cell] = (label==i)
		ax.scatter(y[celltmp, 0], y[celltmp, 1], c=legcolor[i], s=5, edgecolors='none', alpha=0.8, rasterized=True)
		ax.text(np.median(y[celltmp,0]), np.median(y[celltmp,1]), str(i), fontsize=15, horizontalalignment='center', verticalalignment='center')

for k,g in enumerate(['Slco2a1','Astn2']):
	ax = axes[2,k+1]
	cell = (assay=='SMART-Seq')
	ax.scatter(y[cell, 0], y[cell, 1], c='grey', s=5, edgecolors='none', alpha=0.3, rasterized=True)
	mch = rate[:,genedict[g]]
	plot = ax.scatter(y[~cell, 0], y[~cell, 1], s=5, c=mch, alpha=0.8, edgecolors='none', cmap=cm.bwr, rasterized=True, vmin=np.percentile(mch,5), vmax=np.percentile(mch,95))
	# cbar = plt.colorbar(plot, ax=ax, anchor=(0,1))
	# vmin, vmax = np.around([np.percentile(mch,5), np.percentile(mch,95)], decimals=2)
	# cbar.solids.set_clim([vmin, vmax])
	# cbar.set_ticks([vmin, vmax])
	# cbar.draw_all()

for i,ax in enumerate(axes.flatten()):
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(axis='both', which='both', length=0)
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.set_title(['ALM Cluster', 'Retro-Seq L1 Target', 'Retro-Seq L2 Target', 'MOp Subcluster', 'Epi-Retro-Seq Source', 'Epi-Retro-Seq Target', 'SSp Subcluster', 'Slco2a1', 'Astn2'][i], fontsize=15)

for ax in axes[:,0]:
	ax.set_ylabel('t-SNE-2', fontsize=15)

for ax in axes[-1]:
	ax.set_xlabel('t-SNE-1', fontsize=15)

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_4541_L5ET_methyl_rateg_rna_hvg10k_zscore_sigma100_facsfilter.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.close()
