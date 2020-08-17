legcolor = np.array(list(islice(cycle(['#CB4335','#F39C12','#F4D03F','#27AE60','#76D7C4','#3498DB','#8E44AD','#6D4C41','#FFAB91','#1A237E','#FFFF66','#004D40','#00FF33','#CCFFFF','#A1887F','#FFCDD2','#999966','#212121','#FF00FF']), 100)))
tarcolor = {x:y for x,y in zip(['Pons', 'SC', 'TH', 'VTA', 'MOp', 'SSp', 'ACA', 'VISp', 'STR', 'MY'], ['#0082c8','#2ca02c','#ff7f0e','#d62728','#9467bd','#8c564b','#e377c2','#000080','#bcbd22','#ffe119'])}

rateb = np.load(indir + 'cell_11827_rateb.mCH.npy')
rateg = np.load(indir + 'cell_11827_rateg.mCH.npy')
readb = np.load(indir + 'cell_11827_read_bin.mCH.npy')
readg = np.load(indir + 'cell_11827_read_gene.mCH.npy')
binfilter = np.load(indir + 'cell_16971_binfilter90_autosomal.npy')
genefilter = np.load(indir + 'cell_11827_gene_filter80_autosomal.npy')
readb = readb[:,binfilter]
readg = readg[:,genefilter]
metaall = np.load(indir + 'cell_11827_meta.npy')
cluster = np.load(indir + 'cell_11827_posterior_disp2k_pc100_nd50_p50.pc50.knn25.louvain_res1.2_nc25_cluster_label.npy')

metaall = np.concatenate((metaall[:, 12:16], cluster[:, None], metaall[:,[10,8]]), axis=1)

leg = np.array(['L2/3', 'L4', 'L5-IT', 'L6-IT'])
reg = np.array(['MOp', 'SSp', 'ACA', 'AI', 'AUD', 'RSP', 'PTLp', 'VIS'])
tar = np.array(['STR', 'MOp', 'SSp', 'ACA', 'VISp'])
corfilter = np.array([x in tar for x in metaall[:,2]])
itfilter = np.array([x in leg for x in metaall[:,4]])
cellfilter = np.logical_and(corfilter, itfilter)
rate = rateg[cellfilter]
read = readg[cellfilter]
meta = metaall[cellfilter]
bins = np.array([x[0]+'-'+x[1] for x in bin_all[binfilter]])
genes = gene_all[genefilter, 4]

disp = highly_variable_methylation_feature(rate, np.mean(read, axis=0), bins)
idx = np.argsort(disp)[::-1]
data = rate[:, idx[:2000]]
pca = PCA(n_components=50, random_state=0)
# umap = UMAP(n_neighbors=25, random_state=0)
tsne = MulticoreTSNE(perplexity=50, verbose=3, random_state=0, n_jobs=10)
rateb_reduce = pca.fit_transform(rate)
# y = umap.fit_transform(rateb_reduce[:, :50])
# np.savetxt(outdir + 'matrix/cell_5043_rateb_pc50_knn25_umap.txt', y, delimiter='\t', fmt='%s')
y = tsne.fit_transform(rateb_reduce[:, :50])
np.savetxt(outdir + 'matrix/cell_5043_rateg_pc50_p50_tsne.txt', y, delimiter='\t', fmt='%s')


cord = []
ndim = 30
for xx in reg:
	print(xx)
	ratetmp = rate[meta[:,0]==xx]
	readtmp = read[meta[:,0]==xx]
	disp = highly_variable_methylation_feature(ratetmp, np.mean(readtmp, axis=0), genes)
	idx = np.argsort(disp)[::-1]
	data = ratetmp[:, idx[:2000]]
	rate_reduce = pca.fit_transform(data)
	# y = umap.fit_transform(rate_reduce[:,:ndim])
	y = tsne.fit_transform(rate_reduce[:,:ndim])
	cord.append(y)

# np.save(outdir + 'matrix/cell_5043_source_rateb_disp2k_pc'+str(ndim)+'_knn25_umap.npy', cord)
np.save(outdir + 'matrix/cell_5043_source_rategp_disp2k_nozs_pc'+str(ndim)+'_p50_tsne.npy', cord)

# cord = np.load(outdir + 'matrix/cell_5043_source_rateb_pc30_p50_tsne.npy', allow_pickle=True)

fig, axes = plt.subplots(8,3,figsize=(10,16))
for i,xx in enumerate(reg):
	metatmp = meta[meta[:,0]==xx]
	y = cord[i]
	for j in range(3):
		ax = axes[i,j]
		if xx=='VISp':
			ax.set_xlabel('t-SNE-1', fontsize=15)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.tick_params(axis='both', which='both', length=0)
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		if j==1:
			for yy in tar:
				cell = (metatmp[:,2]==yy)
				ax.scatter(y[cell, 0], y[cell, 1], c=tarcolor[yy], s=8, edgecolors='none', alpha=0.8, label='->'+yy, rasterized=True)
		elif j==0:
			ax.set_ylabel(xx + '\nt-SNE-2', fontsize=15)
			for k,x in enumerate(leg):
				cell = (metatmp[:,4]==x)
				ax.scatter(y[cell, 0], y[cell, 1], c=legcolor[k], s=8, edgecolors='none', alpha=0.8, label=leg[k], rasterized=True)
		else:
			mch = np.log10(metatmp[:,5].astype(float))
			plot = ax.scatter(y[:, 0], y[:, 1], s=8, c=mch, alpha=0.8, edgecolors='none', cmap=cm.bwr, rasterized=True)
			cbar = plt.colorbar(plot, ax=ax)
			vmin, vmax = np.around([np.percentile(mch,5), np.percentile(mch,95)], decimals=2)
			cbar.solids.set_clim([vmin, vmax])
			cbar.set_ticks([vmin, vmax])
			cbar.draw_all()

axes[0,0].set_title('Cluster', fontsize=15)
axes[0,1].set_title('Target', fontsize=15)
axes[0,2].set_title('Non-clonal', fontsize=15)
plt.tight_layout()
plt.savefig(outdir + 'plot/cell_5043_source_rategp_disp2k_nozs_pc'+str(ndim)+'_p50_tar.pdf', transparent=True, bbox_inches="tight", dpi=300)
plt.close()


ndim = 30
xx = 'AUD'
# metatmp = meta[meta[:,0]==xx]
metatmp = np.concatenate((metaall[:,6:], metaall[:,2][:,None]), axis=1)
metatmp[~facsfilter, 2] = 'Unknown'
metatmp = metatmp[np.logical_and(np.logical_and(itfilter, corfilter), metatmp[:,0]==xx)]
cord = np.load(outdir + '../matrix/cell_5043_source_rategp_disp2k_pc'+str(ndim)+'_p50_tsne.npy', allow_pickle=True)
y = cord[4]
ds = 10
fig, axes = plt.subplots(1,3,figsize=(12,3), sharex='all', sharey='all')
for ax in axes:
	ax.set_xlabel('t-SNE-1', fontsize=15)
	ax.set_ylabel('t-SNE-2', fontsize=15)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(axis='both', which='both', length=0)
	ax.set_xticklabels([])
	ax.set_yticklabels([])

ax = axes[0]
for k,x in enumerate(leg):
	cell = (metatmp[:,4]==x)
	ax.scatter(y[cell, 0], y[cell, 1], c=legcolor[k], s=ds, edgecolors='none', alpha=1.0, label=leg[k], rasterized=True)
	ax.text(np.median(y[cell,0]), np.median(y[cell,1]), x, fontsize=20, horizontalalignment='center', verticalalignment='center', fontweight='bold')

ax = axes[1]
cell = (metatmp[:,2]=='SSp')
ax.scatter(y[cell, 0], y[cell, 1], c='C1', s=ds, edgecolors='none', alpha=0.8, label='->SSp', rasterized=True)
cell = (metatmp[:,2]=='VISp')
ax.scatter(y[cell, 0], y[cell, 1], c='C0', s=ds, edgecolors='none', alpha=0.8, label='->VISp', rasterized=True)
cell = np.logical_and(metatmp[:,2]!='SSp',metatmp[:,2]!='VISp')
ax.scatter(y[cell, 0], y[cell, 1], c='grey', s=5, edgecolors='none', alpha=0.3, label='Others', rasterized=True)
ax.legend(markerscale=2)

ax = axes[2]
cell = (metatmp[:,2]=='SSp')
ax.scatter(y[cell, 0], y[cell, 1], c='C1', s=ds, edgecolors='none', alpha=0.8, label='->SSp', rasterized=True)
cell = (metatmp[:,2]=='ACA')
ax.scatter(y[cell, 0], y[cell, 1], c='C2', s=ds, edgecolors='none', alpha=0.8, label='->ACA', rasterized=True)
cell = np.logical_and(metatmp[:,2]!='SSp',metatmp[:,2]!='ACA')
ax.scatter(y[cell, 0], y[cell, 1], c='grey', s=5, edgecolors='none', alpha=0.3, label='Others', rasterized=True)
ax.legend(markerscale=2)

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_737_AUDp_rategp_disp2k_pc'+str(ndim)+'_knn25_cluster_proj_facsfilter.pdf', transparent=True, bbox_inches="tight", dpi=300)
plt.close()
