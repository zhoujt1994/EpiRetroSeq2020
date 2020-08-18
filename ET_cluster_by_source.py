legcolor = np.array(list(islice(cycle(['#CB4335','#F39C12','#F4D03F','#27AE60','#76D7C4','#3498DB','#8E44AD','#6D4C41','#FFAB91','#1A237E','#FFFF66','#004D40','#00FF33','#CCFFFF','#A1887F','#FFCDD2','#999966','#212121','#FF00FF']), 100)))
tarcolor = {x:y for x,y in zip(['Pons', 'SC', 'TH', 'MB', 'MOp', 'SSp', 'ACA', 'VISp', 'STR', 'MY'], ['#0082c8','C2','C1','C3','C4','C5','C6','#000080','C8','#ffe119'])}
outdir = '/gale/netapp/home/zhoujt/project/CEMBA/RS2/merge_RS2/cortex/L5PT/'
reg = np.array(['MOp', 'SSp', 'ACA', 'AI', 'AUDp', 'RSP', 'PTLp', 'VISp'])
tar = np.array(['MOp', 'SSp', 'ACA', 'Pons', 'SC', 'TH', 'MB', 'MY', 'STR', 'VISp'])
etfilter = (metaall[:,4]=='L5-ET')
cluster = np.load(indir + 'cell_4176_L5ET_knn30_res1.6_15cluster_label.npy')
rate = rateb[cellfilter]
read = readb[cellfilter]
meta = metaall[cellfilter]
meta[:,4] = cluster.copy()

disp = highly_variable_methylation_feature(rate, np.mean(read, axis=0), bins)
idx = np.argsort(disp)[::-1]
data = rate[:, idx[:2000]]
pca = PCA(n_components=50, random_state=0)
umap = UMAP(n_neighbors=15, random_state=0)
# tsne = MulticoreTSNE(perplexity=50, verbose=3, random_state=0, n_jobs=10)
rateb_reduce = pca.fit_transform(data)
y = umap.fit_transform(rateb_reduce[:, :50])
np.savetxt(outdir + 'matrix/cell_4176_rateb_disp2k_nozs_pc50_knn25_umap.txt', y, delimiter='\t', fmt='%s')
# y = tsne.fit_transform(rateb_reduce[:, :50])
# np.savetxt(outdir + 'matrix/cell_5043_rateg_pc50_p50_tsne.txt', y, delimiter='\t', fmt='%s')
y = np.loadtxt(outdir + 'matrix/cell_4176_rateb_disp2k_nozs_pc50_knn25_umap.txt')

fig, axes = plt.subplots(1, 4, figsize=(25, 4))
for ax in axes:
	ax.set_xlabel('UMAP-1', fontsize=20)
	ax.set_ylabel('UMAP-2', fontsize=20)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(axis='both', which='both', length=0)
	ax.set_xticklabels([])
	ax.set_yticklabels([])

ax = axes[0]
bbox_props = dict(boxstyle='round,pad=0.05', fc='w', ec='k', alpha=0.5)
for i in range(15):
	cell = (meta[:,4]==str(i))
	ax.scatter(y[cell, 0], y[cell, 1], c=legcolor[i], s=8, edgecolors='none', alpha=0.8, label=str(i), rasterized=True)
	ax.text(np.median(y[cell,0]), np.median(y[cell,1]), str(i), fontsize=18, horizontalalignment='center', verticalalignment='center', bbox=bbox_props)

ax.legend(markerscale=5, prop={'size': 10}, bbox_to_anchor=(1,1), loc='upper left', fontsize=20)

ax = axes[1]
for i,x in enumerate(reg):
	cell = (meta[:,0]==x)
	ax.scatter(y[cell, 0], y[cell, 1], c=color[i], s=8, edgecolors='none', alpha=0.8, label=x, rasterized=True)

ax.legend(markerscale=5, prop={'size': 10}, bbox_to_anchor=(1,1), loc='upper left', fontsize=20)

ax = axes[2]
for i,x in enumerate(tar):
	cell = (meta[:,2]==x)
	ax.scatter(y[cell, 0], y[cell, 1], c=tarcolor[x], s=8, edgecolors='none', alpha=0.8, label='->'+x, rasterized=True)

ax.legend(markerscale=5, prop={'size': 10}, bbox_to_anchor=(1,1), loc='upper left', fontsize=20)

ax = axes[3]
mch = np.log10(meta[:,5].astype(float))
plot = ax.scatter(y[:, 0], y[:, 1], s=8, c=mch, alpha=0.8, edgecolors='none', cmap=cm.bwr, rasterized=True)
cbar = plt.colorbar(plot, ax=ax)
vmin, vmax = np.around([np.percentile(mch,5), np.percentile(mch,95)], decimals=2)
cbar.solids.set_clim([vmin, vmax])
cbar.set_ticks([vmin, vmax])
cbar.draw_all()

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_4176_rateb_disp2k_nozs_pc50_knn25.meta.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.close()


for ndim in [30, 50]:
	for dr,tmp in enumerate([[15,25],[30,50]]):
		for nn in tmp:
			if dr==0:
				xl, yl = 'UMAP-1', 'UMAP-2'
				pref = 'cell_4176_source_rateb_disp2k_nozs_pc' + str(ndim) + '_knn' + str(nn)
			else:
				xl, yl = 't-SNE-1', 't-SNE-2'
				pref = 'cell_4176_source_rateb_disp2k_nozs_pc' + str(ndim) + '_p' + str(nn)
			cord = []
			for xx in reg:
				print(xx)
				ratetmp = rate[meta[:,0]==xx]
				readtmp = read[meta[:,0]==xx]
				disp = highly_variable_methylation_feature(ratetmp, np.mean(readtmp, axis=0), bins)
				idx = np.argsort(disp)[::-1]
				data = ratetmp[:, idx[:2000]]
				rate_reduce = pca.fit_transform(data)
				if dr==0:
					umap = UMAP(n_neighbors=nn, random_state=0)
					y = umap.fit_transform(rate_reduce[:,:ndim])
				else:
					tsne = MulticoreTSNE(perplexity=nn, verbose=3, random_state=0, n_jobs=10)
					y = tsne.fit_transform(rate_reduce[:,:ndim])
				cord.append(y)
			np.save(outdir + 'matrix/' + pref + '.npy', cord)
			fig, axes = plt.subplots(8,3,figsize=(10,16))
			for i,xx in enumerate(reg):
				metatmp = meta[meta[:,0]==xx]
				y = cord[i]
				for j in range(3):
					ax = axes[i,j]
					if xx=='VISp':
						ax.set_xlabel(xl, fontsize=15)
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
						ax.set_ylabel(xx + '\n' + yl, fontsize=15)
						label = np.load(indir + clusterfiles[i])
						nc = len(set(label))
						for k in range(nc):
							cell = (label==k)
							ax.scatter(y[cell, 0], y[cell, 1], c=legcolor[k], s=8, edgecolors='none', alpha=0.8, label=str(k), rasterized=True)
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
			plt.savefig(outdir + 'plot/' + pref + '_tar.pdf', transparent=True, bbox_inches='tight', dpi=300)
			plt.close()

xl, yl = 'UMAP-1', 'UMAP-2'
pref = 'cell_4176_source_rateb_pcgiven_knngiven'
cord = []
for i,xx in enumerate(reg):
	print(xx)
	ndim = [50, 50, 30, 30, 25, 25, 25, 30][i]
	nn = [15,15,10,10,25,15,15,15][i]
	ratetmp = rate[meta[:,0]==xx]
	readtmp = read[meta[:,0]==xx]
	disp = highly_variable_methylation_feature(ratetmp, np.mean(readtmp, axis=0), bins)
	idx = np.argsort(disp)[::-1]
	data = ratetmp[:, idx[:2000]]
	rate_reduce = pca.fit_transform(data)
	umap = UMAP(n_neighbors=nn, random_state=0)
	y = umap.fit_transform(rate_reduce[:,:ndim])
	cord.append(y)

np.save(outdir + 'matrix/' + pref + '.npy', cord)
fig, axes = plt.subplots(8,3,figsize=(10,16))
for i,xx in enumerate(reg):
	metatmp = meta[meta[:,0]==xx]
	y = cord[i]
	for j in range(3):
		ax = axes[i,j]
		if xx=='VISp':
			ax.set_xlabel(xl, fontsize=15)
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
			ax.set_ylabel(xx + '\n' + yl, fontsize=15)
			label = np.load(indir + clusterfiles[i])
			nc = len(set(label))
			for k in range(nc):
				cell = (label==k)
				ax.scatter(y[cell, 0], y[cell, 1], c=legcolor[k], s=8, edgecolors='none', alpha=0.8, label=str(k), rasterized=True)
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
plt.savefig(outdir + 'plot/' + pref + '_tar.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.close()



def dmgfind(args):
	i,j = args
	print(i,j)
	global ratetmp, label
	rate1 = ratetmp[label==i]
	rate2 = ratetmp[label==j]
	fc = np.log2((np.mean(rate1, axis=0) + 0.1) / (np.mean(rate2, axis=0) + 0.1))
	pv = np.array([ranksums(rate1[:,k],rate2[:,k])[1] for k in range(ratetmp.shape[1])])
	fdr = FDR(pv, 0.01, 'fdr_bh')
	return [i,j,fc,pv]

dmgcount5 = []
for k, xx in enumerate(reg):
	label = np.load(indir + clusterfiles[k])
	metatmp = meta[meta[:,0]==xx]
	ratetmp = rate[meta[:,0]==xx]
	nc = len(set(label))
	print(xx, nc)
	paras = [[i,j] for i in range(nc-1) for j in range(i+1, nc)]
	p = Pool(ncpus)
	result = p.map(dmgfind, paras)
	p.close()
	tmpcount = np.zeros((nc, nc))
	for x in result:
		tmpcount[x[0],x[1]] = np.sum(np.logical_and(x[2]<-np.log2(1.5), x[3]<0.01))
		tmpcount[x[1],x[0]] = np.sum(np.logical_and(x[2]>np.log2(1.5), x[3]<0.01))
	dmgcount5.append(tmpcount)




indir = '/gale/netapp/home/zhoujt/project/CEMBA/RS1/L5ET/matrix/'
metars1 = pd.read_msgpack(indir + 'CellTidyData.PT-L5.msg')
f = h5py.File(indir + 'PT-L5.mcds', 'r')
metars1 = metars1.loc[f['cell'][()]]
rs1filter = np.array([x in reg for x in metars1['SubRegion']])
mcrs1 = f['chrom100k_da'][:,:,1,0,0]
tcrs1 = f['chrom100k_da'][:,:,1,0,1]
binrs1 = np.array([f['chrom100k_chrom'][()],f['chrom100k_bin_start'][()]]).T.astype(np.str)
bindict = {'-'.join(x):i for i,x in enumerate(binrs1)}
orderbin = np.array([bindict['-'.join(x[:2])] for x in bin_all[binfilter]])
mcrs1 = mcrs1[rs1filter][:,orderbin]
tcrs1 = tcrs1[rs1filter][:,orderbin]
metars1 = metars1[rs1filter]
ratebrs1 = calculate_posterior_mc_rate(mcrs1, tcrs1)
ratemerge = np.concatenate((ratebrs1, rateb[cellfilter]), axis=0)
tc = np.concatenate((tcrs1, readb[cellfilter]), axis=0)
region = np.concatenate((metars1['SubRegion'], meta[:,0]))
study = np.array(['RS1' for i in range(len(metars1))] + ['RS2' for i in range(len(meta))])
cov = np.concatenate((metars1['FinalReads'], meta[:,5])).astype(float)

mcrs1 = f['gene_da'][:,:,1,0,0][rs1filter]
tcrs1 = f['gene_da'][:,:,1,0,1][rs1filter]
geners1 = f['gene'][()]
rs1gdict = {x[4]:-1 for x in gene}
for i,x in enumerate(geners1):
	rs1gdict[x.split('.')[0]] = i

ordergene = np.array([rs1gdict[x[4]] for x in gene])
mcrs1 = mcrs1[:,ordergene]
tcrs1 = tcrs1[:,ordergene]
mcrs1 = mcrs1[:,ordergene!=-1]
tcrs1 = tcrs1[:,ordergene!=-1]
rategrs1 = calculate_posterior_mc_rate(mcrs1, tcrs1)
rate = np.concatenate((rategrs1, rateg[etfilter][:,ordergene!=-1]), axis=0)

nn = 25
umap = UMAP(n_neighbors=nn, random_state=0)
cord = []
for i,xx in enumerate(reg):
	print(xx)
	ratetmp = ratemerge[region==xx]
	readtmp = tc[region==xx]
	disp = highly_variable_methylation_feature(ratetmp, np.mean(readtmp, axis=0), bins)
	idx = np.argsort(disp)[::-1]
	data = zscore(ratetmp[:, idx[:2000]], axis=0)
	rate_reduce = pca.fit_transform(data)
	y = umap.fit_transform(rate_reduce[:,:30])
	cord.append(y)

np.save(outdir + 'matrix/cell_5506_RS1_RS2_L5ET_source_disp2k_pc30_knn' + str(nn) + '_umap.npy', cord)


nn = 25
cord = np.load(outdir + '../L5PT/matrix/cell_5506_RS1_RS2_L5ET_source_disp2k_pc30_knn25_umap.npy', allow_pickle=True)
regcluster = np.load(outdir + '../L5PT/matrix/cell_5506_RS1_RS2_L5ET_source_disp2k_pc30_knn25_cluster.npy', allow_pickle=True)
meta = metaall.copy()
meta[~facsfilter, 8] = 'Unknown'
meta = meta[meta[:,10]=='L5-ET']
meta = meta[:, [6,7,8,9,10,4,2]]
tar = ['Unbiased', 'VISp', 'TH', 'SC', 'VTA', 'Pons', 'MY']
fig, axes = plt.subplots(8, 9, figsize=(27,16), sharex='row', sharey='row')
for i,xx in enumerate(reg):
	studytmp = study[region==xx]
	projtmp = np.array(['Unbiased' for j in range(len(studytmp))])
	projtmp[studytmp=='RS2'] = meta[meta[:,0]==xx, 2]
	covtmp = cov[region==xx]
	label = regcluster[i]
	nc = len(set(label))
	y = cord[i]
	ax = axes[i,0]
	for j in range(nc):
		cell = (label==j)
		ax.scatter(y[cell, 0], y[cell, 1], c=legcolor[j], s=10, edgecolors='none', alpha=0.8, rasterized=True)
		ax.text(np.median(y[cell,0]), np.median(y[cell,1]), str(j), fontsize=15, horizontalalignment='center', verticalalignment='center', rasterized=True)
	# ax = axes[i,1]
	# cell = (studytmp=='RS2')
	# ax.scatter(y[cell, 0], y[cell, 1], c='grey', s=5, edgecolors='none', alpha=0.3, rasterized=True)
	# for j in range(nc):
	# 	cell = np.logical_and(studytmp=='RS1', label==j)
	# 	ax.scatter(y[cell, 0], y[cell, 1], c=legcolor[j], s=10, edgecolors='none', alpha=0.8, rasterized=True)	
	for k,yy in enumerate(tar):
		ax = axes[i,k+1]
		cell = (projtmp==yy)
		ax.scatter(y[~cell, 0], y[~cell, 1], c='grey', s=5, edgecolors='none', alpha=0.3, rasterized=True)
		for j in range(nc):
			celltmp = np.logical_and(cell, label==j)
			ax.scatter(y[celltmp, 0], y[celltmp, 1], c=legcolor[j], s=10, edgecolors='none', alpha=0.8, rasterized=True)	
	ax = axes[i,-1]
	mch = np.log10(covtmp)
	plot = ax.scatter(y[:, 0], y[:, 1], s=8, c=mch, alpha=0.8, edgecolors='none', cmap=cm.bwr, rasterized=True)
	cbar = plt.colorbar(plot, ax=ax)
	vmin, vmax = np.around([np.percentile(mch,5), np.percentile(mch,95)], decimals=2)
	cbar.solids.set_clim([vmin, vmax])
	cbar.set_ticks([vmin, vmax])
	cbar.draw_all()
	for ax in axes[i]:
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.tick_params(axis='both', which='both', length=0)
		ax.set_xticklabels([])
		ax.set_yticklabels([])

for xx,ax in zip(reg,axes[:,0]):
	ax.set_ylabel(xx + '\nUMAP-2', fontsize=15)

for ax in axes[-1]:
	ax.set_xlabel('UMAP-1', fontsize=15)

for yy,ax in zip(['Cluster']+['->'+x for x in tar]+['# reads'], axes[0]):
	ax.set_title(yy, fontsize=15)

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_5506_RS1_RS2_L5ET_source_disp2k_pc30_knn25_study_tar_cluster_facsfilter.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.close()



for i,xx in enumerate(reg):
	print(xx)
	ratetmp = ratemerge[region==xx]
	readtmp = tc[region==xx]
	disp = highly_variable_methylation_feature(ratetmp, np.mean(readtmp, axis=0), bins)
	idx = np.argsort(disp)[::-1]
	data = zscore(ratetmp[:, idx[:2000]], axis=0)
	rate_reduce = pca.fit_transform(data)
	y = cord[i]
	g = knn(rate_reduce[:, :30], n_neighbors=25)
	inter = g.dot(g.T)
	diag = inter.diagonal()
	jac = inter.astype(float)/(diag[None,:]+diag[:,None]-inter)
	adj = nx.from_numpy_matrix(g.multiply(jac).toarray())
	knnjaccluster = {}
	for res in [0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0]:
		partition = community.best_partition(adj,resolution=res)
		label = np.array([k for k in partition.values()])
		knnjaccluster[res] = label
		nc = len(set(label))
		count = np.array([sum(label==k) for k in range(nc)])
		print(xx,res,count)
	np.save(outdir + 'matrix/cell_'+str(len(y))+'_'+xx+'_disp2k_pc30_knn25_louvain.npy', knnjaccluster)

for i,xx in enumerate(reg):
	y = cord[i]
	knnjaccluster = np.load(outdir + 'matrix/cell_'+str(len(y))+'_'+xx+'_disp2k_pc30_knn25_louvain.npy', allow_pickle=True).item()
	fig, axes = plt.subplots(2,4,figsize=(20,8))
	for ax,res in zip(axes.flatten(), [0.8, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0]):
		label = knnjaccluster[res]
		nc = len(set(label))
		ax.set_frame_on(False)
		ax.axis('off')
		ax.set_title(str(res)+' ('+str(nc)+')', fontsize=15)
		for k in range(nc):
			cell = (label==k)
			ax.scatter(y[cell,0], y[cell,1], s=8, c=legcolor[k], alpha=0.8, edgecolors='none')
			ax.text(np.median(y[cell,0]), np.median(y[cell,1]), str(k), fontsize=12, horizontalalignment='center', verticalalignment='center', rasterized=True)
	plt.tight_layout()
	plt.savefig(outdir + 'plot/cell_'+str(len(y))+'_'+xx+'_disp2k_pc30_knn25_louvain.pdf', bbox_inches="tight", dpi=200)
	plt.close()

selres = [1.6, 2.0, 2.5, 1.6, 1.6, 1.6, 2.0, 1.0]
dmgcount = []
for k, xx in enumerate(reg):
	y = cord[k]
	knnjaccluster = np.load(outdir + 'matrix/cell_'+str(len(y))+'_'+xx+'_disp2k_pc30_knn25_louvain.npy', allow_pickle=True).item()
	label = knnjaccluster[selres[k]]
	ratetmp = rate[region==xx]
	nc = len(set(label))
	print(xx, nc)
	paras = [[i,j] for i in range(nc-1) for j in range(i+1, nc)]
	p = Pool(ncpus)
	result = p.map(dmgfind, paras)
	p.close()
	tmpcount = np.zeros((nc, nc))
	for x in result:
		tmpcount[x[0],x[1]] = np.sum(np.logical_and(x[2]<-np.log2(1.5), x[3]<0.01))
		tmpcount[x[1],x[0]] = np.sum(np.logical_and(x[2]>np.log2(1.5), x[3]<0.01))
	dmgcount.append(tmpcount)

dmbcount = []
for k, xx in enumerate(reg):
	y = cord[k]
	knnjaccluster = np.load(outdir + 'matrix/cell_'+str(len(y))+'_'+xx+'_disp2k_pc30_knn25_louvain.npy', allow_pickle=True).item()
	label = knnjaccluster[selres[k]]
	ratetmp = ratemerge[region==xx]
	nc = len(set(label))
	print(xx, nc)
	paras = [[i,j] for i in range(nc-1) for j in range(i+1, nc)]
	p = Pool(ncpus)
	result = p.map(dmgfind, paras)
	p.close()
	tmpcount = np.zeros((nc, nc))
	for x in result:
		tmpcount[x[0],x[1]] = np.sum(np.logical_and(x[2]<-np.log2(1.5), x[3]<0.01))
		tmpcount[x[1],x[0]] = np.sum(np.logical_and(x[2]>np.log2(1.5), x[3]<0.01))
	dmbcount.append(tmpcount)

def dmgrocpr(args):
	i,j = args
	print(i,j)
	global ratetmp, label
	cell = np.logical_or(label==i, label==j)
	rate0 = ratetmp[cell]
	if np.sum(label==i) < np.sum(label==j):
		label0 = (label[cell]==i)
		k = -1
	else:
		label0 = (label[cell]==j)
		k = 1
	ngene = rate0.shape[1]
	rocpr = np.array([[roc_auc_score(label0, k*xx), average_precision_score(label0, k*xx), average_precision_score(label0, -k*xx)] for xx in rate0.T])
	return [i,j,rocpr[:,0],rocpr[:,1],rocpr[:,2]]


dmbcount_rocpr = []
for k, xx in enumerate(reg):
	y = cord[k]
	knnjaccluster = np.load(outdir + 'matrix/cell_'+str(len(y))+'_'+xx+'_disp2k_pc30_knn25_louvain.npy', allow_pickle=True).item()
	label = knnjaccluster[selres[k]]
	ratetmp = ratemerge[region==xx]
	nc = len(set(label))
	print(xx, nc)
	paras = [[i,j] for i in range(nc-1) for j in range(i+1, nc)]
	p = Pool(ncpus)
	result = p.map(dmgrocpr, paras)
	p.close()
	tmpcount = np.zeros((nc, nc))
	for x in result:
		tmpcount[x[0],x[1]] = np.sum(np.logical_and(x[2]>0.85, x[3]>0.6))
		tmpcount[x[1],x[0]] = np.sum(np.logical_and(x[2]<0.15, x[4]>0.6))
	dmbcount_rocpr.append(tmpcount)

np.save(outdir + 'matrix/cell_5506_RS1_RS2_L5ET_source_disp2k_pc30_knn25_cluster_dmbcount_rocpr.npy', dmbcount_rocpr)

regcluster = []
for k, xx in enumerate(reg):
	y = cord[k]
	knnjaccluster = np.load(outdir + 'matrix/cell_'+str(len(y))+'_'+xx+'_disp2k_pc30_knn25_louvain.npy', allow_pickle=True).item()
	label = knnjaccluster[selres[k]]
	if xx=='RSP':
		trans = np.array([0,1,2,1,3,4])
		label = trans[label]
	regcluster.append(label)

np.save(outdir + 'matrix/cell_5506_RS1_RS2_L5ET_source_disp2k_pc30_knn25_cluster.npy', regcluster)
