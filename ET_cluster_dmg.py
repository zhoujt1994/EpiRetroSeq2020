metaall = np.load(indir + 'cell_11827_meta.npy')
rateb = np.load(indir + 'cell_11827_rateb.mCH.npy')
rateg = np.load(indir + 'cell_11827_rateg.mCH.npy')
cluster = np.load(indir + 'cell_4176_L5ET_knn30_res1.6_15cluster_label.npy')
nc = len(set(cluster))

cellfilter = (metaall[:,-1]=='L5-ET')
print(np.sum(cellfilter))
meta = np.concatenate((metaall[cellfilter, 12:16], cluster[:, None], metaall[cellfilter, 8]), axis=1)
rateg = rateg[cellfilter]
rateb = rateb[cellfilter]

def ranksumpv(args):
	i,j = args
	print(i,j)
	global rateb, group
	rate1 = rateb[group==i]
	rate2 = rateb[group==j]
	pv = np.array([ranksums(rate1[:,k], rate2[:,k])[1] for k in range(rateb.shape[1])])
	return pv

reg = np.array(['MOp', 'SSp', 'ACA', 'AI', 'AUD', 'RSP', 'PTLp', 'VIS'])
group = cluster.copy()
paras = [[reg[i],reg[j]] for i in range(len(reg)-1) for j in range(i+1, len(reg))]
ncpus = 10
p = Pool(ncpus)
result = p.map(ranksumpv, paras)
p.close()

selb_reg = np.zeros(rateb.shape[1])
for pv in result:
	idx = np.argsort(pv)
	selb_reg[idx[:100]] = 1

ratereg = np.array([np.mean(rateb[group==x], axis=0) for x in reg])
Z = linkage(ratereg[:, selb_reg==1], method='average', metric='correlation')
fig, ax = plt.subplots(figsize=(1.5,3))
dendrogram(Z, ax=ax, labels=reg, leaf_rotation=0, orientation='right', link_color_func=lambda k:'k')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('Correlation')
plt.tight_layout()
plt.savefig(outdir + 'plot/cell_4176_L5ET_source_meanrateb_100marker_ave_corr_dendrogram.pdf', transparent=True)
plt.close()

y = np.loadtxt(indir + 'matrix/cell_4176_rateb_disp2k_nozs_pc50_knn25_umap.txt')
marker = ['Astn2', 'Fstl4', 'Efna5', 'Ptprt', 'Bace2', 'Galntl6', 'Slc44a5', 'Cadm1', 'Rora', 'Tbc1d1', 'Cacna1e', 'Tenm3', 'Erg', 'Etl4', 'Fam19a2']
fig, axes = plt.subplots(3,5,figsize=(20,9))
for k,ax in zip(marker, axes.flatten()):
	mch = rateg[:,genedict[k]]
	ax.set_frame_on(False)
	ax.axis('off')
	plot = ax.scatter(y[:, 0], y[:, 1], s=8, c=mch, alpha=0.8, edgecolors='none', cmap=cm.bwr, rasterized=True)
	ax.set_title(k, fontsize=20)
	cbar = plt.colorbar(plot, ax=ax)
	vmin, vmax = np.around([np.percentile(mch,5), np.percentile(mch,95)], decimals=2)
	cbar.solids.set_clim([vmin, vmax])
	cbar.set_ticks([vmin, vmax])
	cbar.draw_all()

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_4176_rateb_disp2k_nozs_pc50_knn25.marker.pdf', transparent=True, dpi=300)
plt.close()

data = pd.DataFrame(data=rateg, columns=gene[:,-1])
data['target'] = meta[:,2]
data['gender'] = meta[:,3]
data['cluster'] = meta[:,4]
data['global'] = meta[:,5]
data.to_csv(outdir + 'cell_4176_rateg.meta.txt', index=None, sep='\t')
para_list = np.array([[i,j] for i in range(nc-1) for j in range(i+1, nc)])
np.savetxt(outdir + 'para_list.txt', para_list, fmt='%s', delimiter='\t')


rateg_cluster = np.array([np.mean(rate[cluster==i], axis=0) for i in range(nc)])
selg = np.zeros(len(gene))
result, title = [], []
for i in range(nc-1):
	for j in range(i+1, nc):
		pv = np.loadtxt(indir + 'cluster_'+str(i)+'_'+str(j)+'.post.ga.m.wald.pvalue.txt')
		fdr = FDR(pv, 0.01, 'fdr_bh')[1]
		fc = (rateg_cluster[i] + 0.1) / (rateg_cluster[j] + 0.1)
		tmp = np.logical_and(fdr<0.01, np.abs(np.log2(fc))>np.log2(1.5))
		selg = np.logical_or(selg, tmp)
		print(i,j,np.sum(tmp))
		result = result + [fc, fdr]
		title = title + ['FC_'+str(i)+'_'+str(j), 'FDR_'+str(i)+'_'+str(j)]

print(np.sum(selg))

result = np.concatenate((gene[:,-2:], rateg_cluster.T, np.array(result).T), axis=1)
title = np.array(['Ensemble ID', 'Gene name'] + ['mCH_'+str(i) for i in range(nc)] + title)
np.savetxt(outdir + 'supp_table/L5-ET_proj_dmg.fc.fdr.txt', np.concatenate((title[None,:],result),axis=0), fmt='%s', delimiter='\t')
np.savetxt(indir + 'cell_4176_L5ET_15cluster_ovo_post_ga_m_pv01_fc50_pseudo10.txt', gene[selg,-1], fmt='%s', delimiter='\n')

mch = rateg_cluster[:,selg].T
leg = np.array([str(i) for i in range(nc)])
cg = clustermap(mch, metric='correlation', z_score=0)
rorder = cg.dendrogram_row.reordered_ind.copy()
corder = cg.dendrogram_col.reordered_ind.copy()
data = mch[rorder][:,corder]

fig, ax = plt.subplots(figsize=(5, 7))
plot = ax.imshow(zscore(data, axis=1), cmap='bwr', vmin=-2.5, vmax=2.5, aspect='auto')
ax.set_xticks(np.arange(len(leg)))
# ax.set_yticklabels(leg, fontsize=15, rotation=60, rotation_mode='anchor', ha='right')
ax.set_xticklabels(leg[corder], fontsize=15)
ax.set_yticks([])
ax.set_xlabel('Subclusters', fontsize=15)
ax.set_ylabel(str(len(data)) + ' DMGs', fontsize=15)
cbar = plt.colorbar(plot, ax=ax, shrink=0.3, fraction=0.05, orientation='horizontal')
cbar.set_ticks([-2.5, 2.5])
cbar.set_ticklabels([-2.5, 2.5])
cbar.set_label('Norm mCH', fontsize=15)
cbar.ax.yaxis.set_label_position('left')
cbar.draw_all()
plt.tight_layout()
plt.savefig(outdir + 'plot/cell_4176_L5ET_15cluster_ovo_ga_m_pv01_fc50_pseudo10_mch.pdf', transparent=True)
plt.close()
