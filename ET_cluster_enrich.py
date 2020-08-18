enrich, tarall = [], []
rorder, corder = [], []
for k, xx in enumerate(reg):
	studytmp = study[region==xx]
	metatmp = np.concatenate((meta[meta[:,0]==xx][:,[2,3]], regcluster[k][studytmp=='RS2'][:,None]), axis=1)
	count = np.array([np.sum(metatmp[:,0]==yy) for yy in tar])
	tartmp = tar[count>20]
	cellfilter = np.array([x in tartmp for x in metatmp[:,0]])
	metatmp = metatmp[cellfilter]
	np.savetxt(outdir + 'dmg/enrich/' + xx + '_meta_cluster.txt', metatmp, fmt='%s', delimiter='\t')
	legtmp = np.sort(list(set(metatmp[:,2])))
	count = np.array([[np.sum(np.logical_and(metatmp[:,0]==yy, metatmp[:,2]==xx)) for yy in tartmp] for xx in legtmp]) + 1
	# enrichtmp = np.array([[count[i,j] / np.sum(count[:,j]) / (np.sum(count[i]) - count[i,j]) * (np.sum(count) - np.sum(count[:,j])) for j in range(len(tartmp))] for i in range(len(legtmp))])
	exp = chi2_contingency(count)[3]
	enrichtmp = (count - exp) / exp
	tarall.append(tartmp)
	enrich.append(enrichtmp)
	cg = clustermap(enrichtmp)
	rorder.append(cg.dendrogram_row.reordered_ind.copy())
	corder.append(cg.dendrogram_col.reordered_ind.copy())

from statsmodels.sandbox.stats.multicomp import multipletests as FDR

fdrall = []
fig, axes = plt.subplots(2, 4, figsize=(5, 4))
for i,xx in enumerate(reg):
	legtmp = np.array([str(j) for j in range(len(enrich[i]))])
	tartmp = tarall[i]
	ax = axes.flatten()[i]
	# obs = np.array([[np.sum(np.logical_and(projtmp==yy, label==zz)) for yy in tartmp] for zz in legtmp])
	# exp = chi2_contingency(obs)[3]
	data = enrich[i]
	pv = np.loadtxt(outdir + 'dmg/enrich/' + xx + '_proj_cluster_enrich_lmm_pvalue.txt')
	fdr = FDR(pv.flatten(), 0.1, 'fdr_bh')[1].reshape(pv.shape)
	fdrall.append(fdr)
	sig = np.where(fdr[rorder[i]][:,corder[i]]<0.05)
	plot = ax.imshow(data[rorder[i]][:,corder[i]], cmap='bwr', vmin=-2, vmax=2)
	ax.scatter(sig[1], sig[0], marker='*')
	ax.set_yticks(np.arange(len(legtmp)))
	ax.set_yticklabels(legtmp[rorder[i]])
	ax.set_xticks(np.arange(len(tartmp)))
	ax.set_xticklabels(tartmp[corder[i]], rotation_mode='anchor', rotation=60, ha='right')
	ax.set_title(xx)

plt.tight_layout()
plt.savefig(outdir + 'plot/L5-ET_proj_cluster_enrich_reg_facsfilter.pdf', transparent=True)
plt.close()

