from sklearn.metrics import pairwise_distances

dist = pairwise_distances(rateb_reduce[:, :50])
idx = np.argsort(dist, axis=1)
dist = np.array([dist[i,idx[i]] for i in range(len(dist))])
nn = 25

roc = -np.ones((len(metaall), 4))
meta = metaall[:, 6:].copy()
meta[~facsfilter, 2] = 'Unknown'
for i,x in enumerate([4,0,2,3]):
	for j in range(len(meta)):
		ytmp = (meta[:,x]==meta[j,x])[idx[j]]
		xtmp = dist[j]
		if x==2:
			if meta[j,2]=='Unknown':
				continue
			cellfilter = (meta[:,x]!='Unknown')[idx[j]]
			ytmp = ytmp[cellfilter]
			xtmp = xtmp[cellfilter]
		elif x==3:
			if meta[j,2]=='Pons' and meta[j,0]=='AI':
				continue
			cellfilter = ~np.logical_and(meta[:,2]=='Pons', meta[:,0]=='AI')[idx[j]]
			ytmp = ytmp[cellfilter]
			xtmp = xtmp[cellfilter]
		k = int(np.around(np.sum(~ytmp)/np.sum(ytmp), decimals=0))
		xx = np.concatenate((xtmp[ytmp][1:(nn+1)],xtmp[~ytmp][:nn*k]))
		yy = np.array([0 for i in range(nn)] + [1 for i in range(nn*k)])
		roc[j,i] = roc_auc_score(yy, xx)
	print(i, np.mean(roc[roc[:,i]>-1,i]))

np.save(outdir + 'matrix/cell_11827_rateb_hvg2k_pc50_nn25_roc.npy', roc)


roc = np.load(outdir + 'matrix/cell_11827_rateb_hvg2k_pc50_nn25_roc.npy')

fig, ax = plt.subplots(figsize=(4,2))
sns.despine(ax=ax)
ax = sns.boxplot(data=[x[x>0] for x in roc.T], width=0.6, color='w', showfliers=False, ax=ax)
plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
plt.setp(ax.lines, color='k')
ax.set_xticks(np.arange(4))
ax.set_xticklabels(['Cluster', 'Source', 'Target', 'Replicate'], fontsize=10)
ax.set_ylabel('nAUROC', fontsize=12)
plt.tight_layout()
plt.savefig(outdir + 'plot/cell_11827_rateb_hvg2k_pc50_nn25_roc_facsfilter.pdf', transparent=True)
plt.close()
