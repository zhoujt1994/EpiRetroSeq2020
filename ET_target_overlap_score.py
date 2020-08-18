tar = np.array(['VISp', 'TH', 'SC', 'VTA', 'Pons', 'MY'])
overlap, tarall = [], []
for k,xx in enumerate(reg):
	studytmp = study[region==xx]
	projtmp = meta[meta[:,0]==xx, 2]
	label = regcluster[k][studytmp=='RS2']
	nc = len(set(label))
	count = np.array([[np.sum(np.logical_and(projtmp==xx, label==i)) for i in range(nc)] for xx in tar])
	tarfilter = (np.sum(count, axis=1)>20)
	count = count[tarfilter]
	tartmp = tar[tarfilter]
	ratio = count / np.sum(count, axis=1)[:,None]
	overlap.append(np.array([[np.sum(np.min(ratio[[i,j]], axis=0)) for i in range(len(tartmp))] for j in range(len(tartmp))]))
	tarall.append(tartmp)

rord, cord = [], []
for x in overlap:
	cg = clustermap(x, cmap='bwr', metric='euclidean')
	rord.append(cg.dendrogram_row.reordered_ind)
	cord.append(cg.dendrogram_col.reordered_ind)

fig, axes = plt.subplots(2, 4, figsize=(10,5))
for i,ax in enumerate(axes.flatten()):
	plot = ax.imshow(overlap[i][rord[i]][:, rord[i]], cmap='bwr', vmin=0.35, vmax=1.0)
	ax.set_xticks(range(len(tarall[i])))
	ax.set_yticks(range(len(tarall[i])))
	ax.set_yticklabels(tarall[i][rord[i]], fontsize=12)
	ax.set_xticklabels(tarall[i][rord[i]], fontsize=12, rotation=60, rotation_mode='anchor', va='top', ha='right')
	ax.set_title(reg[i], fontsize=15)
	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes('right', size='5%', pad='5%')
	# cbar = plt.colorbar(plot, cax=cax)
	# vmin, vmax = 0.5, 1.0
	# cbar.solids.set_clim([vmin, vmax])
	# cbar.set_ticks([vmin, vmax])
	# # cbar.set_label('AUROC', fontsize=15)
	# cbar.draw_all()

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_4176_region_tar_overlap_score_facsfilter.pdf', transparent=True)
plt.close()

