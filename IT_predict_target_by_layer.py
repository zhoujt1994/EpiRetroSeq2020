leg = np.array(['L2/3', 'L4', 'L5-IT', 'L6-IT'])
reg = np.array(['MOp', 'SSp', 'ACA', 'AI', 'AUDp', 'RSP', 'PTLp', 'VISp'])
tar = np.array(['STR', 'MOp', 'SSp', 'ACA', 'VISp'])

clf = LogisticRegression(class_weight='balanced')
meta = metait.copy()
perform = []
for k1 in range(len(tar)-1):
	for k2 in range(k1+1, len(tar)):
		tar1, tar2 = tar[[k1,k2]]
		for xx in reg:
			for yy in leg:
				tarfilter = [np.logical_and(np.logical_and(meta[:,0]==xx, meta[:,4]==yy), np.logical_and(meta[:,2]==t, meta[:,3]==g)) for t in [tar1, tar2] for g in ['male', 'female']]
				tot = np.array([np.sum(tarfilter[i]) for i in range(4)])
				if np.sum(tot==0)>0:
					continue
				cellfilter = np.logical_or(np.logical_or(tarfilter[0], tarfilter[2]), np.logical_or(tarfilter[1], tarfilter[3]))
				metatmp = meta[cellfilter]
				ratetmp = rate[cellfilter]
				roctmp = []
				for k in ['male', 'female']:
					trainfilter = (metatmp[:,3]!=k)
					testfilter = (metatmp[:,3]==k)
					pred = clf.fit(ratetmp[trainfilter], metatmp[trainfilter, 2]==tar1).predict_proba(ratetmp[testfilter])[:,1]
					roctmp.append(roc_auc_score(metatmp[testfilter, 2]==tar1, pred))
				perform.append([xx, yy, tar1, tar2] + tot.tolist() + [np.mean(roctmp)])
				print(perform[-1])

perform = np.array(perform)
np.save(outdir + 'matrix/cell_3513_cor_cor_layer_pred_gene.npy', perform)


count = np.array([[[np.logical_and(np.logical_and(meta[:,0]==xx, meta[:,2]==yy), meta[:,4]==zz).sum() for zz in leg] for xx in reg] for yy in tar])
perform = []
for k1 in range(len(tar)-1):
	for k2 in range(k1+1, len(tar)):
		tar1, tar2 = tar[[k1, k2]]
		for i,xx in enumerate(reg):
			for j,yy in enumerate(leg):
				if count[k1,i,j]>5 and count[k2,i,j]>5:
					perform.append([xx, yy, tar1, tar2, count[k1,i,j], count[k2,i,j]])

for t in range(50):
	meta = metait.copy()
	selid = []
	for xx in reg:
		for yy in tar:
			for zz in leg:
				idx = np.where(np.logical_and(np.logical_and(meta[:,0]==xx, meta[:,2]==yy), meta[:,4]==zz))[0]
				if len(idx)>1:
					selid.append(np.random.choice(idx, int(len(idx)//2), False))
	selid = np.sort(np.concatenate(selid))
	meta[:,3] = 'female'
	meta[selid, 3] = 'male'
	tot = 0
	for k1 in range(len(tar)-1):
		for k2 in range(k1+1, len(tar)):
			tar1, tar2 = tar[[k1, k2]]
			for i,xx in enumerate(reg):
				for j,yy in enumerate(leg):
					if count[k1,i,j]>5 and count[k2,i,j]>5:
						cellfilter = np.logical_and(np.logical_and(meta[:,0]==xx, meta[:,4]==yy), np.logical_or(meta[:,2]==tar1, meta[:,2]==tar2))
						metatmp = meta[cellfilter]
						ratetmp = rate[cellfilter]
						roctmp = []
						for k in ['male', 'female']:
							trainfilter = (metatmp[:,3]!=k)
							testfilter = (metatmp[:,3]==k)
							pred = clf.fit(ratetmp[trainfilter], metatmp[trainfilter, 2]==tar1).predict_proba(ratetmp[testfilter])[:,1]
							roctmp.append(roc_auc_score(metatmp[testfilter, 2]==tar1, pred))
						perform[tot].append(np.mean(roctmp))
						print(t, tot, tar1, tar2, xx, yy, perform[tot][-1])
						tot += 1

perform = np.array(perform)
np.save(outdir + 'matrix/cell_3513_cor_cor_layer_pred_gene_cv50.npy', perform)

perform = np.concatenate((perform[:,:6], np.mean(perform[:,6:].astype(float), axis=1)[:,None]), axis=1)
# perform = np.load(outdir + 'matrix/cell_3800_cor_cor_layer_pred.npy')
# tmpfilter = ~np.logical_and(perform[:,0]=='AI', np.logical_or(perform[:,2]=='MOp', perform[:,3]=='MOp'))
# perform = perform[tmpfilter]
label = np.array(['-'.join(x[:4]) for x in perform])
cmap = cm.bwr
cmap.set_bad('k')
tot = 0
fig, axes = plt.subplots(1, 6, figsize=(12,3))
for k1 in range(len(tar)-1):
	for k2 in range(k1+1, len(tar)):
		tar1, tar2 = tar[k1], tar[k2]
		dataall, labelall = [], []
		for xx in reg:
			datatmp = []
			for yy in leg:
				tmp = '-'.join([xx, yy, tar1, tar2])
				if tmp in label:
					datatmp.append(float(perform[label==tmp, -1]))
					# datatmp.append(np.sum(perform[label==tmp, 4:8].astype(int)))
				else:
					datatmp.append(np.nan)
					# datatmp.append(0)
			if np.sum(np.isnan(datatmp))<4:
				dataall.append(datatmp)
				labelall.append(xx)
		ax = axes.flatten()[tot]
		plot = ax.imshow(dataall, cmap=cmap, vmin=0.5, vmax=1.0)
		ax.set_yticks(range(len(labelall)))
		ax.set_yticklabels(labelall, fontsize=12)
		ax.set_xticks(range(4))
		ax.set_xticklabels(leg, fontsize=12, rotation=60, rotation_mode='anchor', va='top', ha='right')
		ax.set_title(tar1 + '/' + tar2)
		# divider = make_axes_locatable(ax)
		# cax = divider.append_axes('right', size='5%', pad='5%')
		# cbar = plt.colorbar(plot, cax=cax)
		# # vmin, vmax = 3, 9
		# vmin, vmax = 0.5, 1.0
		# cbar.solids.set_clim([vmin, vmax])
		# cbar.set_ticks([vmin, vmax])
		# cbar.set_label('AUROC', fontsize=12)
		# # cbar.set_label('log2 sample size', fontsize=15)
		# cbar.draw_all()
		tot += 1

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_3513_cor_cor_layer_pred_gene_cv50_roc_heatmap.pdf', transparent=True)
plt.close()
