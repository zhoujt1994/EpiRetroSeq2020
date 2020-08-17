def upsample(data, num):
	if len(data)==num:
		return data
	idx = np.random.choice(np.arange(len(data)), num, True)
	return data[idx]

meta = metait.copy()
perform = []
for k1 in range(len(tar)-1):
	for k2 in range(k1+1, len(tar)):
		tar1, tar2 = tar[[k1,k2]]
		for xx in reg:
			cellfilter = np.logical_and(meta[:,0]==xx, np.logical_or(meta[:,2]==tar1, meta[:,2]==tar2))
			ratetmp = rate[cellfilter]
			metatmp = meta[cellfilter]
			count = np.array([[[np.sum(np.logical_and(np.logical_and(metatmp[:,2]==yy, metatmp[:,4]==zz), metatmp[:,3]==kk)) for zz in leg] for kk in ['male', 'female']] for yy in [tar1, tar2]])
			tot = np.sum(count, axis=2)[:,:,None]
			legfilter = (np.sum(np.logical_and(count>0, count<tot), axis=(0,1))==4)
			legtmp = leg[legfilter]
			counttmp = np.max(count, axis=0)[:,legfilter]
			nc = len(legtmp)
			if nc<2:
				continue
			roc_tmp = np.zeros((nc,2,50))
			for i,yy in enumerate(['male', 'female']):
				test_x, test_y = [], []
				for k,zz in enumerate(legtmp):
					tarfilter1 = np.logical_and(np.logical_and(metatmp[:,2]==tar1, metatmp[:,4]==zz), metatmp[:,3]!=yy)
					tarfilter2 = np.logical_and(np.logical_and(metatmp[:,2]==tar2, metatmp[:,4]==zz), metatmp[:,3]!=yy)
					test_x.append(np.concatenate([ratetmp[tarfilter1], ratetmp[tarfilter2]], axis=0))
					test_y.append(np.repeat([0,1], [np.sum(tarfilter1), np.sum(tarfilter2)]))
				for t in range(50):
					datatmp = []
					for k,zz in enumerate(legtmp):
						tarfilter1 = np.logical_and(np.logical_and(metatmp[:,2]==tar1, metatmp[:,4]==zz), metatmp[:,3]==yy)
						tarfilter2 = np.logical_and(np.logical_and(metatmp[:,2]==tar2, metatmp[:,4]==zz), metatmp[:,3]==yy)
						datatmp.append(np.concatenate([upsample(ratetmp[tarfilter1], counttmp[i,k]), upsample(ratetmp[tarfilter2], counttmp[i,k])], axis=0))
					for k,zz in enumerate(legtmp):
						train_x = np.concatenate([datatmp[l] for l in range(nc) if (l!=k)])
						train_y = np.concatenate([np.repeat([0,1], counttmp[i,l]) for l in range(nc) if (l!=k)])
						# train_x = datatmp[k]
						# train_y = np.repeat([0,1], counttmp[k])
						# tarfilter1 = np.logical_and(metatmp[:,2]==tar1, metatmp[:,4]!=zz)
						# tarfilter2 = np.logical_and(metatmp[:,2]==tar2, metatmp[:,4]!=zz)
						# test_x = np.concatenate([ratetmp[tarfilter1], ratetmp[tarfilter2]], axis=0)
						# test_y = np.repeat([0,1], [np.sum(tarfilter1), np.sum(tarfilter2)])
						pred = clf.fit(train_x, train_y).predict_proba(test_x[k])[:,1]
						roc_tmp[k,i,t] = roc_auc_score(test_y[k], pred)
			for k,zz in enumerate(legtmp):
				perform.append([xx, zz, tar1, tar2, np.mean(roc_tmp[k])])
				print(perform[-1])

perform = np.array(perform)
np.save(outdir + 'matrix/cell_3513_cor_cor_cross_layer_pred_gene_test1layer_heterogender_upsample50.npy', perform)


count = np.array([[[np.logical_and(np.logical_and(meta[:,0]==xx, meta[:,2]==yy), meta[:,4]==zz).sum() for zz in leg] for xx in reg] for yy in tar])
perform = []
for k1 in range(len(tar)-1):
	for k2 in range(k1+1, len(tar)):
		tar1, tar2 = tar[[k1, k2]]
		for i,xx in enumerate(reg):
			for j,yy in enumerate(leg):
				if count[k1,i,j]>5 and count[k2,i,j]>5 and (np.sum(count[k1,i,:])-count[k1,i,j])>5 and (np.sum(count[k2,i,:])-count[k2,i,j])>5:
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
					if count[k1,i,j]>5 and count[k2,i,j]>5 and (np.sum(count[k1,i,:])-count[k1,i,j])>5 and (np.sum(count[k2,i,:])-count[k2,i,j])>5:
						cellfilter = np.logical_and(meta[:,0]==xx, np.logical_or(meta[:,2]==tar1, meta[:,2]==tar2))
						metatmp = meta[cellfilter]
						ratetmp = rate[cellfilter]
						roctmp = []
						for k in ['male', 'female']:
							trainfilter = np.logical_and(metatmp[:,4]!=yy, metatmp[:,3]!=k)
							testfilter = np.logical_and(metatmp[:,4]==yy, metatmp[:,3]==k)
							pred = clf.fit(ratetmp[trainfilter], metatmp[trainfilter, 2]==tar1).predict_proba(ratetmp[testfilter])[:,1]
							roctmp.append(roc_auc_score(metatmp[testfilter, 2]==tar1, pred))
						perform[tot].append(np.mean(roctmp))
						print(t, tot, tar1, tar2, xx, yy, perform[tot][-1])
						tot += 1

perform = np.array(perform)
np.save(outdir + 'matrix/cell_3513_cor_cor_cross_layer_pred_gene_test1layer_heterogender_cv50.npy', perform)

perform = np.concatenate((perform[:,:6], np.mean(perform[:,6:].astype(float), axis=1)[:,None]), axis=1)


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
plt.savefig(outdir + 'plot/cell_3513_cor_cor_cross_layer_pred_gene_test1layer_heterogender_roc_heatmap_upsample50.pdf', transparent=True)
plt.close()
