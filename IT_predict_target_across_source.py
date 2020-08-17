rocall, regall, comb = [], [], []
for k1 in range(len(tar)-1):
	for k2 in range(k1+1, len(tar)):
		tar1, tar2 = tar[[k1, k2]]
		count = np.array([[[[np.sum(np.logical_and(np.logical_and(meta[:,2]==yy, meta[:,4]==zz), np.logical_and(meta[:,0]==xx, meta[:,3]==kk))) for kk in ['male', 'female']] for zz in leg] for yy in [tar1, tar2]] for xx in reg])
		regtmp = reg[np.min(np.sum(count, axis=2), axis=(1,2))>5]
		print(tar1, tar2, len(regtmp))
		roc = np.zeros((len(regtmp), len(regtmp), 2, 50))
		tarfilter = np.logical_or(meta[:,2]==tar1, meta[:,2]==tar2)
		for i, xx in enumerate(regtmp):
			metatmp = meta[meta[:,0]==xx]
			ratetmp = rate[meta[:,0]==xx]
			for l,gender in enumerate(['male', 'female']):
				count = np.array([[np.sum(np.logical_and(np.logical_and(metatmp[:,2]==yy, metatmp[:,4]==zz), metatmp[:,3]==gender)) for zz in leg] for yy in [tar1, tar2]])
				laytmp = (np.min(count, axis=0)>0)
				count = np.max(count, axis=0)
				count[~laytmp] = 0
				tot = np.sum(count[:4])
				label = np.array([1 for j in range(tot)] + [0 for j in range(tot)])
				test_x, test_y = [], []
				for xx0 in regtmp:
					test_x.append(rate[np.logical_and(np.logical_and(tarfilter, meta[:,0]==xx0), meta[:,3]!=gender)])
					test_y.append((meta[np.logical_and(np.logical_and(tarfilter, meta[:,0]==xx0), meta[:,3]!=gender), 2]==tar1))
				for t in range(50):
					datatmp = []
					for yy in [tar1, tar2]:
						for k,zz in enumerate(leg):
							if not laytmp[k]:
								continue
							obs = ratetmp[np.logical_and(np.logical_and(metatmp[:,2]==yy, metatmp[:,4]==zz), metatmp[:,3]==gender)]
							if len(obs) < count[k]:
								datatmp.append(upsample(obs, count[k]))
							else:
								datatmp.append(obs)
					datatmp = np.concatenate(datatmp, axis=0)
					clf = LogisticRegression(solver='lbfgs', class_weight='balanced').fit(datatmp, label)
					for j in range(len(regtmp)):
						pred = clf.predict_proba(test_x[j])[:,1]
						roc[i,j,l,t] = roc_auc_score(test_y[j], pred)
				print(tar1, tar2, xx, tot)
		roc = np.around(np.mean(roc, axis=(2,3)), decimals=3)
		print(roc)
		rocall.append(roc)
		regall.append(regtmp)
		comb.append([tar1, tar2])

np.save(outdir + 'matrix/cell_3513_cor_cor_cross_region_pred_heterogender_upsample50_genepost.npy', rocall)
np.save(outdir + 'matrix/cell_3513_cor_cor_cross_region_pred_heterogender_upsample50_reglabel.npy', regall)
np.save(outdir + 'matrix/cell_3513_cor_cor_cross_region_pred_heterogender_upsample50_comb.npy', comb)

rocall, regall, comb = [], [], []
for k1 in range(len(tar)-1):
	for k2 in range(k1+1, len(tar)):
		tar1, tar2 = tar[[k1, k2]]
		count = np.array([[[np.sum(np.logical_and(np.logical_and(meta[:,2]==yy, meta[:,4]==zz), meta[:,0]==xx)) for zz in leg] for yy in [tar1, tar2]] for xx in reg])
		regtmp = reg[np.min(np.sum(count, axis=2), axis=1)>11]
		print(tar1, tar2, len(regtmp))
		rocall.append(np.zeros((len(regtmp), len(regtmp), 2, 50)))
		regall.append(regtmp)
		comb.append([tar1, tar2])


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
			regtmp = regall[tot]
			print(tar1, tar2, len(regtmp))
			roc = np.zeros((len(regtmp), len(regtmp), 2, 50))
			tarfilter = np.logical_or(meta[:,2]==tar1, meta[:,2]==tar2)
			for i, xx in enumerate(regtmp):
				for l,gender in enumerate(['male', 'female']):
					train_x = rate[np.logical_and(np.logical_and(tarfilter, meta[:,0]==xx), meta[:,3]==gender)]
					train_y = (meta[np.logical_and(np.logical_and(tarfilter, meta[:,0]==xx), meta[:,3]==gender), 2]==tar1)
					clf = LogisticRegression(solver='lbfgs', class_weight='balanced').fit(train_x, train_y)
					for j,xx0 in enumerate(regtmp):
						test_x = rate[np.logical_and(np.logical_and(tarfilter, meta[:,0]==xx0), meta[:,3]!=gender)]
						test_y = (meta[np.logical_and(np.logical_and(tarfilter, meta[:,0]==xx0), meta[:,3]!=gender), 2]==tar1)
						pred = clf.predict_proba(test_x)[:,1]
						rocall[tot][i,j,l,t] = roc_auc_score(test_y, pred)
			tot += 1
			print(t, tar1, tar2)


rocallnew = [np.around(np.mean(roc, axis=(2,3)), decimals=3) for roc in rocall]
[np.mean(np.diag(roc)) for roc in rocallnew]


np.save(outdir + 'matrix/cell_3513_cor_cor_cross_region_pred_cv50_genepost_indiv.npy', rocall)
np.save(outdir + 'matrix/cell_3513_cor_cor_cross_region_pred_cv50_genepost.npy', rocallnew)
np.save(outdir + 'matrix/cell_3513_cor_cor_cross_region_pred_cv50_reglabel.npy', regall)


rord, cord = [], []
for roc in rocall:
	if len(roc)>2:
		cg = clustermap(roc, cmap='bwr', metric='euclidean')
		rord.append(cg.dendrogram_row.reordered_ind)
		cord.append(cg.dendrogram_col.reordered_ind)
	else:
		rord.append(range(len(roc)))
		cord.append(range(len(roc)))

fig, axes = plt.subplots(1, 6, figsize=(12,3))
for i,ax in enumerate(axes.flatten()):
	plot = ax.imshow(rocall[i][rord[i]][:, rord[i]], cmap='bwr', vmin=0.5, vmax=1.0)
	ax.set_xticks(range(len(regall[i])))
	ax.set_yticks(range(len(regall[i])))
	ax.set_yticklabels(regall[i][rord[i]], fontsize=12)
	ax.set_xticklabels(regall[i][rord[i]], fontsize=12, rotation=60, rotation_mode='anchor', va='top', ha='right')
	ax.set_title('/'.join(comb[i]), fontsize=15)
	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes('right', size='5%', pad='5%')
	# cbar = plt.colorbar(plot, cax=cax)
	# vmin, vmax = 0.5, 1.0
	# cbar.solids.set_clim([vmin, vmax])
	# cbar.set_ticks([vmin, vmax])
	# # cbar.set_label('AUROC', fontsize=15)
	# cbar.draw_all()

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_3513_cor_cor_cross_region_pred_gene_cv50.pdf', transparent=True)
plt.close()

fig, axes = plt.subplots(1,6,figsize=(12,2))
for i in range(len(comb)):
	xx = '/'.join(comb[i])
	data = np.diag(rocall[i])
	# idx = rord[i]
	idx = np.argsort(data)[::-1]
	ax = axes.flatten()[i]
	sns.despine(ax=ax)
	ind = np.arange(len(data))
	ax.bar(ind, data[idx], width=0.5, color=[regcolordict[yy] for yy in regall[i][idx]])
	ax.set_xticks(ind)
	ax.set_xticklabels(regall[i][idx], rotation_mode='anchor', ha='right', rotation=60)
	ax.set_ylim([0.5, 1.0])
	ax.set_title(xx)

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_3513_cor_cor_reg_tarpair_gene_cv50_valueorder.pdf', transparent=True)
plt.close()



fig, ax = plt.subplots(figsize=(3,2))
i = 3
# i = 1
idx = rord[i]
xx = '/'.join(comb[i])
data = np.diag(rocall[i])
sns.despine(ax=ax)
ind = np.arange(len(data))
ax.bar(ind, data[idx], color=[regcolordict[yy] for yy in regall[i][idx]], width=0.5)
ax.set_xticks(ind)
ax.set_xticklabels(regall[i][idx], rotation_mode='anchor', ha='right', rotation=60)
ax.set_ylim([0.5, 1.0])
ax.set_title(xx)
ax.set_ylabel('AUROC')
plt.tight_layout()
plt.savefig(outdir + 'plot/'+'_'.join(comb[i])+'_reg_tarpair_gene.pdf', transparent=True)
plt.close()



from scipy.stats import sem

plotmeta = np.array([[comb[i][0]+'/'+comb[i][1], regall[i][j], rocall[i][j,j]] for i in range(len(comb)) for j in range(len(regall[i])) if not 'STR' in comb[i]])
# combfilter = ~np.array([('MOp' in x[0]) and (x[1]=='AI') for x in plotmeta])
# plotmeta = plotmeta[combfilter]
xleg = [tar[k1]+'/'+tar[k2] for k1 in range(len(tar)-1) for k2 in range(k1+1, len(tar)) if tar[k1]!='STR']
color = np.array(list(islice(cycle(['#e6194b','#3cb44b','#ffe119','#0082c8','#f58231','#911eb4','#46f0f0','#f032e6','#d2f53c','#fabebe','#008080','#e6beff','#aa6e28','#800000','#aaffc3','#808000','#ffd8b1','#000080','#d1b26f','#fffac8','#650021','#808080','#000000']), 100)))
colordict = {x:color[i] for i,x in enumerate(reg)}
ave = np.array([np.mean(plotmeta[plotmeta[:,0]==xx, 2].astype(float)) for xx in xleg])
idx = np.argsort(ave)[::-1]
xleg = np.array(xleg)[idx]
fig, ax = plt.subplots(figsize=(5,4))
ax = sns.stripplot(x=plotmeta[:,0], y=plotmeta[:,2].astype(float), hue=plotmeta[:,1], palette=colordict, ax=ax, order=xleg, jitter=False, hue_order=reg)
for i in range(len(xleg)):
	datatmp = plotmeta[plotmeta[:,0]==xleg[i], 2].astype(float)
	avetmp = np.mean(datatmp)
	semtmp = sem(datatmp)
	ax.plot([i-0.2, i+0.2], [avetmp, avetmp], color='k')
	ax.plot([i, i], [avetmp-semtmp, avetmp+semtmp], color='k')
	ax.plot([i-0.05, i+0.05], [avetmp-semtmp, avetmp-semtmp], color='k')
	ax.plot([i-0.05, i+0.05], [avetmp+semtmp, avetmp+semtmp], color='k')

ax.set_ylabel('AUROC')
ax.set_xlabel('Target pairs')
ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=12)
plt.setp(ax.get_xticklabels(), rotation=60, rotation_mode='anchor', ha='right')
plt.tight_layout()
plt.savefig(outdir + 'plot/cell_3513_cor_cor_pred_roc_reg_scatter_cv50_genepost.pdf', transparent=True, bbox_inches='tight')
plt.close()


xx = 'AUD'
fig, ax = plt.subplots(figsize=(4,2.5))
data = plotmeta[plotmeta[:,1]==xx,2].astype(float)
xleg = plotmeta[plotmeta[:,1]==xx,0]
sns.despine(ax=ax)
idx = np.argsort(data)[::-1]
ind = np.arange(len(data))
ax.bar(ind, data[idx], color=regcolordict[xx], width=0.5)
ax.set_xticks(ind)
ax.set_xticklabels(xleg[idx], rotation_mode='anchor', ha='right', rotation=60)
ax.set_ylim([0.5, 1.0])
ax.set_title(xx)
ax.set_ylabel('AUROC')
plt.tight_layout()
plt.savefig(outdir + 'plot/'+xx+'_tarpair_genepost_new.pdf', transparent=True)
plt.close()
