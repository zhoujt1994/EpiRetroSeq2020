metaall = np.load(indir + 'cell_11827_meta.npy')
rateall = np.load(indir + 'cell_11827_rateb.mCH.npy')
cluster = np.load(indir + 'cell_11827_posterior_disp2k_pc100_nd50_p50.pc50.knn25.louvain_res1.2_nc25_cluster_label.npy')
metaall = np.concatenate((metaall[:, 12:16], cluster[:, None]), axis=1)

reg = np.array(['MOp', 'SSp', 'ACA', 'AI', 'AUDp', 'RSP', 'PTLp', 'VISp'])
tar = np.array(['TH', 'SC', 'VTA', 'Pons', 'MY'])
etfilter = (metaall[:,-1]=='L5-ET')
subfilter = np.array([x in tar for x in metaall[:,8]])
cellfilter = np.logical_and(np.logical_and(etfilter, subfilter), np.logical_and(facsfilter, metaall[:,9]=='female'))
metaet = metaall[cellfilter, 6:]
rate = rateg[cellfilter]

rocall, tarall, comb = [], [], []
for k1 in range(len(reg)-1):
	for k2 in range(k1+1, len(reg)):
		reg1, reg2 = reg[[k1, k2]]
		count = np.array([[[np.sum(np.logical_and(meta[:,0]==yy, np.logical_and(meta[:,2]==xx, meta[:,3]==kk))) for kk in ['male', 'female']] for yy in [reg1, reg2]] for xx in tar])
		tartmp = tar[np.min(count, axis=(1,2))>5]
		print(reg1, reg2, len(tartmp))
		if len(tartmp)==0:
			continue
		roc = np.zeros((len(tartmp), len(tartmp), 2))
		regfilter = np.logical_or(meta[:,0]==reg1, meta[:,0]==reg2)
		for i, xx in enumerate(tartmp):
			for l,gender in enumerate(['male', 'female']):
				train_x = rate[np.logical_and(np.logical_and(regfilter, meta[:,2]==xx), meta[:,3]==gender)]
				train_y = (meta[np.logical_and(np.logical_and(regfilter, meta[:,2]==xx), meta[:,3]==gender), 0]==reg1)
				clf = LogisticRegression(solver='lbfgs', class_weight='balanced').fit(train_x, train_y)
				for j,xx0 in enumerate(tartmp):
					test_x = rate[np.logical_and(np.logical_and(regfilter, meta[:,2]==xx0), meta[:,3]!=gender)]
					test_y = (meta[np.logical_and(np.logical_and(regfilter, meta[:,2]==xx0), meta[:,3]!=gender), 0]==reg1)
					pred = clf.predict_proba(test_x)[:,1]
					roc[i,j,l] = roc_auc_score(test_y, pred)
				print(reg1, reg2, xx, len(train_x))
		roc = np.around(np.mean(roc, axis=2), decimals=3)
		print(roc)
		rocall.append(roc)
		tarall.append(tartmp)
		comb.append([reg1, reg2])

np.save(outdir + 'matrix/cell_3772_sub_sub_cross_target_pred_source_heterogender_binpost.npy', rocall)
np.save(outdir + 'matrix/cell_3772_sub_sub_cross_target_pred_source_heterogender_tarlabel.npy', tarall)
np.save(outdir + 'matrix/cell_3772_sub_sub_cross_target_pred_source_heterogender_comb.npy', comb)



rocall, tarall, comb = [], [], []
for k1 in range(len(reg)-1):
	for k2 in range(k1+1, len(reg)):
		reg1, reg2 = reg[[k1, k2]]
		count = np.array([[np.sum(np.logical_and(meta[:,0]==yy, meta[:,2]==xx)) for yy in [reg1, reg2]] for xx in tar])
		tartmp = tar[np.min(count, axis=1)>11]
		print(reg1, reg2, len(tartmp))
		rocall.append(np.zeros((len(tartmp), len(tartmp), 2, 50)))
		tarall.append(tartmp)
		comb.append([reg1, reg2])


for t in range(50):
	meta = metaet.copy()
	selid = []
	for xx in reg:
		for yy in tar:
			idx = np.where(np.logical_and(meta[:,0]==xx, meta[:,2]==yy))[0]
			if len(idx)>1:
				selid.append(np.random.choice(idx, int(len(idx)//2), False))
	selid = np.sort(np.concatenate(selid))
	meta[:,3] = 'female'
	meta[selid, 3] = 'male'
	tot = 0
	for (reg1,reg2), tartmp in zip(comb, tarall):
		roc = np.zeros((len(tartmp), len(tartmp), 2, 50))
		regfilter = np.logical_or(meta[:,0]==reg1, meta[:,0]==reg2)
		for i, xx in enumerate(tartmp):
			for l,gender in enumerate(['male', 'female']):
				train_x = rate[np.logical_and(np.logical_and(regfilter, meta[:,2]==xx), meta[:,3]==gender)]
				train_y = (meta[np.logical_and(np.logical_and(regfilter, meta[:,2]==xx), meta[:,3]==gender), 0]==reg1)
				clf = LogisticRegression(solver='lbfgs').fit(train_x, train_y)
				for j,xx0 in enumerate(tartmp):
					test_x = rate[np.logical_and(np.logical_and(regfilter, meta[:,2]==xx0), meta[:,3]!=gender)]
					test_y = (meta[np.logical_and(np.logical_and(regfilter, meta[:,2]==xx0), meta[:,3]!=gender), 0]==reg1)
					pred = clf.predict_proba(test_x)[:,1]
					rocall[tot][i,j,l,t] = roc_auc_score(test_y, pred)
		tot += 1
		print(t, reg1, reg2)


rocallnew = [np.around(np.mean(roc, axis=(2,3)), decimals=3) for roc in rocall]

np.save(outdir + 'matrix/cell_3772_sub_sub_cross_target_pred_source_cv50_genepost_indiv.npy', rocall)
np.save(outdir + 'matrix/cell_3772_sub_sub_cross_target_pred_source_cv50_genepost.npy', rocallnew)
np.save(outdir + 'matrix/cell_3772_sub_sub_cross_target_pred_source_cv50_tarlabel.npy', tarall)

