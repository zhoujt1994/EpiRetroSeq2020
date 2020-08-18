rocall, regall, comb = [], [], []
for k1 in range(len(tar)-1):
	for k2 in range(k1+1, len(tar)):
		tar1, tar2 = tar[[k1, k2]]
		count = np.array([[[np.sum(np.logical_and(meta[:,2]==yy, np.logical_and(meta[:,0]==xx, meta[:,3]==kk))) for kk in ['male', 'female']] for yy in [tar1, tar2]] for xx in reg])
		regtmp = reg[np.min(count, axis=(1,2))>5]
		print(tar1, tar2, len(regtmp))
		roc = np.zeros((len(regtmp), len(regtmp), 2))
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
					roc[i,j,l] = roc_auc_score(test_y, pred)
				print(tar1, tar2, xx, len(train_x))
		roc = np.around(np.mean(roc, axis=2), decimals=3)
		print(roc)
		rocall.append(roc)
		regall.append(regtmp)
		comb.append([tar1, tar2])

np.save(outdir + 'matrix/cell_3772_sub_sub_cross_region_pred_heterogender_binpost.npy', rocall)
np.save(outdir + 'matrix/cell_3772_sub_sub_cross_region_pred_heterogender_reglabel.npy', regall)


cv50

meta = metaet.copy()

rocall, regall, comb = [], [], []
for k1 in range(len(tar)-1):
	for k2 in range(k1+1, len(tar)):
		tar1, tar2 = tar[[k1, k2]]
		count = np.array([[np.sum(np.logical_and(meta[:,2]==yy, meta[:,0]==xx)) for yy in [tar1, tar2]] for xx in reg])
		regtmp = reg[np.min(count, axis=1)>11]
		print(tar1, tar2, len(regtmp))
		rocall.append(np.zeros((len(regtmp), len(regtmp), 2, 50)))
		regall.append(regtmp)
		comb.append([tar1, tar2])


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
					clf = LogisticRegression(solver='lbfgs').fit(train_x, train_y)
					for j,xx0 in enumerate(regtmp):
						test_x = rate[np.logical_and(np.logical_and(tarfilter, meta[:,0]==xx0), meta[:,3]!=gender)]
						test_y = (meta[np.logical_and(np.logical_and(tarfilter, meta[:,0]==xx0), meta[:,3]!=gender), 2]==tar1)
						pred = clf.predict_proba(test_x)[:,1]
						rocall[tot][i,j,l,t] = roc_auc_score(test_y, pred)
			tot += 1
			print(t, tar1, tar2)


rocallnew = [np.around(np.mean(roc, axis=(2,3)), decimals=3) for roc in rocall]

np.save(outdir + 'matrix/cell_3772_sub_sub_cross_region_pred_cv50_binpost_indiv.npy', rocall)
np.save(outdir + 'matrix/cell_3772_sub_sub_cross_region_pred_cv50_binpost.npy', rocallnew)
np.save(outdir + 'matrix/cell_3772_sub_sub_cross_region_pred_cv50_reglabel.npy', regall)
