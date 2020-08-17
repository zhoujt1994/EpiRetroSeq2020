leg = np.array(['L2/3', 'L4', 'L5-IT', 'L6-IT'])
tar = np.array(['MOp', 'SSp', 'ACA', 'VISp'])
cellfilter = np.logical_and(np.logical_and(corfilter, itfilter), facsfilter)
rate = rategp[cellfilter]
metait = metaall[cellfilter]
meta = metait.copy()

### Biological replicates

rocall, tarall, comb = [], [], []
for k1 in range(len(reg)-1):
	for k2 in range(k1+1, len(reg)):
		reg1, reg2 = reg[[k1, k2]]
		count = np.array([[[[np.sum(np.logical_and(np.logical_and(meta[:,0]==yy, meta[:,4]==zz), np.logical_and(meta[:,2]==xx, meta[:,3]==kk))) for kk in ['male', 'female']] for zz in leg] for yy in [reg1, reg2]] for xx in tar])
		tartmp = tar[np.min(np.sum(count, axis=2), axis=(1,2))>5]
		print(reg1, reg2, len(tartmp))
		if len(tartmp)==0:
			continue
		roc = np.zeros((len(tartmp), len(tartmp), 2, 50))
		regfilter = np.logical_or(meta[:,0]==reg1, meta[:,0]==reg2)
		for i, xx in enumerate(tartmp):
			metatmp = meta[meta[:,2]==xx]
			ratetmp = rate[meta[:,2]==xx]
			for l,gender in enumerate(['male', 'female']):
				count = np.array([[np.sum(np.logical_and(np.logical_and(metatmp[:,0]==yy, metatmp[:,4]==zz), metatmp[:,3]==gender)) for zz in leg] for yy in [reg1, reg2]])
				laytmp = (np.min(count, axis=0)>0)
				count = np.max(count, axis=0)
				count[~laytmp] = 0
				tot = np.sum(count[:4])
				label = np.array([1 for j in range(tot)] + [0 for j in range(tot)])
				test_x, test_y = [], []
				for xx0 in tartmp:
					test_x.append(rate[np.logical_and(np.logical_and(regfilter, meta[:,2]==xx0), meta[:,3]!=gender)])
					test_y.append((meta[np.logical_and(np.logical_and(regfilter, meta[:,2]==xx0), meta[:,3]!=gender), 0]==reg1))
				for t in range(50):
					datatmp = []
					for yy in [reg1, reg2]:
						for k,zz in enumerate(leg):
							if not laytmp[k]:
								continue
							obs = ratetmp[np.logical_and(np.logical_and(metatmp[:,0]==yy, metatmp[:,4]==zz), metatmp[:,3]==gender)]
							if len(obs) < count[k]:
								datatmp.append(upsample(obs, count[k]))
							else:
								datatmp.append(obs)
					datatmp = np.concatenate(datatmp, axis=0)
					clf = LogisticRegression(solver='lbfgs', class_weight='balanced').fit(datatmp, label)
					for j in range(len(tartmp)):
						pred = clf.predict_proba(test_x[j])[:,1]
						roc[i,j,l,t] = roc_auc_score(test_y[j], pred)
				print(reg1, reg2, xx, tot)
		roc = np.around(np.mean(roc, axis=(2,3)), decimals=3)
		print(roc)
		rocall.append(roc)
		tarall.append(tartmp)
		comb.append([reg1, reg2])

np.save(outdir + 'matrix/cell_3513_cor_cor_cross_target_pred_source_heterogender_upsample50_binpost.npy', rocall)
np.save(outdir + 'matrix/cell_3513_cor_cor_cross_target_pred_source_heterogender_upsample50_tarlabel.npy', tarall)
np.save(outdir + 'matrix/cell_3513_cor_cor_cross_target_pred_source_heterogender_upsample50_comb.npy', comb)

### Computational replicates

rocall, tarall, comb = [], [], []
for k1 in range(len(reg)-1):
	for k2 in range(k1+1, len(reg)):
		reg1, reg2 = reg[[k1, k2]]
		count = np.array([[[np.sum(np.logical_and(np.logical_and(meta[:,0]==yy, meta[:,4]==zz), meta[:,2]==xx)) for zz in leg] for yy in [reg1, reg2]] for xx in tar])
		tartmp = tar[np.min(np.sum(count, axis=2), axis=1)>11]
		print(reg1, reg2, len(tartmp))
		if len(tartmp)==0:
			continue
		rocall.append(np.zeros((len(tartmp), len(tartmp), 2, 50)))
		tarall.append(tartmp)
		comb.append([reg1, reg2])

for t in range(50):
	meta = metait.copy()
	selid = []
	for xx in tar:
		for yy in reg:
			for zz in leg:
				idx = np.where(np.logical_and(np.logical_and(meta[:,2]==xx, meta[:,0]==yy), meta[:,4]==zz))[0]
				if len(idx)>1:
					selid.append(np.random.choice(idx, int(len(idx)//2), False))
	selid = np.sort(np.concatenate(selid))
	meta[:,3] = 'female'
	meta[selid, 3] = 'male'
	tot = 0
	for reg1,reg2 in comb:
		tartmp = tarall[tot]
		print(reg1, reg2, len(tartmp))
		roc = np.zeros((len(tartmp), len(tartmp), 2, 50))
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
					rocall[tot][i,j,l,t] = roc_auc_score(test_y, pred)
		tot += 1
		print(t, reg1, reg2)


rocallnew = [np.around(np.mean(roc, axis=(2,3)), decimals=3) for roc in rocall]

np.save(outdir + 'matrix/cell_3513_cor_cor_cross_target_pred_source_cv50_binpost_indiv.npy', rocall)
np.save(outdir + 'matrix/cell_3513_cor_cor_cross_target_pred_source_cv50_binpost.npy', rocallnew)
np.save(outdir + 'matrix/cell_3513_cor_cor_cross_target_pred_source_cv50_tarlabel.npy', tarall)
np.save(outdir + 'matrix/cell_3513_cor_cor_cross_target_pred_source_cv50_comb.npy', comb)
