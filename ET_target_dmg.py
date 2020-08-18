tar = np.array(['TH', 'SC', 'VTA', 'Pons', 'MY'])
outdir = '/gale/netapp/home/zhoujt/project/CEMBA/RS2/merge_RS2/cortex/L5PT/proj_dmg/r_lmertest/'
data = pd.DataFrame(data=rate, columns=gene[:,-2])
data['target'] = meta[:,2]
data['gender'] = meta[:,3]
data['global'] = meta[:,6]
fout = open(outdir + 'dmg/ET/para_list.txt', 'w')
for xx in reg:
	for i in range(len(tar)-1):
		for j in range(i+1, len(tar)):
			tar1, tar2 = tar[[i,j]]
			tarfilter1 = np.logical_and(meta[:,0]==xx, meta[:,2]==tar1)
			tarfilter2 = np.logical_and(meta[:,0]==xx, meta[:,2]==tar2)
			if np.sum(tarfilter1)==0 or np.sum(tarfilter2)==0:
				continue
			print(xx, tar1, tar2)
			fout.write('{0}\n'.format(' '.join([xx, tar1, tar2])))
	data[meta[:,0]==xx].to_csv(outdir + 'dmg/ET/' + xx + '.rate.meta.txt', index=None, sep='\t')

fout.close()


count = np.array([[[np.sum(np.logical_and(meta[:,3]==zz, np.logical_and(meta[:,2]==yy, meta[:,0]==xx))) for zz in ['male', 'female']] for yy in tar] for xx in reg])
fcall, regall = [], []
for i in range(len(tar)-1):
	for j in range(i+1, len(tar)):
		tar1, tar2 = tar[[i,j]]
		regtmp = reg[np.sum(count[:,[i,j],:]>0, axis=(1,2))>2]
		fctmp, dftmp = [], []
		for k,xx in enumerate(regtmp):
			tarfilter1 = np.logical_and(meta[:,0]==xx, meta[:,2]==tar1)
			tarfilter2 = np.logical_and(meta[:,0]==xx, meta[:,2]==tar2)
			ave1 = np.mean(rate[tarfilter1], axis=0)
			ave2 = np.mean(rate[tarfilter2], axis=0)
			fc = (ave1 + 0.1) / (ave2 + 0.1)
			fctmp.append(fc)
		fcall.append(np.array(fctmp))
		regall.append(regtmp)

np.save(outdir + 'matrix/cell_3772_post_fcall.npy', fcall)
np.save(outdir + 'matrix/cell_3772_regall.npy', regall)

fcall = np.load(outdir + 'matrix/cell_3772_post_fcall.npy', allow_pickle=True)
regall = np.load(outdir + 'matrix/cell_3772_regall.npy', allow_pickle=True)

effect = '.post.ga.m.wald'
pvall = []
for i in range(len(tar)-1):
	for j in range(i+1, len(tar)):
		tar1, tar2 = tar[[i,j]]
		regtmp = reg[np.sum(count[:,[i,j],:]>0, axis=(1,2))>2]
		pvtmp = []
		for k,xx in enumerate(regtmp):
			pv = np.loadtxt(outdir + 'dmg/ET/' + '_'.join([xx, tar1, tar2]) + effect + '.pvalue.txt')
			fdr = FDR(pv, 0.1, 'fdr_bh')[1]
			pvtmp.append(fdr)
		pvall.append(np.array(pvtmp))

np.save(outdir + 'matrix/cell_3772' + effect.replace('.','_') + '.fdr.npy', pvall)

tot = 0
for i in range(4):
	for j in range(i+1, 5):
		tar1, tar2 = tar[[i,j]]
		selg = np.zeros(len(gene))
		result, title = [], []
		tmp = 0
		for k,xx in enumerate(reg):
			if xx in regall[tot]:
				tarfilter1 = np.logical_and(meta[:,0]==xx, meta[:,2]==tar1)
				tarfilter2 = np.logical_and(meta[:,0]==xx, meta[:,2]==tar2)
				fdr = pvall[tot][tmp]
				fc = fcall[tot][tmp]
				seltmp = np.logical_and(fdr<0.01, np.abs(np.log2(fc))>np.log2(1.25))
				selg = np.logical_or(selg, seltmp)
				result = result + [fc, fdr]
				title = title + ['FC_'+xx, 'FDR_'+xx]
				tmp += 1
				print(tar1, tar2, xx)
		result = np.array(result).T
		result = np.concatenate((gene[:,-2:], result), axis=1)[selg==1]
		title = np.array(['Ensemble ID', 'Gene name'] + title)
		np.savetxt(outdir + 'matrix/'+'-'.join(tar[[i,j]])+'.fc.fdr.txt', np.concatenate((title[None,:],result), axis=0), fmt='%s', delimiter='\t')
		tot += 1

selghypo = np.zeros(len(gene))
for i in [3,6,8,9]:
	tmp1 = np.logical_and(np.log2(fcall[i][0])>np.log2(1.25), pvall[i][0]<0.01)
	tmp2 = np.logical_and(np.log2(fcall[i][1])>np.log2(1.25), pvall[i][1]<0.01)
	print(np.sum(np.logical_and(tmp1, tmp2)))
	selghypo = selghypo + np.logical_and(tmp1, tmp2)

selghyper = np.zeros(len(gene))
for i in [3,6,8,9]:
	tmp1 = np.logical_and(np.log2(fcall[i][0])<-np.log2(1.25), pvall[i][0]<0.01)
	tmp2 = np.logical_and(np.log2(fcall[i][1])<-np.log2(1.25), pvall[i][1]<0.01)
	print(np.sum(np.logical_and(tmp1, tmp2)))
	selghyper = selghyper + np.logical_and(tmp1, tmp2)



exp = ['Kcnh1', 'Astn2', 'Celf4', 'Dscaml1', 'Cadm1', 'Bdnf', 'Mpped2', 'Nr4a3']

k, xx = 1, 'SSp'

selg = np.zeros(len(gene))
for i in [3,6,8,9]:
	tmp = np.logical_and(np.log2(fcall[i][k])>np.log2(1.25), pvall[i][k]<0.01)
	print(np.sum(tmp))
	selg = selg + tmp

np.sum(selg>0)
np.sum(selg==np.max(selg))

selg = np.zeros(len(gene))
for i in [3,6,8,9]:
	tmp = np.logical_and(np.abs(np.log2(fcall[i][k]))>np.log2(1.25), pvall[i][k]<0.01)
	print(np.sum(tmp))
	selg = selg + tmp

np.sum(selg>0)

tartmp = np.array(['MY', 'VTA', 'SC', 'Pons', 'TH'])
mch = np.array([np.mean(rate[np.logical_and(meta[:,0]==xx, meta[:,2]==yy)], axis=0) for yy in tartmp])
mch = mch[:,selg>0].T
cg = clustermap(mch, col_cluster=False, metric='correlation', z_score=0)
rorder = cg.dendrogram_row.reordered_ind.copy()
# corder = cg.dendrogram_col.reordered_ind.copy()
if np.mean(zscore(mch, axis=1)[rorder[:50],0])>0:
	rorder = rorder[::-1]

data = mch[rorder]
gtmp = [np.where(gene[selg>0,-1][rorder]==x)[0][0] for x in exp]
fig, ax = plt.subplots(figsize=(5, 4))
plot = ax.imshow(zscore(data, axis=1), cmap='bwr', vmin=-2.5, vmax=2.5, aspect='auto')
ax.set_xticks(np.arange(len(tartmp)))
ax.set_xticklabels(tartmp, fontsize=15, rotation=60, rotation_mode='anchor', ha='right')
ax.set_yticks(gtmp)
ax.set_yticklabels(exp)
ax.set_xlabel('Target', fontsize=15)
ax.set_ylabel(str(len(data)) + ' MY DMGs', fontsize=15)
cbar = plt.colorbar(plot, ax=ax, shrink=0.3, fraction=0.05, pad=0.3, orientation='horizontal')
cbar.set_ticks([-2.5, 2.5])
cbar.set_ticklabels([-2.5, 2.5])
cbar.set_label('Norm mCH', fontsize=15)
cbar.ax.yaxis.set_label_position('left')
cbar.draw_all()
plt.tight_layout()
plt.savefig(outdir + 'plot/'+xx+'_target_ovo_ga_m_pv01_fc25_pseudo10_mch_facsfilter.pdf', transparent=True)
plt.close()


k, xx = 3, 'AI'

exp = ['Adgrl3', 'Tenm2', 'Epha6', 'Lingo2', 'Cdh11', 'Epha4', 'Pcdh7', 'Grm3', 'Sox5', 'Pde1a']
selg = np.logical_and(np.abs(np.log2(fcall[5][k]))>np.log2(1.25), pvall[5][k]<0.01)
tartmp = ['SC', 'Pons', 'TH']
mch = np.array([np.mean(rate[np.logical_and(meta[:,0]==xx, meta[:,2]==yy)], axis=0) for yy in tartmp])
mch = mch[:,selg].T
cg = clustermap(mch, col_cluster=False, metric='correlation', z_score=0)
rorder = cg.dendrogram_row.reordered_ind.copy()
# corder = cg.dendrogram_col.reordered_ind.copy()
if np.mean(zscore(mch, axis=1)[rorder[:50]])>0:
	rorder = rorder[::-1]

data = mch[rorder]
gtmp = [np.where(gene[selg,-1][rorder]==x)[0][0] for x in exp]

fig, ax = plt.subplots(figsize=(3, 4))
plot = ax.imshow(zscore(data, axis=1), cmap='bwr', vmin=-2.5, vmax=2.5, aspect='auto')
ax.set_xticks(np.arange(len(tartmp)))
ax.set_xticklabels(tartmp, fontsize=15, rotation=60, rotation_mode='anchor', ha='right')
ax.set_yticks(gtmp)
ax.set_yticklabels(exp)
ax.set_xlabel('Target', fontsize=15)
ax.set_ylabel(str(len(data)) + ' SC/Pons DMGs', fontsize=15)
cbar = plt.colorbar(plot, ax=ax, shrink=0.3, fraction=0.05, pad=0.3, orientation='horizontal')
cbar.set_ticks([-2.5, 2.5])
cbar.set_ticklabels([-2.5, 2.5])
cbar.set_label('Norm mCH', fontsize=15)
cbar.ax.yaxis.set_label_position('left')
cbar.draw_all()
plt.tight_layout()
plt.savefig(outdir + 'plot/'+xx+'_target_ovo_ga_m_pv01_fc25_pseudo10_mch_facsfilter.pdf', transparent=True)
plt.close()
