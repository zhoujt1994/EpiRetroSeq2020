meta = metait.copy()
data = pd.DataFrame(data=rate, columns=gene[:,-2])
data['target'] = meta[:,2]
data['gender'] = meta[:,3]
data['cluster'] = meta[:,4]
data['global'] = meta[:,5]
# data.to_csv(outdir + 'cell_11827_rateg.meta.txt', index=None, sep='\t')
fout = open(outdir + 'dmg/IT/para_list.txt', 'w')
# data['proj'] = (meta[:,2]==tar1).astype(int)
for xx in reg:
	for i in range(len(tar)-1):
		for j in range(i+1, len(tar)):
			tar1, tar2 = tar[[i,j]]
			tarfilter1 = np.logical_and(meta[:,0]==xx, meta[:,2]==tar1)
			tarfilter2 = np.logical_and(meta[:,0]==xx, meta[:,2]==tar2)
			if np.sum(tarfilter1)==0 or np.sum(tarfilter2)==0:
				continue
			# tarfilter = np.logical_or(tarfilter1, tarfilter2)
			# datatmp = data[tarfilter]
			# atatmp.to_csv(outdir + '_'.join([xx, tar1, tar2]) + '.rate.txt', index=None, sep='\t')
			print(xx, tar1, tar2)
			fout.write('{0}\n'.format(' '.join([xx, tar1, tar2])))
	data[meta[:,0]==xx].to_csv(outdir + 'dmg/IT/' + xx + '.rate.meta.txt', index=None, sep='\t')

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
			# df = ave1 - ave2
			fctmp.append(fc)
			# dftmp.append(df)
		fcall.append(np.array(fctmp))
		regall.append(regtmp)
		# dfall.append(np.array(dftmp))

np.save(outdir + 'matrix/cell_3513_post_fcall.npy', fcall)
np.save(outdir + 'matrix/cell_3513_regall.npy', regall)

fcall = np.load(outdir + 'matrix/cell_3513_post_fcall.npy', allow_pickle=True)
regall = np.load(outdir + 'matrix/cell_3513_regall.npy', allow_pickle=True)

effect = '.post.gca.m.wald'
pvall = []
for i in range(len(tar)-1):
	for j in range(i+1, len(tar)):
		tar1, tar2 = tar[[i,j]]
		regtmp = reg[np.sum(count[:,[i,j],:]>0, axis=(1,2))>2]
		pvtmp = []
		for k,xx in enumerate(regtmp):
			pv = np.loadtxt(outdir + 'dmg/IT/' + '_'.join([xx, tar1, tar2]) + effect + '.pvalue.txt')
			fdr = FDR(pv, 0.1, 'fdr_bh')[1]
			pvtmp.append(fdr)
		pvall.append(np.array(pvtmp))

np.save(outdir + 'matrix/cell_3513' + effect.replace('.','_') + '.fdr.npy', pvall)


pvall = np.load(outdir + 'matrix/cell_3513' + effect.replace('.','_') + '.fdr.npy', allow_pickle=True)

selg = np.zeros(len(gene))
for k in range(6):
	tmp1 = np.sum(np.logical_and(np.log2(fcall[k])>np.log2(1.25), pvall[k]<0.01), axis=0)
	tmp2 = np.sum(np.logical_and(np.log2(fcall[k])<-np.log2(1.25), pvall[k]<0.01), axis=0)
	print(np.sum(tmp1>0), np.sum(tmp2>0), np.sum(np.logical_or(tmp1>0, tmp2>0)))
	# selg = np.logical_or(selg, (tmp1+tmp2)>1)
	selg = np.logical_or(selg, np.logical_or(tmp1>0, tmp2>0))
	# selg = np.logical_or(selg, np.logical_or(np.logical_and(tmp1>1, tmp2==0), np.logical_and(tmp1==0, tmp2>1)))

np.sum(selg)

np.savetxt(outdir + 'dmg/IT/cell_3513_dmg' + effect.replace('.','_') + '_pseudo10.fc25.pv01.txt', gene[selg, -1], fmt='%s', delimiter='\n')
np.savetxt(outdir + 'gene_bkg_12261.txt', gene[:, -1], fmt='%s', delimiter='\n')

plotg = [[0, ['Gnb1', 'Bsn']], [5, ['Scn2a1', 'Dcps']]]

[np.sum(np.logical_and(np.abs(dfall[k])>0.1, pvall[k]<0.01), axis=0).max() for k in range(10)]

gene[np.sum(np.logical_and(np.abs(np.log2(fcall[k]))>np.log2(1.25), pvall[k]<0.05), axis=0)>1, -1]
gene[np.sum(np.logical_and(np.abs(dfall[k])>0.1, pvall[k]<0.01), axis=0)==4, -1]

tarpalette = {'MOp_rep1':'#ff00ff', 'MOp_rep2':'#bf00ff', 'SSp_rep1':'#ffbf00', 'SSp_rep2':'#ff8000', 'ACA_rep1':'#95D100', 'ACA_rep2':'#A1B900', 'VISp_rep1':'#00bfff', 'VISp_rep2':'#0040ff'}
tot = 0
fig, axes = plt.subplots(4,1,figsize=(6,8))
for gtmp in plotg:
	tar1, tar2 = comb[gtmp[0]]
	selreg = regall[gtmp[0]]
	tarfilter = np.logical_or(meta[:,2]==tar1, meta[:,2]==tar2)
	regfilter = np.array([x in selreg for x in meta[:,0]])
	ratetmp = rate[np.logical_and(tarfilter, regfilter)]
	metatmp = meta[np.logical_and(tarfilter, regfilter)]
	metatmp[metatmp[:,3]=='male',3] = 'rep1'
	metatmp[metatmp[:,3]=='female',3] = 'rep2'	
	count = np.array([[np.sum(np.logical_and(np.logical_and(metatmp[:,0]==xx, metatmp[:,2]==yy), metatmp[:,3]==zz)) for yy in [tar1,tar2] for zz in ['rep1','rep2']] for xx in selreg])
	for g in gtmp[1]:
		pvtmp = pvall[gtmp[0]][:,genedict[g]]
		mch = ratetmp[:,genedict[g]]
		ax = axes[tot]
		sns.despine(ax=ax)
		mice = np.array([xx+'_'+yy for xx,yy in zip(metatmp[:,2],metatmp[:,3])])
		ax = sns.boxplot(x=metatmp[:,0], y=mch, hue=mice, palette=tarpalette, ax=ax, showfliers=False, order=selreg, hue_order=[yy+'_'+zz for yy in [tar1,tar2] for zz in ['rep1','rep2']])
		ax.set_title(g, fontsize=15)
		ax.legend(bbox_to_anchor=(1,1), loc='upper left', fontsize=12)
		if g==gtmp[1][-1]:
			ax.set_xticklabels([x+'\n'+str(count[i][0])+'/'+str(count[i][1])+'\n'+str(count[i][2])+'/'+str(count[i][3]) for i,x in enumerate(selreg)], rotation=0, fontsize=12)
		else:
			ax.set_xticklabels([])
		for i,xx in enumerate(selreg):
			if pvtmp[i]<0.1:
				mchtmp = [mch[np.logical_and(metatmp[:,0]==xx, mice==yy)] for yy in [tar1+'_rep2', tar2+'_rep1']]
				q1 = [np.percentile(x, 25) if len(x)>0 else 0 for x in mchtmp]
				q3 = [np.percentile(x, 75) if len(x)>0 else 0 for x in mchtmp]
				q = np.max([np.max(x[x < (q3[j]+1.5*(q3[j]-q1[j]))]) if len(x)>0 else 0 for j,x in enumerate(mchtmp)])
				if pvtmp[i] < 0.01:
					ax.text(i, q, '**', horizontalalignment='center', verticalalignment='bottom')
				else:
					ax.text(i, q, '*', horizontalalignment='center', verticalalignment='bottom')
		tot += 1

plt.tight_layout()
plt.savefig(outdir + 'plot/cor_cor_dmg_sharereg_boxplot.pdf', transparent=True, bbox_inches='tight')
plt.close()
