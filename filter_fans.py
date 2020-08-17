metaglia = np.load(indir + 'cell_13414_meta.npy')
gliafilter = np.loadtxt(indir + 'cell_13414_gliafilter.txt')
clusterglia = np.array(['NonN' for i in range(len(metaglia))], dtype='<U20')
clusterglia[gliafilter==1] = metaall[:,-1]
metaglia = np.concatenate((metaglia[:,0][:,None], metaglia[:,7:16], clusterglia[:,None]), axis=1)

tar = ['MOp', 'SSp', 'ACA', 'VISp', 'STR', 'TH', 'SC', 'Pons', 'VTA', 'MY']
leg = ['L2/3', 'L4', 'L5-IT', 'L6-IT', 'L5-ET', 'L6-CT', 'L6b', 'NP', 'CLA', 'Inh']
slc = ['3C', '4B', '6B', '7B', '4A', '5A', '3D', '9D', '10C', '9A', '10A', '9B', '11B', '12B']
fig, axes = plt.subplots(10, 28, figsize=(56,10))
for j,yy in enumerate(tar):
	tot = 0
	for xx in slc:
		for zz in ['male', 'female']:
			ax = axes[j, tot]
			cellfilter = np.logical_and(np.logical_and(metaglia[:,8]==yy, metaglia[:,7]==xx), metaglia[:,9]==zz)
			if np.sum(cellfilter)>0:
				metatmp = metaglia[cellfilter]
				count = [np.sum(metatmp[:,-1]==x) for x in leg]
				tmp = np.sum(count)
				count = count / tmp
				sns.despine(ax=ax)
				ax.spines['top'].set_visible(False)
				ax.bar(range(len(leg)), count, color=legcolor[:len(leg)], width=0.66)
				ax.set_title(metatmp[0,5].split('_')[0] + '(n=' + str(tmp) + ')')
				ax.set_xticks(range(len(leg)))
				ax.set_xticklabels([])
			else:
				sns.despine(ax=ax, left=True, bottom=True)
				ax.set_xticks([])
				ax.set_yticks([])
			tot += 1

# for ax in axes[-1]:
# 	ax.set_xticklabels([x.replace('-', ' ') for x in leg], fontsize=12, rotation=60, ha='right')

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_13414_FACS.layer_ratio_neu.pdf', transparent=True)
plt.close()



rmax = 256
ratio, xleg, count = [], [], []
for xx in slc:
	for yy in tar[:5]:
		for zz in ['male', 'female']:
			cellfilter = np.logical_and(np.logical_and(metaglia[:,8]==yy, metaglia[:,7]==xx), metaglia[:,9]==zz)
			if np.sum(cellfilter)>0:
				metatmp = metaglia[cellfilter]
				counttmp = [np.sum(metatmp[:,-1]==x) for x in leg]
				counttmp = [np.sum(counttmp[:4]), counttmp[-2]+counttmp[5]]
				count.append(counttmp)
				if counttmp[0]==0:
					ratio.append(0)
				elif counttmp[1]==0:
					ratio.append(rmax)
				else:
					ratio.append(min(counttmp[0]/counttmp[1], rmax))
				xleg.append(metatmp[0,5].split('_')[0] + '(' + str(counttmp[0]) + ',' + str(counttmp[1])+')')

count = np.array(count)
ratio = np.array(ratio)
xleg = np.array(xleg)
idx = np.argsort(ratio)
ratio = ratio[idx]
xleg = xleg[idx]
fig, ax = plt.subplots(figsize=(35, 3))
sns.despine(ax=ax)
ax.spines['top'].set_visible(False)
ax.bar(range(len(ratio)), ratio, color='C0', width=0.66)
ax.plot([-0.5, len(ratio)-0.5], [8571/5274, 8571/5274], 'k')
ax.plot([-0.5, len(ratio)-0.5], [8571/5274*3, 8571/5274*3], 'k')
ax.plot([-0.5, len(ratio)-0.5], [8571/5274*5, 8571/5274*5], 'k')
ax.set_xticks(range(len(ratio)))
ax.set_xticklabels(xleg, rotation=60, ha='right', fontsize=12)
ax.set_xlim([-0.5, len(ratio)-0.5])
ax.legend(bbox_to_anchor=(1,1), loc='upper left')

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_13414_FACS.IT2ICT.pdf', bbox_inches='tight', transparent=True)
plt.close()


pv = np.array([stats.binom(x[0]+x[1], 8571/(8571+5274)).sf(x[0]-1) for x in count])
fdr = FDR(pv, 0.01, 'fdr_bh')[1]
np.sum(np.logical_and(fdr<0.05, (ratio/8571*5274)>3))
itfacs = [x.split('(')[0] for x in xleg[np.logical_and(fdr<0.05, (ratio/8571*5274)>3)]]


fig, axes = plt.subplots(1,2,figsize=(6,3))
ax = axes[0]
ax.plot(np.log2([8571/5274*3, 8571/5274*3]), [0,30], 'b--')
ax.plot([-2,8], -np.log10([0.05, 0.05]), 'b--')
tmpfilter = ~np.logical_and(fdr<0.05, (ratio/8571*5274)>3)
ax.scatter(np.log2(ratio[tmpfilter]), -np.log10(fdr[tmpfilter]), color='none', marker='o', s=count[tmpfilter,0], edgecolors='r', label='Removed', zorder=50)
tmpfilter = np.logical_and(fdr<0.05, (ratio/8571*5274)>3)
ax.scatter(np.log2(ratio[tmpfilter]), -np.log10(fdr[tmpfilter]), color='none', marker='o', s=count[tmpfilter,0], edgecolors='k', label='Included', zorder=100)
ax.set_xlabel('log2 Fold Change', fontsize=12)
ax.set_ylabel('-log10 FDR', fontsize=12)
ax.legend(loc='upper left')
ax = axes[1]
ax.axis('off')
ax.scatter([0,0,0,0], [0,10,20,30], s=[1,10,50,100], color='none', marker='o', edgecolors='k')
for i in range(4):
	ax.text(-1, 10*i, str([1,10,50,100][i]), fontsize=12, verticalalignment='center', horizontalalignment='center')

ax.set_xlim([-5,5])
plt.tight_layout()
plt.savefig(outdir + 'plot/cell_13414_FACSfilter.IT2ICT.pdf', transparent=True)
plt.close()


rmax = 256
ratio, xleg, count = [], [], []
for xx in slc:
	for yy in tar[5:]:
		for zz in ['male', 'female']:
			cellfilter = np.logical_and(np.logical_and(metaglia[:,8]==yy, metaglia[:,7]==xx), metaglia[:,9]==zz)
			if np.sum(cellfilter)>0:
				metatmp = metaglia[cellfilter]
				counttmp = [np.sum(metatmp[:,-1]==x) for x in leg]
				counttmp = [counttmp[4], np.sum(counttmp[:4])+counttmp[-2]]
				count.append(counttmp)
				if counttmp[0]==0:
					ratio.append(0)
				elif counttmp[1]==0:
					ratio.append(rmax)
				else:
					ratio.append(min(counttmp[0]/counttmp[1], rmax))
				xleg.append(metatmp[0,5].split('_')[0] + '(' + str(counttmp[0]) + ',' + str(counttmp[1])+')')

count = np.array(count)
ratio = np.array(ratio)
xleg = np.array(xleg)
idx = np.argsort(ratio)
ratio = ratio[idx]
xleg = xleg[idx]
fig, ax = plt.subplots(figsize=(35, 3))
sns.despine(ax=ax)
ax.spines['top'].set_visible(False)
ax.bar(range(len(ratio)), ratio, color='C1', width=0.66)
ax.plot([-0.5, len(ratio)-0.5], [887/10626, 887/10626], 'k')
ax.plot([-0.5, len(ratio)-0.5], [887/10626*5, 887/10626*5], 'k')
ax.plot([-0.5, len(ratio)-0.5], [887/10626*10, 887/10626*10], 'k')
ax.set_xticks(range(len(ratio)))
ax.set_xticklabels(xleg, rotation=60, ha='right', fontsize=12)
ax.set_xlim([-0.5, len(ratio)-0.5])
ax.set_ylim([0,2])
ax.legend(bbox_to_anchor=(1,1), loc='upper left')

plt.tight_layout()
plt.savefig(outdir + 'plot/cell_13414_FACS.ET2IIT.pdf', bbox_inches='tight', transparent=True)
plt.close()


pv = np.array([stats.binom(x[0]+x[1], 887/(887+10626)).sf(x[0]-1) for x in count])
fdr = FDR(pv, 0.01, 'fdr_bh')[1]
np.sum(np.logical_and(fdr<0.01, (ratio/887*10626)>5))
etfacs = [x.split('(')[0] for x in xleg[np.logical_and(fdr<0.01, (ratio/887*10626)>5)]]


fig, axes = plt.subplots(1,2,figsize=(6,3))
ax = axes[0]
ax.plot(np.log2([887/10626*5, 887/10626*5]), [0,100], 'b--')
ax.plot([-2,8], -np.log10([0.01, 0.01]), 'b--')
tmpfilter = ~np.logical_and(fdr<0.01, (ratio/887*10626)>5)
ax.scatter(np.log2(ratio[tmpfilter]), -np.log10(fdr[tmpfilter]), color='none', marker='o', s=count[tmpfilter,0], edgecolors='r', label='Removed', zorder=50)
tmpfilter = np.logical_and(fdr<0.01, (ratio/887*10626)>5)
ax.scatter(np.log2(ratio[tmpfilter]), -np.log10(fdr[tmpfilter]), color='none', marker='o', s=count[tmpfilter,0], edgecolors='k', label='Included', zorder=100)
ax.set_xlabel('log2 Fold Change', fontsize=12)
ax.set_ylabel('-log10 FDR', fontsize=12)
ax.legend(loc='upper left')
ax = axes[1]
ax.axis('off')
ax.scatter([0,0,0,0], [0,10,20,30], s=[1,10,50,100], color='none', marker='o', edgecolors='k')
for i in range(4):
	ax.text(-1, 10*i, str([1,10,50,100][i]), fontsize=12, verticalalignment='center', horizontalalignment='center')

ax.set_xlim([-5,5])
plt.tight_layout()
plt.savefig(outdir + 'plot/cell_13414_FACSfilter.ET2I.pdf', transparent=True)
plt.close()

selfacs = itfacs + etfacs
facsfilter = np.array([x.split('_')[0] in selfacs for x in metaall[:,5]])
