from collections import Counter

def buildgraph(i):
    global cluster_dmr, motif_dmr, dmrgene, motif, gene_all, c, genedict, geneidx
    print(i)
    if  dmrgene[list(cluster_dmr.intersection(motif_dmr[motif[i]]))]!= np.array([], dtype=object):
        motifgene = np.concatenate(dmrgene[list(cluster_dmr.intersection(motif_dmr[motif[i]]))])
        
    else:
        motifgene = np.array([], dtype=object)
        
    count = Counter(motifgene)
    return [geneidx[motif[i]], np.array([count[x] for x in gene_all[:,-1]]) * mch[c, genedict[motif[i]]]]

def normgraph(A):
    ngene= len(A)
    print('Normalize')
    A = A + A.T
    A = A - np.diag(np.diag(A))
    A = A + np.diag(np.sum(A, axis=0) == 0)
    P = A / np.sum(A, axis=1)[:, None]
    return P

def pagerank(P, node_weight, alpha=0.85):
    print('Propagate')
    pr = np.ones(len(P)) / len(P)
    for i in range(300):
        pr_new = (1 - alpha) * node_weight + alpha * np.dot(pr, P)
        delta = np.linalg.norm(pr - pr_new)
        pr = pr_new.copy()
        print('Iter ', i, ' delta = ', delta)
        if delta < 1e-6:
            break
    return pr

nc = 15
label = np.load(f'cell_4176_L5PT_{nc}cluster_label.npy')
read_gene = np.load('cell_4176_L5PT_read_gene.mCH.npy')
rate_gene = np.load('cell_4176_L5PT_rate_gene.mCH.npy')

gene_all = np.loadtxt('gencode.vM10.bed', dtype=np.str)
geneidx = {x:i for i,x in enumerate(gene_all[:,-1])}
chrfilter = (gene_all[:,0]!='chrX')

mc = (read_gene * rate_gene)[:, chrfilter]
tc = read_gene[:, chrfilter]
clustermch = np.array([np.sum(mc[label==i], axis = 0)/np.sum(tc[label==i], axis = 0) for i in range(nc)])
clusterglobalch = np.array([np.sum(meta[label==i, 3].astype(float))/np.sum(meta[label==i, 4].astype(float)) for i in range(nc)])
clusterrateg = clustermch / clusterglobalch[:,None]
clusterreadg = np.array([np.sum(read_gene[label==i], axis = 0) for i in range(nc)])[:,genefilter]

covfilter = (np.sum((clusterreadg>100), axis = 0)==len(clusterrateg))
rateg = clusterrateg[:, covfilter]
gene = gene[covfilter]
genefilter = np.where(chrfilter)[0][covfilter]

minmaxclusterrateg = np.array([(rateg[i] - min(rateg[i])) / (max(rateg[i]) - min(rateg[i])) for i in range(nc)])
nodeweight = 1 - minmaxclusterrateg

motif_map = np.array([np.load('motif_gene_name.npy'), np.loadtxt('filename_to_motifname.txt', dtype=np.str)[:,0]]).T
motif_map = np.array([x for x in motif_map if x[0] in gene[:,-1]])
motif_dmr = {x:[] for x in motif_map[:,0]}

tot = 0
for x,y in motif_map:
    if motif_dmr[x]==[]:
        motif_dmr[x] = set(np.loadtxt(f'DMR_CG_comb/assign_motif/DMR_all_hypo.slop100.{y}.txt', dtype=np.int))
    else:
        motif_dmr[x] = motif_dmr[x].union(np.loadtxt(f'DMR_CG_comb/assign_motif/DMR_all_hypo.slop100.{y}.txt', dtype=np.int))
    tot += 1
    print(tot, x, len(motif_dmr[x]))

cluster_dmr_all = [np.loadtxt(f'DMR_CG_comb/DMR_cluster{i}_hypo.bed', usecols=(3), dtype=np.int) for i in range(nc)]
dmrgene = [[] for i in range(341748)]
fin = open('DMR_CG_comb/DMR_all_hypo.slop100.greatTSS.txt')
for line in fin:
    tmp = line.strip().split('\t')
    dmrgene[int(tmp[3])].append(tmp[8])

fin.close()
dmrgene = np.array(dmrgene)
motif = np.sort(list(motif_dmr.keys()))

prscore = []
isonodefilter = []
geneallrank = []
generank = []

ncpus = 5
for c in range(nc):
    print('Processing cluster'+str(c))
    cluster_dmr = set(cluster_dmr_all[c])
    print('Reading edge')
    p = Pool(ncpus)
    result = p.map(buildgraph, np.arange(len(motif)))
    p.close()
    print('Building graph')
    graph = np.zeros((len(gene_all), len(gene_all)))
    for x in result:
        graph[x[0]] = x[1]
    graph = graph[genefilter][:, genefilter]
    print('Normalization')
    ngraph = normgraph(graph)
    save_npz(f'cell_4176_L5PT_cluster{c}_ngraph.npz', csr_matrix(ngraph))
    print('Pagerank')
    pr = pagerank(ngraph, node_weight[c])
    print('Saving')
    prscore.append(pr)
    isonode = (np.diag(ngraph)==1)
    isonodefilter.append(isonode)
    geneallrank.append(gene[np.argsort(pr)[::-1], -1])
    idx = np.argsort(pr[~isonode])[::-1]
    generank.append(gene[~isonode][idx, -1])
    print(c)

prscore = np.array(prscore)
isonodefilter = np.array(isonodefilter)
geneallrank = np.array(geneallrank)
generank = np.array(generank)
