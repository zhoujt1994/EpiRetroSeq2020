require(data.table)
library(lmerTest)
for (infile in c('MOp', 'SSp', 'ACA', 'AI', 'AUD', 'RSP', 'PTLp', 'VIS')){
fin <- paste(indir, infile, '_meta_cluster.txt', sep='')
data <- fread(fin)
colnames(data) <- c('target', 'gender', 'cluster')
data$mice <- apply(data[,c('gender', 'target')], 1, paste, collapse="-" )
pv <- c()
for (x in c('VISp', 'TH', 'SC', 'VTA', 'Pons', 'MY'))
	# if (is.element(x, unique(data$target)))
	if (sum(data$target==x)>20)
		for (y in sort(unique(data$cluster))){
			datatmp <- data
			datatmp$target <- (data$target==x)
			datatmp$cluster <- (data$cluster==y)
			model <- lmer(cluster ~ target + (1 | mice), data=datatmp)
			pv <- append(pv, anova(model)[6])
		}
pv <- matrix(pv, nrow=length(unique(data$cluster)))
write.table(pv, file=paste(indir, infile, '_proj_cluster_enrich_lmm_pvalue.txt', sep=''), quote=FALSE, sep='\t', row.names=FALSE, col.names=FALSE)
}
