require(data.table)
library(lmerTest)
args = commandArgs(trailingOnly=TRUE)
fin <- paste(indir, 'cell_4176_rateg.meta.txt', sep='')
data <- fread(fin)
data$mice <- apply(data[,c('gender', 'target')], 1, paste, collapse="-")
data <- data[which((data$cluster==args[1]) | (data$cluster==args[2]))]
pvl <- c()
pvw <- c()
ngene <- 12261
for (i in 1:ngene){
	datatmp = data[,c(i, ngene+1, ngene+2, ngene+3, ngene+4, ngene+5), with=FALSE]
	colnames(datatmp)[1] = 'mCH'
	model <- lmer(mCH ~ cluster + gender + global + (1 | mice), data=datatmp)
  null <- lmer(mCH ~ gender + global + (1 | mice), REML=FALSE, data=datatmp)
  summary <- anova(model, null, test='Chisq')
  pvl <- append(pvl, summary$Pr[2])
  pvw <- append(pvw, anova(model)$Pr[1])
}
fout <- paste(indir, paste('cluster', args[1], args[2], sep='_'), '.ga.m.lrt.pvalue.txt', sep='')
write.table(pvl, file=fout, quote=FALSE, sep='\t', row.names=FALSE, col.names=FALSE)
fout <- paste(indir, paste('cluster', args[1], args[2], sep='_'), '.ga.m.wald.pvalue.txt', sep='')
write.table(pvw, file=fout, quote=FALSE, sep='\t', row.names=FALSE, col.names=FALSE)
