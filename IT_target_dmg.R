require(data.table)
library(lmerTest)
args = commandArgs(trailingOnly=TRUE)
fin <- paste(indir, args[1], '.rate.meta.txt', sep='')
data <- fread(fin)
data$mice <- apply(data[,c('gender', 'target')], 1, paste, collapse="-")
data <- data[which((data$target==args[2]) | (data$target==args[3]))]
pvl <- c()
pvw <- c()
ngene <- 12261
for (i in 1:ngene){
        datatmp = data[,c(i, ngene+1, ngene+2, ngene+3, ngene+4, ngene+5), with=FALSE]
        colnames(datatmp)[1] = 'mCH'
        model <- lmer(mCH ~ target + gender + cluster + global + (1 | mice), REML=FALSE, data=datatmp)
        null <- lmer(mCH ~ gender + cluster + global + (1 | mice), REML=FALSE, data=datatmp)
        summary <- anova(model, null, test='Chisq')
        pvl <- append(pvl, summary$Pr[2])
        pvw <- append(pvw, anova(model)$Pr[1])
}
fout <- paste(indir, paste(args[1], args[2], args[3], sep='_'), '.post.gca.m.lrt.pvalue.txt', sep='')
write.table(pvl, file=fout, quote=FALSE, sep='\n', row.names=FALSE, col.names=FALSE)
fout <- paste(indir, paste(args[1], args[2], args[3], sep='_'), '.post.gca.m.wald.pvalue.txt', sep='')
write.table(pvw, file=fout, quote=FALSE, sep='\n', row.names=FALSE, col.names=FALSE)
