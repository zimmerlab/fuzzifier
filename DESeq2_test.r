suppressMessages (library (DESeq2))


metadata <- read.table ("./data/metadata.tsv", header = TRUE, row.names = NULL, sep = "\t", check.names = FALSE)
allClusters <- unique (metadata$context); allClusters <- allClusters[order (allClusters)]
miRNA <- read.table ("./data/raw_counts.tsv", header = TRUE, row.names = 1, sep = "\t", check.names = FALSE)
colData <- data.frame (sample = c(metadata$tumor, metadata$normal),
                       cluster = c(metadata$context, metadata$context),
                       type = c(rep ("tumor", nrow (metadata)), rep ("normal", nrow (metadata))))
colData <- colData[!duplicated (colData), ]
rownames (colData) <- colData$sample; colData <- colData[colnames (miRNA), ]
colData$condition <- unlist (lapply (rownames (colData), function (x) { paste (colData[x, c("cluster", "type")], collapse = "_") }))
colData$condition <- factor (colData$condition, levels = sort (unique (colData$condition)))
dds <- DESeqDataSetFromMatrix (miRNA, colData = colData, design = ~ condition)
dds <- estimateSizeFactors (dds); dds <- DESeq (dds)
effect_size <- data.frame (row.names = rownames (miRNA)); significance <- data.frame (row.names = rownames (miRNA))
for (cluster in allClusters)
{
    print (cluster)
    res <- results (dds, contrast = c("condition", paste (cluster, "tumor", sep = "_"), paste (cluster, "normal", sep = "_")))
    effect_size[, cluster] <- res$log2FoldChange; significance[, cluster] <- res$padj
}
significance[significance < 1e-10] <- 1e-10; significance <- -log10 (significance)
write.table (counts (dds, normalized = TRUE), "./data/expression_matrix.tsv", sep = "\t", quote = FALSE)
write.table (effect_size, "./data/DESeq2_log2FC.tsv", sep = "\t", quote = FALSE)
write.table (significance, "./data/DESeq2_padj.tsv", sep = "\t", quote = FALSE)


