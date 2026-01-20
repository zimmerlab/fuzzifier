# Fuzzifier
## Capability
Instead of significant feature identification with statistical tests, we demonstrate methods based on fuzzification and fuzzy logic, which provides a robust and comprehensive categorization of raw values in multiple approaches. Moreover, the results of our fuzzy-based methods extend the results of statistical tests with a higher sensitivity and focus on differences in the value distributions between both conditions in the comparison, while providing further data- and distribution-based validation.

## Requirements
### Python packages (Python version 3.13.5)
```
numpy
pandas
scipy
seaborn
matplotlib
```

### R packages (R version 4.5.0)
```
DESeq2
tidyverse
ggridges
dplyr
ggplot2
ggVennDiagram
UpSetR
ComplexUpset
argparse
VennDiagram
```

## Example input data
The miRNA counts are downloaded from TCGA (The Cancer Genome Atlas Project) as an example. Pairs of primary tumor and solid normal tissue samples collected from the same patient are built. Nine cancer types containing at least 30 such tumor-normal sample pairs remain for the analysis. The miRNAs are filtered by their sparsity, with either of the following criterion:

1. at least expressed in over 50% of primary tumor and solid normal tissue samples in each cancer type,
2. at least expressed in over 70% of either primary tumor or solid normal tissue samples in exactly one cancer type, otherweise expressed in less than 10% of both types of samples in other cancer types

This results in a raw count matrix containing 359 miRNAs in 968 samples. A mapping of sample IDs to their originated cancer types is listed in the metadata.

Information regarding benchmark miRNAs and CMC scores are extracted from the supplemental table of <cite>Suszynska et al.</cite> as a reference for the identified cancer-specific miRNAs.

## Pipeline
### DESeq2 normalization statistical test
The raw count matrix is first normalized by DESeq2, followed by statistical test comparing tumor and normal expression in each cancer type, respectively.
```
make DESeq2_test
```

### DESeq2 2-aspect fuzzification
The results from the statistical test, namely log2 foldchange and corrected _p_-values, are fuzzified separately in a matrix-wise manner.
```
make DESeq2_log2FC_concept
make DESeq2_log2FC_fuzzify
make DESeq2_padj_concept
make DESeq2_padj_fuzzify
```

### preprocessing
The raw count matrix is log2-transformed and splitted into two submatrices for primary tumor samples (numerator) and solid normal tissue samples (denominator). Pair-wise raw log2 foldchanges (raw log2FC) are calculated from both submatrices.
```
make matrix_prepare
```

### raw log2FC fuzzification

```
make raw_log2FC_concept
make raw_log2FC_fuzzify
```

### fuzzy rule fuzzification
```
make fuzzy_rule_concept
make fuzzy_rule_numerator
make fuzzy_rule_denominator
make fuzzy_rule_combine
```

### identification and validation of cancer-specific and marker miRNAs
```
make comparison
```

### visualization
```
make visualization
```

## Citation
Suszynska M, Machowska M, Fraszczyk E, Michalczyk M, Philips A, Galka-Marciniak P, Kozlowski P. CMC: Cancer miRNA Census - a list of cancer-related miRNA genes. Nucleic Acids Res. 2024 Feb 28;52(4):1628-1644. doi: 10.1093/nar/gkae017. PMID: 38261968; PMCID: PMC10899758.
