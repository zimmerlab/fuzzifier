# Fuzzifier
## Capability
Instead of significant feature identification with statistical tests, we demonstrate methods based on fuzzification and fuzzy logic, which provides a robust and comprehensive categorization of raw values in multiple approaches. Moreover, the results of our fuzzy-based methods extend the results of statistical tests with a higher sensitivity and focus on differences in the value distributions between both conditions in the comparison, while providing further data- and distribution-based validation.

## Requirements
```Python packages (Python version 3.13.5)
numpy
pandas
scipy
seaborn
matplotlib
```
```R packages (R version 4.5.0)
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

## Pipeline
### DESeq2 normalization statistical test
```
make DESeq2_test
```

### DESeq2 2-aspect fuzzification
```
make DESeq2_log2FC_concept
make DESeq2_log2FC_fuzzify
make DESeq2_padj_concept
make DESeq2_padj_fuzzify
```

### preprocessing
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
