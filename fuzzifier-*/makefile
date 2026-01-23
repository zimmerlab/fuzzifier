PYTHON=python3.13


# DESeq2 2-aspect fuzzficiation
DESeq2_test: ./data/raw_counts.tsv
	Rscript DESeq2_test.r

DESeq2_log2FC_concept: ./data/DESeq2_log2FC.tsv ./config/concepts_DESeq2FC.json
	$(PYTHON) main_concepts.py --mtx ./data/DESeq2_log2FC.tsv \
		--config ./config/concepts_DESeq2FC.json \
		--output ./FV_DESeq2/concepts_DESeq2_log2FC.json

DESeq2_log2FC_fuzzify: ./data/DESeq2_log2FC.tsv ./FV_DESeq2/concepts_DESeq2_log2FC.json ./config/fuzzifier_DESeq2FC.json
	$(PYTHON) main_fuzzifier.py --mtx ./data/DESeq2_log2FC.tsv \
		--concept ./FV_DESeq2/concepts_DESeq2_log2FC.json \
		--config ./config/fuzzifier_DESeq2FC.json \
		--output ./FV_DESeq2/log2FC/

DESeq2_padj_concept: ./data/DESeq2_padj.tsv ./config/concepts_DESeq2padj.json
	$(PYTHON) main_concepts.py --mtx ./data/DESeq2_padj.tsv \
		--config ./config/concepts_DESeq2padj.json \
		--output ./FV_DESeq2/concepts_DESeq2_padj.json

DESeq2_padj_fuzzify: ./data/DESeq2_padj.tsv ./FV_DESeq2/concepts_DESeq2_padj.json ./config/fuzzifier_DESeq2padj.json
	$(PYTHON) main_fuzzifier.py --mtx ./data/DESeq2_padj.tsv \
		--concept ./FV_DESeq2/concepts_DESeq2_padj.json \
		--config ./config/fuzzifier_DESeq2padj.json \
		--output ./FV_DESeq2/padj/


# preparation
matrix_prepare: ./data/expression_matrix.tsv ./config/rawFoldChange.json
	$(PYTHON) main_rawFoldChange.py --mtx ./data/expression_matrix.tsv \
		--metadata ./data/metadata.tsv \
		--config ./config/rawFoldChange.json \
		--output ./data/


# raw log2FC fuzzification
raw_log2FC_concept: ./data/paired_log2FC.tsv ./config/concepts_defaultRFC.json
	$(PYTHON) main_concepts.py --mtx ./data/paired_log2FC.tsv \
		--config ./config/concepts_defaultRFC.json \
		--output ./FV_paired_log2FC/concepts_paired_log2FC.json

raw_log2FC_fuzzify: ./data/paired_log2FC.tsv ./FV_paired_log2FC/concepts_paired_log2FC.json ./config/fuzzifier_defaultRFC.json
	$(PYTHON) main_fuzzifier.py --mtx ./data/paired_log2FC.tsv \
		--concept ./FV_paired_log2FC/concepts_paired_log2FC.json \
		--config ./config/fuzzifier_defaultRFC.json \
		--output ./FV_paired_log2FC/


# fuzzy rule fuzzification
fuzzy_rule_concept: ./data/numerator_log.tsv ./data/denominator_log.tsv ./config/concepts_defaultLog.json
	$(PYTHON) main_concepts.py --mtx ./data/numerator_log.tsv \
		--metadata ./data/metadata.tsv \
		--config ./config/concepts_defaultLog.json \
		--perCluster \
		--output ./FV_fuzzy_log2FC/concepts_log_numerator.json
	$(PYTHON) main_concepts.py --mtx ./data/denominator_log.tsv \
		--metadata ./data/metadata.tsv \
		--config ./config/concepts_defaultLog.json \
		--perCluster \
		--output ./FV_fuzzy_log2FC/concepts_log_denominator.json
	$(PYTHON) main_mergeConcepts.py --data ./data/ \
		--concepts ./FV_fuzzy_log2FC/ \
		--metadata ./data/metadata.tsv \
		--config ./config/concepts_defaultLog.json

fuzzy_rule_numerator: ./data/numerator_log.tsv ./FV_fuzzy_log2FC/concepts_log_feature-wise.json ./config/fuzzifier_defaultLog.json
	$(PYTHON) main_fuzzifier.py --mtx ./data/numerator_log.tsv \
		--concept ./FV_fuzzy_log2FC/concepts_log_feature-wise.json \
		--metadata ./data/metadata.tsv \
		--config ./config/fuzzifier_defaultLog.json \
		--perCluster \
		--output ./FV_fuzzy_log2FC/numerator/

fuzzy_rule_denominator: ./data/denominator_log.tsv ./FV_fuzzy_log2FC/concepts_log_feature-wise.json ./config/fuzzifier_defaultLog.json
	$(PYTHON) main_fuzzifier.py --mtx ./data/denominator_log.tsv \
		--concept ./FV_fuzzy_log2FC//concepts_log_feature-wise.json \
		--metadata ./data/metadata.tsv \
		--config ./config/fuzzifier_defaultLog.json \
		--perCluster \
		--output ./FV_fuzzy_log2FC/denominator/

fuzzy_rule_combine: ./FV_fuzzy_log2FC/numerator/ ./FV_fuzzy_log2FC/denominator/
	$(PYTHON) main_fuzzyRule.py --numerator ./FV_fuzzy_log2FC/numerator/ \
		--denominator ./FV_fuzzy_log2FC/denominator/ \
		--output ./FV_fuzzy_log2FC/fuzzy_rule/


# identification and validation of cancer-specific or marker miRNAs
comparison: ./data/ ./FV_paired_log2FC/ ./FV_fuzzy_log2FC/ ./FV_DESeq2/ ./config/comparison.json ./gkae017_supplemental_files/
	$(PYTHON) main_comparison.py --standard ./data/ \
		--raw ./FV_paired_log2FC/ \
		--fuzzy ./FV_fuzzy_log2FC/ \
		--DESeq2 ./FV_DESeq2/ \
		--metadata ./data/metadata.tsv \
		--config ./config/comparison.json \
		--output ./results/
	$(PYTHON) main_CMC-validation.py --data ./data/ \
		--metadata ./data/metadata.tsv \
		--result ./results/comparison_results.tsv \
		--reference ./gkae017_supplemental_files/ \
		--cmcCut 3


# visualization
visualization: ./data/ ./results/ ./config/visualization.json
	$(PYTHON) main_visualization.py --data ./data/ \
		--result ./results/ \
		--metadata ./data/metadata.tsv \
		--config ./config/visualization.json \
		--output ./visualization/
	mkdir -p ./upset_plots/
	Rscript visOverlap.R \
		--methods "DESeq2 2-aspect,DESeq2 standard,fuzzy rule,raw log2FC" \
		--compPath ./results/comparison_results.tsv \
		--output ./upset_plots/ \
		--cmcCut 3
	mkdir -p ./ridgelines/
	bash run_ridgeline.sh ./results/comparison_results.tsv ./ridgelines/

