R = R/R-4.5.0/bin/Rscript
PYTHON = biopython
DIR = command_line_tool
DATASET = miRNA
BASEDIR = FlowSets_selection/$(DATASET)


DESeq2_2aspect: DESeq2_log2FC_concept DESeq2_log2FC_fuzzy DESeq2_padj_concept DESeq2_padj_fuzzy
expression: expression_concept numerator_fuzzy denominator_concept denominator_fuzzy
RFC: expression RFC_concept RFC_fuzzy RFC_markers
FFC: expression FFC_rule FFC_markers
.PHONY: generate_inputs DESeq2_test DESeq2_2aspect prepare_matrix expression RFC FFC visualization


generate_inputs:
	$(PYTHON) main_input.py --min_pairs 30


DESeq2_test:	$(BASEDIR)/data/raw_counts.tsv \
				$(BASEDIR)/data/metadata.tsv
	$(R) DESeq2_test.r $(BASEDIR)/data/raw_counts.tsv $(BASEDIR)/data/metadata.tsv $(BASEDIR)/data/


DESeq2_log2FC_concept:	$(BASEDIR)/data/DESeq2_log2FC.tsv \
						./config/DESeq2_log2FC.json
	$(PYTHON) $(DIR)/main_concepts.py --mtx $(BASEDIR)/data/DESeq2_log2FC.tsv \
		--config ./config/DESeq2_log2FC.json \
		--output $(BASEDIR)/FV_DESeq2/concepts_log2FC/


DESeq2_log2FC_fuzzy:	$(BASEDIR)/data/DESeq2_log2FC.tsv \
						$(BASEDIR)/FV_DESeq2/concepts_log2FC/concepts_detailed.json \
						./config/DESeq2_log2FC.json
	$(PYTHON) $(DIR)/main_fuzzify.py --mtx $(BASEDIR)/data/DESeq2_log2FC.tsv \
		--concept $(BASEDIR)/FV_DESeq2/concepts_log2FC/concepts_detailed.json \
		--config ./config/DESeq2_log2FC.json \
		--output $(BASEDIR)/FV_DESeq2/log2FC/


DESeq2_padj_concept:	$(BASEDIR)/data/DESeq2_padj.tsv \
						./config/DESeq2_padj.json
	$(PYTHON) $(DIR)/main_concepts.py --mtx $(BASEDIR)/data/DESeq2_padj.tsv \
		--config ./config/DESeq2_padj.json \
		--output $(BASEDIR)/FV_DESeq2/concepts_padj/


DESeq2_padj_fuzzy:	$(BASEDIR)/data/DESeq2_padj.tsv \
					$(BASEDIR)/FV_DESeq2/concepts_padj/concepts_detailed.json \
					./config/DESeq2_padj.json
	$(PYTHON) $(DIR)/main_fuzzify.py --mtx $(BASEDIR)/data/DESeq2_padj.tsv \
		--concept $(BASEDIR)/FV_DESeq2/concepts_padj/concepts_detailed.json \
		--config ./config/DESeq2_padj.json \
		--output $(BASEDIR)/FV_DESeq2/padj/


prepare_matrix:	$(BASEDIR)/data/raw_counts.tsv \
				$(BASEDIR)/data/metadata.tsv \
				./config/rawFoldChange.json
	$(PYTHON) main_rawFoldChange.py --mtx $(BASEDIR)/data/raw_counts.tsv \
		--metadata $(BASEDIR)/data/metadata.tsv \
		--config ./config/rawFoldChange.json \
		--output $(BASEDIR)/data/


expression_concept:	$(wildcard $(BASEDIR)/data/denominator_submatrices/submatrix_*.tsv) \
					./config/expression.json
	for f in `ls $(BASEDIR)/data/denominator_submatrices/ | grep -E submatrix_.+\.tsv`; do \
		g="`echo $$f | sed -e 's/\.tsv//g'`"; \
		$(PYTHON) $(DIR)/main_concepts.py --mtx $(BASEDIR)/data/denominator_submatrices/$$f \
			--config ./config/expression.json \
			--output $(BASEDIR)/FV_expression/concepts/$$g/; \
	done


numerator_fuzzy:	$(wildcard $(BASEDIR)/data/numerator_submatrices/submatrix_*.tsv) \
					$(wildcard $(BASEDIR)/FV_expression/concepts/submatrix_*/concepts_detailed.json) \
					./config/expression.json
	for f in `ls $(BASEDIR)/data/numerator_submatrices/ | grep -E submatrix_.+\.tsv`; do \
		g="`echo $$f | sed -e 's/\.tsv//g'`"; \
		$(PYTHON) $(DIR)/main_fuzzify.py --mtx $(BASEDIR)/data/numerator_submatrices/$$f \
			--concept $(BASEDIR)/FV_expression/concepts/$$g/concepts_detailed.json \
			--config ./config/expression.json \
			--output $(BASEDIR)/FV_expression/numerator/$$g/; \
	done


denominator_fuzzy:	$(wildcard $(BASEDIR)/data/denominator_submatrices/submatrix_*.tsv) \
					$(wildcard $(BASEDIR)/FV_expression/concepts/submatrix_*/concepts_detailed.json) \
					./config/expression.json
	for f in `ls $(BASEDIR)/data/denominator_submatrices/ | grep -E submatrix_.+\.tsv`; do \
		g="`echo $$f | sed -e 's/\.tsv//g'`"; \
		$(PYTHON) $(DIR)/main_fuzzify.py --mtx $(BASEDIR)/data/denominator_submatrices/$$f \
			--concept $(BASEDIR)/FV_expression/concepts/$$g/concepts_detailed.json \
			--config ./config/expression.json \
			--output $(BASEDIR)/FV_expression/denominator/$$g/; \
	done


RFC_concept:	$(BASEDIR)/data/paired_log2FC.tsv \
				./config/RFC.json
	$(PYTHON) $(DIR)/main_concepts.py --mtx $(BASEDIR)/data/paired_log2FC.tsv \
		--config ./config/RFC.json \
		--output $(BASEDIR)/FV_paired_log2FC/


RFC_fuzzy:	$(BASEDIR)/data/paired_log2FC.tsv \
			$(BASEDIR)/FV_paired_log2FC/concepts_detailed.json \
			./config/RFC.json
	$(PYTHON) $(DIR)/main_fuzzify.py --mtx $(BASEDIR)/data/paired_log2FC.tsv \
		--concept $(BASEDIR)/FV_paired_log2FC/concepts_detailed.json \
		--config ./config/RFC.json \
		--output $(BASEDIR)/FV_paired_log2FC/


RFC_markers:	$(wildcard $(BASEDIR)/FV_expression/numerator/submatrix_*/fuzzyValues_*.tsv) \
				$(wildcard $(BASEDIR)/FV_expression/denominator/submatrix_*/fuzzyValues_*.tsv) \
				$(wildcard $(BASEDIR)/FV_paired_log2FC/fuzzyValues_*.tsv) \
				./config/FlowSets_$(DATASET)_RFC.json
	$(PYTHON) main_FlowSets.py --expression $(BASEDIR)/FV_expression/ \
		--log2FC $(BASEDIR)/FV_paired_log2FC/ \
		--metadata $(BASEDIR)/data/metadata.tsv \
		--config ./config/FlowSets.json \
		--output $(BASEDIR)/results/raw_log2FC/


FFC_rule:	$(wildcard $(BASEDIR)/FV_expression/numerator/submatrix_*/fuzzyValues_*.tsv) \
			$(wildcard $(BASEDIR)/FV_expression/denominator/submatrix_*/fuzzyValues_*.tsv) \
			./config/fuzzy_rules.csv
	$(PYTHON) main_fuzzyFoldChange.py --numerator $(BASEDIR)/FV_expression/numerator/ \
		--denominator $(BASEDIR)/FV_expression/denominator/ \
		--rules ./config/fuzzy_rules.csv \
		--savedBy feature \
		--fuzzySets "NA,-INF,+INF,--,-,o,+,++" \
		--output $(BASEDIR)/FV_fuzzy_log2FC/


FFC_markers:	$(wildcard $(BASEDIR)/FV_expression/numerator/submatrix_*/fuzzyValues_*.tsv) \
				$(wildcard $(BASEDIR)/FV_expression/denominator/submatrix_*/fuzzyValues_*.tsv) \
				$(wildcard $(BASEDIR)/FV_fuzzy_log2FC/fuzzyValues_*.tsv) \
				$(BASEDIR)/data/metadata.tsv \
				./config/FlowSets_$(DATASET)_FFC.json
	$(PYTHON) main_FlowSets.py --expression $(BASEDIR)/FV_expression/ \
		--log2FC $(BASEDIR)/FV_fuzzy_log2FC/ \
		--metadata $(BASEDIR)/data/metadata.tsv \
		--config ./config/FlowSets.json \
		--output $(BASEDIR)/results/fuzzy_log2FC/


visualization:	$(wildcard $(BASEDIR)/data/DESeq2_*.tsv) \
				$(BASEDIR)/data/numerator_log.tsv \
				$(BASEDIR)/data/denominator_log.tsv \
				$(BASEDIR)/data/paired_log2FC.tsv \
				$(BASEDIR)/data/metadata.tsv \
				$(wildcard ./config/*.json) \
				$(wildcard $(BASEDIR)/FV_DESeq2/concepts_*.json) \
				$(wildcard $(BASEDIR)/FV_DESeq2/*/*.tsv) \
				$(wildcard $(BASEDIR)/FV_expression/numerator/submatrix_*/*.tsv) \
				$(wildcard $(BASEDIR)/FV_expression/denominator/submatrix_*/*.tsv) \
				$(wildcard $(BASEDIR)/FV_expression/concepts/submatrix_*/concepts.detailed.json) \
				$(wildcard $(BASEDIR)/FV_fuzzy_log2FC/*.tsv) \
				$(wildcard $(BASEDIR)/FV_paired_log2FC/*.tsv) \
				$(BASEDIR)/results/raw_log2FC/cluster-specific_features.tsv \
				$(BASEDIR)/results/fuzzy_log2FC/cluster-specific_features.tsv
	$(PYTHON) main_visualization.py --baseDir $(BASEDIR)/ \
		--config ./config/figures.json \
		--output $(BASEDIR)/visualization/
	$(PYTHON) main_ridgeline.py --data $(BASEDIR)/data/ \
		--concepts $(BASEDIR)/FV_expression/concepts/ \
		--result $(BASEDIR)/results/ \
		--indexCol sample \
		--clusterCol context \
		--defaultName DEFAULT \
		--output $(BASEDIR)/results/ridgeline/


