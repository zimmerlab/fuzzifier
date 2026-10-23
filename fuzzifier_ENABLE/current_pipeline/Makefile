PYTHON = biopython
FUZZSYS = /mnt/extproj/projekte/modelling/interactive_fuzzifier/command_line
PIPELINE = /mnt/extproj/prepare/modelling/interactive_fuzzifier/FlowSets_selection


all: prepare fuzzify report
.PHONY: prepare fuzzify report

prepare:	prepare_input.py \
			prepare_submatrix.py \
			./excel_tables/ \
			all_comparisons.tsv
	$(PYTHON) prepare_input.py --excel ./excel_tables/ \
		--base 2 \
		--pseudo 0 \
		--output ./data/; \
	tail -n +2 all_comparisons.tsv | while read line; do \
		IFS=$$'\t' read -r -a content <<< $$line; \
		$(PYTHON) prepare_submatrix.py --data ./data/ \
			--groups "$${content[1]}" \
			--output $${content[3]}/$${content[4]}.tsv; \
		$(PYTHON) prepare_submatrix.py --data ./data/ \
			--groups "$${content[2]}" \
			--output $${content[3]}/$${content[5]}.tsv; \
	done


fuzzify:	$(FUZZSYS)/main_concepts.py \
			$(FUZZSYS)/main_fuzzify.py \
			all_comparisons.tsv \
			config.json
	tail -n +2 all_comparisons.tsv | while read line; do \
		IFS=$$'\t' read -r -a content <<< $$line; \
		$(PYTHON) $(FUZZSYS)/main_concepts.py --mtx $${content[3]}/$${content[4]}.tsv \
			--config config.json \
			--threads 4 \
			--output $${content[3]}/fuzzy_$${content[4]}/; \
		$(PYTHON) $(FUZZSYS)/main_fuzzify.py --mtx $${content[3]}/$${content[4]}.tsv \
			--concept $${content[3]}/fuzzy_$${content[4]}/concepts_detailed.json \
			--config config.json \
			--output $${content[3]}/fuzzy_$${content[4]}/; \
		$(PYTHON) $(FUZZSYS)/main_concepts.py --mtx $${content[3]}/$${content[5]}.tsv \
			--config config.json \
			--threads 4 \
			--output $${content[3]}/fuzzy_$${content[5]}/; \
		$(PYTHON) $(FUZZSYS)/main_fuzzify.py --mtx $${content[3]}/$${content[5]}.tsv \
			--concept $${content[3]}/fuzzy_$${content[5]}/concepts_detailed.json \
			--config config.json \
			--output $${content[3]}/fuzzy_$${content[5]}/; \
	done


report:	generate_report.py \
		all_comparisons.tsv
	tail -n +2 all_comparisons.tsv | while read line; do \
		IFS=$$'\t' read -r -a content <<< $$line; \
		$(PYTHON) generate_report.py --dir $${content[3]} \
			--label1 $${content[4]} \
			--label2 $${content[5]} \
			--output $${content[3]}/comparison_reports/; \
	done


