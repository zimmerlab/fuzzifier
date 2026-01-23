#!/usr/bin/env Rscript

metadata_path="/mnt/extproj/projekte/modelling/interactive_fuzzifier/pipelines/data/metadata.tsv"
countdata_path="/mnt/extproj/projekte/modelling/interactive_fuzzifier/pipelines/data/normalized/DESeq2_normalized_counts.tsv"

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggridges)
  library(ggnewscale)
  library(tibble)
})

usage <- function() {
  cat(
    "Usage:\n",
    "  Rscript ridgeline_plot.R \"FEATURE1,FEATURE2,FEATURE3\" output_plot.png [OVERLAP_MAP_TSV]\n\n",
    "Inputs (expected in working directory):\n",
    "  metadata.tsv\n",
    "  counts.tsv   (header starts with a tab; first column is feature rownames)\n\n",
    "Arguments:\n",
    "  1) Comma-separated feature IDs (input_features)\n",
    "  2) Output plot filename (e.g., plot.png, plot.pdf)\n",
    "  3) Optional TSV mapping feature_id+cluster to overlap level (all/some/none)\n",
    sep = ""
  )
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  usage()
  quit(status = 1)
}

features_arg <- args[[1]]
out_plot <- args[[2]]
overlap_map_path <- if (length(args) >= 3) args[[3]] else ""

input_features <- str_split(features_arg, pattern = ",", simplify = TRUE) %>%
  as.character() %>%
  str_trim() %>%
  discard(~ .x == "")

if (length(input_features) == 0) {
  stop("No input features parsed from argument 1.")
}

overlap_by_feature_cluster <- NULL
if (overlap_map_path != "" && file.exists(overlap_map_path)) {
  overlap_df <- readr::read_tsv(
    overlap_map_path,
    col_names = c("feature_id", "cluster", "overlap_level"),
    show_col_types = FALSE
  )
  overlap_df <- overlap_df %>%
    filter(!is.na(feature_id), feature_id != "", !is.na(cluster), cluster != "")
  overlap_by_feature_cluster <- overlap_df
}

# -----------------------
# Read inputs
# -----------------------
meta_data <- readr::read_tsv(metadata_path, show_col_types = FALSE)

# counts.tsv: leading tab in header; first column are feature IDs -> rownames
counts <- read.delim(
  countdata_path,
  header = TRUE,
  sep = "\t",
  row.names = 1,
  check.names = FALSE
) %>%
  as.data.frame() %>%
  rownames_to_column(var = "feature_id") %>%
  as_tibble()

# -----------------------
# Pivot metadata to sample key (tumor/normal columns contain sample IDs)
# -----------------------
if (!all(c("tumor", "normal") %in% colnames(meta_data))) {
  stop("metadata.tsv must contain columns named 'tumor' and 'normal'.")
}

meta_long <- meta_data %>%
  pivot_longer(
    cols = c(tumor, normal),
    names_to = "condition",
    values_to = "sample_id"
  ) %>%
  filter(!is.na(sample_id), sample_id != "") %>%
  distinct(sample_id, condition, .keep_all = TRUE)

# -----------------------
# Filter counts to features and pivot longer to (feature_id, sample, count)
# -----------------------
counts_filtered <- counts %>%
  filter(feature_id %in% input_features)

counts_long <- counts_filtered %>%
  pivot_longer(
    cols = -feature_id,
    names_to = "sample_id",
    values_to = "count"
  ) %>%
  mutate(count = as.numeric(count))

# -----------------------
# Merge counts with metadata (expects context + condition in metadata)
# -----------------------
if (!all(c("context", "condition") %in% colnames(meta_long))) {
  stop("After pivot_longer, metadata must include columns named 'context' and 'condition'.")
}

merged <- counts_long %>%
  left_join(meta_long, by = "sample_id")

# -----------------------
# Plot data + annotations
# -----------------------
df <- merged %>%
  mutate(
    x = log2(count + 1),
    context = factor(context, levels = sort(unique(context)))
  )

# Fraction non-zero per feature_id × context × condition,
# collapsed into one label per feature_id × context
# Only keep labels where not all conditions are 100% (i.e., any frac_nonzero < 1)
ann <- df %>%
  group_by(feature_id, context, condition) %>%
  summarise(frac_nonzero = mean(count > 0, na.rm = TRUE), .groups = "drop") %>%
  mutate(line = sprintf("%s: %.1f%%", condition, 100 * frac_nonzero)) %>%
  arrange(feature_id, context, condition) %>%
  group_by(feature_id, context) %>%
  summarise(
    label = paste(line, collapse = " "),
    any_not_full = any(frac_nonzero < 1),
    .groups = "drop"
  )# %>%filter(any_not_full)

#p <- ggplot(df, aes(x = x, y = context, fill = condition, color = condition)) +
#  geom_density_ridges(alpha = 0.35, scale = 0.75, rel_min_height = 0.01) +
  #geom_label(
  #  data = ann,
  #  aes(x = -Inf, y = context, label = label),
  #  inherit.aes = FALSE,
  #  hjust = -0.1,
  #  vjust = 1.2,       # slightly lower
  #  label.size = 0,
  #  size = 2.2
  #) +
#  facet_wrap(~ feature_id, scales = "free_x", ncol = length(input_features)) +
#  scale_x_continuous(expand = expansion(mult = c(0.10, 0.05))) +
#  scale_y_discrete(
#    drop = FALSE,
#    expand = expansion(mult = c(0.05, 0.20))  # small space above each facet
#  ) +
#  labs(
#    x = "log2(count + 1)",
#    y = "context",
#    fill = "condition",
#    color = "condition"
#  ) +
#  theme_classic() +
#  theme(
#    strip.background = element_blank(),
#    legend.position = "bottom"
#  )

p <- ggplot(df, aes(x = x)) +
  {
    if (!is.null(overlap_by_feature_cluster) && nrow(overlap_by_feature_cluster) > 0) {
      bg_df <- df %>%
        distinct(feature_id, context) %>%
        left_join(
          overlap_by_feature_cluster,
          by = c("feature_id" = "feature_id", "context" = "cluster")
        ) %>%
        filter(!is.na(overlap_level), overlap_level != "")
      if (nrow(bg_df) > 0) {
        bg_df <- bg_df %>%
          mutate(
            bg_fill = case_when(
              overlap_level == "all" ~ "#1f77b4",
              overlap_level == "some" ~ "#a6cee3",
              overlap_level == "none" ~ "white",
              TRUE ~ "white"
            )
          )
        geom_rect(
          data = bg_df,
          aes(xmin = -Inf, xmax = Inf, ymin = -Inf, ymax = Inf, fill = bg_fill),
          inherit.aes = FALSE,
          alpha = 0.5
        )
      }
    }
  } +
  scale_fill_identity(guide = "none") +
  ggnewscale::new_scale_fill() +
  geom_histogram(
    position = "identity",
    bins = 30,
    aes(fill = condition),
    alpha = 0.4
  ) +
  facet_grid(context ~ feature_id, scales = "free_x") +
  labs(
    x = "log2(count + 1)",
    y = "count",
    fill = "condition"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.background = element_rect(fill = "white", color = NA)
  )
print(ann)
ggsave(filename = out_plot, plot = p, width = 3.5 * max(1, length(input_features)), height = 7, dpi = 300)
