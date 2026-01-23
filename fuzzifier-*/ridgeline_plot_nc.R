#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggridges) # retained for notebook parity (not strictly required by this plot)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat(
    "Usage: Rscript ridgeline_plot.R FEATURE_OR_FEATURES_CSV OUT_PNG\n\n",
    "FEATURE_OR_FEATURES_CSV: a single feature id (e.g., 'hsa-let-7c') or comma-separated list\n",
    "OUT_PNG: output path for the PNG\n\n",
    "Inputs are discovered in this order:\n",
    "  1) Environment variables COMPARISON_TSV, METADATA_TSV, COUNTS_TSV\n",
    "  2) The absolute paths used in the notebook (if they exist on this machine)\n",
    "  3) Local defaults in the current working directory\n\n",
    sep = ""
  )
  quit(status = 2)
}

features_arg <- args[[1]]
out_png      <- args[[2]]

input_features <- str_split(features_arg, ",", simplify = FALSE)[[1]] %>%
  str_trim() %>%
  discard(~ .x == "")

if (length(input_features) == 0) {
  stop("No feature ids provided in first argument.")
}

# ------------------------------------------------------------------
# Input path discovery (env var -> notebook absolute -> local default)
# ------------------------------------------------------------------
first_folder <- strsplit(out_png, "/")[[1]][2]
noemd_files <- list.files(
  path = first_folder,
  pattern = "_noEMD\\.tsv$",
  full.names = TRUE
)
print(noemd_files)
if (length(noemd_files) == 0) {
  stop("No *_noEMD.tsv file found in ", first_folder)
}

# Take the first one (deterministic: alphabetical order)
nb_comparison <- sort(noemd_files)[1]
nb_metadata   <- "/mnt/extproj/projekte/modelling/interactive_fuzzifier/pipelines/data/metadata.tsv"
nb_counts     <- "/mnt/extproj/projekte/modelling/interactive_fuzzifier/pipelines/data/normalized/DESeq2_normalized_counts.tsv"

pick_path <- function(env_name, nb_path, local_default) {
  p_env <- Sys.getenv(env_name, unset = "")
  if (nzchar(p_env)) return(p_env)
  if (file.exists(nb_path)) return(nb_path)
  return(local_default)
}

comparison_path <- pick_path("COMPARISON_TSV", nb_comparison, "comparison_results.tsv")
metadata_path   <- pick_path("METADATA_TSV",   nb_metadata,   "metadata.tsv")
counts_path     <- pick_path("COUNTS_TSV",     nb_counts,     "DESeq2_normalized_counts.tsv")

if (!file.exists(comparison_path)) stop("Comparison TSV not found: ", comparison_path)
if (!file.exists(metadata_path))   stop("Metadata TSV not found: ", metadata_path)
if (!file.exists(counts_path))     stop("Counts TSV not found: ", counts_path)

# -------------------------
# Helpers
# -------------------------
safe_name <- function(x) {
  x <- as.character(x)
  x <- gsub("[^A-Za-z0-9._-]+", "_", x)
  x <- gsub("^_+|_+$", "", x)
  ifelse(nchar(x) == 0, "feature", x)
}

popcount <- function(bitstring) {
  stringr::str_count(bitstring, "1")
}

# -------------------------
# Load comparison results and compute method bitstrings per (feature, context)
# -------------------------
comp <- readr::read_tsv(comparison_path, show_col_types = FALSE) %>%
  rename_with(tolower)

# Accept either `context` or `cluster` as the context column
if (!"context" %in% colnames(comp) && "cluster" %in% colnames(comp)) {
  comp <- comp %>% rename(context = cluster)
}

required_cols <- c("feature", "context", "method")
missing <- setdiff(required_cols, colnames(comp))
if (length(missing) > 0) {
  stop("comparison file is missing required columns: ", paste(missing, collapse = ", "))
}

comp <- comp %>%
  mutate(
    feature = as.character(feature),
    context = as.character(context),
    method  = as.character(method)
  ) %>%
  distinct(feature, context, method)

# Stable method order: first appearance in file
method_levels <- comp %>%
  distinct(method) %>%
  pull(method)
print(method_levels)
M <- length(method_levels)
method_to_pos <- setNames(seq_len(M), method_levels)

row_bits <- comp %>%
  mutate(pos = method_to_pos[method]) %>%
  group_by(feature, context) %>%
  summarise(
    bitstring = {
      present <- sort(unique(pos))
      bits <- rep("0", M)
      bits[present] <- "1"
      paste0(bits, collapse = "")
    },
    .groups = "drop"
  ) %>%
  rename(feature_id = feature) %>%
  filter(feature_id %in% input_features)

if (nrow(row_bits) == 0) {
  stop(
    "None of the requested features were found in the comparison TSV. Requested: ",
    paste(input_features, collapse = ", ")
  )
}

row_bits <- row_bits %>%
  mutate(
    overlap_k = map_int(bitstring, popcount),
    symbol_string = chartr("01", "\u25cb\u25cf", bitstring)
  )

# -------------------------
# Load metadata + counts and merge
# -------------------------
meta_data <- readr::read_tsv(metadata_path, show_col_types = FALSE)

# counts.tsv: first column are feature IDs -> rownames
counts <- read.delim(
  counts_path,
  header = TRUE,
  sep = "\t",
  row.names = 1,
  check.names = FALSE
) %>%
  as.data.frame() %>%
  tibble::rownames_to_column(var = "feature_id") %>%
  as_tibble()

# Pivot metadata to sample key (tumor/normal columns contain sample IDs)
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

if (!all(c("context", "condition") %in% colnames(meta_long))) {
  stop("metadata.tsv must contain a 'context' column (plus tumor/normal sample columns).")
}

counts_filtered <- counts %>%
  filter(feature_id %in% input_features)

if (nrow(counts_filtered) == 0) {
  stop("None of the requested features were found in the counts TSV.")
}

counts_long <- counts_filtered %>%
  pivot_longer(
    cols = -feature_id,
    names_to = "sample_id",
    values_to = "count"
  ) %>%
  mutate(count = suppressWarnings(as.numeric(count)))

merged <- counts_long %>%
  left_join(meta_long, by = "sample_id") %>%
  filter(!is.na(context), !is.na(condition))

if (nrow(merged) == 0) {
  stop("After joining counts to metadata, there were no rows with non-missing context/condition.")
}

df <- merged %>%
  mutate(
    x = log2(count + 1),
    context = factor(context, levels = sort(unique(context))),
    condition = factor(condition, levels = c("normal", "tumor"))
  )

# -------------------------
# Plot (histogram facets) + per-facet method bitstring annotation
# -------------------------
bg_df <- row_bits %>%
  mutate(
    xmin = -Inf, xmax = Inf,
    ymin = -Inf, ymax = Inf
  )

p <- ggplot(df, aes(x = x, fill = condition)) +
  geom_histogram(position = "identity", alpha = 0.4, bins = 30) +
  geom_rect(
    data = bg_df,
    aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
    fill = "orange",
    alpha = 0.05,
    inherit.aes = FALSE
  ) +
  geom_text(
    data = row_bits,
    aes(label = symbol_string),
    x = Inf, y = Inf,
    hjust = 1, vjust = 1.3,
    size=8,
    family = "mono",
    inherit.aes = FALSE
  ) +
  facet_grid(context ~ feature_id, scales = "free_x") +
  labs(
    x = "log2(count + 1)",
    y = "count",
    fill = "condition"#,
    #title = if (length(input_features) == 1) input_features[[1]] else paste0(length(input_features), " features")
  ) +
  theme_minimal() +
  theme(legend.position = "bottom",text=element_text(size=20))

out_dir <- dirname(out_png)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

n_feat <- length(unique(df$feature_id))
contexts_present <- df %>% distinct(context) %>% nrow()

w <- max(6, 3.5 * n_feat)
h <- max(3, 2 * contexts_present)

ggsave(filename = out_png, plot = p, width = w, height = h, units = "in", dpi = 150)
