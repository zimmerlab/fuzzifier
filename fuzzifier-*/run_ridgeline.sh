#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF2'
Usage:
  ./run_ridgelines_by_method_combo.sh INPUT_TSV OUTDIR [R_SCRIPT]

INPUT_TSV must contain columns:
  - feature
  - method
  - cluster

Outputs:
  OUTDIR/
    methods_index.tsv              # bit position -> method
    <bitstring>/                   # one dir per chosen overlap combination
      single_features/<feature>__<cluster>.png
      ridgeline_combined.png       # only if <10 features in this combo
      features.txt                 # list of features in this combo

Notes:
  - This version plots each feature into EVERY bitstring directory derived from each cluster (not only one).
  - Bit order is the order of first appearance of each distinct method in the input file.
EOF2
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

INPUT_TSV="$1"
OUTDIR="$2"
R_SCRIPT="${3:-./ridgeline_plot_nc.R}"

[[ -f "$INPUT_TSV" ]] || { echo "ERROR: Input TSV not found: $INPUT_TSV" >&2; exit 2; }
[[ -f "$R_SCRIPT" ]]  || { echo "ERROR: R script not found: $R_SCRIPT" >&2; exit 3; }

mkdir -p "$OUTDIR"

# ----------------------------------------------------------
# Filter out EMD methods (write to OUTFILE)
# ----------------------------------------------------------
BASENAME="$(basename "$INPUT_TSV")"
OUTFILE="${OUTDIR%/}/${BASENAME%.tsv}_noEMD.tsv"

awk -F'\t' '$NF !~ /EMD/' "$INPUT_TSV" > "$OUTFILE"

# Temporary mapping file: feature \t cluster \t bitstring
MAP_TSV="$(mktemp)"
trap 'rm -f "$MAP_TSV"' EXIT

# Temporary mapping file: feature \t cluster \t overlap_level (all/some/none)
OVERLAP_MAP_TSV="$(mktemp)"
trap 'rm -f "$OVERLAP_MAP_TSV"' EXIT

# -------------------------------------------------------------------
# Build:
# 1) methods_index.tsv (bit -> method)
# 2) MAP_TSV: feature -> cluster -> bitstring (ALL clusters)
# 3) OVERLAP_MAP_TSV: feature -> cluster -> overlap_level (all/some/none)
#
# IMPORTANT: use OUTFILE (noEMD) as input to AWK
# -------------------------------------------------------------------
awk -F'\t' -v outdir="$OUTDIR" -v overlap_map="$OVERLAP_MAP_TSV" '
function trim(s){ gsub(/^[ \t\r\n]+|[ \t\r\n]+$/, "", s); return s }

NR==1 {
  for (i=1; i<=NF; i++) {
    h=tolower(trim($i))
    if (h=="feature") fcol=i
    if (h=="method")  mcol=i
    if (h=="cluster") ccol=i
  }
  if (!fcol || !mcol || !ccol) {
    print "ERROR: header must contain columns named feature, method, and cluster" > "/dev/stderr"
    exit 10
  }
  next
}

{
  feat=trim($fcol)
  meth=trim($mcol)
  clus=trim($ccol)

  if (feat=="" || meth=="" || clus=="") next

  # method index in order of first appearance
  if (!(meth in mid)) {
    mid[meth]=++M
    mlist[M]=meth
  }

  # cluster order in order of first appearance
  if (!(clus in cid)) {
    cid[clus]=++C
    clist[C]=clus
  }

  # preserve feature order (first appearance)
  if (!(feat in fseen)) {
    fseen[feat]=1
    forder[++F]=feat
  }

  # mark presence of method for feature+cluster
  key=feat SUBSEP clus SUBSEP mid[meth]
  present[key]=1

  # track method presence per cluster (for total method count)
  ckey=clus SUBSEP mid[meth]
  cluster_has_method[ckey]=1
}

END {
  # write methods index
  idxfile = outdir "/methods_index.tsv"
  print "bit\tmethod" > idxfile
  for (i=1; i<=M; i++) {
    print i "\t" mlist[i] >> idxfile
  }

  # count methods per cluster (for overlap_level)
  for (c=1; c<=C; c++) {
    clus = clist[c]
    total = 0
    for (i=1; i<=M; i++) {
      if ((clus SUBSEP i) in cluster_has_method) total++
    }
    cluster_total[clus]=total
  }

  # For each feature AND each cluster: output bitstring (ALL clusters)
  for (j=1; j<=F; j++) {
    feat = forder[j]

    for (c=1; c<=C; c++) {
      clus = clist[c]
      bits = ""
      count = 0

      for (i=1; i<=M; i++) {
        key = feat SUBSEP clus SUBSEP i
        has = (key in present)
        bits = bits (has ? "1" : "0")
        if (has) count++
      }

      # write MAP_TSV row: feature, cluster, bits
      print feat "\t" clus "\t" bits

      # overlap level for feature x cluster
      total = cluster_total[clus]
      level = "none"
      if (count <= 0) level = "none"
      else if (total > 0 && count == total) level = "all"
      else level = "some"

      print feat "\t" clus "\t" level >> overlap_map
    }
  }
}
' "$OUTFILE" > "$MAP_TSV"

# -------------------------------------------------------------------
# For each (feature, cluster), create directory by its bitstring and write plot there.
# Also build per-directory features.txt for combined plotting.
# -------------------------------------------------------------------
while IFS=$'\t' read -r feat clus bits; do
  [[ -n "${feat:-}" && -n "${clus:-}" && -n "${bits:-}" ]] || continue
   # Skip empty bitstrings (all zeros)
  [[ "$bits" =~ ^0+$ ]] && continue

  combo_dir="${OUTDIR%/}/${bits}"
  single_dir="${combo_dir}/single_features"
  mkdir -p "$single_dir"

  # track features for this combo (will be de-duped later)
  echo "$feat" >> "${combo_dir}/features.txt"

  # safe filenames
  safe_feat="$(printf '%s' "$feat" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
  safe_clus="$(printf '%s' "$clus" | sed -E 's/[^A-Za-z0-9._-]+/_/g')"
  out_png="${single_dir}/${safe_feat}.png"

  echo "Plotting feature '$feat' (cluster '$clus') -> ${bits}/single_features/${safe_feat}.png"
  Rscript "$R_SCRIPT" "$feat" "$out_png" "$OVERLAP_MAP_TSV"
done < "$MAP_TSV"

# -------------------------------------------------------------------
# Combined plot per combination directory ONLY if < 10 features
# (uses unique feature names, regardless of how many clusters contributed)
# -------------------------------------------------------------------
shopt -s nullglob
for d in "${OUTDIR%/}"/[01]*; do
  [[ -d "$d" ]] || continue
  [[ -f "$d/features.txt" ]] || continue

  sort -u "$d/features.txt" -o "$d/features.txt"

  n="$(wc -l < "$d/features.txt" | tr -d '[:space:]')"
  if [[ "$n" -lt 11 ]]; then
    csv="$(paste -sd, "$d/features.txt")"
    combined_out="$d/ridgeline_combined.png"
    echo "Creating combined plot in $(basename "$d") ($n features)"
    Rscript "$R_SCRIPT" "$csv" "$combined_out" "$OVERLAP_MAP_TSV"
  else
    echo "Skipping combined plot in $(basename "$d") ($n features > 10)"
  fi
done

echo "Done."
echo "Filtered TSV: $OUTFILE"
echo "Methods index: ${OUTDIR%/}/methods_index.tsv"
echo "Combo directories created under: $OUTDIR"