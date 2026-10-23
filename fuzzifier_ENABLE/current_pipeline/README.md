# Current Pipeline for Overweight Data

## Updates

1. Used FuzzSys for fuzzy concept definition and fuzzification, using default fuzzification with five fuzzy sets instead of three.

2. Used automatic Makefile with a for-loop iterating over all comparisons.

3. Generated fuzzy reports for all features.

## Usage:

1. Copy the three Excel tables in `excel_tables` subdirectory.

2. Call `make all`. The program extracts counts and metadata from the Excel tables, computes fuzzification per feature per comparison and generates the corresponding fuzzy reports.

