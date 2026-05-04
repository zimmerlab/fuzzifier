# Command-Line Fuzzifier

## Overview

The Fuzzifier tool is used to define fuzzy concepts based on a given raw value matrix, for example, expression values from high-throughput sequencing. Both dense and sparse matrices are accepted, namely with extension `.tsv` or `.h5ad`.

The tool consists of two parts. First, fuzzy concepts are derived from the raw value matrix by `main_concepts.py`. The other script `main_fuzzify.py` calculates fuzzy values to the raw values according to the fuzzy concepts. It is not required that one same raw value matrix should be imported for both scripts in one analysis run.

## Fuzzy Concept Definition

Fuzzy concepts are defined for the raw values using either of the two approaches per feature (row) or per sample (column). Otherwise it is derived based on all values in the raw value matrix, which is always calculated as the backup fuzzy concept in default.

Two types of membership functions are currently implemented. For one trapezoidal membership function, the four x-coordinates of the trapezoid are required as function parameter, while the Gaussian membership function requires parameter $\mu$ and $\sigma$. These can be acquired either from the mean and standard deviation, or from fitting a Gaussian function to the underlying raw value distribution. No Gaussian function will be fitted for the whole matrix. Moreover, for the $\sigma$ of a Gaussian membership function, the function parameter is displayed as a factor to be multiplied with $\sigma$.


### Input Files and Formats

Two input files are necessary, namely the raw value matrix `mtx` and the config file `config`.

- `mtx`: Raw value matrix, either in `.tsv` or `.h5ad` format. The raw value matrix contains all values for fuzzy concept definition, which are required to be numeric values.

- `config`: Config file containing detailed parameter settings in `.json` format.

    - `label_zero`: Whether to exclude zeros in fuzzy concept derivation and to add an extra fuzzy set for zeros in fuzzification. Other specific values such as $\pm\infty$ and missing values (NaN) are detected automatically.

    - Cutoff for minimally accepted values: All values no larger than this cutoff are regarded as noise and discarded in the fuzzy concept definition. Two types of cutoffs are available, and the larger one is chosen as cutoff.

        - `left_noise_cutoff_constant`: Constant cutoff for small values. This value is applied for all features or samples in the raw value matrix.

        - `left_noise_cutoff_percent`: Percent cutoff for small values. For each feature or sample, the real cutoff is calculated as the corresponding percentile of its raw value distribution.

    - Cutoff for maximally accepted values: All values no smaller than this cutoff are regarded as noise and discarded in the fuzzy concept definition. Two types of cutoffs are available, and the smaller one is chosen as cutoff.

        - `right_noise_cutoff_constant`: Constant cutoff for large values. This value is applied for all features or samples in the raw value matrix.

        - `right_noise_cutoff_percent`: Percent cutoff for large values. For each feature or sample, the real cutoff is calculted as the corresponding percentil of its raw value distribuion.

    - `key_default_concept`: Name of the default fuzzy concept.

    - `define_concept_per`: Whether the fuzzy concept is defined for each feature (`feature`) or for each sample (`sample`) or for the whole raw value matrix (`matrix`).

    - `define_concept_by`: Whether to define fuzzy concepts by given constraints (`constraint`) or by default fuzzification (`default`). Detailed parameters are specified in the entry with key starting with `parameters`.

        - `constraint`: The fuzzy concepts are derived from a set of given constraints. Three types of constraints are allowed, which is specified by `constraint_type`. It is required that one constraint should be specified for each function parameter and given in `constraints`. An additional parameter, `use_scipy_optimization`, specifies whether SciPy optimization is to be applied for estimating $\sigma$ for z-scores as constraints.

            - `fixed`: The constraints can be directly used as a completed fuzzy concept with concrete values.

            - `proportion`: Percentiles are calculated from the underlying raw value distribution, excluding all specific and noise values. For the coordinates of trapezoidal membership functions and $\mu$ of Gaussian membership functions, percentages are required as constraints. On the other hand, the constraint for $\sigma$ of a Gaussian membership function is the factor to be multiplied to the standard deviation of the underlying raw value distribution.

            - `z-score`: The $\mu$ and $\sigma$ for z-score calculation are acquired either from the mean and standard distribution, or from fitting of a Gaussian function to the underlying raw value distribution. The constraints should be given in the same way as for the percentiles, namely z-scores for the coordinates of trapezoidal membership functions and $\mu$ for Gaussian membership functions, while multiplication factor is required for $\sigma$ of Gaussian membership functions.

        - `default`: A Gaussian function is fitted to the underlying raw value distribution. This fitting is calculated on the density curve estimated from a given band width factor (`band_width_factor`). Then a fuzzy concept symmetric to $\mu$ is defined based on the fitted Gaussian parameters, with the fitted Gaussian membership function in the middle and the same number of trapezoidal membership functions on both sides, as specified by `number_fuzzy_sets_per_side`. It is optional to adjust $\sigma$ globally by a multiplication factor (`width_scale_factor`). The width of each of these symmetric trapezoids is the same and derived from $\mu$ and the width factor (`slope_percentage`). Additionally, SciPy optimization is available for estimating $\sigma$ by setting `use_scipy_optimization` here to `true`.

    - `fuzzy_variables`: Ordered list of fuzzy variables, excluding those for specific values or noise.


### Output Files and Formats

Two output files are generated. `concepts_detailed.json` contains all fuzzy concepts with concrete values, while `concepts_constraints.json` consists only of the (derived) constraints. Each fuzzy concept, regardless of constraints or concrete values, are generally a dictionary in Python.

#### Concrete Fuzzy Concept

- `number_fuzzy_sets`: Number of fuzzy sets, excluding labeling fuzzy sets for specific values or noise.

- `label_values`: List of specific values that should be excluded from fuzzy concept definition. $\pm\infty$ and NaN are stored as strings.

- `MIN-NOISE`: Cutoff for the smallest raw value allowed for fuzzy concept definition. It is calculated from the maximum of the constant cutoff and the percentile cutoff.

- `MAX-NOISE`: Cutoff for the largest raw value allowed for fuzzy concept definition. It is calculated from the minimum of the constant cutoff and the percentile cutoff.

- Fuzzy variable: List consisting of a sublist of function parameters, type of membership function (`trapezoidal` or `Gaussian`), color of the membership funcion for plotting and the expected percentage of raw values, which have their highest membership in this fuzzy set. It is defined for each fuzzy set.

#### Constrainted Fuzzy Concept

- `value_type`: Type of the constraints (`fixed` or `proportion` or `z-score`).

- `number_of_fuzzy_sets`: Number of fuzzy sets, excluding labeling fuzzy sets for specific values or noise.

- `label_values`: List of specific values that should be excluded from fuzzy concept definition. $\pm\infty$ and NaN are stored as strings.

- `fit_Gaussian_curve`: Whether the required $\mu$ and $\sigma$ are approximated from the mean and standard deviation, or derived from fitting of a Gaussian function.

- `use_scipy_optimization`: Whether to use SciPy optimization functions to approximate $\sigma$.

- `band_width_factor`: Factor for band width in density estimation and Gaussian funciton fitting.

- Fuzzy variable: List consisting of a sublist of constraints for function parameters, type of membership function (`trapezoidal` or `Gaussian`), color of the membership funcion for plotting and the expected percentage of raw values, which have their highest membership in this fuzzy set. It is defined for each fuzzy set.

## Fuzzification

Memberships are calculated for the given raw value matrix based on the fuzzy concepts. Selected specific values and noise are labelled in an additional boolean fuzzy set, respectively, where each specific value has membership 1 in the corresponding labelling fuzzy set and membership 0 otherwise.

The results are evaluated after each fuzzification by comparing the observed percentage of raw values for each fuzzy set to the expected one. If the absolute difference for one fuzzy set is lower than a given cutoff, e.g. 10%, then it can be assumed that the fuzzy concept fits the raw value distribution in this particular fuzzy set.

### Input Files and Formats

- `mtx`: Raw value matrix, either in `.tsv` or `.h5ad` format. The raw value matrix contains all values for fuzzy concept definition, which are required to be numeric values. It is neither required that the raw value matrix remains the same as that used for fuzzy concept definition, nor should all entries match the row or column names in the matrix.

- `concept`: Fuzzy concepts in `.json` format. These can either be constraints of fuzzy concepts, or complete fuzzy concepts with concrete values.

    - Constraints of fuzzy concepts: Concrete fuzzy concepts will be derived for each feature or sample, with an additional default fuzzy concept defined on the whole raw value matrix. The derivation follows the fuzzy concept definition with option `constraint` in `define_concept_by`. The derived fuzzy concepts are delivered as an additional output in `.json` format.
    
    - Concrete fuzzy concepts: The fuzzy concepts are directly used for fuzzification. For each fuzzification, the script searches for the fuzzy concept that matches the current row or column name in the raw value matrix. Otherwise the default fuzzy concept is applied if the corresponding name is not found. It is also recorded whether an individual fuzzy concept is available or not.

- `config`: Config file containing detailed parameter settings in `.json` format.

    - Cutoff for minimally accepted values: All values no larger than this cutoff are regarded as noise and discarded in the fuzzy concept definition. Two types of cutoffs are available, and the larger one is chosen as cutoff.

        - `left_noise_cutoff_constant`: Constant cutoff for small values. This value is applied for all features or samples in the raw value matrix.

        - `left_noise_cutoff_percent`: Percent cutoff for small values. For each feature or sample, the real cutoff is calculated as the corresponding percentile of its raw value distribution.

    - Cutoff for maximally accepted values: All values no smaller than this cutoff are regarded as noise and discarded in the fuzzy concept definition. Two types of cutoffs are available, and the smaller one is chosen as cutoff.

        - `right_noise_cutoff_constant`: Constant cutoff for large values. This value is applied for all features or samples in the raw value matrix.

        - `right_noise_cutoff_percent`: Percent cutoff for large values. For each feature or sample, the real cutoff is calculted as the corresponding percentil of its raw value distribuion.

    - `key_default_concept`: Name of the default fuzzy concept.

    - `fuzzify_per`: Whether to fuzzify per feature (`feature`) or per sample (`sample`) or for the whole raw value matrix (`matrix`). In the last case, feature-wise fuzzification is calculated and the default fuzzy concept is applied. This also determines whether the fuzzy value output is generated per feature or per sample, since it is delivered after each fuzzification.

    - `rename_labels`: Dictionary for renaming of the labelling fuzzy sets.

    - `generate_report_plots`: Whether to generate fuzzy report figure for each feature or sample after fuzzification.

### Output Files and Formats

Two subdirectories are generated for fuzzy values (`fuzzy_values`) and evaluations (`evaluations`), respectively. Given a constrained fuzzy concept as input, feature-wise or sample-wise fuzzy concepts will automatically be derived based on the input raw value matrix and stored in `concepts_detailed.json` in the main output directory. A third subdirectory, `reports`, is created if individual fuzzy report figures are to be generated.

- `fuzzy_values`: Subdirectory for fuzzy values. The fuzzy values are delivered as `.tsv` files after each feature-wise or sample-wise fuzzification, starting with prefix `fuzzyValues_`. Each of these files contains fuzzy values in the additional labelling fuzzy sets and the defined fuzzy sets as columns. For a feature-wise (sample-wise) fuzzificaiton, each row represents a sample (feature) from the raw value matrix. All fuzzy values are rounded to 3 decimals in default.

- `evaluations`: Subdirectory for evaluation tables and figures.

    - `expected_percentage.tsv`: Table of expected percentage of raw values per fuzzy set per feature / sample (expectation). These values are summarized from the corresponding fuzzy concept.

    - `expected_percentage.png`: Clustered heatmap of expectation.

    - `observed_percentage.tsv`: Table of observed percentage of raw values per fuzzy set per feature / sample (observation). It is defined as the percentage of raw values with their highest fuzzy value in one fuzzy set.

    - `observed_percentage.png`: Clustered heatmap of observation.

    - `deviation.png`: Clustered heatmap of the difference between observation and expectation.

    - `summary.tsv`: Summary table for each fuzzification.
    
        - `deviation`: RMSD between observation and expectation.

        - `individual_concept`: Whether the feature / sample was fuzzified with a fuzzy concept that has the same name as the feature / sample name.

- `reports`: Subdirectory for fuzzy report per feature / sample. It is a figure with 2 subplots, and a prefix `report_` followed by the corresponding feature / sample name.

    - Top panel: Fuzzy concept with the raw value distribution in background. A dashed line is added on the left or the right side of the subplot if there is any noise value, indicating the corresponding noise cutoff.

    - Bottom panel: Categorical heatmap for observation, expectation and their difference in each fuzzy set. The corresponding percentages are shown by the percentage labels in the first 2 rows above the red line. On the other hand, the difference in the last row below the red line is categorized into 3 groups by comparing to a given cutoff $\pm$ 10%.
