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

    - `label_values`: List of specific values to be excluded from fuzzy concept definition, such as $\pm\infty$, zero and NA.

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

### Input Files and Formats

- `mtx`: Raw value matrix, either in `.tsv` or `.h5ad` format. The raw value matrix contains all values for fuzzy concept definition, which are required to be numeric values. It is neither required that the raw value matrix remains the same as that used for fuzzy concept definition, nor should all entries match the row or column names in the matrix.

- `concept`: Fuzzy concepts in `.json` format. These can either be constraints of fuzzy concepts, or complete fuzzy concepts with concrete values.

    - Constraints of fuzzy concepts: Concrete fuzzy concepts will be derived for each feature or sample, with an additional default fuzzy concept defined on the whole raw value matrix. The derivation follows the fuzzy concept definition with option `constraint` in `define_concept_by`. The derived fuzzy concepts are delivered as an additional output in `.json` format.
    
    - Concrete fuzzy concepts: The fuzzy concepts are directly used for fuzzification. For each fuzzification, the script searches for the fuzzy concept that matches the current row or column name in the raw value matrix. Otherwise the default fuzzy concept is applied if the corresponding name is not found.

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

### Output Files and Formats

The fuzzy values are delivered as `.tsv` files after each feature-wise or sample-wise fuzzification, starting with prefix `fuzzyValues_`. Each of these files contains fuzzy values in the additional labelling fuzzy sets and the defined fuzzy sets as columns. For a feature-wise (sample-wise) fuzzificaiton, each row represents a sample (feature) from the raw value matrix. All fuzzy values are rounded to 3 decimals in default.
