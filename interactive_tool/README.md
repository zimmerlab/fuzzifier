# Interactive Fuzzifier

## Overview

The Fuzzifier tool is used to define fuzzy concepts based on a given raw value matrix, for example, expression values from high-throughput sequencing. Only dense matrices are accepted, namely with extension `.tsv`.

The tool consists of two parts. First, fuzzy concepts are derived from the raw value matrix in section `Initial Fuzzy Concept Definition`. The other section `Fuzzy Concepts Visualization and Comparison` visualizes fuzzy concepts on a given raw value matrix. Both can be directly taken from the previous section, or uploaded by the user. It is also not required that both raw value matrix and fuzzy concepts should match each other.

## Fuzzy Concept Definition

Fuzzy concepts are defined for the raw values using either of the two approaches per feature (row). Otherwise it is derived based on all values in the raw value matrix, which is always calculated as the backup fuzzy concept in default. It is also possible to invert the matrix so that the fuzzy concept is derived sample-wise.

Two types of membership functions are currently implemented. For one trapezoidal membership function, the four x-coordinates of the trapezoid are required as function parameter, while the Gaussian membership function requires parameter $\mu$ and $\sigma$. These can be acquired either from the mean and standard deviation, or from fitting a Gaussian function to the underlying raw value distribution. No Gaussian function will be fitted for the whole matrix.


### Input Files and Formats

Only one input file, the raw value matrix, is necessary. It is required to be in `.tsv` format and contains all values for fuzzy concept definition, which sould be all numeric values. In default, fuzzy concepts will be derived for each feature. It is still optional to invert the raw value matrix to perform sample-wise fuzzy concept definition by clicking the button `Invert`. Specific values, namely $\pm\infty$ and zero, can be selected so that they are excluded in the fuzzy concept definition.

After the raw value matrix is uploaded, the user can add cutoffs for noise values for both extremely small or large values. Two types of cutoffs are available, namely constant cutoffs or percent cutoffs. The constant cutoffs are applied on all features in the matrix, while the percent cutoffs are calculated for each feature in the raw value matrix. i.e. feature-specific. For extreme small (large) values, the larger (smaller) value among the constant and percent cutoffs is chosen as final cutoff for each feature. All raw values no larger (smaller) than the noise cutoff for extreme small (large) values are regarded as noise and discarded.

A brief overview table of the raw value matrix is also available. It includes the number of rows (features) and columns (samples), the minimum and maximum of all raw values excluding specific values and noise, number and percent of noise, missing values (NA), $-\infty$, $+\infty$ and zeros in the raw value matrix. Additionally, the raw value distribution is visualized in a histogram.


### Initial Fuzzy Concept Derivation

Two main methods are available for fuzzy concept derivation, either from constraints such as constant values or percentiles, or by fitting of a Gaussian function to the raw value distribution. The derived fuzzy concept is visualized, where the corresponding raw value distribution is plotted on the background in the same figure. The direction (`direction`) determines whether the fuzzy concept is defined per feature or for the whole raw value matrix. In the latter case, the derived fuzzy concept remains the same in the visualization, while the raw value distribution changes according to user selection.

Two types of membership functions are currently implemented. For one trapezoidal membership function, the four x-coordinates of the trapezoid are required as function parameter, while the Gaussian membership function requires parameter $\mu$ and $\sigma$. These can be acquired either from the mean and standard deviation, or from fitting a Gaussian function to the underlying raw value distribution. No Gaussian function will be fitted for the whole matrix. Moreover, for the $\sigma$ of a Gaussian membership function, the function parameter is displayed as a factor to be multiplied with $\sigma$.

#### Fuzzy Concepts from Constraints

By selecting the number of desired fuzzy concepts and clicking `Estimate` to proceed, a default fuzzy concept with trapezoidal membership functions is derived. In this case, all fuzzy sets are expected to have the same percentage of raw values with their highest fuzzy values there. The visualization is generated from the percentiles according to the percent inputs. The raw values are updated on change of the corresponding percents and the selected feature for visualization. Alternatively, the raw values corresponding to the percents can be manually specified and transferred into percents based on the current raw value distribution by clicking `Value to percent`.

For each fuzzy set, the name (fuzzy variable), type of membership function and the color in the visualization can be customized.

#### Fuzzy Concepts from Fitting (Default Fuzzification)

A Gaussian function is fitted to the raw value distribution of each feature in the raw value matrix, while the function parameters $\mu$ and $\sigma$ of the whole raw value matrix are approximated from the mean and standard deviation directly. It is also optional for the features to have mean and standard deviation directly as the fitted $\mu$ and $\sigma$ by setting band width factor to 0.

If the parameters are to be fitted, it is calculated on the density curve estimated from a given band width factor after clicking `Estaimate`. Then a fuzzy concept symmetric to $\mu$ is defined based on the fitted Gaussian parameters, with the fitted Gaussian membership function in the middle and the same number of trapezoidal membership functions on both sides as specified by the user. The fitted Gaussian function lies in the middle, while all other derived membership functions are trapezoids. The x-coordinates of the trapezoids and $\mu$ of the fitted Gaussian function is displayed as z-scores, namely $\frac{x-\mu}{\sigma}$.

In comparison to the other derivation method, it is not allowed to change the type of membership functions here.

### Output Files and Formats

Two output files can be generated. `concepts_detailed.json` contains all fuzzy concepts with concrete values, while `concepts_constraints.json` consists only of the (derived) constraints. Each fuzzy concept, regardless of constraints or concrete values, are generally a dictionary in Python. These fuzzy concepts for download can either be derived from constraints or from fitting. Additionally, the name of the default fuzzy concept, i.e. that defined for the whole raw value distribution, can be customized and checked if the input already exists in the raw value matrix as row names.

#### Concrete Fuzzy Concept

- `number_fuzzy_sets`: Number of fuzzy sets, excluding labeling fuzzy sets for specific values or noise.

- `label_values`: List of specific values that should be excluded from fuzzy concept definition. $\pm$ $\infty$ and NaN are stored as strings.

- `MIN-NOISE`: Cutoff for the smallest raw value allowed for fuzzy concept definition. It is calculated from the maximum of the constant cutoff and the percentile cutoff.

- `MAX-NOISE`: Cutoff for the largest raw value allowed for fuzzy concept definition. It is calculated from the minimum of the constant cutoff and the percentile cutoff.

- Fuzzy variable: List consisting of a sublist of function parameters, type of membership function (`trapezoidal` or `Gaussian`), color of the membership funcion for plotting and the expected percentage of raw values, which have their highest membership in this fuzzy set. It is defined for each fuzzy set.

#### Constrainted Fuzzy Concept

- `value_type`: Type of the constraints (`fixed` or `proportion` or `z-score`).

- `number_of_fuzzy_sets`: Number of fuzzy sets, excluding labeling fuzzy sets for specific values or noise.

- `label_values`: List of specific values that should be excluded from fuzzy concept definition. $\pm$ $\infty$ and NaN are stored as strings.

- `fit_Gaussian_curve`: Whether the required $\mu$ and $\sigma$ are approximated from the mean and standard deviation, or derived from fitting of a Gaussian function.

- `use_scipy_optimization`: Whether to use SciPy optimization functions to approximate $\sigma$.

- `band_width_factor`: Factor for band width in density estimation and Gaussian funciton fitting.

- Fuzzy variable: List consisting of a sublist of constraints for function parameters, type of membership function (`trapezoidal` or `Gaussian`), color of the membership funcion for plotting and the expected percentage of raw values, which have their highest membership in this fuzzy set. It is defined for each fuzzy set.

## Fuzzy Concepts Visualization and Comparison

A raw value matrix and a set of concrete fuzzy concepts are either uploaded by the user, or taken directly from the previous section. Only `.tsv` files are accepted for the raw value matrix and the fuzzy concepts should be given in `.json` format, which contain the same content as the derived ones. Both inputs are not required to match each other, nor should they contain any overlapping features.

One feature is selected for the fuzzy concept and the raw value distribution in the background, respectively. The percentage of raw values with their highest fuzzy values in each fuzzy set, excluding the additional ones for specific values and noise, is calculated as the observed percentage. This is then compared to the expected percentage in these fuzzy sets, as has been stored in the input fuzzy concept. The deviation, namely difference between observed and expected percentages, is calculated and the absolute value is compared to a specified deviation cutoff. If the absolute deviation is lower, then the underlying raw value distribution meets the expectation of the defined fuzzy concept in this region. The expected and observed percentage, as well as the deviation, are visualized in the categorial heatmap.
