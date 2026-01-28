# Fuzzifier APP

## Interactive Fuzzifier

### Interactive Fuzzy Concept Definition
Given a raw value matrix, fuzzy concepts are derived from the data and interactively adjusted parameters. Two types of membership functions, namely trapezoidal and Gaussian, are allowed.

#### Define fuzzy concepts by fixed parameters
The membership functions are directly defined by the user.

#### Derive fuzzy concepts by percent of width in the value range
The raw value range is splitted into a certain number of subsets (fuzzy sets) by percent in width. Slopes are also required for trapezoidal membership functions, while the parameters of Gaussian membership functions will be derived automatically.

#### Derive fuzzy concepts by percent of samples
Each fuzzy set is expected to have a certain percent of samples, which have their highest membership there. Therefore, cutoffs are defined by the corresponding percentiles of the raw value distribution per feature or per matrix. Slopes are also required for trapezoidal membership functions, while the parameters of Gaussian membership functions will be derived automatically.

### Derive fuzzy concepts by mode estimation
TODO: to be replaced by / updated to default fuzzification

### Fuzzification
Fuzzy values are calculated using the defined fuzzy concepts on the same raw value matrix.

TODO: to be removed, replaced by command line fuzzifier that allows for usage of fuzzy concepts that were defined on different raw value matrices.

### Evaluation
TODO: to be removed

## Command Line Fuzzifier
The command line fuzzifier has basically the same functions as the interactive fuzzifier. Additionally, it is able to perform default fuzzification based on fitted normal distributions and allows for using fuzzy concepts that were defined on a different raw value matrix in the fuzzification.

