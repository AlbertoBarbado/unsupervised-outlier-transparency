# Rule extraction in unsupervised outlier detection for XAI

OneClass SVM is a popular method to perform unsupervised outlier detection on the features that compose a dataset. However, this method is generally considered a "black box"
due to the fact that it's difficult to justify in an intuitive and simple way why the decision frontier is classifying the data points in the different categories. This problem 
appears in supervised classification tasks and even more in unsupervised learning. To be able to obtain intuitive explanations this library offers a method to infer rules that
justify why a point is labeled as an outlier based on [1], [2].

The library performs the outlier analysis using scikit-learn [3].

This method offers an algorithmic transparency method for variables of any kind (they can be numerical or categorical)


## Getting Started

These instructions will explain how to use the library and be able to obtain the results indicated.

### Prerequisites

The dependencies included in the file requirements.txt are needed to be able to use the library directly or to execute the example (Example.ipynb)

```
$ pip install -r requirements.txt 
```

### Usage

The main function that acts as a wrapper over the class OneClassSVM from scikit-learn is *ocsvm_rule_extractor*. This function uses as parameters the following ones:
- dataset: pandas dataframe containing the dataset
- numerical_cols: list of the columns that contain numerical variables
- categorical_cols: list of the columns that contain categorical (non ordinal) variables
- dct_params: dictionary that contain the parameters used in the generic model creation of OneClassSVM. See [3] for more info.

NOTE: categorical columns should be onehot encoded.

```
>>> ocsvm_rule_extractor(dataset, numerical_cols, categorical_cols, dct_params)
```

The function then returns both the model trained and a dataframe with the rules infered. These rules look like the following example:

```
NOT anomaly...
Rule Nº 1: IF sex = 0 AND school = 0 AND studytime <= 4 AND G3 <= 15 AND studytime >= 1 AND G3 >= 8 
Rule Nº 2: IF sex = 0 AND school = 1 AND studytime <= 2 AND G3 <= 0 AND studytime >= 2 AND G3 >= 0 
Rule Nº 3: IF sex = 1 AND school = 0 AND studytime <= 4 AND G3 <= 13 AND studytime >= 2 AND G3 >= 8 
```

These rules indicates the limit values that justify why a data point should not be considered an anomaly, so any other case would
mean that the data is anomalous.
 

## More Information

More information regarding the theory behind this method can be found in the file unsupervised_outlier_algorithmic_transparency.pdf


## Authors

* **Alberto Barbado González** - (https://github.com/AlbertoBarbado/)

* Barbado González, Alberto. 2019. *Rule extraction in unsupervised outlier detection for algorithmic transparency*. Madrid. Telefónica. https://github.com/AlbertoBarbado/unsupervised-outlier-transparency [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3387762.svg)](https://doi.org/10.5281/zenodo.3387762)

To cite it (BibTeX):

@article{ abg2019,
       author = "Barbado, Alberto",
       title = "Rule extraction in unsupervised outlier detection for algorithmic transparency",
       year = "2019",
       address = "Spain",
       doi= "10.5281/zenodo.3387762"}

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details


## References

* [1]: H. Núñez, C. Angulo, and A. Català. Rule extraction from support vector machines. In European Symposium on Artificial Neural Networks (ESANN), pages 107–112, 2002.
* [2]: D. Martens, J. Huysmans, R. Setiono, J. Vanthienen, and B. Baesens. Rule Extraction from Support Vector Machines: An Overview of Issues and Application in Credit Scoring. 2008.
* [3]: scikit-learn library for OneClassSVM: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
