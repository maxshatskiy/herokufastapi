# Model Card
The model for classification if salary is >50K or <=50K based on the personal data.

## Model Details
The model is build using logistic regression. The model was developed in October, 2022.

## Intended Use
The primary intended use is to predict if person salary above or below 50K based on the provided information.

## Training Data
Publicly available Census Bureau data were used to build this model. Additional information on dataset can be found
here https://archive.ics.uci.edu/ml/datasets/census+income. 80% of the data were used for training purpose.
Categorical variables were preprocessed using OneHotEncoding.

## Evaluation Data
20% of the data were used for evaluation purpose.

## Metrics
The following metrics are reported for the model: F1-beta score, precision, and recall.
The performance was as following:
precision: 0.714
recall: 0.264
F1-beta: 0.385


## Caveats and Recommendations
This model should additionally be tested for racial and gender biases.
