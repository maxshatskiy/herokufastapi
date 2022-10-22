from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = LogisticRegression(random_state=0).fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds

def compute_model_metrics_on_cat_feature(preds, y_test, test, cat_feature):

    output_dict = {}
    unique_categories_in_cat_feature = test[cat_feature].unique()
    try:
            for category_in_cat_feature in unique_categories_in_cat_feature:
                y_test_slice = y_test[test[cat_feature] == category_in_cat_feature]
                preds_slice = preds[test[cat_feature] == category_in_cat_feature]
                precision, recall, fbeta = compute_model_metrics(y_test_slice, preds_slice)
                output_dict[category_in_cat_feature] = (precision, recall, fbeta)
            return output_dict
    except Exception as e:
        print(e)

def compute_model_metrics_on_cat_features(X_test, y_test, preds, cat_features=[]):

        metrics_for_cat_features = {}
        for cat_feature in cat_features:
            precision, recall, fbeta = compute_model_metrics_on_cat_feature(X_test, y_test, preds, cat_feature)
            metrics_for_cat_features[cat_feature] = [precision, recall, fbeta]

        return metrics_for_cat_features