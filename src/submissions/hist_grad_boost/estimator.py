from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier


class FeatureExtractor(BaseEstimator):

    def fit(self, X, y):
        return self

    def transform(self, X):
        return compute_rolling_std(X, 'Beta', '2h', center=True)


class Classifier(BaseEstimator):

    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            max_iter=1000,
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict_proba(X)
        return y_pred


def get_estimator():

    beta_feature_extractor = FeatureExtractor()
    standard_scaler = StandardScaler()

    classifier = Classifier()

    pipe = make_pipeline(beta_feature_extractor, standard_scaler, classifier)
    return pipe


def compute_rolling_std(X_df, feature, time_window, center=False):
    """
    For a given dataframe, compute the standard deviation over
    a defined period of time (time_window) of a defined feature

    Parameters
    ----------
    X : dataframe
    feature : str
        feature in the dataframe we wish to compute the rolling std from
    time_window : str
        string that defines the length of the time window passed to `rolling`
    center : bool
        boolean to indicate if the point of the dataframe considered is
        center or end of the window
    """
    name = '_'.join([feature, time_window, 'std'])
    X_df[name] = X_df[feature].rolling(time_window, center=center).std()
    X_df[name] = X_df[name].ffill().bfill()
    X_df[name] = X_df[name].astype(X_df[feature].dtype)
  
    return X_df