import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression


class BetaFeatureExtractor(BaseEstimator):

    def fit(self, X, y):
        return self

    def transform(self, X):
        return compute_rolling_std(X, 'Beta', center=True)


class Classifier(BaseEstimator):

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        y_pred = self.model.predict_proba(X)
        return y_pred


def get_estimator():

    beta_feature_extractor = BetaFeatureExtractor()
    standard_scaler = StandardScaler()

    classifier = Classifier()

    pipe = make_pipeline(beta_feature_extractor, standard_scaler, classifier)
    return pipe


def compute_rolling_std(X_df, feature, center=False):
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
    res_df = pd.DataFrame({})
    # name = '_'.join([feature, time_window, 'std'])

    seuil = 50
    var_s = X_df[feature].map(lambda x: min(x, seuil))

    var_s = var_s.rolling('30 min', center=center).mean()
    var_small_s = var_s.rolling('4h', center=center).mean()
    var_long_s = var_s.rolling('4 d', center=center).mean()

    res_df['base'] = var_s
    res_df['small'] = var_small_s
    res_df['long'] = var_long_s

    for name in res_df.columns:
        res_df[name] = res_df[name].ffill().bfill()
        res_df[name] = res_df[name].astype(X_df[feature].dtype)
  
    return res_df
