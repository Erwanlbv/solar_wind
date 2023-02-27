import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, classification_report
from problem import turn_prediction_to_event_list, overlap_with_list, find


def plot_event(start, end, data, labels, delta=36, figsize=(10, 60)):
    """
    Plot data starting at start, ending at end.
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    subset = data[(start - pd.Timedelta(hours=delta)):(end + pd.Timedelta(hours=delta))]
    label_subset = labels.loc[subset.index]

    n_data_cols = data.shape[1]

    fig, axes = plt.subplots(nrows=n_data_cols, ncols=1, figsize=figsize, sharex=True)
    
    if n_data_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, data.columns):

        l = 0.1
        low = np.ma.masked_where(label_subset > l, subset[col])
        high = np.ma.masked_where(label_subset < l, subset[col])
        # print('Max of the subset for ', col, subset[col].max())
        
        ax.plot(subset.index, low)
        ax.plot(subset.index, high, color='orange')
        ax.set_ylabel(col)


    # add vertical lines
    for ax in axes:
        ax.axvline(start, color='k')
        ax.axvline(end, color='k')
        ax.xaxis.grid(True, which="minor")

    return fig, axes
    


def multiple_plots(data, labels, n_events, events, delta=36, random_state=1, figsize=(10, 60)):
    """
    Plot multiples events using the plot_event function.
    Each event are randomly taken in the events variable
    Random state is for plotting the same events for different calls
    """

    rng = np.random.RandomState(random_state)

    for i in rng.randint(0, len(events), n_events):
        plot_event(events[i].begin, events[i].end, data, labels, delta=delta, figsize=figsize)


def display_timeline(data, labels):
    """
    Display all data masked by the value of the labels in labels
    data and labels must start and end at the same time
    """
    l = 0.1
    low = np.ma.masked_where(labels > l, data)
    high = np.ma.masked_where(labels < l, data)
    # print('Max of the subset for ', col, subset[col].max())

    _, ax = plt.subplots(figsize=(17, 3))
    ax.set_ylabel('Valeurs')

    ax.plot(data.index, low)
    ax.plot(data.index, high, color='orange')
    ax.axhline(y=data.mean(), color='g', label='Valeur moyenne')

    ax.legend()

def show_densities(data_with_labels, thresh_data_with_labels=None):
    """
    Display densities of the data according to the labels
    data_with_label : Dataframe ['var_name', 'label']

    If not None :
    thresh_data_with_labels : Dataframe with expected format ['var_name', 'label']
    """

    var_name = data_with_labels.columns[0]
    for density in [True, False]:
        _, ax = plt.subplots(ncols=2, figsize=(7, 2))

        ax[0].set_title("Densité : " + str(density) + " (sans seuillage)", fontsize=7)
        data_with_labels.groupby(['label'])[var_name].plot(kind='hist', alpha=0.7, ax=ax[0], legend=True, density=density, logx=True)
        
        if thresh_data_with_labels is not None:
            ax[1].set_title("Densité :" + str(density)+ " (avec seuillage)", fontsize=7)
            thresh_data_with_labels.groupby(['label'])[var_name].plot(kind='hist', alpha=0.7, ax=ax[1], legend=True, density=density, logx=True)

        plt.tight_layout()


def precision(y_true, y_pred):
    event_true = turn_prediction_to_event_list(y_true)
    event_pred = turn_prediction_to_event_list(y_pred)
    FP = [x for x in event_pred
          if max(overlap_with_list(x, event_true, percent=True)) < 0.5]
    if len(event_pred):
        score = 1-len(FP)/len(event_pred)
    else:
        # no predictions -> precision not defined, but setting to 0
        score = 0
    return score


def recall(y_true, y_pred):
    event_true = turn_prediction_to_event_list(y_true)
    event_pred = turn_prediction_to_event_list(y_pred)
    if not event_pred:
        return 0.
    FN = 0
    for event in event_true:
        corresponding = find(event, event_pred, 0.5, 'best')
        if corresponding is None:
            FN += 1
    score = 1 - FN / len(event_true)
    return score

def display_res(data_test, labels_test, smooth=False, models=[]):
    """
    For each model in models, display all the metrics used to defined the leaderboard.
    Smooth=False is for quantile rolling
    Smooth=True is for "n-hours" rolling
    """

    for model in models:
        y_pred = model.predict_proba(data_test)
        s_y_pred = pd.Series(y_pred[:, 1], index=labels_test.index)

        if not smooth:
            s_y_pred = s_y_pred.rolling('18 h', center=True).mean()
        else :
            s_y_pred = s_y_pred.rolling('12 h', min_periods=0, center=True).quantile(0.90)

        y_pred[:, 1] = s_y_pred.to_numpy()

        # Précision point par point
        loss = log_loss(labels_test, s_y_pred)
        print("Loss :", loss)
        print(classification_report(labels_test, y_pred.argmax(axis=1)))

        # Précision sur les évènements

        print("ev_prec", precision(labels_test, s_y_pred))
        print('ev_rec', recall(labels_test, s_y_pred))
        print('-------------')