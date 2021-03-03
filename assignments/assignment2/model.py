#!/usr/bin/env python3


import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics


def _seq_forward_selection(dataf, ycolumn, xcolumns, test_size, method, seed=99):
    """Sequential forward selection of available features based on holdout performance"""

    selected_full, selected_xcolumns = [], []
    while len(xcolumns) != len(selected_xcolumns):
        output_xcolumns, output_scores = [], []
        for f in xcolumns:
            if f not in selected_xcolumns:
                X = dataf[selected_xcolumns + [f]].values
                y = dataf[ycolumn].values
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                _score = _model(X_train, X_test, y_train, y_test, method, seed=seed)
                output_xcolumns.append(f)
                output_scores.append(_score)
        best_next_score = max(output_scores)
        best_index = output_scores.index(best_next_score)
        best_next_f = output_xcolumns[best_index]
        selected_full.append([len(selected_xcolumns + [best_next_f]), selected_xcolumns + [best_next_f], best_next_score])
        selected_xcolumns.append(best_next_f)

    return selected_full


def _model(X_train, X_test, y_train, y_test, method, seed=99):
    """Fit model and predict on test set"""

    if method == 'tree':
        classifier = DecisionTreeClassifier(random_state=seed)
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)

    if method == 'svm':
        classifier = LinearSVC(max_iter=100000, dual=True, random_state=seed)
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)

    if method == 'naivebayes':
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)

    if method == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=8)
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)

    if method == 'lg':
        classifier = LogisticRegression(random_state=seed, solver='sag', multi_class='ovr', max_iter=100000)
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)

    if method == 'randomforest':
        classifier = RandomForestClassifier(n_estimators=100, random_state=seed)
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)

    if method == 'boosting':
        params = {
            'n_estimators': 500,
            'max_depth': 7,
            'random_state': 10,
            'min_samples_split': 5,
            'learning_rate': 0.01,
        }
        classifier = GradientBoostingClassifier(**params)
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)

    return round(metrics.accuracy_score(y_test, y_predict), 7)

def execute_models(models):
    """Loop model execution on forward feature selection"""

    model_comparison = {}
    outputdfs = []
    for m in models:
        try:
            print(f'==> running {m}')
            output = _seq_forward_selection(df, ycolumn, xcolumns, test_size, method=m, seed=seed)
            dfoutput = pd.DataFrame(output, columns=['n_features', 'features', 'score'])
            dfoutput.index = dfoutput['n_features']
            outputdfs.append(dfoutput)
            max_score = dfoutput['score'].max()
            optimal_f_selection = dfoutput.loc[lambda x: x['score']==x['score'].max()].loc[lambda x: x['n_features']==x['n_features'].min()]['features'].values[0]
            model_comparison[m] = {
                'score': max_score,
                'n_features': len(optimal_f_selection),
                'features': optimal_f_selection
            }
        except:
            print(f'error occured during {m} model')
            pass

    return model_comparison, outputdfs

def tree_fit_score(X_train, y_train, X_test=None, y_test=None, n_nodes=None, use_testset=True, method=None):
    """Fit model on training data, and calculate accuracy based on training or test data"""

    if method == 'tree':
        classifier = DecisionTreeClassifier(max_depth=n_nodes, random_state=99)
    elif method == 'randomforest':
        classifier = RandomForestClassifier(n_estimators=100, max_depth=n_nodes, random_state=seed)
    classifier.fit(X_train, y_train)
    if use_testset == False:
        y_predict = classifier.predict(X_train)
        score = metrics.accuracy_score(y_train, y_predict)
    else:
        y_predict = classifier.predict(X_test)
        score = metrics.accuracy_score(y_test, y_predict)
    return score

def n_node_increase_scores(X_train, y_train, X_test, y_test, depth_range, method):
    """Get model accuracies for the given n nodes range"""

    rows = []
    for n in depth_range:
        _score_full = tree_fit_score(X_train, y_train, n_nodes=n, use_testset=False, method=method)
        _score_testtrain = tree_fit_score(X_train, y_train, X_test, y_test, n_nodes=n, method=method)
        rows.append([n, _score_full, _score_testtrain])

    return rows

def plot_fitting_graph(rows):
    """Plot n nodes accuracies based on both training and test data"""

    plt.plot([y[0] for y in rows], [y[1] for y in rows], label='train_performance', color='k')
    plt.plot([y[0] for y in rows], [y[2] for y in rows], label='test_performance', color='k', linestyle='dashed')
    plt.xticks(np.arange(1, max([y[0] for y in rows])+1, step=1))
    plt.title('Train & holdout performance vs tree depth')
    plt.legend()


def main():

    df = pd.read_csv('data/prepped/spotify_featured.csv')

    ycolumn = 'target'
    xcolumns = [c for c in df.columns if c != ycolumn]
    seed = 99
    test_size = 0.2
    models = [
        'tree',
        'svm',
        'lg',
        'naivebayes',
        'knn',
        'randomforest',
        'boosting'
    ]

    model_comparison, outputdfs = execute_models(models)

    nowstamp = datetime.now().strftime('%Y%m%d%H%M%S')
    with open(f'data/output/model_outputs_{nowstamp}.json', 'w') as stream:
        json.dump(model_comparison, stream)

if __name__ == '__main__':
    main()
