import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def load_wineing_data():
    return pd.read_csv('winequality-red.csv')


if __name__ == "__main__":
    wineing = load_wineing_data()

    # 데이터 확인
    # print(wineing.info())
    # wineing.hist()
    # plt.show()

    # 트레인세트, 테스트세트 나누기
    train_set, test_set = train_test_split(wineing, test_size=0.2, random_state=42)
    # print("train:", train_set.head())
    # print("test:", test_set.head())

    cor_matrix = train_set.corr()
    print(cor_matrix["quality"].sort_values(ascending=False))

    train = train_set.drop("quality", axis=1)  # x_train
    train_labels = train_set["quality"].copy()  # y_train
    test = test_set.drop("quality", axis=1)  # x_test
    test_labels = test_set["quality"].copy()  # y_test


    # 모델 선택 및 트레이닝
    print('------------train_set로 학습,예측---------------')
    model = RandomForestClassifier(random_state=42)
    model.fit(train, train_labels)

    y_predict = model.predict(train)
    # print("정답:", test_labels)
    # print("예측:", y_predict)

    accuracy = accuracy_score(train_labels, y_predict)
    print("정확도: {0} %".format(accuracy * 100))

    print('----------------Random Grid-----------')

    # param_grid = {
    #             'max_depth': [10, 15, 20, 25, None],
    #             'criterion': ['gini', 'entropy'],
    #             'n_estimators': [45, 50, 55]
    # } #1차 65.3125 % {'criterion': 'gini', 'max_depth': 25, 'n_estimators': 55}

    # param_grid = {
    #     'max_depth': [21,22,23,24],
    #     'criterion': ['gini'],
    #     'n_estimators': [64,65, 66]
    # } #2차 65.9375 % {'criterion': 'gini', 'max_depth': 22, 'n_estimators': 64}

    param_grid = {
        'max_depth': [22],
        'criterion': ['gini'],
        'n_estimators': [64],
    } #3차 최종  65.9375 % {'criterion': 'gini', 'max_depth': 22, 'n_estimators': 64}

    grid_search = GridSearchCV(model, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               return_train_score=True, n_jobs=-1)

    grid_search.fit(train, train_labels)
    # print(grid_search.best_params_)
    # print(grid_search.best_estimator_)
    forest_best = grid_search.best_estimator_
    forest_best_pre = forest_best.predict(train)

    accuracy = accuracy_score(train_labels, forest_best_pre)
    print("정확도: {0} %".format(accuracy * 100))










