import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, ShuffleSplit
from prettytable import PrettyTable
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import VotingRegressor, BaggingClassifier, StackingClassifier, RandomTreesEmbedding
from sklearn.metrics import accuracy_score

# import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots


cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

df['target'] = cancer.target

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
score_line_reg = lin_reg.score(X_test, y_test)
cross_result_lin_reg = cross_val_score(lin_reg, X, y, cv=cv)
avr_cross_result_lin_reg = np.average(cross_result_lin_reg)

parametrs = {'n_jobs':[i for i in range(0, 100)]}
# print(lin_reg.get_params().keys())
grid_result_lin_reg = GridSearchCV(lin_reg, parametrs).fit(X_train, y_train).best_score_

print(f'Обычное обучение: {score_line_reg}')
print(f'Кросс-валидация: {avr_cross_result_lin_reg}')
print(f'Решётчатый поиск: {grid_result_lin_reg}')



lin_reg_ridge = Ridge()
lin_reg_ridge.fit(X_train, y_train)
score_lin_reg_ridge = lin_reg_ridge.score(X_test, y_test)

cross_result_lin_reg_ridge = cross_val_score(lin_reg_ridge, X, y, cv=cv)
avr_cross_result_lin_reg_ridge = np.average(cross_result_lin_reg_ridge)

parametrs2 = {'alpha': [i for i in range(0, 100)]}
# print(lin_reg_ridge.get_params().keys())
grid_result_lin_reg_ridge = GridSearchCV(lin_reg, parametrs).fit(X_train, y_train).best_score_

print(f'Обычное обучение: {score_lin_reg_ridge}')
print(f'Кросс-валидация: {avr_cross_result_lin_reg_ridge}')
print(f'Решётчатый поиск: {grid_result_lin_reg_ridge}')



lin_reg_lasso = Lasso()
lin_reg_lasso.fit(X_train, y_train)
score_lin_reg_lasso = lin_reg_lasso.score(X_test, y_test)

cross_result_lin_reg_lasso = cross_val_score(lin_reg_lasso, X, y, cv=cv)
avr_cross_result_lin_reg_lasso = np.average(cross_result_lin_reg_lasso)

parametrs3 = {'alpha':[1, 100]}
# print(lin_reg_lasso.get_params().keys())
lin_reg_lasso.get_params().keys()
grid_result_lin_reg_lasso = GridSearchCV(lin_reg_lasso, parametrs3).fit(X_train, y_train).best_score_

print(f'Обычное обучение: {score_lin_reg_lasso}')
print(f'Кросс-валидация: {avr_cross_result_lin_reg_lasso}')
print(f'Решётчатый поиск: {grid_result_lin_reg_lasso}')



model_mean = np.mean([score_line_reg, score_lin_reg_ridge, score_lin_reg_lasso])



estimators = [('lin_reg', lin_reg), ('lin_reg_ridge', lin_reg_ridge), ('lin_reg_lasso', lin_reg_lasso)]
ensemble = VotingRegressor(estimators)
ensemble.fit(X_train, y_train)
score_voting = ensemble.score(X_test, y_test)



# plt.figure(figsize=(12,6))

# plt.plot(ensemble.predict(X_test))
# plt.plot(lin_reg_lasso.predict(X_test))
# plt.plot(lin_reg.predict(X_test))
# plt.plot(lin_reg_ridge.predict(X_test))

# plt.show()



lin_reg_pred = lin_reg.predict(X_test)
lin_reg_ridge_pred = lin_reg_ridge.predict(X_test)
lin_reg_lasso_pred = lin_reg_lasso.predict(X_test)

fig = make_subplots(rows=1, cols=1, subplot_titles=('Сравнение линейных регрессий'))

fig.add_trace(
    go.Scatter(x=[i for i in range(len(lin_reg_pred))], y=lin_reg_pred, mode='lines', name='Linear Regression'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=[i for i in range(len(lin_reg_ridge_pred))], y=lin_reg_ridge_pred, mode='lines', name='Ridge Regression'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=[i for i in range(len(lin_reg_lasso_pred))], y=lin_reg_lasso_pred, mode='lines', name='Lasso Regression'),
    row=1, col=1
)

fig.update_xaxes(title_text="Индекс", row=1, col=1)
fig.update_yaxes(title_text="Предсказание", row=1, col=1)
fig.update_layout(height=600, width=800, legend=dict(orientation="h", yanchor="bottom", y=1.02))

fig.show()
