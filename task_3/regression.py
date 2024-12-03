from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots


df = pd.read_csv('/home/ilya/Downloads/abalone/abalone.csv', delimiter=',')

mapping = {
    'sex': {'M': 0, 'F': 1, 'I': 2}
}

for col, mapping_dict in mapping.items():
    df[col] = df[col].map(mapping_dict)

X = df.drop('rings', axis=1)
Y = df['rings']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_regressior = LinearRegression()
linear_regressior.fit(X_train, Y_train)

y_pred_linear = linear_regressior.predict(X_test)

mse_linear = mean_squared_error(Y_test, y_pred_linear)
r2_linear = r2_score(Y_test, y_pred_linear)
print(f'Лучший MSE (Linear): {mse_linear}') # Ошибка: 0,05; Точность модели: 0,95.
print(f'Лучший R^2 (Linear): {r2_linear}') # Точность модели: 0,52; Ошибка: 0,48.



ridge_regressor = Ridge()
param_grid_ridge = {'alpha': np.logspace(-3, 3, 7)}

grid_search_ridge = GridSearchCV(
    ridge_regressor,
    param_grid_ridge,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid_search_ridge.fit(X_train, Y_train)

print('Лучшие параметры гребневой регрессии: ', grid_search_ridge.best_params_)
print('Лучшая оценка на кросс-валидации (MSE) (Ridge):', grid_search_ridge.best_score_)

best_ridge_regressor = grid_search_ridge.best_estimator_
y_pred_ridge = best_ridge_regressor.predict(X_test)

mse_ridge = mean_squared_error(Y_test, y_pred_ridge)
r2_ridge = r2_score(Y_test, y_pred_ridge)
print(f'Лучший MSE (Ridge): {mse_ridge}') # Ошибка: 0,05; Точность модели: 0,95.
print(f'Лучший R^2 (Ridge): {r2_ridge}') # Точность модели: 0,52; Ошибка: 0,48.



lasso_regressor = Lasso()
param_grid_lasso = {'alpha': np.logspace(-3, 3, 7)}

grid_search_lasso = GridSearchCV(
    lasso_regressor,
    param_grid_lasso,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid_search_lasso.fit(X_train, Y_train)

print('Лучшие параметры лассо-регрессии: ', grid_search_lasso.best_params_)
print('Лучшая оценка на кросс-валидации (MSE) (Lasso):', grid_search_lasso.best_score_)

best_lasso_regressor = grid_search_lasso.best_estimator_
y_pred_lasso = best_lasso_regressor.predict(X_test)

mse_lasso = mean_squared_error(Y_test, y_pred_lasso)
r2_lasso = r2_score(Y_test, y_pred_lasso)
print(f'Лучший MSE (Lasso): {mse_lasso}') # Ошибка: 0,05; Точность модели: 0,95.
print(f'Лучший R^2 (Lasso): {r2_lasso}') # Точность модели: 0,53; Ошибка: 0,47.



results = pd.DataFrame(
    {
        'Model': ['Linear Regression', 'Ridge Regression', 'Lasso Regression'],
        'MSE': [mse_linear, mse_ridge, mse_lasso],
        'R^2': [r2_linear, r2_ridge, r2_lasso]
    }
)



fig = make_subplots(rows=1, cols=2, subplot_titles=('Сравнение MSE моделей', 'Сравнение R^2 моделей'))

fig.add_trace(
    go.Bar(x=results['Model'], y=results['MSE'], marker_color=['blue', 'orange', 'green']),
    row=1, col=1
)

fig.add_trace(
    go.Bar(x=results['Model'], y=results['R^2'], marker_color=['blue', 'orange', 'green']),
    row=1, col=2
)

fig.update_xaxes(title_text="Модель", row=1, col=[1, 2])
fig.update_yaxes(title_text="Mean Squared Error", row=1, col=1)
fig.update_yaxes(title_text="R^2 Score", row=1, col=2)
fig.update_layout(height=600, width=1200)

fig.show()



if r2_linear > r2_lasso and r2_linear > r2_ridge:
    print('Линейная регрессия показала наилучшие результаты')
elif r2_ridge > r2_lasso and r2_ridge > r2_linear:
    print('Гребневая регрессия показала наилучшие результаты')
else:
    print('Лассо-регрессия показала наилучшие результаты')