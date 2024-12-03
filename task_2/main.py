import dash
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd

import plotly.express as px
import plotly.io as pio

from sklearn.datasets import load_iris


pio.renderers.default = 'browser'

cancer = load_iris()

data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
target = pd.DataFrame(cancer.target, columns=['target'])
df = pd.concat([data, target], axis=1)

fig1 = px.scatter(df, x='sepal width (cm)', y='sepal length (cm)', color='target')
fig2 = px.scatter(df, x='petal length (cm)', y='sepal length (cm)', color='target')
fig3 = px.scatter(df, x='petal width (cm)', y='sepal length (cm)', color='target')
fig4 = px.scatter(df, x='petal length (cm)', y='sepal width (cm)', color='target')
fig5 = px.scatter(df, x='petal width (cm)', y='sepal width (cm)', color='target')

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Hello Dash!'),
    html.Div('Exoplanets chart'),
    dcc.Graph(figure=fig1),
    dcc.Graph(figure=fig2),
    dcc.Graph(figure=fig3),
    dcc.Graph(figure=fig4),
    dcc.Graph(figure=fig5)
])
if __name__ == '__main__':
    app.run_server(debug=True)
