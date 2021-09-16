# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
from webapp import app
import pickle

with open("data/tmp/ファッション_SOM.pickle", "rb") as f:
    som = pickle.load(f)


csv_df = pd.read_csv("ファッション.csv")


labels = np.load("data/tmp/ファッション_label.npy")


Z = som.history['z'][-1]
data_num = 10
demo_z = np.random.randint(0, 10, (data_num, 2))
#検索結果順に色つけるやつ
color = Z[:, 0]
df_demo = pd.DataFrame({
    "x": Z[:, 0],
    "y": Z[:, 1],
    "c": color,
    "page_title": labels,
})
fig = px.scatter(df_demo, x="x", y="y",
                 width=800, height=800, color="c",
                 hover_name="page_title"
                 )


#sampleコード
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 20, 20, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = dbc.Container(children=[
    html.H1(id='title', children='Hello Dash'),
    html.Div(children='''
        Dash: A web application framework for Python.
    '''),
    html.Hr(),

    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='example-graph',
                figure=fig
        ), md=12),
    ], align="center"),

    html.A(
        id='link',
        href='#',
        children="ahiahi",
        target="_blank",
        className="btn btn-outline-primary btn-lg",
    )
])

@app.callback([
        Output('link', 'children'),
        Output('link', 'href')
    ],
    Input('example-graph', 'hoverData'))
def update_title(hoverData):
    if hoverData:
        index = hoverData['points'][0]['pointIndex']
        retvalue = labels[index]
        print(csv_df['URL'][index][12:-2])
        url = csv_df['URL'][index][12:-2]
    else:
        retvalue = "ahiahi"
        url = "#"
    return retvalue, url
