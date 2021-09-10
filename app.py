# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

N = 100
Z = np.random.normal(scale=1, size=(N, 2))
zeros = np.zeros(N)
fig = go.Figure(
    data=[go.Scatter(x=zeros, y=zeros)],
    layout=go.Layout(
       xaxis=dict(range=[-3, +3], autorange=False),
       yaxis=dict(range=[-3, +3], autorange=False),
       width=800,
       height=800,
       updatemenus=[dict(
           type="buttons",
           buttons=[dict(
               label="Play",
               method="animate",
               args=[None]
           )]
       )]
    ),
    frames=[
        go.Frame(data=[go.Scatter(x=zeros, y=zeros, mode="markers", marker=dict(size=10))]),
        go.Frame(data=[go.Scatter(x=Z[:, 0], y=Z[:, 1], mode="markers", marker=dict(size=10))]),
    ]
)

app.layout = html.Div(children=[
    html.H1(id='title', children='Hello Dash'),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
])


import argparse
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
