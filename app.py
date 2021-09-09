# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

Z = np.random.normal(scale=1, size=(100, 2))
fig = px.scatter(
    x=Z[:, 0],
    y=Z[:, 1],
    range_x=[-3, +3],
    range_y=[-3, +3],
    width=800,
    height=800
)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

import argparse
parser = argparse.ArgumentParser()
args, unknown = parser.parse_known_args()

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
