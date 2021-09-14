# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
from webapp import app

data_num = 10
demo_z = np.random.randint(0, 10, (data_num, 2))
color = np.arange(1, data_num+1)
df_demo = pd.DataFrame({
    "x": demo_z[:, 0],
    "y": demo_z[:, 1],
    "c": color,
    "page_title": ["af","gd","gaj","fdfd","olo","maaa","fd","qrer","ddd","lkl"]
})
# fig = px.scatter(x=demo_data[:, 0], y=demo_data[:, 1],
#                  width=800, height=800, color=color)
fig = px.scatter(df_demo, x="x", y="y",
                 width=800, height=800, color="c", size="c",
                 hover_name="page_title")

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 20, 20, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])
