# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.CardBody import CardBody
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
from webapp import app
import pickle
from fit import SOM

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
                #  width=800, height=800,
                 color="c",
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
link_card = dbc.Card([
    # dbc.CardImg(
    #     id="card-img",
    #     src="https://1.bp.blogspot.com/-9DCMH4MtPgw/UaVWN2aRpRI/AAAAAAAAUE4/jRRLie86hYI/s800/columbus.png",
    #     top=True,
    #     className="img-fluid img-thumbnail",
    #     style="height: 50px; width: 50px;",
    # ),
    html.P("", id="card-text", className="h4"),
    html.A(
        id='link',
        href='#',
        children="マウスを当ててみよう",
        target="_self",
        className="btn btn-outline-primary btn-lg",
    )
], id="link-card")


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
        ), md=8),
        dbc.Col(link_card, md=4)
    ], align="center"),
    dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'ファッション', 'value': 'ファッション'},
            {'label': '機械学習', 'value': '機械学習'},
            {'label': 'あっぷる', 'value': 'あっぷる'}
        ],
        value='ファッション'
    ),
])

@app.callback([
        Output('link', 'children'),
        Output('link', 'href'),
        Output('link', 'target'),
        Output('card-text', 'children'),
        # Output('card-img', 'src'),
    ],
    Input('example-graph', 'hoverData'))

def update_title(hoverData):
    if hoverData:
        index = hoverData['points'][0]['pointIndex']
        print(csv_df['URL'][index][12:-2])

        link_title = "サイトへ Go"
        url = csv_df['URL'][index][12:-2]
        target = "_blank"
        page_title = labels[index]
        # favicon_url = f"https://s2.googleusercontent.com/s2/favicons?domain_url={url}"
    else:
        link_title = "マウスを当ててみよう"
        url = "#"
        target = "_self"
        page_title = ""
        # favicon_url = "https://1.bp.blogspot.com/-9DCMH4MtPgw/UaVWN2aRpRI/AAAAAAAAUE4/jRRLie86hYI/s800/columbus.png"
    return link_title, url, target, page_title #, favicon_url

@app.callback(
    Output('example-graph', 'figure'),
    Input('dropdown', 'value'))

def load_learning(value):
    global csv_df
    print("関数呼び込んでます")
    keyword=value
    csv_df = pd.read_csv(keyword+".csv")
    labels = np.load("data/tmp/"+keyword+"_label.npy")

    feature_file = 'data/tmp/'+keyword+'.npy'
    X = np.load(feature_file)

    nb_epoch = 50
    resolution = 10
    sigma_max = 2.2
    sigma_min = 0.3
    tau = 50
    latent_dim = 2
    seed = 1

    np.random.seed(seed)

    som = SOM(X, latent_dim=latent_dim, resolution=resolution, sigma_max=sigma_max, sigma_min=sigma_min, tau=tau,
              init='PCA')
    som.fit(nb_epoch=nb_epoch)
    print("学習終わりました")
    Z = som.history['z'][-1]

    # 検索結果順に色つけるやつ
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
    return fig
