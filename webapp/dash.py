# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import numpy as np
from webapp import app
from fit import SOM


def make_figure(keyword):
    csv_df = pd.read_csv(keyword+".csv")
    labels = csv_df['site_name']
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

    som = SOM(
        X,
        latent_dim=latent_dim,
        resolution=resolution,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        tau=tau,
        init='PCA'
    )
    som.fit(nb_epoch=nb_epoch)
    print("Learning finished.")
    Z = som.history['z'][-1]

    # 検索結果順に色つけるやつ
    color = np.arange(Z[:, 0].shape[0])
    df_demo = pd.DataFrame({
        "x": Z[:, 0],
        "y": Z[:, 1],
        "c": color,
        "page_title": labels,
    })
    fig = px.scatter(
        df_demo,
        x="x",
        y="y",
        color="c",
        hover_name="page_title"
    )
    return fig


@app.callback([
        Output('link', 'children'),
        Output('link', 'href'),
        Output('link', 'target'),
        Output('card-text', 'children'),
        # Output('card-img', 'src'),
    ],
    [
        Input('example-graph', 'hoverData'),
    ],
    [
        State('dropdown', 'value'),
    ])
def update_title(hoverData, keyword):
    # print(hoverData)
    if hoverData:
        csv_df = pd.read_csv(keyword+".csv")
        # print("df:", csv_df)
        # print("labels:", labels)
        index = hoverData['points'][0]['pointIndex']
        # print(csv_df['URL'][index][12:-2])

        link_title = "サイトへ Go"
        labels = csv_df['site_name']
        url = csv_df['URL'][index][12:-2]
        print(csv_df["URL"])
        print(index, url)
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
    return make_figure(value)


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
                figure=make_figure("ファッション"),
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
