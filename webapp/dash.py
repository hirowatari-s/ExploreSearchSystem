# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from webapp import app
from dev.Grad_norm import Grad_Norm
from fit import SOM


def make_figure(keyword):
    # Load data
    csv_df = pd.read_csv(keyword+".csv")
    labels = csv_df['site_name']
    X = np.load("data/tmp/" + keyword + ".npy")

    # Learn model
    nb_epoch = 50
    resolution = 100
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

    # Make U-Matrix
    umatrix = Grad_Norm(
        X=X,
        Z=Z,
        sigma=som.history['sigma'][-1],
        labels=labels, resolution=100, title_text="dammy"
    )
    U_matrix, resolution, _ = umatrix.calc_umatrix()

    # Build figure
    fig = go.Figure(
        layout=go.Layout(
            xaxis={
                'range': [Z[:, 0].min() - 0.1, Z[:, 0].max() + 0.1],
                'visible': False
            },
            yaxis={
                'range': [Z[:, 1].min() - 0.1, Z[:, 1].max() + 0.1],
                'visible': False,
                'scaleanchor': 'x',
                'scaleratio': 1.0
            },
            showlegend=False,
            # **self.params_figure_layout
        )
    )
    fig.add_trace(
        go.Contour(
            x=np.linspace(-1, 1, resolution),
            y=np.linspace(-1, 1, resolution),
            z=U_matrix.reshape(resolution, resolution),
            name='contour',
            colorscale="viridis",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=Z[:, 0],
            y=Z[:, 1],
            mode="markers",
            name='lv',
            marker=dict(
                size=13,
                # line=dict(
                #     width=1.5,
                #     color="white"
                # ),
            ),
            text=labels)
        )
    return fig
# fig.update_layout(hovermode="lv")
# fig.update_layout(legend_title_text='Trend')


@app.callback(
    Output('example-graph', 'figure'),
    Input('dropdown', 'value'))
def load_learning(value):
    return make_figure(value)


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
        State('link', 'children'),
        State('link', 'href'),
        State('link', 'target'),
        State('card-text', 'children'),
    ])
def update_title(hoverData, keyword, prev_linktext, prev_url, prev_target, prev_page_title):
    print(hoverData)
    if hoverData:
        if not ("points" in hoverData and "pointIndex" in hoverData["points"][0]):
            link_title = prev_linktext
            url = prev_url
            target = prev_target
            page_title = prev_page_title
        else:
            csv_df = pd.read_csv(keyword+".csv")
            index = hoverData['points'][0]['pointIndex']
            link_title = "サイトへ Go"
            labels = csv_df['site_name']
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
