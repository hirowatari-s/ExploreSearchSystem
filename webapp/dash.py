# -*- coding: utf-8 -*-

from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from webapp import app
from Grad_norm import Grad_Norm
from tsom import ManifoldModeling as MM
import pathlib
from fetch_arxiv import fetch_search_result
from make_BoW import make_bow
from sklearn.decomposition import NMF
from scipy.spatial import distance as dist
import pickle


resolution = 10


def prepare_materials(keyword, model_name):
    # Learn model
    nb_epoch = 50
    sigma_max = 2.2
    sigma_min = 0.2
    tau = 50
    latent_dim = 2
    seed = 1

    # Load data
    if pathlib.Path(keyword+".csv").exists():
        print("Data exists")
        csv_df = pd.read_csv(keyword+".csv")
        labels = csv_df['site_name']
        rank = csv_df['ranking']
        X = np.load("data/tmp/" + keyword + ".npy")
    else:
        print("Fetch data to learn")
        csv_df = fetch_search_result(keyword)
        X , labels, _ = make_bow(csv_df)
        rank = np.arange(1, X.shape[0]+1)  # FIXME
        csv_df.to_csv(keyword+".csv")
        feature_file = 'data/tmp/'+keyword+'.npy'
        label_file = 'data/tmp/'+keyword+'_label.npy'
        np.save(feature_file, X)
        np.save(label_file, labels)


    model_save_path = 'data/tmp/'+ keyword +'_'+ model_name +'_history.pickle'
    if pathlib.Path(model_save_path).exists():
        print("Model already learned")
        with open(model_save_path, 'rb') as f:
            history = pickle.load(f)
    else:
        print("Model learning")
        np.random.seed(seed)
        mm = MM(
            X,
            latent_dim=latent_dim,
            resolution=resolution,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            model_name=model_name,
            tau=tau,
            # init='PCA'
        )
        mm.fit(nb_epoch=nb_epoch)
        history = dict(
            Z1=mm.history['z1'][-1],
            Z2=mm.history['z2'][-1],
            Y=mm.history['y'][-1],
            sigma=mm.history['sigma'][-1],
            Zeta=mm.Zeta1,
            resolution=mm.resoluton
        )
        print("Learning finished.")
        with open(model_save_path, 'wb') as f:
            pickle.dump(history, f)
    return csv_df, labels, X, history, rank


def draw_umatrix(fig, X, Z, sigma, u_resolution, labels):
    umatrix = Grad_Norm(
        X=X,
        Z=Z,
        sigma=sigma,
        labels=labels,
        resolution=u_resolution,
        title_text="dammy"
    )
    U_matrix, _, _ = umatrix.calc_umatrix()
    fig.add_trace(
        go.Contour(
            x=np.linspace(-1, 1, u_resolution),
            y=np.linspace(-1, 1, u_resolution),
            z=U_matrix.reshape(u_resolution, u_resolution),
            name='contour',
            colorscale="gnbu",
            hoverinfo='skip',
            showscale=False,
        )
    )
    return fig


def draw_topics(fig, Y, n_components):
    # decomposed by Topic
    model_t3 = NMF(
        n_components=n_components,
        init='random',
        random_state=2,
        max_iter=300,
        solver='cd'
    )
    W = model_t3.fit_transform(Y)

    # For mask and normalization(min:0, max->1)
    mask_std = np.zeros(W.shape)
    mask = np.argmax(W, axis=1)
    for i, max_k in enumerate(mask):
        mask_std[i, max_k] = 1 / np.max(W)
    W_mask_std = W * mask_std
    DEFAULT_PLOTLY_COLORS=[
        'rgb(31, 119, 180)', 'rgb(255, 127, 14)',
        'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
        'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
        'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
        'rgb(188, 189, 34)', 'rgb(23, 190, 207)'
    ]
    alpha = 0.1
    DPC_with_Alpha = [k[:-1]+', '+str(alpha)+k[-1:] for k in DEFAULT_PLOTLY_COLORS]
    for i in range(n_components):
        fig.add_trace(
            go.Contour(
                x=np.linspace(-1, 1, resolution),
                y=np.linspace(-1, 1, resolution),
                z=W_mask_std[:, i].reshape(resolution, resolution),
                name='contour',
                colorscale=[
                [0, "rgba(0, 0, 0,0)"],
                [1.0, DPC_with_Alpha[i]]],
                hoverinfo='skip',
                showscale=False,
            )
        )
    return fig

def draw_ccp(fig, Y, Zeta, resolution, clickedData, viewer_id):
    print('ccp')
    if viewer_id == 'viewer_1':
        # viewer_1 ってことはviewer_2をクリックした．
        y = Y[:, get_bmu(Zeta, clickedData)].reshape(resolution, resolution)
    elif viewer_id == 'viewer_2':
        y = Y[get_bmu(Zeta, clickedData), :].reshape(resolution, resolution)
    fig.add_trace(
        go.Contour(
            x=np.linspace(-1, 1, resolution),
            y=np.linspace(-1, 1, resolution),
            z=y,
            name='contour',
            colorscale="gnbu",
            hoverinfo='skip',
            showscale=False,
        )
    )
    return fig


def get_bmu(Zeta, clickData):
    clicked_point = [[clickData['points'][0]['x'], clickData['points'][0]['y']]] if clickData else [[0, 0]]
    clicked_point = np.array(clicked_point)
    dists = dist.cdist(Zeta, clicked_point)
    unit = np.argmin(dists, axis=0)
    return unit[0]


def draw_scatter(fig, Z, labels, rank):
    fig.add_trace(
        go.Scatter(
            x=Z[:, 0],
            y=Z[:, 1],
            mode="markers",
            name='lv',
            marker=dict(
                size=rank[::-1],
                sizemode='area',
                sizeref=2. * max(rank) / (40. ** 2),
                sizemin=4,
            ),
            text=labels,
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.75)",
            ),
        )
    )
    return fig


def make_figure(keyword, model_name, viewer_name="U_matrix", viewer_id=None, clicked_z=None):
    csv_df, labels, X, history, rank = prepare_materials(keyword, model_name)
    print(viewer_id)
    if viewer_id == 'viewer_1':
        Z, Y, sigma = history['Z1'], history['Y'], history['sigma']
    elif viewer_id == 'viewer_2':
        Z, Y, sigma = history['Z2'], history['Y'], history['sigma']
    else:
        print("Set viewer_id")

    # Build figure
    fig = go.Figure(
        layout=go.Layout(
            xaxis=dict(
                range=[Z[:, 0].min() - 0.1, Z[:, 0].max() + 0.1],
                visible=False,
                autorange=True,
            ),
            yaxis=dict(
                range=[Z[:, 1].min() - 0.1, Z[:, 1].max() + 0.1],
                visible=False,
                scaleanchor='x',
                scaleratio=1.0,
            ),
            showlegend=False,
            autosize=True,
            margin=dict(
                b=0,
                t=0,
                l=0,
                r=0,
            ),
        ),
    )

    if viewer_name=="topic":
        n_components = 5
        fig = draw_topics(fig, Y, n_components)
    elif viewer_name=="CCP":
        fig = draw_ccp(fig, Y, history['Zeta'], history['resolution'], clicked_z, viewer_id)
    else:
        print("U-matrix not implemented")
        NotImplemented
        # u_resolution = 100
        # fig = draw_umatrix(fig, X, Z, sigma, u_resolution, labels)

    fig = draw_scatter(fig, Z, labels, rank)

    fig.update_coloraxes(
        showscale=False
    )
    fig.update_layout(
        plot_bgcolor="white",
    )
    fig.update(
        layout_coloraxis_showscale=False,
        layout_showlegend=False,
    )
    fig.update_yaxes(
        fixedrange=True,
    )
    fig.update_xaxes(
        fixedrange=True,
    )

    return fig


@app.callback(
    Output('example-graph', 'figure'),
    [
        Input('explore-start', 'n_clicks'),
        Input('model-selector', 'value'),
        Input('viewer-selector', 'value'),
        Input('example-graph2', 'clickData'),
    ],
    [
        State('search-form', 'value'),
        State('example-graph', 'figure'),
    ])

def load_learning(n_clicks, model_name, viewer_name, clickData, keyword, prev_fig):
    print("clicked")
    print(clickData)
    print(keyword)

    if not keyword:
        keyword = "Machine Learning"

    if clickData and "points" in clickData and "pointIndex" in clickData["points"][0]:
        print("clicked_from_map2")
        # index = clickData['points'][0]['pointIndex']
        return make_figure(keyword, model_name, "CCP", "viewer_1", clickData)

    # return make_figure(keyword, model_name, "viewer_1", viewer_name)
    return make_figure(keyword, model_name, viewer_name, "viewer_1", None)


@app.callback(
    Output('example-graph2', 'figure'),
    [
        Input('explore-start', 'n_clicks'),
        Input('model-selector', 'value'),
        Input('viewer-selector', 'value'),
        Input('example-graph', 'clickData'),
    ],
    [
        State('search-form', 'value'),
        State('example-graph2', 'figure'),
    ])

def load_learning(n_clicks, model_name, viewer_name, clickData, keyword, prev_fig):
    if not keyword:
        keyword = "Machine Learning"

    if clickData:
        if not ("points" in clickData and "pointIndex" in clickData["points"][0]):
            pass
        else:
            print("clicked_from_map1")
            return make_figure(keyword, model_name, "CCP", "viewer_2", clickData)

    # return make_figure(keyword, model_name, "viewer_1", viewer_name)
    return make_figure(keyword, model_name, viewer_name, "viewer_2", None)


# def ccp_given_z2(clickData):
#         print("clicked")
#         # if not ("points" in Z1 and "pointIndex" in Z1["points"][0]) :
#         # if clicked viewer_2 variable, change viewer_1
#         return make_figure("ファッション", "TSOM", "viewer_1", viewer_name="CCP")



@app.callback([
        Output('link', 'children'),
        Output('link', 'href'),
        Output('link', 'target'),
        Output('card-text', 'children'),
        Output('snippet-text', 'children'),
    ],
    [
        Input('example-graph', 'hoverData'),
    ],
    [
        State('search-form', 'value'),
        State('link', 'children'),
        State('link', 'href'),
        State('link', 'target'),
        State('card-text', 'children'),
        State('snippet-text', 'children')
    ])
def update_title(hoverData, keyword, prev_linktext, prev_url, prev_target, prev_page_title, prev_snippet):
    if hoverData:
        if not ("points" in hoverData and "pointIndex" in hoverData["points"][0]) \
            or keyword == None:
            link_title = prev_linktext
            url = prev_url
            target = prev_target
            page_title = prev_page_title
            snippet = prev_snippet
        else:
            csv_df = pd.read_csv(keyword+".csv")
            index = hoverData['points'][0]['pointIndex']
            link_title = "サイトへ Go"
            labels = csv_df['site_name']
            url = csv_df['URL'][index]
            target = "_blank"
            page_title = labels[index]
            snippet = csv_df['snippet'][index]
    else:
        link_title = "マウスを当ててみよう"
        url = "#"
        target = "_self"
        page_title = ""
        snippet = ""
    return link_title, url, target, page_title, snippet

# app.clientside_callback(
#     "onLatentClicked",
#     Output('explore-start', 'outline'),
#     Input('example-graph', 'clickData'), prevent_initial_call=True)


# U-Matrix の説明用のモーダル
umatrix_modal = dbc.Modal([
    dbc.ModalHeader("U-Matrix 表示とは？"),
    dbc.ModalBody("青い領域がクラスタ，赤い領域がクラスタ境界を表す"),
    dbc.ModalFooter(
        dbc.Button(
            "Close", id="close-umatrix-modal", className="ml-auto", n_clicks=0
        )
    ),
], id="umatrix-modal", is_open=False, centered=True)


# app.callback(
#     Output('umatrix-modal', 'is_open'),
#     [
#         Input('open-umatrix-modal', 'n_clicks'),
#         Input('close-umatrix-modal', 'n_clicks'),
#     ],
#     State('umatrix-modal', 'is_open'))(toggle_modal)


link_card = dbc.Card([
    # html.P("", id="card-text", className="h4"),
    dbc.CardHeader("", id="card-text", className="h4"),
    html.P("", id="snippet-text", className="h5",style={"min-height":"100px"}),
    html.A(
        id='link',
        href='#',
        children="マウスを当ててみよう",
        target="_self",
        className="btn btn-outline-primary btn-lg",
    ),
    dbc.CardFooter(
        "マップ中の丸をクリックしても該当ページへ飛べます．",
        className="font-weight-light",
    )],
    id="link-card",
)


search_component = dbc.Col([
    dbc.Row([
        dbc.Col(
            id='search-form-div',
            children=dcc.Input(
                id='search-form',
                type="text",
                placeholder="検索ワードを入力してください",
                style=dict(width="100%"),
                className="input-control"),
            width=10,
        ),
        dbc.Col(
            html.Div(
                id='explore-start',
                children="検索！",
                className="btn btn-primary btn-lg",
            ),
            width=2,
        )],
        align="center")],
    style={"min-height":"100px"},
    md=12,
    xl=6,
    className="card",
)

view_options = dbc.Col([
    dbc.Row(
        dbc.RadioItems(
            options=[
                {'label': 'SOM', 'value': 'SOM'},
                {'label': 'UKR', 'value': 'UKR'},
            ],
            value='TSOM',
            id="model-selector",
            style={'textAlign': "center", "display": "none"},
            className="h3",
        ),
    ),
    dbc.Row(
        dbc.RadioItems(
            options=[
                {'label': 'U-matrix 表示', 'value': 'U-matrix'},
                {'label': 'クラスタ表示', 'value': 'topic'},
            ],
            value='U-matrix',
            id="viewer-selector",
            inline=True,
            className="h3",
        ),
        style=dict(height="60%", width="100%", padding="10"),
        align="center",
    )],
    md=12,
    xl=6,
    style={"min-height":"100px", "padding-left":"30px"},
    className="card",
)


result_component = dbc.Row(
    [
        dbc.Col(
            dcc.Loading(
                dcc.Graph(
                    id='example-graph',
                    figure=make_figure("Machine Learning", "TSOM", viewer_id="viewer_1"),
                    config=dict(displayModeBar=False)
                ),
                id="loading"
            ),
            style={"height": "100%",},
            md=12,
            xl=9,
            className="card",
        ),
        dbc.Col(
            dcc.Loading(
                dcc.Graph(
                    id='example-graph2',
                    figure=make_figure("Machine Learning", "TSOM", viewer_id="viewer_2"),
                    config=dict(displayModeBar=False)
                ),
                id="loading2"
            ),
            style={"height": "100%",},
            md=12,
            xl=9,
            className="card",
        ),
        dbc.Col(
            link_card,
            md=12,
            xl=3
        )
    ],
    align="center",
    className="h-75",
    style={"min-height": "70vh",},
    no_gutters=True
)


app.layout = dbc.Container(children=[
    dbc.Row([
        dbc.Col(
            html.H1(
                id='title',
                children='情報探索エンジン',
                className="display-2",
                style=dict(
                    fontFamily="Oswald, sans-serif"
                )
            ),
            md=12,
            xl=6
        ),
        dbc.Col(
            html.Div(
            children=[
                "情報探索をサポートする Web アプリケーションです．", html.Br(),
                "Google 検索結果を2次元にマッピングし，", html.Br(),
                "さらに勾配計算やクラスタリングをすることによって", html.Br(),
                "情報探索をサポートします．",
            ],
            className="h4"),
            md=12,
            xl=6
        )
    ], style={"min-height":"10vh", "margin-top":"10px"},
    align="end"),
    html.Hr(),
    # dbc.Button(
    #     "U-Matrix 表示とは？", id="open-umatrix-modal", className="ml-auto", n_clicks=0
    # ),
    umatrix_modal,
    dbc.Row([
        search_component,
        view_options
        ],
        style={"min-height":"10vh"}),
    result_component,
])
