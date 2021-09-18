# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.


from re import search
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from webapp import app, FILE_UPLOAD_PATH
import requests
import io
from PIL import Image
from Grad_norm import Grad_Norm
from som import ManifoldModeling as MM
import pathlib
from scraperbox import fetch_gsearch_result
from make_BoW import make_bow
from sklearn.decomposition import NMF
import tldextract
import pickle


PROJECT_ROOT = pathlib.Path('.')
SAMPLE_DATASETS = [
    csv_file.stem for csv_file in PROJECT_ROOT.glob("./*.csv")
]
domain_favicon_map = dict()
resolution = 20


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
        csv_df = fetch_gsearch_result(keyword)
        X , labels, df = make_bow(csv_df)
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
            init='PCA'
        )
        mm.fit(nb_epoch=nb_epoch)
        history = dict(
            Z=mm.history['z'][-1],
            Y=mm.history['y'][-1],
            sigma=mm.history['sigma'][-1],
        )
        print("Learning finished.")
        with open(model_save_path, 'wb') as f:
            pickle.dump(history, f)
    return csv_df, labels, X, history, rank


def draw_umatrix(fig, X, Z, sigma, u_resolution, labels):
    print(X.shape, Z.shape)
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
            colorscale="Greens",
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
                # line=dict(
                #     width=1.5,
                #     color="white"
                # ),
            ),
            text=labels,
            hoverlabel=dict(
                bgcolor="rgba(255, 255, 255, 0.75)",
            ),
        )
    )
    return fig


def make_figure(keyword, model_name, enable_favicon=False, viewer_name="U_matrix"):
    csv_df, labels, X, history, rank = prepare_materials(keyword, model_name)
    Z, Y, sigma = history['Z'], history['Y'], history['sigma']

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
                autorange=True,
            ),
            showlegend=False,
            margin=dict(
                b=0,
                t=0,
                l=0,
                r=0,
            )
        ),
    )

    if viewer_name=="topic":
        n_components = 5
        fig = draw_topics(fig, Y, n_components)
    else:
        u_resolution = 100
        fig = draw_umatrix(fig, X, Z, sigma, u_resolution, labels)

    fig = draw_scatter(fig, Z, labels, rank)

    if enable_favicon:
        for i, z in enumerate(Z[::-1]):
            url = csv_df['URL'][i]
            parser = tldextract.extract(url)
            image_filepath = pathlib.Path(FILE_UPLOAD_PATH, parser.domain + '.png')
            print("image path:", image_filepath.resolve())
            if not parser.domain in domain_favicon_map:
                if not image_filepath.exists():
                    print("From API")
                    favicon_url = f"https://s2.googleusercontent.com/s2/favicons?domain_url={url}"
                    res = requests.get(favicon_url)
                    logo_img = Image.open(io.BytesIO(res.content))
                    logo_img.save(image_filepath)
                else:
                    print("From local")
                    logo_img = Image.open(image_filepath)
                domain_favicon_map[parser.domain] = logo_img
            else:
                print("From cache")
                logo_img = domain_favicon_map[parser.domain]
            print("fetched:", url)
            fig.add_layout_image(
                    x=z[0],
                    sizex=0.1,
                    y=z[1],
                    sizey=0.1,
                    xref="x",
                    yref="y",
                    opacity=1,
                    xanchor="center",
                    yanchor="middle",
                    layer="above",
                    source=logo_img
            )
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


def make_search_form(style):
    form_id = 'search-form'
    if style == 'selection':
        return dcc.Dropdown(
            id=form_id,
            options=[
                {'label': name, 'value': name} for name in SAMPLE_DATASETS
            ],
            value='ファッション'
        )
    else:
        return dcc.Input(
            id=form_id,
            type="text",
            placeholder="検索ワードを入力してください",
            style=dict(width="100%")
        )

@app.callback(
    Output('example-graph', 'figure'),
    [
        Input('explore-start', 'n_clicks'),
        Input('model-selector', 'value'),
        Input('viewer-selector', 'value'),
    ],
    [
        State('search-form', 'value'),
        State('favicon-enabled', 'checked'),
    ])
def load_learning(n_clicks, model_name, viewer_name,  keyword, favicon):
    return make_figure(keyword, model_name, favicon, viewer_name)


@app.callback(
    Output('search-form-div', 'children'),
    Input('search-style-selector', 'value'))
def search_form_callback(style):
    return make_search_form(style)


@app.callback([
        Output('link', 'children'),
        Output('link', 'href'),
        Output('link', 'target'),
        Output('card-text', 'children'),
        Output('snippet-text', 'children'),
        # Output('card-img', 'src'),
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
        if not ("points" in hoverData and "pointIndex" in hoverData["points"][0]):
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
            # favicon_url = f"https://s2.googleusercontent.com/s2/favicons?domain_url={url}"
    else:
        link_title = "マウスを当ててみよう"
        url = "#"
        target = "_self"
        page_title = ""
        snippet = ""
        # favicon_url = "https://1.bp.blogspot.com/-9DCMH4MtPgw/UaVWN2aRpRI/AAAAAAAAUE4/jRRLie86hYI/s800/columbus.png"
    return link_title, url, target, page_title, snippet #, favicon_url


app.clientside_callback(
    "onLatentClicked",
    Output('explore-start', 'outline'),
    Input('example-graph', 'clickData'))


# モーダルの Toggler
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


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


app.callback(
    Output('umatrix-modal', 'is_open'),
    [
        Input('open-umatrix-modal', 'n_clicks'),
        Input('close-umatrix-modal', 'n_clicks'),
    ],
    State('umatrix-modal', 'is_open'))(toggle_modal)


link_card = dbc.Card([
    # dbc.CardImg(
    #     id="card-img",
    #     src="https://1.bp.blogspot.com/-9DCMH4MtPgw/UaVWN2aRpRI/AAAAAAAAUE4/jRRLie86hYI/s800/columbus.png",
    #     top=True,
    #     className="img-fluid img-thumbnail",
    #     style="height: 50px; width: 50px;",
    # ),
    html.P("", id="card-text", className="h4"),
    html.P("", id="snippet-text", className="h5"),
    html.A(
        id='link',
        href='#',
        children="マウスを当ててみよう",
        target="_self",
        className="btn btn-outline-primary btn-lg",
    )
], id="link-card")


search_component = dbc.Col([
    dbc.Row(
        dbc.RadioItems(
            options=[
                {'label': 'サンプルのデータセット', 'value': 'selection'},
                {'label': '自由に検索', 'value': 'freedom'},
            ],
            value='selection',
            id="search-style-selector",
            style={"background-color": "purple"}
        ),
    ),
    dbc.Row([
        dbc.Col(
            html.Div(
                id='search-form-div',
                children=make_search_form('selection'),
            ),
            width=10,
        ),
        dbc.Col(
            dbc.Button(
                "検索！",
                outline=True,
                id="explore-start",
                n_clicks=0,
            ),
            width=2,
        ),
    ], align="center")
], style={"background-color":"blue"})


view_options = dbc.Col([
    dbc.RadioItems(
        options=[
            {'label': 'SOM', 'value': 'SOM'},
            {'label': 'UKR', 'value': 'UKR'},
        ],
        value='UKR',
        id="model-selector",
        style={'textAlign': "center", "background-color": "pink"}
    ),
    dbc.FormGroup([
        dbc.Checkbox(
            id="favicon-enabled",
            children="ファビコンを表示する",
            checked=False,
        ),
        dbc.Label(
            "ロゴを表示する",
            html_for="favicon-enabled",
            className="form-check-label",
        ),
    ], check=True),
    dbc.RadioItems(
        options=[
            {'label': 'U-matrix', 'value': 'U-matrix'},
            {'label': 'topic', 'value': 'topic'},
        ],
        value='U-matrix',
        id="viewer-selector",
        style={'textAlign': "center", "background-color": "cyan"}
    ),
])


result_component = dbc.Row(
    [
        dbc.Col(
            dcc.Loading(
                dcc.Graph(
                    id='example-graph',
                    figure=make_figure("ファッション", "UKR"),
                    config=dict(displayModeBar=False)
                ),
                id="loading"
            ),
            width=8,
            style={"height": "100%", "background-color": "blue"}),
        dbc.Col(link_card, width=4, style={"background-color": "yellow"})
    ],
    align="center",
    className="h-75",
    style={"min-height": "70vh", "background-color": "green"},
    no_gutters=True
)


app.layout = dbc.Container(children=[
    dbc.Row([
        html.H1(id='title', children='情報探索エンジン'),
        html.Div(children='''
            情報探索を行うツールです．
        '''),
        html.Hr(),
    ], style={"background-color":"red"}),
    # dbc.Button(
    #     "U-Matrix 表示とは？", id="open-umatrix-modal", className="ml-auto", n_clicks=0
    # ),
    umatrix_modal,

    dbc.Row([
        search_component,
        view_options
    ]),
    result_component,
])
