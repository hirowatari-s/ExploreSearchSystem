# -*- coding: utf-8 -*-

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from webapp import app
from webapp.figure_maker import make_figure


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
    className="card bg-danger",
)

view_options = dbc.Col([
    dbc.Row(
        dbc.RadioItems(
            options=[
                {'label': 'U-matrix 表示', 'value': 'U-matrix'},
                {'label': 'CCP 表示', 'value': 'CCP'},
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
    style={"padding-left":"30px"},
    className="card bg-success",
)


make_map = lambda id, viewer_id: dbc.Col(
    dcc.Loading(
        dcc.Graph(
            id=id,
            figure=make_figure("Machine Learning", "TSOM", viewer_id=viewer_id),
            config=dict(displayModeBar=False)
        ),
    ),
    style={"height": "100%",},
    md=12,
    xl=6,
    className="card",
)


result_component = dbc.Row(
    [
        make_map('paper-map', 'viewer_1'),
        make_map('word-map',  'viewer_2'),
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
                children='論文探索エンジン',
                className="display-2",
                style=dict(
                    fontFamily="Oswald, sans-serif"
                )
            ),
            md=12,
            xl=6
        ),
        search_component,
        ],
    style={"min-height":"10vh", "margin-top":"10px"},
    align="end"),
    html.Hr(),
    umatrix_modal,
    dbc.Row([
        view_options
        ],
        style={"min-height":"5vh"}),
    result_component,
], className="bg-light")
