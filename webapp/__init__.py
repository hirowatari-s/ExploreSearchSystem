import dash
import dash_bootstrap_components as dbc
import pathlib


external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    dbc.themes.BOOTSTRAP,
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


FILE_UPLOAD_PATH = pathlib.Path("webapp", "uploads")
DEFAULT_FAVICON_PATH = pathlib.Path("webapp", "assets", "default.png")
app.server.config['UPLOAD_FOLDER'] = str(FILE_UPLOAD_PATH)

import webapp.dash
