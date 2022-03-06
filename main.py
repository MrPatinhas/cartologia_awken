import pandas as pd
from scipy import stats
from bokeh.io import show, output_file
from bokeh.plotting import figure, save, output_file, show
from bokeh.models import ColumnDataSource, ranges, LabelSet, Label, Range1d, PolyAnnotation, Band
from bokeh.layouts import gridplot, row, column, layout
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.sampledata.perceptions import probly
from datetime import timedelta
from bokeh.models import Range1d,ImageURL
import math
from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead
from bokeh.models import Span
from bokeh.models import BoxAnnotation
from bokeh.models import Title
from bokeh.models import Arrow, OpenHead, NormalHead, VeeHead
from bokeh.models import Span
from bokeh.models import BoxAnnotation
from collections import OrderedDict
from io import StringIO
from math import log, sqrt
from bokeh.io import export_png
import numpy as np
import pandas as pd
from bokeh.document import Document
from bokeh.embed import file_html
from bokeh.layouts import gridplot
from bokeh.models import (BasicTicker, Circle, ColumnDataSource, DataRange1d,
                          Grid, LinearAxis, PanTool, Plot, WheelZoomTool,)
from bokeh.resources import INLINE
from bokeh.sampledata.iris import flowers
from bokeh.util.browser import view
import numpy as np
from scipy import optimize, interpolate
from scipy import stats as st
from bokeh.models import Legend
from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, HBar, LinearAxis, Plot, Text
from bokeh.models import FixedTicker,Wedge
import math
from datetime import date
from bokeh.document import Document
from bokeh.models import (Circle, ColumnDataSource, Div, Grid,
                          Line, LinearAxis, Plot, Range1d,)
from bokeh.resources import INLINE
from bokeh.util.browser import view
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings("ignore")
from bokeh.models import Arrow, NormalHead, OpenHead, VeeHead
from bokeh.plotting import figure, output_file, show
from PIL import Image
import requests
from io import BytesIO
import io
from pathlib import Path
from PIL import Image
import urllib.request
import io
import random
from difflib import SequenceMatcher
from pathlib import Path
import streamlit as stream
from functions_storage import *

@stream.cache
def load_data(rodada_atual_, country_of_league, year):

    df_data, df_liga = generate_df_data_dataframe(rodada_atual_, 'Spain','2021')
    dataframe_pontuacao_relativa = get_pontuacao_relativa(df_liga)
    dataframe_player = get_datatframe_player_quantile(df_data)
    team_table = df_data[['home_team_id','home_team_nome']].drop_duplicates(keep='first')

    team_name_list = df_data.team_name.unique().tolist()
    team_name_list = [""] + team_name_list

    return df_data, df_liga, dataframe_pontuacao_relativa, dataframe_player, team_table, team_name_list

stream.markdown("### 🎲 The Application")
rodada_atual_ = 27
country_of_league = 'Spain'
year = '2021'

df_data, df_liga, dataframe_pontuacao_relativa, dataframe_player, team_table, team_name_list = load_data(rodada_atual_, country_of_league, year)

cb1, cb2, cb3 = stream.columns(3)
flag_plot_mapas = cb1.checkbox('Mapa de Confronto')
flag_plot_gol = cb2.checkbox('Análise de Goleiros')
flag_plot_decisivo = cb3.checkbox('Análise de Decisivo')

cb1, cb2 = stream.columns(2)
flag_plot_time_conteiner = cb1.checkbox('Análise de Equipe')
flag_plot_campeonato_conteiner = cb2.checkbox('Análise de Campeonato')

team_selector = ""
if(flag_plot_time_conteiner):
    team_selector = stream.selectbox(
    'How would you like to be contacted?',
    team_name_list
    )

container_time = stream.container()

container_campeonato = stream.container()

if(team_selector!="" and flag_plot_time_conteiner):
    pt = get_team_res(team_selector, "M", df_data, df_liga, rodada_atual_, dataframe_pontuacao_relativa, dataframe_player)
    container_time.bokeh_chart(pt)

    tid__ = team_table[team_table['home_team_nome']==team_selector].home_team_id.tolist()[0]

    p5 = get_round_plot(df_data, tid__,
                       scout1='Avaliação Jogador',
                       scout2='Nota Cartola Jogador',
                       sc1 = 'player_grade',
                       sc2 = 'prev_cartola',
                       min_num_jogos=10)

    container_time.bokeh_chart(p5)

if(flag_plot_campeonato_conteiner):

    p2 = get_plot_scout_decisivo(df_data, df_liga, alpha_y=1.75, photo_height=30)
    container_campeonato.bokeh_chart(p2)

    p3 = get_goleiro_plot(df_data, df_liga, alpha_y = 1.7, photo_height = 20, legend_desloc = 1.8, w_ = 0.9)
    container_campeonato.bokeh_chart(p3)

    p4_1 = get_sg_plot(df_data, df_liga,'CASA')
    p4_1.background_fill_color = 'white'
    p4_1.border_fill_color = 'white'
    p4_1.outline_line_color = 'white'

    p4_2 = get_sg_plot(df_data, df_liga,'FORA')
    p4_2.background_fill_color = 'white'
    p4_2.border_fill_color = 'white'
    p4_2.outline_line_color = 'white'

    grid_sg = gridplot([[p4_1], [p4_2]])
    container_campeonato.bokeh_chart(grid_sg)

    mapa_plot_F = get_mapa_confronto(df_data, rodada_atual_, df_liga, 'F')
    mapa_plot_F.plot_width = 800
    mapa_plot_F.plot_height = 600

    mapa_plot_M = get_mapa_confronto(df_data, rodada_atual_, df_liga, 'M')
    mapa_plot_M.plot_width = 800
    mapa_plot_M.plot_height = 600

    mapa_plot_D = get_mapa_confronto(df_data, rodada_atual_, df_liga, 'D')
    mapa_plot_D.plot_width = 800
    mapa_plot_D.plot_height = 600

    mapa_plot_G = get_mapa_confronto(df_data, rodada_atual_, df_liga, 'G')
    mapa_plot_G.plot_width = 800
    mapa_plot_G.plot_height = 600

    grid_confronto = gridplot([[mapa_plot_F, mapa_plot_M], [mapa_plot_D, mapa_plot_G]])

    container_campeonato.bokeh_chart(grid_confronto)
