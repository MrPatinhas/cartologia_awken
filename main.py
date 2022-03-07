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

stream.set_page_config(
     page_title="Cartologia | Soccer Analysis",
     page_icon="🎲",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )

main_parameters_dict = dict({
     'Premier League (ENG)':{
         "rodada_atual": 28,
         "country_name": "England"
     },
     'Serie A (ITA)':{
         "rodada_atual": 28,
         "country_name": "Italy"
     },
     'Bundesliga (GER)':{
         "rodada_atual": 25,
         "country_name": "Germany"
     },
     'La Liga (SPA)':{
         "rodada_atual": 27,
         "country_name": "Spain"
     },
     'Paulistao (BRA)':{
         "rodada_atual": 10,
         "country_name": "Paulista"
     },

 })

minor_parameters_dict = dict({
      'Atacantes':"F",
      'Meio Campo': "M",
      'Defensores': "D"
  })

@stream.cache
def load_data(rodada_atual_, country_of_league, year):

    df_data, df_liga = generate_df_data_dataframe(rodada_atual_, country_of_league,year)
    dataframe_pontuacao_relativa = get_pontuacao_relativa(df_liga)
    dataframe_player = get_datatframe_player_quantile(df_data)
    team_table = df_data[['home_team_id','home_team_nome']].drop_duplicates(keep='first')

    team_name_list = df_data.team_name.unique().tolist()
    team_name_list = [""] + team_name_list

    return df_data, df_liga, dataframe_pontuacao_relativa, dataframe_player, team_table, team_name_list

stream.markdown("### 🎲 Cartologia | Soccer Analysis | Fantasy Games")

if 'key' not in stream.session_state:
    stream.session_state['league_decision'] = 'UNDEFINED'

league_selection = stream.selectbox(
'Selecione a Liga',
["", 'Premier League (ENG)', 'Serie A (ITA)', 'Bundesliga (GER)','La Liga (SPA)','Paulistao (BRA)']
)

if(league_selection!=""):
    rodada_atual_ = main_parameters_dict[league_selection]['rodada_atual']
    country_of_league = main_parameters_dict[league_selection]['country_name']
    year = '2021'
    df_data, df_liga, dataframe_pontuacao_relativa, dataframe_player, team_table, team_name_list = load_data(rodada_atual_, country_of_league, year)

    stream.session_state['league_decision'] = 'DEFINED'

if(stream.session_state['league_decision'] == 'DEFINED'):
    cb1, cb2 = stream.columns(2)
    flag_plot_time_conteiner = cb1.checkbox('Análise de Equipe')
    flag_plot_campeonato_conteiner = cb2.checkbox('Análise de Campeonato')

    team_selector = ""
    if(flag_plot_time_conteiner):
        team_selector = stream.selectbox(
        'Selecione o Time',
        team_name_list
        )

        position_selector = stream.selectbox(
        'Selecione a Posição',
        ["", "Atacantes", "Meio Campo", "Defensores"]
        )

    container_time = stream.container()

    container_campeonato = stream.container()

    if(team_selector!="" and flag_plot_time_conteiner):

        container_time.markdown("### 👀 Análise de Time | {}".format(team_selector))

        if(position_selector!=""):

            pt = get_team_res(team_selector,
                             minor_parameters_dict[position_selector],
                             df_data,
                             df_liga,
                             rodada_atual_,
                             dataframe_pontuacao_relativa,
                             dataframe_player)

            container_time.bokeh_chart(pt)

        tid__ = team_table[team_table['home_team_nome']==team_selector].home_team_id.tolist()[0]

        p5 = get_round_plot(df_data, tid__,
                           scout1='Pontuação Média',
                           scout2='Desvio Padrão Pontuação',
                           sc1 = 'prev_cartola',
                           sc2 = 'prev_cartola',
                           min_num_jogos=5)

        container_time.bokeh_chart(p5)

    if(flag_plot_campeonato_conteiner):

        container_campeonato.markdown("### 🌎 Análise da Liga | {}".format(league_selection))

        p2 = get_plot_scout_decisivo(df_data, df_liga, alpha_y=1.75, photo_height=30)
        container_campeonato.bokeh_chart(p2)

        p3 = get_goleiro_plot(df_data, df_liga, alpha_y = 1.6, photo_height = main_parameters_dict[league_selection]['rodada_atual']*0.9, legend_desloc = 1.6, w_ = 0.9)
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
        mapa_plot_F.plot_width = 750
        mapa_plot_F.plot_height = 600

        mapa_plot_M = get_mapa_confronto(df_data, rodada_atual_, df_liga, 'M')
        mapa_plot_M.plot_width = 750
        mapa_plot_M.plot_height = 600

        mapa_plot_D = get_mapa_confronto(df_data, rodada_atual_, df_liga, 'D')
        mapa_plot_D.plot_width = 750
        mapa_plot_D.plot_height = 600

        mapa_plot_G = get_mapa_confronto(df_data, rodada_atual_, df_liga, 'G')
        mapa_plot_G.plot_width = 750
        mapa_plot_G.plot_height = 600

        grid_confronto = gridplot([[mapa_plot_F, mapa_plot_M], [mapa_plot_D, mapa_plot_G]])

        container_campeonato.bokeh_chart(grid_confronto)
