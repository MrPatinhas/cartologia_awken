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
import os
import random
from difflib import SequenceMatcher
from pathlib import Path
import streamlit as stream
from functions_storage import *
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_hello = "https://assets3.lottiefiles.com/private_files/lf30_rg5wrsf4.json"
lottie_hello = load_lottieurl(lottie_url_hello)

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

c1, c2, c3, c4, c5, c6, c7, c8 = stream.columns([1, 1, 0.2, 3, 1, 0.5, 0.5, 0.5])

with c1:
    st_lottie(lottie_hello, key="hello", height=150,
        width=150)

c2.image('https://i.ibb.co/fvbCyHB/Imagem1.png', caption='Cartologia', width=150)

url_telegram = 'https://logodownload.org/wp-content/uploads/2017/11/telegram-logo.png'
url_insta = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Instagram_icon.png/640px-Instagram_icon.png'
url_yt = 'https://logodownload.org/wp-content/uploads/2014/10/youtube-logo-5-2.png'

c3.image(url_telegram, width=40)
c3.image(url_insta, width=40)
c3.image(url_yt, width=40)

c4.write("Telegram: https://t.me/cartologia")
c4.write("")
c4.write("Instagram: https://www.instagram.com/cartologiafc/")
c4.write("")
c4.write("Youtube: https://www.youtube.com/channel/UCgwkcq7yeW1mOXdNXhszmOg")


main_parameters_dict = dict({
     'Premier League (ENG)':{
         "rodada_atual": 29,
         "country_name": "England"
     },
     'Serie A (ITA)':{
         "rodada_atual": 29,
         "country_name": "Italy"
     },
     'Bundesliga (GER)':{
         "rodada_atual": 26,
         "country_name": "Germany"
     },
     'La Liga (SPA)':{
         "rodada_atual": 28,
         "country_name": "Spain"
     },
     'Paulistao (BRA)':{
         "rodada_atual": 11,
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

    if((country_of_league=='England') or (country_of_league=='Italy')):
        stream.session_state['flag_understat'] = True
    else:
        stream.session_state['flag_understat'] = False

    if(stream.session_state['flag_understat']):
        dataframe_lances_finalizacao_global__, depara_understat = get_understat_dataframe(country_of_league)
        end_df = get_list_top_shooters(dataframe_lances_finalizacao_global__)
    else:
        dataframe_lances_finalizacao_global__ = None
        depara_understat = None
        end_df = None

    team_name_list = df_data.team_name.unique().tolist()
    team_name_list = [""] + team_name_list

    return df_data, df_liga, dataframe_pontuacao_relativa, dataframe_player, team_table, team_name_list, dataframe_lances_finalizacao_global__, depara_understat, end_df

stream.markdown("### 🎲 Cartologia | Soccer Analysis | Fantasy Games")

with stream.expander("Glossário & Premissas"):
     stream.write("""
         Scouts = Eventos que ocorrem durantse uma partida de futebol \n
         A = Assistência - (Passes que resultam imediatamente em gol) \n
         G = Gols \n
         FS = Faltas sofridas \n
         FD = Finalizações Defendidados - (Chutem que vão no gol mas são defendidas) \n
         DS = Desarmes - (Admitidos com Interceptações declaradas + 50% de dividias vencidas) \n
         FF = Finalizações para Fora - (Chutes que não vão no gol) \n
         F =  Finalizações - (Representa a soma de FF e FD) \n
         xG = Expected Goals - Métrica probabilistica de conversão de finalizações em gols \n
         xA = Expected Assist - Métrica probabilistica de conversão de passes em finalizações \n
     """)

if 'flag_understat' not in stream.session_state:
    stream.session_state['flag_understat'] = False

if 'league_decision' not in stream.session_state:
    stream.session_state['league_decision'] = 'UNDEFINED'

league_selection = stream.selectbox(
'Selecione a Liga',
["", 'Premier League (ENG)', 'Serie A (ITA)', 'Bundesliga (GER)','La Liga (SPA)','Paulistao (BRA)']
)

if(league_selection!=""):
    rodada_atual_ = main_parameters_dict[league_selection]['rodada_atual']
    country_of_league = main_parameters_dict[league_selection]['country_name']
    year = '2021'
    df_data, df_liga, dataframe_pontuacao_relativa, dataframe_player, team_table, team_name_list, dataframe_lances_finalizacao_global__, depara_understat, end_df = load_data(rodada_atual_, country_of_league, year)
    stream.session_state['league_decision'] = 'DEFINED'

if(stream.session_state['league_decision'] == 'DEFINED'):

    stream.markdown("### Liga | {0} | Rodada {1} ".format(league_selection, main_parameters_dict[league_selection]['rodada_atual']))
    stream.write("")

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
        cb4,xx = container_time.columns([1,0.1])
        cb1, cb2, cb3 = container_time.columns([1,0.5,1])
        cb5,xx = container_time.columns([1,0.1])

        if(position_selector!=""):

            pt = get_team_res(team_selector,
                             minor_parameters_dict[position_selector],
                             df_data,
                             df_liga,
                             rodada_atual_,
                             dataframe_pontuacao_relativa,
                             dataframe_player)

            cb4.markdown("##### Fantasy Evaluation | {}".format(team_selector))
            cb4.write("""
             Relatório com foco em avaliar as peças possívies para escalação no Fantasy Game de acordo com a posição selecionada \n
             A primeira linha mostra da direita para esquerda: Evolução temporal relativa da pontuação na liga dos times em confronto, \n
             O segundo gráfico traz as médias de pontuação dos jogadores da equipe na posição bem como as quantidades de jogos participados (total e últimos 5 jogos) \n
             O terceiro gráfico que mostra a comparação com a média do campenato no ponto de vista do executado e cedido pelo adversário por scout. \n
             Bolas vermelhas indicam um scout que a equipe executa abaixo da média do campeonato e o adversário cede também abaixo da média, \n
             Bolas azuis indicam scout com execução acima mas cedido abaixo da médio, amarelo o contrário, e por fim o verde indica execução e cedido acima da média \n
             As duas últimas linhas do relatório mostram respectivamente a distribuição de probabilidade da pontuação e o mapa de quartis por scout do jogador \n
            """)

            cb4.bokeh_chart(pt)

        tid__ = team_table[team_table['home_team_nome']==team_selector].home_team_id.tolist()[0]

        cb1.markdown("##### Team Wheel | {}".format(team_selector))
        cb1.write("""
         O gráfico abaixo mostra qual a média de pontuação dos jogadores, no Fantasy Rei do Pitaco, bem como o desvio padrão dos mesmos.
         Considera-se jogadores com no mínimo 5 jogos de participação
        """)

        p5 = get_round_plot(df_data, tid__,
                           scout1='Pontuação Média',
                           scout2='Desvio Padrão Pontuação',
                           sc1 = 'prev_cartola',
                           sc2 = 'prev_cartola',
                           min_num_jogos=5)

        cb1.bokeh_chart(p5)

        if((country_of_league=='England') or (country_of_league=='Italy')):

            cb3.markdown("##### Corner Masters | {}".format(team_selector))
            cb3.write("""
             O gráfico abaixo mostra a posição das finalizações de cabeça obtidas em escanteio pelo time selecionado.
             Os pontos vermelhos mostram as finalizações que resultaram em gols. Em destaque temos os TOP3 jogadores em volume de finalização de cabeça em escanteio
            """)

            p6 = get_team_corner_plot(df_data, dataframe_lances_finalizacao_global__, depara_understat, team_selector, iqr_multiple = 0.45)
            p6.plot_width = 600
            p6.plot_height = 800
            cb3.bokeh_chart(p6)

            players_df_shots = end_df[end_df['team_name_base']==team_selector].head(3)

            plots = []
            row_ = []
            row_count = 0

            for p_id__ in players_df_shots.player_id.unique():

                if len(depara_understat[depara_understat['understat_id']==p_id__]['player_id'].tolist()) > 0:
                    #df_def_aux = df_def__[df_def__['team_name']==team_name__]
                    plt_ = get_player_shot_xray(df_data, dataframe_lances_finalizacao_global__, depara_understat, p_id__, iqr_multiple = 0.5)
                    #plt_.min_border_bottom = 50
                    #plt_.plot_width = 800
                    #plt_.plot_height = 600

                    row_.append(plt_)
                    row_count = row_count + 1
                    if(row_count==3):
                        row_count = 0
                        plots.append(row_)
                        row_ = []

            if(len(row_)>0):
                plots.append(row_)
                row_ = []

            grid = gridplot(plots)

            cb5.markdown("##### Shot XRay | {}".format(team_selector))
            cb5.write("""
             O gráfico abaixo mostra a posição das finalizações dos TOP3 Jogadores em xG/90 acumulados.
             As bolas verdes indicam finalizações que resultaram em gols e o polígono convexo atribui um cerco com 80% da mediana de posição de finalização.  Destacamos a % dos chutes por faixas verticais do campo

            """)
            cb5.bokeh_chart(grid)


    if(flag_plot_campeonato_conteiner):

        container_campeonato.markdown("### 🌎 Análise da Liga | {}".format(league_selection))

        p2 = get_plot_scout_decisivo(df_data, df_liga, alpha_y=1.75, photo_height=30)
        container_campeonato.bokeh_chart(p2)

        p3 = get_goleiro_plot(df_data, df_liga, alpha_y = 1.6, photo_height = main_parameters_dict[league_selection]['rodada_atual']*0.9, legend_desloc = 1.6, w_ = 0.9)
        p3.plot_width = 1600
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
