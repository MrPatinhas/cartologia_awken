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

stream.markdown("### 🎲 The Application")

#rodada_atual_, country_of_league, year
rodada_atual_ = 27
df_data, df_liga = generate_df_data_dataframe(rodada_atual_, 'Spain','2021')
dataframe_pontuacao_relativa = get_pontuacao_relativa(df_liga)

team_name_list = df_data.team_name.unique().tolist()
team_name_list = [""] + team_name_list

option = stream.selectbox(
'How would you like to be contacted?',
team_name_list
)

#if(option!=""):
#    p1 =  get_player_dispersion(df_data, df_liga, option, param_size = 0.1, min_num_jogos=2)
#    stream.bokeh_chart(p1)

p2 = get_plot_scout_decisivo(df_data, df_liga, alpha_y=1.75, photo_height=30)
stream.bokeh_chart(p2)

p3 = get_goleiro_plot(df_data, df_liga, alpha_y = 1.7, photo_height = 20, legend_desloc = 1.8, w_ = 0.9)
stream.bokeh_chart(p3)


p4_1 = get_sg_plot(df_data, df_liga,'CASA')
p4_1.background_fill_color = 'white'
p4_1.border_fill_color = 'white'
p4_1.outline_line_color = 'white'

p4_2 = get_sg_plot(df_data, df_liga,'FORA')
p4_2.background_fill_color = 'white'
p4_2.border_fill_color = 'white'
p4_2.outline_line_color = 'white'

grid_sg = gridplot([[p4_1], [p4_2]])
stream.bokeh_chart(grid_sg)

team_table = df_data[['home_team_id','home_team_nome']].drop_duplicates(keep='first')

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

stream.bokeh_chart(grid_confronto)

if(option=="Blocked"):
    tid__ = team_table[team_table['home_team_nome']==option].home_team_id.tolist()[0]

    p5 = get_round_plot(df_data, tid__,
                       scout1='Avaliação Jogador',
                       scout2='Nota Cartola Jogador',
                       sc1 = 'player_grade',
                       sc2 = 'prev_cartola',
                       min_num_jogos=10)

    stream.bokeh_chart(p5)

    p1 =  get_player_dispersion(df_data, df_liga, option, param_size = 0.1, min_num_jogos=2)
    stream.bokeh_chart(p1)
