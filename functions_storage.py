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


def get_opponent(t,h,a):
    if(t==h):
        return a
    elif(t==a):
        return h

def apply_sg(pos,sg_f):
    if (pos=='D' or pos=='G'):
        return sg_f
    else:
        return 0

def get_local(t,h,a):
    if(t==h):
        return 'CASA'
    elif(t==a):
        return 'FORA'


def classify_score(score):
    if(score<0):
        return "F"
    elif(score>=0 and score<3):
        return "E"
    elif(score>=3 and score<5):
        return "C"
    elif(score>=5 and score<8):
        return "B"
    elif(score>=8):
        return "A"

def get_rod_num(x):
    return int(x[-2:])


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_most_similar(nome, df_scout):
    df_aux = df_scout
    df_aux['comp'] = nome

    df_aux['ratio_'] = df_aux.apply(lambda x: similar(x['apelido'], x['comp']), axis=1)

    df_aux = df_aux.sort_values(by='ratio_').tail(1)

    return df_aux.atleta_id.tolist()[0]


def generate_df_data_dataframe(rodada_atual_, country_of_league, year):

    if(country_of_league=='England'):
        code = 39
    elif(country_of_league=='Germany'):
        code = 78
    elif(country_of_league=='Italy'):
        code = 135
    elif(country_of_league=='Spain'):
        code = 140
    elif(country_of_league=='Paulista'):
        code = 475
    elif(country_of_league=='Champions'):
        code = 2
    elif(country_of_league=='France'):
        code = 61
    elif(country_of_league=='Serie B'):
        code = 72
    elif(country_of_league=='Serie A'):
        code = 71
    elif(country_of_league=='Portugal'):
        code = 94
    else:
        code = -1

    path = r"\local_dbs\{0}\{1}_dados_jogadores.csv".format(country_of_league, country_of_league.lower())
    path_old_partidas = r"\local_dbs\{0}\{1}_dados_partidas.csv".format(country_of_league, country_of_league.lower())
    path_old_predictions = r"\local_dbs\{0}\{1}_dados_bets.csv".format(country_of_league, country_of_league.lower())
    #if Path(path).is_file():
    df_data = pd.read_csv(path).drop('Unnamed: 0',axis=1)
    df_liga = pd.read_csv(path_old_partidas).drop('Unnamed: 0',axis=1)
    #if Path(path_old_predictions).is_file():
    df_pred_players = pd.read_csv(path_old_predictions).drop('Unnamed: 0',axis=1)

    df_data = df_data.fillna(0)

    try:
        df_data['passes_incompletos'] = df_data['passes'] - df_data['acuracia_passes']
    except:
        df_data['acuracia_passes'] = [int(x.split("%")[0]) for x in df_data['acuracia_passes'].tolist()]
        df_data['passes_incompletos'] = (100-df_data['acuracia_passes'])*df_data['passes']/100

    df_data['passes_incompletos'] = [math.ceil(x) for x in df_data['passes_incompletos'].tolist()]

    df_data['chutes_no_gol'] = df_data['chutes_no_gol'] - df_data['gols']

    df_data['chutes_fora'] = df_data['chutes'] - df_data['chutes_no_gol']

    df_data['desarmes_executado'] = df_data['desarmes_executado'] + 0*df_data['desarmes_cedidos'] + 1*df_data['divididas_vencidas']/2

    df_data['desarmes_executado'] = [math.ceil(x) for x in df_data['desarmes_executado'].tolist()]

    df_data['prev_cartola'] = df_data['chutes_fora']*0.8 + df_data['chutes_no_gol']*1.2 + df_data['impedimentos']*(-0.5) + df_data['gols']*8 + df_data['gols_sofridos']*(-2) + df_data['assistencias']*5 + df_data['defesas_no_gol']*2 + df_data['passes_incompletos']*(-0.1)

    df_data['prev_cartola'] = df_data['prev_cartola'] + df_data['desarmes_executado']*1 + df_data['faltas_sofridas']*(0.5) + df_data['faltas_cometidas']*(-0.5) + df_data['cartao_amarelo']*(-2) + df_data['cartao_vermelho']*(-5)

    df_data['positive'] = df_data['chutes_fora']*0.8 + df_data['chutes_no_gol']*1.2 + df_data['gols']*8 + df_data['assistencias']*5 + df_data['defesas_no_gol']*2+ df_data['desarmes_executado']*1 + df_data['faltas_sofridas']*(0.5)

    df_data['negative'] = df_data['prev_cartola'] - df_data['positive']

    df_data['negative'] = -df_data['negative']

    df_sg = df_data.groupby(['fix_id','team_id'])['gols_sofridos'].sum().reset_index()

    df_sg['SG'] = [1 if x == 0 else 0 for x in df_sg['gols_sofridos'].tolist()]

    df_data = pd.merge(df_data,df_sg[['fix_id','team_id','SG']], how='left',on=['fix_id','team_id'])

    df_data['SG'] = df_data.apply(lambda x: apply_sg(x['player_position'],x['SG']), axis = 1)

    df_data['prev_cartola'] = df_data['prev_cartola'] + df_data['SG']*5 + df_data['penalti_defendido']*7

    depara_round = df_liga[['id_','round_','date','referee']]

    de_para_time_logo = df_liga[['home_team_nome','home_team_logo']].drop_duplicates(subset=['home_team_nome','home_team_logo'],keep='first')

    df_data = pd.merge(df_data, df_liga, how='left', left_on='fix_id', right_on='id_')

    df_data['oponente_id'] = df_data.apply(lambda x: get_opponent(x['team_id'],x['home_team_id'],x['away_team_id']), axis=1)
    df_data['local_'] = df_data.apply(lambda x: get_local(x['team_id'],x['home_team_id'],x['away_team_id']), axis=1)

    df_data['pontos_por_minuto'] = df_data['prev_cartola']/df_data['minutos_jogados']

    df_data['class_score'] = df_data.apply(lambda x: classify_score(x['prev_cartola']), axis=1)

    df_data['foi_substituido_'] = [1 if x else 0 for x in df_data['foi_substituido'].tolist()]

    df_data = df_data.fillna(0)

    df_data['oponente_id'] = df_data.apply(lambda x: get_opponent(x['team_id'],x['home_team_id'],x['away_team_id']), axis=1)
    df_data['local_'] = df_data.apply(lambda x: get_local(x['team_id'],x['home_team_id'],x['away_team_id']), axis=1)

    df_data['pontos_por_minuto'] = df_data['prev_cartola']/df_data['minutos_jogados']

    df_data['class_score'] = df_data.apply(lambda x: classify_score(x['prev_cartola']), axis=1)

    depara_rod = df_data[['round_','date']].sort_values(by=['date']).drop_duplicates(subset='round_', keep='first')

    df_liga['date'] = pd.to_datetime(df_liga['date'])
    df_liga['date'] = [x.tz_localize(None) for x in df_liga['date'].tolist()]

    df_liga['rod_num'] = df_liga.apply(lambda x: get_rod_num(x['round_']), axis=1)

    rt = df_liga[(df_liga['status_short']=='FT') & (df_liga['date']>=pd.to_datetime(date.today()))].groupby(['rod_num'])['id_'].count().reset_index()

    rodada_atual_l = rt['rod_num'].tolist()

    if(len(rodada_atual_l)>0):
        rodada_atual = rodada_atual_l[-1]+1
    else:
        rodada_atual = min(max(df_liga[(df_liga['status_short']=='FT')]['rod_num'].unique().tolist()) + 1,
                           max(df_liga['rod_num'].unique().tolist()))

    rodada_atual = rodada_atual_

    df_r = df_liga[df_liga['rod_num']==rodada_atual]
    fix_ids_list = df_r.id_.tolist()

    dict_local = {}
    for t in df_r['home_team_id']:
        dict_local[t] = 'CASA'

    for t in df_r['away_team_id']:
        dict_local[t] = 'FORA'

    df_data['curr_rod_exec'] = df_data['team_id'].map(dict_local)
    df_data['curr_rod_ced'] = df_data['oponente_id'].map(dict_local)

    df_data['curr_rod_exec_code'] = [1 if x==y else 0 for x,y in zip(df_data['local_'].tolist(), df_data['curr_rod_exec'].tolist())]

    df_data['curr_rod_ced_code'] = [1 if x!=y else 0 for x,y in zip(df_data['local_'].tolist(), df_data['curr_rod_ced'].tolist())]

    df_data['player_grade'] = [float(x) if x!='–' else 0 for x in df_data['player_grade'].tolist()]

    df_data['rod_num'] = df_data.apply(lambda x: get_rod_num(x['round_']), axis=1)

    depara = df_data[['team_id','team_name']].drop_duplicates()

    depara.columns = ['oponente_id','oponente_name']

    df_data = pd.merge(df_data, depara, how='left', on='oponente_id')

    return df_data, df_liga

def get_pontuacao_relativa(df_liga):
    jogos_ = df_liga[df_liga['status_short']=='FT']
    jogos_['rod_num'] = jogos_.apply(lambda x: get_rod_num(x['round_']), axis=1)

    jogos_['diff_goals_home'] = jogos_['home_team_goals'] - jogos_['away_team_goals']
    jogos_['points_home'] = [3 if x>0 else 1 if x==0 else 0 for x in jogos_['diff_goals_home'].tolist()]
    jogos_['points_away'] = [3 if x==0 else 1 if x==1 else 0 for x in jogos_['points_home'].tolist()]

    lista_jogos = jogos_['rod_num'].unique().tolist()

    lista_times = jogos_['home_team_id'].unique().tolist()

    dataframe_evol_pontos = pd.DataFrame(columns=['rod_num','team_id','team_name','points'])

    for test_ in lista_times:
        dataframe_evol_pontos_aux = pd.DataFrame(columns=['rod_num','team_id','team_name','points'])
        resp = []
        nome = jogos_[(jogos_['home_team_id']==test_)].home_team_nome.unique().tolist()[0]
        for rd_ in lista_jogos:
            total_pontos = jogos_[(jogos_['home_team_id']==test_) & (jogos_['rod_num']<=rd_)].points_home.sum() + jogos_[(jogos_['away_team_id']==test_) & (jogos_['rod_num']<=rd_)].points_away.sum()
            resp.append(total_pontos)
        dataframe_evol_pontos_aux['rod_num'] = lista_jogos
        dataframe_evol_pontos_aux['team_id'] = test_
        dataframe_evol_pontos_aux['team_name'] = nome
        dataframe_evol_pontos_aux['points'] = resp

        dataframe_evol_pontos = dataframe_evol_pontos.append(dataframe_evol_pontos_aux)

    dataframe_pontuacao_relativa = pd.DataFrame(columns = ['rod_num', 'team_id', 'team_name', 'points', 'points_relative'])
    for rd_ in lista_jogos:
        relative_df = dataframe_evol_pontos[dataframe_evol_pontos['rod_num']==rd_]
        relative_df['points_relative'] = relative_df['points']/relative_df['points'].max()
        dataframe_pontuacao_relativa = dataframe_pontuacao_relativa.append(relative_df)

    return dataframe_pontuacao_relativa

def get_player_dispersion(df_data, df_liga, team_name, param_size = 0.05, min_num_jogos=2):

    de_para_time_logo = df_liga[['home_team_nome','home_team_logo']].drop_duplicates(subset=['home_team_nome','home_team_logo'],keep='first')

    pl = df_data[df_data['player_grade']>0].groupby('player_id')['round_'].count().reset_index()
    pl = pl[pl['round_']>=min_num_jogos]['player_id'].tolist()
    df_scout = df_data[(df_data['player_grade']>0) &
                       (df_data['player_id'].isin(pl))].groupby(['player_id',
                                    'team_name',
                                     'player_name',
                                     'player_photo']).agg(media = ('prev_cartola','mean'),
                                                          std = ('prev_cartola','std')).reset_index(
                                                            ).fillna(0).sort_values(by='media', ascending=False)

    df_scout = pd.merge(df_scout,de_para_time_logo,how='left',left_on='team_name', right_on='home_team_nome')

    df_scout['home_team_logo'] = ['' if y != team_name else x for x,y in zip(df_scout.player_photo.tolist(),
                                                                             df_scout.team_name.tolist())]

    p = figure(plot_width=600, plot_height=600, output_backend="webgl")
    sc1 = 'media'
    sc2 = 'std'

    max_x = max(df_scout[sc1].tolist())
    max_y = max(df_scout[sc2].tolist())

    med_x = df_scout[sc1].mean()
    med_y = df_scout[sc2].mean()


    source = ColumnDataSource(df_scout)


    p.cross(x=sc1, y=sc2,
             color='navy', fill_alpha=0.1, source=source)

    p.add_layout(Title(text="Dispersão dos Jogadores - Destaque {0}".format(team_name.upper()), align="center"), "above")

    s_ = min(param_size*(max_x), param_size*(max_y))
    image2 = ImageURL(url="home_team_logo", x=sc1, y=sc2, w=s_, h =s_, anchor="center")
    p.add_glyph(source, image2)

    med_cedido = Span(location=med_x,
                                      dimension='height', line_color='green',
                                      line_dash='dashed', line_width=3)
    p.add_layout(med_cedido)

    med_exec = Span(location=med_y,
                                  dimension='width', line_color='green',
                                  line_dash='dashed', line_width=3)
    p.add_layout(med_exec)


    q_1 = BoxAnnotation(top=med_y, right=med_x, fill_alpha=0.1, fill_color='blue')
    q_2 = BoxAnnotation(bottom=med_y, left = med_x, fill_alpha=0.1, fill_color='yellow')

    q_3 = BoxAnnotation(top=med_y, left=med_x, fill_alpha=0.1, fill_color='green')
    q_4 = BoxAnnotation(bottom=med_y, right = med_x, fill_alpha=0.1, fill_color='red')

    p.add_layout(q_1)
    p.add_layout(q_2)
    p.add_layout(q_3)
    p.add_layout(q_4)
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.yaxis.axis_label = "Oscilação na Pontuação do Cartola"
    p.xaxis.axis_label = "Média na Pontuação do Cartola"

    return p


def get_plot_scout_decisivo(df_data,df_liga, alpha_y=1.4, photo_height=30):

    de_para_time_logo = df_liga[['home_team_nome','home_team_logo']].drop_duplicates(subset=['home_team_nome','home_team_logo'],keep='first')

    df_decisivo = df_data[(df_data['player_grade']>0) & (df_data['player_position'].isin(['M','F','D']))].groupby(['player_id','player_photo','team_name','team_id']).agg(
        player_name = ('player_name','first'),
        min_ = ('minutos_jogados','sum'),
        gols_ = ('gols','sum'),
        assist_ = ('assistencias','sum')
    ).reset_index()

    df_decisivo = df_decisivo[(df_decisivo['gols_']>0) | (df_decisivo['assist_']>0)]

    df_decisivo['r'] = (df_decisivo['gols_'] + df_decisivo['assist_'])/(df_decisivo['min_'])

    df_decisivo['1_r'] = 1/df_decisivo['r']

    df_mais_decisivos_media = df_decisivo[df_decisivo['min_']>=df_decisivo['min_'].mean()].sort_values(by='1_r').head(12)
    df_mais_decisivos_media['1_r'] = df_mais_decisivos_media['1_r'].astype(int)

    df_mais_decisivos_media = pd.merge(df_mais_decisivos_media,de_para_time_logo,how='left',left_on='team_name', right_on='home_team_nome')

    df_mais_decisivos_media['label_pos'] = [x + photo_height for x in df_mais_decisivos_media['1_r'].tolist()]

    df_mais_decisivos_media['label_pos_clube'] = [x + photo_height*2 for x in df_mais_decisivos_media['1_r'].tolist()]

    df_mais_decisivos_media['label_pos_data'] = [x*0.9 for x in df_mais_decisivos_media['1_r'].tolist()]
    df_mais_decisivos_media['label_data'] = ['{0:.0f}'.format(x) for x in df_mais_decisivos_media['1_r'].tolist()]

    df_mais_decisivos_media['label_pos_data_2'] = [x for x in df_mais_decisivos_media.gols_.tolist()]
    df_mais_decisivos_media['label_data_2'] = ['{0:.0f}'.format(x) for x in df_mais_decisivos_media['gols_'].tolist()]

    df_mais_decisivos_media['label_pos_data_3'] = [x for x in df_mais_decisivos_media.assist_.tolist()]
    df_mais_decisivos_media['label_data_3'] = ['{0:.0f}'.format(x) for x in df_mais_decisivos_media['assist_'].tolist()]

    source = ColumnDataSource(df_mais_decisivos_media)
    f = figure(x_range=df_mais_decisivos_media.player_name.tolist(),
                        y_range=Range1d(0,max(df_mais_decisivos_media['1_r'].tolist())*alpha_y),
                    plot_height=600,
                    plot_width = 1200)


    b1 = f.vbar(x='player_name', bottom=0, top='1_r', width=0.5, source=source, color='#FFCD58')
    f.hex(x='player_name', y='label_pos_data', size=40, source=source, color='#010100', fill_alpha=0.5)
    f.text(x='player_name', y='label_pos_data', source=source, text='label_data', x_offset=-12, y_offset=+10, text_color='white')

    f.yaxis.axis_label = "Minutos até Scout Decisivo"

    image2 = ImageURL(url="player_photo", x="player_name", y='label_pos', w=0.8, h = photo_height, anchor="center")
    f.add_glyph(source, image2)

    image2 = ImageURL(url="home_team_logo", x="player_name", y='label_pos_clube', w=0.8, h = photo_height, anchor="center")
    f.add_glyph(source, image2)

    # Setting the second y axis range name and range
    l_2 = df_mais_decisivos_media['gols_'].tolist()
    f.extra_y_ranges = {"foo": Range1d(start=min(l_2)*0.5, end=max(l_2)*4)}

    # Setting the rect glyph params for the second graph.
    # Using the aditional y range named "foo" and "right" y axis here.
    l1 = f.line(x='player_name', y='gols_',color="green", line_dash='dashed',line_width=2 ,y_range_name="foo", source=source)

    f.circle(x='player_name', y='gols_', size=5, source=source, color='#010100', fill_alpha=0.5, y_range_name="foo")

    f.text(x='player_name', y='label_pos_data_2', source=source, text='label_data_2', x_offset=-10, y_range_name="foo")

    l2 = f.line(x='player_name', y='assist_',color="black",line_dash='dashed', y_range_name="foo", source=source)

    f.circle(x='player_name', y='assist_', size=5, source=source, color='#010100', fill_alpha=0.5, y_range_name="foo")

    f.text(x='player_name', y='label_pos_data_3', source=source, text='label_data_3', x_offset=-10, y_range_name="foo")

    legend = Legend(items=[(fruit, [r]) for (fruit, r) in zip(['Minutos até scout decisivo','Gols','Assistências'], [b1,l1,l2])], location=(10, 100))
    f.add_layout(legend, 'right')

    f.xaxis.major_label_orientation = math.pi/4

    f.ygrid.grid_line_color = None

    return f

def get_goleiro_plot(df_data, df_liga, alpha_y = 2, photo_height = 20, legend_desloc = 1.8, w_ = 0.5):

    de_para_time_logo = df_liga[['home_team_nome','home_team_logo']].drop_duplicates(subset=['home_team_nome','home_team_logo'],keep='first')


    df_goleiros = df_data[(df_data['player_position']=='G') & (df_data['player_grade']>0)].groupby(['player_id','player_name','player_photo','team_name']).agg(dds = ('defesas_no_gol','sum'),
                                                                                      gs = ('gols_sofridos','sum'),
                                                                                      round_nums = ('round_','count')
                                                                                     ).reset_index().sort_values(by='round_nums', ascending=False).drop_duplicates(subset=['team_name'], keep='first')

    df_goleiros['dds_gs'] = df_goleiros['dds']/df_goleiros['gs']

    df_goleiros['label_pos'] = [x+photo_height for x in df_goleiros.dds.tolist()]

    df_goleiros['label_pos_clube'] = [x+2.5*photo_height for x in df_goleiros.dds.tolist()]

    df_goleiros['label_pos_data'] = [x-1 for x in df_goleiros.dds.tolist()]

    df_goleiros['label_data'] = ['{0:.0f}'.format(x) for x in df_goleiros['dds'].tolist()]

    df_goleiros = pd.merge(df_goleiros,de_para_time_logo,how='left',left_on='team_name', right_on='home_team_nome')

    df_goleiros['label_pos_data_2'] = [x-legend_desloc for x in df_goleiros.dds_gs.tolist()]

    df_goleiros['label_data_2'] = ['{0:.1f}'.format(x) for x in df_goleiros['dds_gs'].tolist()]

    df_goleiros = df_goleiros.sort_values(by='dds', ascending=False)

    source = ColumnDataSource(df_goleiros)
    f = figure(x_range=df_goleiros.team_name.tolist(),
                        y_range=Range1d(0,max(df_goleiros.dds.tolist())*alpha_y),
                    plot_height=600,
                    plot_width = 1200)


    f.vbar(x='team_name', bottom=0, top='dds', width=0.5, source=source, color='#FFCD58')
    f.text(x='team_name', y='label_pos_data', source=source, text='label_data', x_offset=-10)
    f.yaxis.axis_label = "Quantidade de Defesas"

    image2 = ImageURL(url="player_photo", x="team_name", y='label_pos', w=w_, h = photo_height, anchor="center")
    f.add_glyph(source, image2)

    image2 = ImageURL(url="home_team_logo", x="team_name", y='label_pos_clube', w=w_, h = photo_height, anchor="center")
    f.add_glyph(source, image2)

    # Setting the second y axis range name and range
    f.extra_y_ranges = {"foo": Range1d(start=0, end=max(df_goleiros.dds_gs.tolist())*5)}

    # Adding the second axis to the plot.
    f.add_layout(LinearAxis(y_range_name="foo", axis_label='Média de Defesas por Gol Sofridos'), 'right')

    # Setting the rect glyph params for the second graph.
    # Using the aditional y range named "foo" and "right" y axis here.
    f.line(x='team_name', y='dds_gs',color="green", y_range_name="foo", source=source)

    f.hex(x='team_name', y='dds_gs', size=10, source=source, color='#010100', fill_alpha=0.5, y_range_name="foo")

    f.text(x='team_name', y='label_pos_data_2', source=source, text='label_data_2', x_offset=-10, y_range_name="foo")



    f.xaxis.major_label_orientation = math.pi/4

    f.ygrid.grid_line_color = None
    return f

def get_sg_plot(df_data, df_liga, local):

    de_para_time_logo = df_liga[['home_team_nome','home_team_logo']].drop_duplicates(subset=['home_team_nome','home_team_logo'],keep='first')

    df_sg = df_data[df_data['local']==local].groupby(['team_id', 'team_name','round_'])['SG'].sum().reset_index()
    df_sg['SG'] = ['SG_' if x > 0 else 'NSG_' for x in df_sg['SG'].tolist()]

    df_sg = df_sg.drop(['round_'], axis=1).groupby(['team_name','SG']).size().reset_index().pivot('team_name','SG',0).reset_index().fillna(0)

    df_sg['tot_'] = df_sg['NSG_'] + df_sg['SG_']

    df_sg['label_pos_top'] = df_sg['NSG_'] + df_sg['SG_']/2

    df_sg['label_pos_bottom'] = df_sg['NSG_']/2

    df_sg['perc_'] = df_sg['SG_']/df_sg['tot_']

    df_sg = df_sg.sort_values(by='perc_', ascending=False)

    df_sg['label_pos_perc'] = max(df_sg.tot_.tolist())*1.1

    df_sg['perc_label'] = ['{0:.0f}%'.format(x*100) for x in df_sg['perc_'].tolist()]

    df_sg['label_pos_image'] = max(df_sg.tot_.tolist())*1.38

    df_sg = pd.merge(df_sg,de_para_time_logo,how='left',left_on='team_name', right_on='home_team_nome')

    source = ColumnDataSource(df_sg)
    f = figure(x_range=df_sg.team_name.tolist(),
                    y_range=Range1d(0,max(df_sg.tot_.tolist())*1.7),
                    plot_height=500,
                    plot_width = 1300)

    s = f.vbar(x='team_name', bottom=0, top='NSG_', width=0.5, source=source, color='#FFCD58')
    p = f.vbar(x='team_name', bottom='NSG_', top='tot_', width=0.5, source=source, color='#FF5C4D')


    f.hex(x='team_name', y='label_pos_bottom', size=40, source=source, color='#010100', fill_alpha=0.5)
    s_label = f.text(x='team_name', y='label_pos_bottom',
                     source=source, text='NSG_', x_offset=-8, y_offset = 10, text_color='white', text_font_size='15pt')

    f.hex(x='team_name', y='label_pos_top', size=40, source=source, color='#010100', fill_alpha=0.5)
    p_label = f.text(x='team_name',
                     y='label_pos_top',
                     source=source, text='SG_', x_offset=-8, y_offset = 10, text_color='white', text_font_size='15pt')

    f.text(x='team_name', y='label_pos_perc', source=source, text='perc_label', x_offset=-15, text_font_size='16pt')

    image2 = ImageURL(url="home_team_logo", x="team_name", y='label_pos_image', w=0.95, h = df_sg['tot_'].max()/4, anchor="center")
    f.add_glyph(source, image2)

    f.xaxis.major_label_orientation = math.pi/4

    f.ygrid.grid_line_color = None
    f.xgrid.grid_line_color = None
    f.yaxis.minor_tick_line_width = 0
    f.yaxis.major_tick_line_width = 0
    f.yaxis.minor_tick_out = 1
    f.ygrid.grid_line_color = None
    f.outline_line_color = None
    f.axis.axis_line_color=None
    f.yaxis.major_label_text_font_size = '0pt'
    f.xaxis.major_label_text_font_size = '0pt'

    legend = Legend(items=[(fruit, [r]) for (fruit, r) in zip(['Sem SG','Com SG'], [s,p])], location=(10, 100))
    #f.add_layout(legend, 'right')

    f.add_layout(Title(text='% de SG jogando em {0}'.format(local.upper()), align="center"), "above")

    return f

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS_FAST = [
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
    ]

    DISTRIBUTIONS_SLOW = [
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS_FAST:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

def get_round_plot(df_data, tid,
                   scout1='Avaliação Jogador',
                   scout2='Nota Cartola Jogador',
                   sc1 = 'player_grade',
                   sc2 = 'prev_cartola',
                   min_num_jogos=10):

    pl = df_data.groupby('player_id')['round_'].count().reset_index()
    pl = pl[pl['round_']>=min_num_jogos]['player_id'].tolist()
    df_bayern = df_data[(df_data['team_id']==tid) & (df_data['player_id'].isin(pl)) & (df_data['player_grade']>0)].groupby(['player_id','player_name','player_photo']).agg(pg_ = (sc1, 'mean'),
                                                                                            pc_ = (sc2,'std'),
                                                                                             nj_ = ('round_','count')).reset_index().sort_values(by='pc_', ascending=False).head(20).sort_values(by='pg_', ascending=False).reset_index().head(10)

    depara_clube_logo = df_data[['home_team_logo','home_team_id','home_team_nome']].drop_duplicates(keep='first')
    logo = depara_clube_logo[depara_clube_logo['home_team_id']==tid]['home_team_logo']
    nome = depara_clube_logo[depara_clube_logo['home_team_id']==tid]['home_team_nome'].tolist()[0]


    drug_color = OrderedDict([
        ("pg_", "#0d3362"),
        ("pc_", "#c64737")
    ])

    gram_color = OrderedDict([
    ("negative", "#e69584"),
    ("positive", "#aeaeb8"),
    ])


    df = df_bayern.copy()

    width = 650
    height = 650
    inner_radius = 30
    outer_radius = inner_radius + 15

    minr = sqrt(log(0.1 * 1E4))
    maxr = sqrt(log(1000 * 1E4))
    a = (outer_radius - inner_radius) / (minr - maxr)
    b = inner_radius - a * maxr

    def rad(mic):
        return mic + inner_radius

    big_angle = 2.0 * np.pi / (len(df) + 1)
    small_angle = big_angle / 5

    p = figure(plot_width=width, plot_height=height, title="",
        x_axis_type=None, y_axis_type=None,
        x_range=(-60, 60), y_range=(-60, 60),
        min_border=0, outline_line_color="black",
        background_fill_color=None)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # annular wedges
    angles = np.pi/2 - big_angle/2 - df.index.to_series()*big_angle
    colors = ["#F7CF88" for gram in df.player_id]
    p.annular_wedge(
        0, 0, inner_radius, outer_radius, -big_angle+angles, angles, color=colors,
    )

    # small wedges
    p.annular_wedge(0, 0, inner_radius, rad(df.pg_),
                    -big_angle+angles+3*small_angle, -big_angle+angles+4*small_angle,
                    color=drug_color['pg_'])

    p.annular_wedge(0, 0, inner_radius,rad(df.pc_),
                    -big_angle+angles+1*small_angle, -big_angle+angles+2*small_angle,
                    color=drug_color['pc_'])


    # circular axes and lables
    labels = np.power(10.0, np.arange(1, 2))
    radii = []
    radii.append(inner_radius + 10)
    p.circle(0, 0, radius=inner_radius + 10, fill_color=None, line_color="white")
    p.text(0, inner_radius + 10, ['10'],
           text_font_size="11px", text_align="center", text_baseline="middle")

    source_ = ColumnDataSource(dict(
        url = [r"local_dbs\pictures\Imagem1.png"],
        x_  = [0],
        y_  = [inner_radius + 10]
    ))

    image3 = ImageURL(url='url', x='x_', y='y_', w=14, h = 14, anchor="center")
    p.add_glyph(source_, image3)

    source_l = ColumnDataSource(dict(
        url = [logo],
        x_  = [0],
        y_  = [0]
    ))

    image4 = ImageURL(url='url', x='x_', y='y_', w=14, h = 14, anchor="center")
    p.add_glyph(source_l, image4)

    p.text(0, inner_radius + 20, ['Cartologia'],
           text_font_size="20px", text_align="center", text_baseline="middle")



    # radial axes
    p.annular_wedge(0, 0, inner_radius-5, outer_radius+10,
                    -big_angle+angles, -big_angle+angles, color="black")

    # bacteria labels
    xr = radii[0]*np.cos(np.array(-big_angle/2 + angles))*1.3
    yr = radii[0]*np.sin(np.array(-big_angle/2 + angles))*1.3
    label_angle=np.array(-big_angle/2+angles)
    label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
    p.text(xr, yr, df.player_name, angle=label_angle,
           text_font_size="12px", text_align="center", text_baseline="middle")

    xr = radii[0]*np.cos(np.array(-big_angle/2 + angles+1*small_angle))*1.3
    yr = radii[0]*np.sin(np.array(-big_angle/2 + angles+1*small_angle))*1.3
    label_angle=np.array(-big_angle/2+angles+1*small_angle)
    label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side

    source = ColumnDataSource(dict(
        url = df.player_photo.tolist(),
        x_  = xr,
        y_  = yr,
        angle_ = label_angle
    ))

    image2 = ImageURL(url='url', x='x_', y='y_', w=7, h = 7, anchor="center")
    p.add_glyph(source, image2)

    # bacteria labels
    xr = rad(df.pg_)*np.cos(np.array(-big_angle+angles+3.5*small_angle))*1.15
    yr = rad(df.pg_)*np.sin(np.array(-big_angle+angles+3.5*small_angle))*1.15
    label_angle=np.array(-big_angle+angles+3.5*small_angle)
    label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
    p.text(xr, yr, ['{0:.1f}'.format(x) for x in df.pg_.tolist()], angle=label_angle,
           text_font_size="18px", text_align="center", text_baseline="middle")


    # bacteria labels
    xr = rad(df.pc_)*np.cos(np.array(-big_angle+angles+1.5*small_angle))*1.15
    yr = rad(df.pc_)*np.sin(np.array(-big_angle+angles+1.5*small_angle))*1.15
    label_angle=np.array(-big_angle+angles+1.5*small_angle)
    label_angle[label_angle < -np.pi/2] += np.pi # easier to read labels on the left side
    p.text(xr, yr, ['{0:.1f}'.format(x) for x in df.pc_.tolist()], angle=label_angle,
           text_font_size="18px", text_align="center", text_baseline="middle")


    # OK, these hand drawn legends are pretty clunky, will be improved in future release
    p.circle([-20, -20], [-370, -390], color=list(gram_color.values()), radius=5)
    p.text([-40, -30], [-370, -390], text=["Gram-" + gr for gr in gram_color.keys()],
           text_font_size="9px", text_align="left", text_baseline="middle")

    p.rect([0, 0], [10, -10], width=35, height=5,
           color=list(drug_color.values()))
    p.text([-10, -15], [10, -10], text=[scout1, scout2], text_color='white',
           text_font_size="12px", text_align="left", text_baseline="middle")

    p.outline_line_color = None
    p.background_fill_color = 'white'
    p.border_fill_color = 'white'
    p.outline_line_color = 'white'


    return p

def get_mapa_confronto(df_data, rodada_atual, df_liga, pos__):
    pos_ = [pos__]
    df_r = df_liga[df_liga['rod_num']==rodada_atual]
    df_pconquistado = df_data[(df_data['player_grade']>0) &
                              (df_data['rod_num']>=(rodada_atual-10)) &
                             (df_data['player_position'].isin(pos_))].groupby('team_id').agg(pontos_conquistados = ('prev_cartola','mean')
                                                                                                    ).reset_index()

    df_pcedido = df_data[(df_data['player_grade']>0) &
                         (df_data['rod_num']>=(rodada_atual-10)) &
                        (df_data['player_position'].isin(pos_))].groupby('oponente_id').agg(pontos_cedidos = ('prev_cartola','mean')
                                                                                                    ).reset_index()
    #limite = df_pcedido.pontos_cedidos.mean() + 1.5*df_pcedido.pontos_cedidos.std()

    #df_pcedido['pontos_cedidos'] = [limite if x>= limite else x for x in df_pcedido['pontos_cedidos'].tolist()]


    df_1 = df_r[['home_team_id','away_team_id']]

    df_2 = df_1[['away_team_id','home_team_id']]

    df_1.columns = df_2.columns = ['team_id','oponente_id']

    df_confrontos = df_1.append(df_2)

    df_confrontos = pd.merge(df_confrontos, df_pconquistado, how='left', on='team_id')

    df_confrontos = pd.merge(df_confrontos, df_pcedido, how='left', on='oponente_id')

    df_confrontos = pd.merge(df_confrontos,
                        df_liga[['home_team_id','home_team_logo']].drop_duplicates(),
                        how='left',left_on='team_id', right_on='home_team_id')
    pt_ = 0.1

    p = figure(plot_width=500, plot_height=700, output_backend="webgl")
    source = ColumnDataSource(df_confrontos)

    #p.circle(x='pontos_cedidos', y='pontos_conquistados', radius=0.1,
    #         color='navy', fill_alpha=0.1, source=source)

    pos_t = pos_[0].upper()

    title_pos = ""
    if(pos__=='D'):
        title_pos = 'Defensores'
    elif(pos__=='M'):
        title_pos = 'Meio Campo'
    elif(pos__=='F'):
        title_pos = 'Atacante'
    elif(pos__=='G'):
        title_pos = 'Goleiros'

    p.add_layout(Title(text=" - " + str(title_pos) + " - ", align="center"), "above")

    #p.add_layout(Title(text="{0}".format(pos_t), align="center"), "above")
    defl = max(df_confrontos['pontos_cedidos'].tolist())/max(df_confrontos['pontos_conquistados'].tolist())
    if(defl<=1):
        defl = 1
    image2 = ImageURL(url="home_team_logo", x='pontos_cedidos', y='pontos_conquistados',
                      w=max(df_confrontos['pontos_cedidos'].tolist())*pt_*(defl), h = max(df_confrontos['pontos_conquistados'].tolist())*pt_,anchor="center")

    p.add_glyph(source, image2)

    p.yaxis.axis_label = 'Pontuação Conquistada no Local de Jogo da Rodada'
    p.xaxis.axis_label = 'Pontuação Cedida pelo Adversário no Local de Jogo da Rodada'

    med_x = df_confrontos['pontos_cedidos'].mean()
    med_y = df_confrontos['pontos_conquistados'].mean()

    med_cedido = Span(location=med_x,
                                  dimension='height', line_color='green',
                                  line_dash='dashed', line_width=3)
    p.add_layout(med_cedido)

    med_exec = Span(location=med_y,
                                  dimension='width', line_color='green',
                                  line_dash='dashed', line_width=3)
    p.add_layout(med_exec)


    q_1 = BoxAnnotation(top=med_y, right=med_x, fill_alpha=0.1, fill_color='red')
    q_2 = BoxAnnotation(bottom=med_y, left = med_x, fill_alpha=0.1, fill_color='green')

    q_3 = BoxAnnotation(top=med_y, left=med_x, fill_alpha=0.1, fill_color='blue')
    q_4 = BoxAnnotation(bottom=med_y, right = med_x, fill_alpha=0.1, fill_color='yellow')

    p.add_layout(q_1)
    p.add_layout(q_2)
    p.add_layout(q_3)
    p.add_layout(q_4)

    p.background_fill_color = 'white'
    p.border_fill_color = None
    return p


def get_pont_distribution(df_data, team_id_ = -1, player_id_=-1, dist_user = st.norm, flag_std=4):
    p = figure(title='Distribuição de Probabilidade',
               tools='',
               background_fill_color="#fafafa",
               plot_width=500,
               plot_height=500)
    dist = dist_user


    #Obter distribuição do Jogador
    if(team_id_>0 or player_id_>0):
        if(team_id_>0):

            mean_ = df_data[(df_data['team_id']==team_id_) & ((df_data['minutos_jogados']>0))]['prev_cartola'].mean()
            std_ = df_data[(df_data['team_id']==team_id_) & ((df_data['minutos_jogados']>0))]['prev_cartola'].std()

            ponts_player = df_data[(df_data['team_id']==team_id_) &
                                   ((df_data['minutos_jogados']>0)) &
                                   (abs(df_data['prev_cartola']-mean_)<=flag_std*std_)]['prev_cartola'].tolist()

            depara_clube_logo = df_data[['home_team_logo','home_team_id','home_team_nome']].drop_duplicates(keep='first')
            logo = depara_clube_logo[depara_clube_logo['home_team_id']==team_id_]['home_team_logo']

        elif(player_id_>0):

            mean_ = df_data[(df_data['player_id']==player_id_) & ((df_data['minutos_jogados']>0))]['prev_cartola'].mean()
            std_ = df_data[(df_data['player_id']==player_id_) & ((df_data['minutos_jogados']>0))]['prev_cartola'].std()

            ponts_player = df_data[(df_data['player_id']==player_id_) &
                                   ((df_data['minutos_jogados']>0))
                                   & (abs(df_data['prev_cartola']-mean_)<=flag_std*std_)]['prev_cartola'].tolist()

            depara_clube_logo = df_data[['player_id','player_photo','player_name']].drop_duplicates(keep='first')
            logo = depara_clube_logo[depara_clube_logo['player_id']==player_id_]['player_photo']
            nome = depara_clube_logo[depara_clube_logo['player_id']==player_id_]['player_name']

        hist, edges = np.histogram(ponts_player, density=True)
        data = ponts_player
        h1 = max(hist)

        params  = dist.fit(data)
        size=10000
        # Separate parts of parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Build PDF and turn into pandas Series
        x_p = np.linspace(int(min(edges)-1), int(max(edges)+1),size)
        y_p = dist.pdf(x_p, loc=loc, scale=scale, *arg)

        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="orange", line_color="white", alpha=0.5)

        p.line(x_p, y_p, line_color="red", line_width=2, alpha=0.7)


    if(player_id_>0):
        player_pos = df_data[(df_data['player_id']==player_id_)]['player_position'].unique().tolist()[0]
        list_ponts = df_data[(df_data['player_position']==player_pos) &
                             ((df_data['minutos_jogados']>0)) &
                            (df_data['minutos_jogados']!=player_id_)]['prev_cartola'].tolist()
    else:
        list_ponts = df_data[((df_data['minutos_jogados']>0)) & (df_data['team_id']!=team_id_)]['prev_cartola'].tolist()


    hist, edges = np.histogram(list_ponts, density=True, bins=50)
    h2 = max(hist)
    # Load data from statsmodels datasets
    data = list_ponts
    params  = dist.fit(data)
    size=10000
    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Build PDF and turn into pandas Series
    x = np.linspace(int(min(edges)-1), int(max(edges)+1),size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)


    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="gray", line_color="white", alpha=0.5)

    p.line(x, y, line_color="black", line_width=2, alpha=0.7)
    list_xs = [int(x) for x in range(int(min(edges)-1), int(max(edges)+1),3)]
    source_l = ColumnDataSource(dict(
            url = [logo],
            x_  = [list_xs[-3]],
            y_  = [max(h1,h2)*0.65]
        ))

    image4 = ImageURL(url='url', x='x_', y='y_', anchor="center")
    p.add_glyph(source_l, image4)

    if(player_id_>0):
            #p.rect(params[0] + 2.5, max(h1,h2), width=30, height=0.025, color="#0d3362", fill_alpha=0.5)
            p.text(list_xs[-3], max(h1,h2)*0.9, nome, x_offset = -30, text_color='#273746', text_font_style="bold")

    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'
    p.grid.grid_line_color="white"
    p.xaxis.ticker = [int(x) for x in range(int(min(edges)-1), int(max(edges)+1),3)]

    return p
