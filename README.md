# cartologia_awken
Soccer Analysis Portal for main european leagues

Welcome ! This is a project dedicated for the visualization of soccer KPIs from clubs in the main european leagues. The project can be defined in a ETL configuration as:
Extraction - The data is retrieved from webscrapping of public websites such as FBREF, UNDERSTAT and the free-range API-Sports
Transformation - The data is reconfigured in a table such that the unique key is the match ID and Player ID so that we can have multiple indicators of performance like passing, kicks, tackles, fouls and much more
Loading - The loading is here loosely defined as the construction of visualization that are then passed for the website

The main goal here is to also generated visualization that are benefitial in a FANTASY SPORTS perspective. Because in those games is common for this KPIS to be transformed in points that users accumulate for a given choosen team
As a old player of a famous fanatasy game (Cartola FC) here in brazil I thoight would be fun to construct something for the the Europe Leagues

The website is live in the link https://share.streamlit.io/mrpatinhas/cartologia_awken/main.py by the use of streamlit lib and cloud service
