#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:51:06 2022

@author: ilham
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import locale
locale.setlocale(locale.LC_TIME,'')
import time
import datetime 
from datetime import datetime



#Création de differents partie
#st.title(' Quosh & United Boss AED & Visualisation')



st.markdown("<h1 style='text-align: center; color: blue;'>QuoshStat</h1>", unsafe_allow_html=True)

#st.markdown("<h3 style='text-align: center; color: orange;'>Cette API analyse les conversations des membres Quosh & United Boss sur WathsApp</h3>", unsafe_allow_html=True)
st.info('Cette API analyse les conversations des membres Quosh & United Boss sur WhatsApp')
#st.markdown('''
#Cette API analyse les conversations des membres Quosh & United Boss sur WathsApp
#* **libraries utilisés:** Streamlit, Pandas,numpy,matplotlib,seaborn
#* **Data Source:** Kaggle
#''')

col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image("image.jpg")

with col3:
    st.write(' ')
    
#st.image('image_quosh.jpg', caption="")
df=pd.read_csv("chat.txt",sep="\t", header=1,encoding
='utf-8') 
@st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)  
def preprocessing(data):
    pattern='\d{1,2}/\d{1,2}/\d{2,4}\s\w{1}\s\d{1,2}:\d{2}\s-\s'
    df=data
    #df.columns=['date_message']
    df.rename(columns={'24/03/2019 à 20:19 - Vous avez été ajouté(e)': 'date_message'}, inplace=True)
    message=[]
    date=[]
    for i in range(len(df)):
        message.append(re.split(pattern,df.date_message[i])[1:])
        date.append(re.findall(pattern,df.date_message[i]))
    df['date']=pd.DataFrame(date)
    df['Message']=pd.DataFrame(message)
    df=df.drop('date_message',axis=1)
    df['date']=df.date.apply(lambda x:re.sub('\s\à',  '',str(x)))
    df=df[df.date!='None']
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M - ')
    utilisateur = []
    messages = []

    for message in df['Message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:
            utilisateur.append(entry[1])
            messages.append(entry[2])
        else:
            utilisateur.append('group_notification')
            messages.append(entry[0])
    df['membre'] = utilisateur
    df['message'] = messages
    
    df.drop(columns=['Message'], inplace=True)
    
    df['année']=df['date'].dt.year
    #df['mois']=df['date'].dt.month_name()
    df['mois']=df['date'].dt.strftime("%B")
    df['jour']=df['date'].dt.day
    df['heure']=df['date'].dt.hour
    df['minute']=df['date'].dt.minute
    df['num_mois']=df['date'].dt.month
    df['date_seul']=df['date'].dt.date
    df['nom_du_jour'] =df['date'].dt.strftime("%A")
    #df['nom_du_jour'] =df['date'].strftime("%A")
    
    
    period = []

    for heure in df[['nom_du_jour', 'heure']]['heure']:
        if heure == 23:
            period.append(str(heure) + "-" + str('00'))
        elif heure == 0:
            period.append(str('00') + "-" + str(heure + 1))
        else:
            period.append(str(heure) + "-" + str(heure + 1))

    df['period'] = period
    return df

df =preprocessing(df)
#st.dataframe(df)

#st.title("conversation journalière")

#st.pyplot(fig)

## On définit les codes des emoji

emoji_code=["\U0001f600","\U0001f603","\U0001f604","\U0001f601","\U0001f606","\U0001f605","\U0001f923","\U0001f602",
           "\U0001f642","\U0001f643","\U0001fAE0","\U0001f609","\U0001f60A","\U0001f607","\U0001f972",
           "\U0001f60B","\U0001f61B","\U0001f61C","\U0001f92A","\U0001f61D","\U0001f917","\U0001f92D","\U0001f914",
           "\U0001f973","\U0001f613","\U0001f970","\U0001f60D","\U0001f929","\U0001f61A","\U0001f972","\U0001f61F",
           "\U0001f625","\U0001f493","\U0001f44B","\U0001f44C","\U0001f64F",r"\U0002764",r"\U000263A" ]

#import streamlit as st
liste_membre = df['membre'].unique().tolist()
liste_membre.remove('group_notification')
liste_membre.sort()
liste_membre.insert(0, "Overall")
#selected_user = st.sidebar.selectbox("Afficher l'analyse par rapport", liste_membre)
#st.sidebar.header("User Input")
# Creating selectbox for Graphs & Plots
#graphs = st.sidebar.selectbox("Graphs & Plots", ("Bar Graph", "Scatter Plot", "HeatMap", "Pie Chart"))



from urlextract import URLExtract
from wordcloud import WordCloud
extract = URLExtract()
import pandas as pd
from collections import Counter
import emoji


def fetch_stats(selected_user, df):

    if selected_user != 'Overall':
        #df = df[df['user'] == selected_user]
        df=df[df.membre.isin(selected_user)]

    num_messages = df.shape[0]
    words = []
    for message in df['message']:
        words.extend(message.split())

    #num_media_msgs = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []

    for message in df['message']:
        links.extend(extract.find_urls(message))
    return num_messages, len(words),len(links)


def fetch_most_busy_users(df):
    x = df['membre'].value_counts()[0:10]
    df = round((df['membre'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'nom', 'membre': 'pourcentage'})
    return x, df

def create_word_cloud(selected_user,df):

    f = open('stop_frensh.txt', 'r')
    stop_words = f.read()

    if isinstance(selected_user, list):
       if selected_user != 'Overall':
           df=df[df.membre.isin(selected_user)]
    else:
       df = df[df['membre'] ==selected_user]

    temp = df[df['membre'] != 'group_notification']
    temp = temp[temp['message'] !='<Médias omis>']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    if temp['message'].shape[0]==0:
       res='Pas assez de mot'
       df_wc=res
      # print(res)
    else:
        df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_frensh.txt','r')
    stop_words = f.read()

    if isinstance(selected_user, list):
       if selected_user != 'Overall':
           df=df[df.membre.isin(selected_user)]
    else:
       df = df[df['membre'] ==selected_user]

    temp = df[df['membre'] != 'group_notification']
    temp = temp[temp['message'] !='<Médias omis>']
   
    if temp['message'].shape[0]==0:
        res='Pas assez de mot'
        most_common_df=res
       # print(res)
    else:
        words = []

        for message in temp['message']:
            for word in message.lower().split():
                if word not in stop_words:
                    words.append(word)

        most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        #df = df[df['user'] == selected_user]
        df=df[df.membre.isin(selected_user)]
    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji_code])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(20))
    return emoji_df

def monthly_timeline(selected_user, df):
    if isinstance(selected_user, list):
       if selected_user != 'Overall':
           df=df[df.membre.isin(selected_user)]
    else:
       df = df[df['membre'] ==selected_user]

    timeline = df.groupby(['année', 'num_mois', 'mois']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['mois'][i] + "-" + str(timeline['année'][i]))
    timeline['time'] = time
    return timeline

def daily_timeline(selected_user, df):
    if isinstance(selected_user, list):
        if selected_user != 'Overall':
            df=df[df.membre.isin(selected_user)]
    else:
        df = df[df['membre'] ==selected_user]

    daily_timeline = df.groupby('date_seul').count()['message'].reset_index()
    return daily_timeline

def week_activity_map(selected_user, df):
    if isinstance(selected_user, list):
       if selected_user != 'Overall':
           df=df[df.membre.isin(selected_user)]
    else:
       df = df[df['membre'] ==selected_user]

    return df['nom_du_jour'].value_counts()

def month_activity_map(selected_user, df):
    if isinstance(selected_user, list):
       if selected_user != 'Overall':
           df=df[df.membre.isin(selected_user)]
    else:
       df = df[df['membre'] ==selected_user]

    return df['mois'].value_counts()


def activity_heatmap(selected_user, df):
    if isinstance(selected_user, list):
       if selected_user != 'Overall':
           df=df[df.membre.isin(selected_user)]
    else:
       df = df[df['membre'] ==selected_user]

    activity_heatmap = df.pivot_table(index='nom_du_jour', columns='period', values='message', aggfunc='count').fillna(0)
    return activity_heatmap
#st.title("conversation journalière")
#st.markdown("<h3 style='text-align: center; color: grey;'>conversation journalière</h3>", unsafe_allow_html=True)
#jour=daily_timeline(liste_membre, df)
#fig, ax = plt.subplots()
#ax.plot(jour['date_seul'], jour['message'], color='green')
#plt.xticks(rotation='vertical');
#st.pyplot(fig)

#if st.sidebar.button("Afficher l'analyse par rapport"):
num_msgs, words, num_links =fetch_stats(liste_membre, df)
st.markdown("<h1 style='text-align: center; color: grey;'>Principaux chiffres</h1>", unsafe_allow_html=True)
#st.title("Principaux chiffres")
col1, col2, col3 = st.columns(3)

with col1:
          st.header("Messages")
          st.title(num_msgs)

with col2:
          st.header("Mots")
          st.title(words)


with col3:
          st.header("Lien partagé")
          st.title(num_links)
colors = sns.color_palette('bright')



dif_parti=["DataViz Quosh & United", "DataViz de chaque membre"]

Partie=st.sidebar.selectbox('Menu',options=dif_parti)


if Partie==dif_parti[0]:
    st.info('Visualisation de DataViz de Quosh & United Boss')

    options=['les membres les plus actifs','journée et mois les plus actifs','Heatmap du groupe',
             'nuage des mots','emojis les plus utilisés','les mots les plus utilisés']
    choix=st.sidebar.selectbox('Choisisez votre DataViz', options=options)
    #st.write('Analyse: ', choix)
    
    if choix==options[0]:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            #plt.figure(figsize=(8, 4))
            x, new_df =fetch_most_busy_users(df)
            fig, ax = plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x.values, color = colors)
                plt.xticks(rotation = 'vertical')
                plt.title('Les membres les plus actifs de Quosh & United Boss',
              fontsize=15,color='grey')
                st.pyplot(fig)
            with col2:
              st.dataframe(new_df)
    elif choix==options[1]: 
        col1, col2 = st.columns(2)
        with col1:
             echange_journalier=week_activity_map(liste_membre, df)
             fig, ax = plt.subplots(figsize=(12,8))
             ax.bar(echange_journalier.index, echange_journalier.values, color=colors)
             plt.title('Les jours où le groupe est plus actif',
              fontsize=15,color='grey')
             plt.ylabel('nombre de messages')
             st.pyplot(fig)
        with col2:
             echange_mois=month_activity_map(liste_membre, df)
             fig, ax = plt.subplots(figsize=(12,8))
             ax.bar(echange_mois.index, echange_mois.values, color=colors)
             plt.title('Les mois où le groupe est plus actif',
              fontsize=15,color='grey')
             plt.ylabel('nombre de messages')
             st.pyplot(fig);
    elif choix==options[2]:
        #st.markdown("<h3 style='text-align: center; color: grey;'>Les heures des conversations du groupe en fonction des jours de la semaines.</h3>", unsafe_allow_html=True)
        carte_chaleur_echange=activity_heatmap(liste_membre,df)
        fig, ax = plt.subplots(figsize=(14,8))
        ax = sns.heatmap(carte_chaleur_echange)
        plt.xlabel('Période de la journée')
        plt.ylabel('jour de la semaine')
        plt.title('Heatmap du groupe',fontsize=15,color='grey')
        #plt.title('Carte de chaleur: Les heures des conversations du groupe en fonction des jours de la semaines.',
             # fontsize=15,color='grey')
        
        st.pyplot(fig)
        
    elif choix==options[3]:
        df_wc =create_word_cloud(liste_membre, df)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(df_wc);
        plt.title('Nuages des mots  des conversation du groupes Quosh & United Boss',
              fontsize=15,color='grey')
        st.pyplot(fig);
    elif choix==options[4]:
    #emoji analysis
        emoji_df =emoji_helper(liste_membre, df)
    #st.title("Emojis Analysis")
    
        col1, col2 = st.columns(2)
    
        with col1:
             st.dataframe(emoji_df)
        with col2:
             fig, ax = plt.subplots()
             ax.pie(emoji_df[1].head(), labels = emoji_df[0].head(), autopct = "%0.2f")
             st.pyplot(fig)    
    else:
        mot_plus_courant_df = most_common_words(liste_membre, df)
        fig, ax = plt.subplots(figsize=(8,8))
        ax.barh(mot_plus_courant_df[0],mot_plus_courant_df[1],color=colors)
        plt.xticks(rotation='vertical');
        plt.title('Les mots les plus frequents dans les conversations du groupe',
              fontsize=15,color='grey');
        st.pyplot(fig)

else:
    st.info(' Vous visualisez vos DataViz')  
    liste=liste_membre[1:]
    membre=st.sidebar.selectbox('Membre', options=liste)
    if membre in liste:
        options=['courbe de conversation','journée et mois les plus actifs','Heatmap de chaque membre',
                 'nuage des mots','Les mots les plus utilisés']
        choix=st.sidebar.selectbox('DataViz', options=options)
        if choix==options[0]:
                st.set_option('deprecation.showPyplotGlobalUse', False)
                daily_m=daily_timeline(membre, df)
                df_max=daily_m.sort_values('message',ascending = False)
                table= pd.DataFrame(list(zip(list(df_max.date_seul[0:3]),df_max.message[0:3])),
                    columns = ['Date','nombre de message'])
                col1, col2 = st.columns(2)
                if daily_m.shape[0]>5:
                    with col1:
                        #if daily_m.shape[0]>3:
                         fig, ax = plt.subplots()
                         ax.plot(daily_m['date_seul'], daily_m['message'], color='green')
                         plt.xticks(rotation='vertical');
                         plt.title(f'Courbe de conversation de Mr {membre}',
                                  fontsize=15,color='grey')
                         st.pyplot(fig)
                    with col2:
                         st.dataframe(table)
                else:
                    st.write(f"Désolé Mr {membre}, vous n'etes pas un membre actif du groupe Quosh.",color='red')
                    
        elif choix==options[1]:
            echange_journalier=week_activity_map(membre, df)
            col1, col2 = st.columns(2)
            if echange_journalier.shape[0]>2:
                with col1:
                     fig, ax = plt.subplots(figsize=(7,4))
                     ax.bar(echange_journalier.index, echange_journalier.values, color=colors)
                     plt.title(f'conversation hebdomadaire de Mr {membre} ',
                               fontsize=15,color='grey')
                     plt.ylabel('nombre de messages')
                     st.pyplot(fig)
                with col2:
                     echange_mois=month_activity_map(membre, df)
                     fig, ax = plt.subplots(figsize=(7,4))
                     ax.bar(echange_mois.index, echange_mois.values, color=colors)
                     plt.title(f'conversation mensuelle de Mr {membre} ',
                      fontsize=15,color='grey')
                     plt.ylabel('nombre de messages')
                     st.pyplot(fig);                    
            else:
                st.write(f"Désolé Mr {membre}, vous n'etes pas un membre active du groupe Quosh.",color='red')
                    
        elif choix==options[2]:
            #st.markdown("<h3 style='text-align: center; color: grey;'>Les heures des conversations du groupe en fonction des jours de la semaines.</h3>", unsafe_allow_html=True)
            carte_chaleur_echange=activity_heatmap(membre,df)
            if carte_chaleur_echange.shape[0]>2:
                fig, ax = plt.subplots(figsize=(14,8))
                ax = sns.heatmap(carte_chaleur_echange)
                plt.xlabel('Période de la journée')
                plt.ylabel('jour de la semaine')
                plt.title(f'Heatmap de Mr {membre}',fontsize=15,color='grey')
                st.pyplot(fig); 
            else:
                st.write(f"Désolé Mr {membre}, vous n'etes pas un membre active du groupe Quosh",color='red')    
        elif choix==options[3]:
            df_wc =create_word_cloud(membre, df)
            if isinstance(df_wc,str):
                st.write(f"Mr {membre}, vous n'avez pas assez de mot pour un nuage des mots.")
            else:
                fig, ax = plt.subplots(figsize=(12, 12))
                ax.imshow(df_wc);
                plt.title(f'Nuages des mots  des conversation de Mr {membre}',
                      fontsize=15,color='grey')
                st.pyplot(fig);                 
        else:
            mot_plus_courant_df= most_common_words(membre, df)
            if isinstance(mot_plus_courant_df,str):
                st.write(f"Mr {membre}, vous n'avez pas assez de mots.")
            else:
                fig, ax = plt.subplots(figsize=(8,8))
                ax.barh(mot_plus_courant_df[0],mot_plus_courant_df[1],color=colors)
                plt.xticks(rotation='vertical');
                plt.title(f'Les mots le plus utilisés par Mr {membre}',
                          fontsize=15,color='grey');
                st.pyplot(fig)
            
    
    
    
    