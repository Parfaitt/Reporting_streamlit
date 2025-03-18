import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
from itertools import combinations
from collections import Counter
import zipfile
import io
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from streamlit_extras.stylable_container import stylable_container


# --- Configuration de la page ---
st.set_page_config(
    page_title="Reporting RA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Injection CSS----
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');
        * { font-family: 'Inter', sans-serif; box-sizing: border-box; }
        .main { background: #f4f6f8; color: #333; }
        /* Sidebar avec gradient froid (bleu foncé) */
        .stSidebar { background: linear-gradient(135deg, #002F6C, #00509E); color: white; }
        /* Header avec une ambiance américaine et froide */
        .banking-header {
            background: linear-gradient(135deg, #002F6C 0%, #00509E 100%);
            padding: 2rem; 
            border-radius: 0 0 25px 25px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        .stPlotlyChart { border: none; }
        .dataframe { border-radius: 10px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)


# --- En-tête personnalisé ---
with stylable_container(key="header", css_styles=".banking-header { color: white !important; }"):
    st.markdown("""
        <div class='banking-header'>
            <h1 style='margin:0;'>Reporting Revenu Assurance</h1>
            <p style='opacity:0.8;'>Interface interactive des Opérations du revenu Assurance</p>
        </div>
    """, unsafe_allow_html=True)

# --- Fonction utilitaire pour créer des "metric cards" compactes ---
def metric_card(title, value, bg_color):
    html = f"""
    <div style="
        background-color: {bg_color};
        padding: 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
        box-shadow: 0 3px 5px rgba(0,0,0,0.1);
        ">
        <h4 style="margin: 0; font-weight: 600; font-size: 1rem;">{title}</h4>
        <p style="font-size: 1.5rem; margin: 5px 0 0; font-weight: bold;">{value}</p>
    </div>
    """
    return html

# --- Chargement du fichier ---
file_path = st.sidebar.file_uploader("Choisir un fichier CSV", type="csv")
if file_path is not None:
    data = pd.read_csv(file_path, encoding="ISO-8859-1")
else:
    st.sidebar.write("Veuillez charger un fichier CSV.")
    st.stop()
        
# --- Nettoyage & transformation ---
def extractday(dated):
    parts=dated.split(' ')
    return parts[0]
data['Date']= data['created_at'].apply(extractday)

data["amount"] = pd.to_numeric(data["amount"], errors="coerce")
data = data.drop_duplicates(subset='transaction_id', keep='first')
payin=data[data['operation_origin']=='payment']
payout=data[data['operation_origin']=='transfer']


# --- Filtres dans la barre latérale ---
st.sidebar.header("🔎 Filtres Stratégiques")
dated = st.sidebar.multiselect('Date', options=sorted(data['Date'].unique()))
statuts = st.sidebar.multiselect('Statut', options=sorted(data['statut'].unique()))
operation = st.sidebar.multiselect('Operation', options=sorted(data['operation_origin'].unique()))
pays = st.sidebar.multiselect('Pays', options=sorted(data['country'].unique()))
partenaire = st.sidebar.multiselect('Provider Name', options=sorted(data['provider_name'].unique()))

if dated:
    data = data[data['Date'].isin(dated)]
    data = data[data['statut'].isin(statuts)]
    data = data[data['operation_origin'].isin(operation)]
    data = data[data['country'].isin(pays)]
    data = data[data['provider_name'].isin(partenaire)]
    
# --- Création des onglets ---
tabs = st.tabs(["📊 Vue Globale", "👥 Opérations", "🔄 Transactions"])

# Onglet Vision 360 (contenu à enrichir ultérieurement)
if len(tabs) > 2:
    with tabs[2]:
        st.subheader("Transactions")
        st.write("Contenu de l'onglet Vision 360 à définir.")
else:
    st.error("L'onglet 'Vision 360' n'a pas été créé correctement.")

# =========================
    # Onglet 1 : Vue Globale
# =========================

with tabs[0]:
    st.subheader("Vue Globale")
        # Calcul des KPI
    montant_total = data["amount"].sum()
    nombre_transaction=data['transaction_id'].count()
    nombre_payin=payin['transaction_id'].count()
    Nombre_payout=payout['transaction_id'].count()
    montant_total_payin = payin["amount"].sum()
    montant_total_payout = payout["amount"].sum()

# Affichage dans des metric cards
col1, col2= st.columns(2)
col1.markdown(metric_card("Nombre Total Transaction", nombre_transaction, "#1E90FF"), unsafe_allow_html=True)
col2.markdown(metric_card("Montant Total", f"{montant_total:,.2f} XOF", "#2E8B57"), unsafe_allow_html=True)

#affichage des graphes
st.markdown("---")
st.markdown("#### Evololutions des transactions par Opérateur")
monthly_sales = data.groupby("provider_name")["amount"].sum().reset_index()
fig_month = px.bar(monthly_sales, x="provider_name", y="amount",
    text_auto=True,
    color="amount",
    color_continuous_scale=["#1E90FF", "#4682B4"],
    template="plotly_white")
fig_month.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_month, use_container_width=True, config={"displayModeBar": False})

chart1, chart2= st.columns((2))
with chart1:
    st.subheader('Vue globale par Statut')
    fig=px.pie(data, values="amount",names="statut", template="plotly_dark")
    fig.update_traces(text=data["statut"], textposition="inside")
    st.plotly_chart(fig,use_container_width=True)

with chart2:
    st.subheader('Vue globale par Pays')
    monthly_statut = data.groupby("country")["amount"].sum().reset_index()
    fig_month = px.bar(monthly_statut, x="country", y="amount",
        text_auto=True,
        color="amount",
        color_continuous_scale=["#1E90FF", "#4682B4"],
        template="plotly_white")
    fig_month.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_month, use_container_width=True, config={"displayModeBar": False})
   




























