import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import modules
from modules.preprocessing import preprocess_pipeline
from modules.retention import create_cohort_data, compute_retention_matrix
from modules.churn import (
    compute_global_churn,
    churn_by_subscription,
    churn_by_region,
    churn_reason_distribution
)
from modules.prediction import (
    monthly_active_users,
    project_users,
    scenario_projection
)
from modules.metrics import (
    calculate_mrr,
    calculate_arpu,
    calculate_ltv
)

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLES
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SaaS Analytics · Rétention & Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,700;1,300&family=DM+Serif+Display:ital@0;1&display=swap');

  /* ── Root palette ── */
  :root {
    --navy:    #0F2C4C;
    --teal:    #0D9488;
    --teal2:   #14B8A6;
    --mint:    #CCFBF1;
    --accent:  #F97316;
    --purple:  #7C3AED;
    --slate:   #475569;
    --light:   #F1F5F9;
    --white:   #FFFFFF;
  }

  /* ── Base ── */
  html, body, [data-testid="stAppViewContainer"] {
    background-color: #F8FAFC !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #1E293B !important;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  [data-testid="stDecoration"] { display: none; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: none;
  }
  [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: #FFFFFF !important;
    font-family: 'DM Serif Display', serif !important;
  }
  [data-testid="stSidebar"] [data-testid="stFileUploader"] label { color: #94A3B8 !important; }

  /* ── Top bar ── */
  .top-banner {
    background: linear-gradient(135deg, #0F2C4C 0%, #0D9488 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .top-banner::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
  }
  .top-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; right: 80px;
    width: 240px; height: 240px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
  }
  .top-banner h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.2rem !important;
    color: #FFFFFF !important;
    margin: 0 0 0.4rem 0 !important;
    line-height: 1.2;
  }
  .top-banner p {
    color: #99F6E4 !important;
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
  }
  .top-banner .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #FFFFFF !important;
    margin-right: 8px;
    margin-top: 12px;
    backdrop-filter: blur(4px);
  }

  /* ── Section headers ── */
  .section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.55rem;
    color: var(--navy);
    margin: 2.2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--mint);
  }

  /* ── KPI Cards ── */
  .kpi-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
  .kpi-card {
    flex: 1;
    min-width: 160px;
    background: var(--white);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    box-shadow: 0 2px 12px rgba(15,44,76,0.08);
    border-top: 4px solid var(--teal);
    position: relative;
    overflow: hidden;
  }
  .kpi-card.orange { border-top-color: var(--accent); }
  .kpi-card.purple { border-top-color: var(--purple); }
  .kpi-card.blue   { border-top-color: #0891B2; }
  .kpi-card .kpi-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
  .kpi-card .kpi-value { font-family: 'DM Serif Display', serif; font-size: 2rem; color: var(--navy); margin: 0; line-height: 1; }
  .kpi-card .kpi-label { font-size: 0.82rem; color: var(--slate); margin-top: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }
  .kpi-card .kpi-glow { position: absolute; width: 120px; height: 120px; border-radius: 50%; background: var(--teal); opacity: 0.05; top: -30px; right: -30px; }

  /* ── Info cards ── */
  .info-card {
    background: var(--white);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(15,44,76,0.07);
    margin-bottom: 1rem;
    border-left: 4px solid var(--teal);
  }
  .info-card.accent { border-left-color: var(--accent); }
  .info-card.purple { border-left-color: var(--purple); }
  .info-card h4 { font-family: 'DM Serif Display', serif; color: var(--navy); margin: 0 0 0.5rem 0; font-size: 1.1rem; }
  .info-card p  { color: var(--slate); margin: 0; font-size: 0.92rem; line-height: 1.6; }

  /* ── Chart wrapper ── */
  .chart-wrap {
    background: var(--white);
    border-radius: 14px;
    padding: 1.2rem;
    box-shadow: 0 2px 12px rgba(15,44,76,0.07);
    margin-bottom: 1.2rem;
  }
  .chart-title { font-weight: 600; color: var(--navy); font-size: 0.95rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.04em; }

  /* ── Upload area ── */
  [data-testid="stFileUploader"] {
    background: rgba(13,148,136,0.05);
    border: 2px dashed var(--teal);
    border-radius: 12px;
    padding: 1rem;
  }

  /* ── Metric ── */
  [data-testid="metric-container"] {
    background: var(--white) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    box-shadow: 0 2px 8px rgba(15,44,76,0.07) !important;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(15,44,76,0.07); }

  /* ── Plotly charts ── */
  .js-plotly-plot .plotly { border-radius: 12px; }

  /* ── Buttons ── */
  .stDownloadButton > button {
    background: var(--teal) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.5rem !important;
  }
  .stDownloadButton > button:hover { background: #0F766E !important; }

  /* ── Slider ── */
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background: var(--teal) !important; }

  /* ── Alert success ── */
  [data-testid="stAlert"] { border-radius: 10px !important; }

  /* ── Tabs ── */
  [data-baseweb="tab-list"] { border-bottom: 2px solid var(--mint) !important; }
  [data-baseweb="tab"][aria-selected="true"] { color: var(--teal) !important; border-bottom: 2px solid var(--teal) !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--light); }
  ::-webkit-scrollbar-thumb { background: var(--teal2); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem 0">
      <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#FFFFFF;line-height:1.2">SaaS Analytics</div>
      <div style="color:#64748B;font-size:0.8rem;margin-top:0.3rem">Rétention & Churn Platform</div>
    </div>
    <hr style="border-color:#1E3A5F;margin:0.8rem 0">
    """, unsafe_allow_html=True)

    st.markdown("###  Données")
    uploaded_file = st.file_uploader("Charger un fichier CSV", type=["csv"], label_visibility="collapsed")

    st.markdown("""
    <hr style="border-color:#1E3A5F;margin:1rem 0">
    <div style="font-size:0.78rem;color:#475569;line-height:1.8">
      <b style="color:#94A3B8">Colonnes requises</b><br>
      user_id · signup_date<br>
      last_activity_date<br>
      subscription_type<br>
      monthly_fee · usage_frequency<br>
      region
    </div>
    <hr style="border-color:#1E3A5F;margin:1rem 0">
    <div style="font-size:0.78rem;color:#475569">
      <b style="color:#94A3B8">Modules actifs</b><br>
       Preprocessing<br>
       Rétention Cohortes<br>
       Analyse Churn<br>
       Prédiction<br>
       Métriques Business
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP BANNER
# ─────────────────────────────────────────────
st.markdown("""
<div class="top-banner">
  <div class="kpi-glow"></div>
  <h1> SaaS Retention & Churn Analytics</h1>
  <p>Analyse comportementale des utilisateurs</p>
  <div>
    <span class="badge"> Rétention</span>
    <span class="badge"> Churn</span>
    <span class="badge"> Prédiction</span>
    <span class="badge"> Motifs de départ</span>
    <span class="badge"> Métriques Business</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────
TEAL_PALETTE = ["#0D9488", "#14B8A6", "#5EEAD4", "#0891B2", "#7C3AED", "#F97316"]

def style_plotly(fig, title=""):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="DM Sans, sans-serif", color="#475569", size=12),
        title=dict(text=title, font=dict(family="DM Serif Display, serif", size=16, color="#0F2C4C")),
        margin=dict(l=20, r=20, t=45, b=20),
        colorway=TEAL_PALETTE,
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0", tickcolor="#CBD5E1")
    fig.update_yaxes(gridcolor="#F1F5F9", linecolor="#E2E8F0", tickcolor="#CBD5E1")
    return fig

# ─────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────
if not uploaded_file:
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem">
      <div style="font-size:4rem;margin-bottom:1rem">📂</div>
      <div style="font-family:'DM Serif Display',serif;font-size:1.8rem;color:#0F2C4C;margin-bottom:0.6rem">
        Chargez votre dataset pour commencer
      </div>
      <div style="color:#64748B;font-size:1rem;max-width:480px;margin:0 auto;line-height:1.7">
        Glissez-déposez votre fichier CSV dans la barre latérale.<br>
        Le dashboard se mettra à jour automatiquement.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
df_raw = pd.read_csv(uploaded_file)
try:
    df = preprocess_pipeline(df_raw)
    st.success("✅ Prétraitement effectué avec succès — données prêtes pour l'analyse")
except Exception as e:
    st.error(f"⚠️ Erreur dans le prétraitement : {e}")
    st.stop()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Métriques",
    "Rétention",
    "Churn",
    "Prédiction",
    "Export"
])

# ═══════════════════════════════════════════════
# TAB 1 — MÉTRIQUES
# ═══════════════════════════════════════════════
with tab1:
    mrr  = calculate_mrr(df)
    arpu = calculate_arpu(df)
    ltv  = calculate_ltv(df)
    churn_rate = compute_global_churn(df)
    total_users = len(df)
    active_users = (df["churn"] == 0).sum()

    st.markdown('<div class="section-header">Indicateurs Clés de Performance</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-glow"></div>
        <div class="kpi-icon">💰</div>
        <div class="kpi-value">${mrr:,.0f}</div>
        <div class="kpi-label">MRR — Revenu Mensuel Récurrent</div>
      </div>
      <div class="kpi-card orange">
        <div class="kpi-icon">👤</div>
        <div class="kpi-value">${arpu:,.2f}</div>
        <div class="kpi-label">ARPU — Revenu Moyen / Utilisateur</div>
      </div>
      <div class="kpi-card purple">
        <div class="kpi-icon">⭐</div>
        <div class="kpi-value">${ltv:,.0f}</div>
        <div class="kpi-label">LTV — Valeur Vie Client</div>
      </div>
      <div class="kpi-card blue">
        <div class="kpi-icon">📉</div>
        <div class="kpi-value">{churn_rate:.1%}</div>
        <div class="kpi-label">Taux de Churn Global</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-header">Aperçu des Données Traitées</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True, height=320)
    with col2:
        st.markdown('<div class="section-header">Résumé</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-card">
          <h4>👥 Utilisateurs</h4>
          <p>Total : <b>{total_users:,}</b><br>
             Actifs : <b>{active_users:,}</b><br>
             Churnés : <b>{total_users - active_users:,}</b></p>
        </div>
        <div class="info-card accent">
          <h4>📊 Données</h4>
          <p>{len(df.columns)} colonnes générées<br>
             Période couverte : {df['signup_date'].min().strftime('%b %Y')} → {df['last_activity_date'].max().strftime('%b %Y')}</p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 2 — RÉTENTION
# ═══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Analyse de Rétention par Cohortes</div>', unsafe_allow_html=True)

    df = create_cohort_data(df)
    retention_matrix = compute_retention_matrix(df)

    if retention_matrix is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Heatmap de Rétention</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(11, 5))
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            sns.heatmap(
                retention_matrix,
                annot=True,
                fmt=".0%",
                cmap=sns.light_palette("#0D9488", as_cmap=True),
                ax=ax,
                linewidths=0.5,
                linecolor="#F1F5F9",
                cbar_kws={"shrink": 0.8}
            )
            ax.set_title("Matrice de Rétention par Cohorte", fontsize=14, color="#0F2C4C", pad=12, fontweight="bold")
            ax.set_xlabel("Ancienneté (mois)", color="#64748B")
            ax.set_ylabel("Mois d'inscription", color="#64748B")
            ax.tick_params(colors="#64748B")
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="info-card">
              <h4>💡 Lecture</h4>
              <p>Chaque ligne = une cohorte d'utilisateurs inscrits le même mois.<br><br>
                 Les colonnes indiquent leur taux de présence M+1, M+2, etc.</p>
            </div>
            <div class="info-card accent">
              <h4>Objectif</h4>
              <p>Identifier les cohortes avec une forte chute dès M+1 pour agir rapidement sur l'onboarding.</p>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 3 — CHURN
# ═══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Analyse du Churn</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        sub_churn = churn_by_subscription(df).reset_index()
        fig = px.bar(
            sub_churn, x="subscription_type", y="mean",
            color="mean",
            color_continuous_scale=["#CCFBF1", "#0D9488", "#0F2C4C"],
            labels={"mean": "Taux de Churn", "subscription_type": "Type d'abonnement"},
        )
        fig = style_plotly(fig, "Churn par Type d'Abonnement")
        fig.update_traces(marker_line_width=0)
        fig.update_coloraxes(showscale=False)
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        region_churn = churn_by_region(df).reset_index()
        fig2 = px.bar(
            region_churn, x="region", y="mean",
            color="mean",
            color_continuous_scale=["#FEF3C7", "#F97316", "#7C3AED"],
            labels={"mean": "Taux de Churn", "region": "Région"},
        )
        fig2 = style_plotly(fig2, "Churn par Région")
        fig2.update_traces(marker_line_width=0)
        fig2.update_coloraxes(showscale=False)
        fig2.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)

    # Churn reasons
    reasons = churn_reason_distribution(df)
    if reasons is not None:
        st.markdown('<div class="section-header">Distribution des Raisons de Churn</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            fig_reason = px.pie(
                values=reasons.values,
                names=reasons.index,
                color_discrete_sequence=TEAL_PALETTE,
                hole=0.45,
            )
            fig_reason = style_plotly(fig_reason, "Raisons de Départ des Utilisateurs")
            fig_reason.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_reason, use_container_width=True)
        with col2:
            for reason, val in reasons.items():
                st.markdown(f"""
                <div class="info-card" style="padding:0.9rem 1.2rem;margin-bottom:0.6rem">
                  <h4 style="font-size:0.95rem">{reason}</h4>
                  <p>{val:.1%} des utilisateurs churnés</p>
                </div>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# TAB 4 — PRÉDICTION
# ═══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Prédiction des Abonnements</div>', unsafe_allow_html=True)

    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df = df.dropna(subset=["signup_date"])
    df["month"] = df["signup_date"].dt.to_period("M")
    monthly_users_series = df.groupby("month")["user_id"].nunique()

    monthly_users = monthly_users_series.reset_index()
    monthly_users.columns = ["month", "users"]
    monthly_users["month"] = monthly_users["month"].astype(str)

    fig_users = px.area(
        monthly_users, x="month", y="users",
        color_discrete_sequence=["#0D9488"],
        labels={"users": "Abonnés uniques", "month": "Mois"},
    )
    fig_users = style_plotly(fig_users, "Croissance des Abonnements par Mois")
    fig_users.update_traces(fill="tozeroy", fillcolor="rgba(13,148,136,0.12)", line_width=2.5)
    st.plotly_chart(fig_users, use_container_width=True)

    st.markdown('<div class="section-header">Projection par Scénarios</div>', unsafe_allow_html=True)

    months_to_predict = st.slider("Nombre de mois à projeter", 3, 24, 6)
    scenarios = scenario_projection(monthly_users_series, months_to_predict)
    scenario_df = pd.DataFrame(scenarios)
    scenario_df.index = [f"M+{i+1}" for i in range(months_to_predict)]

    fig_scenario = go.Figure()
    colors = {"optimistic": "#0D9488", "realistic": "#0891B2", "pessimistic": "#F97316"}
    labels = {"optimistic": "🟢 Optimiste", "realistic": "🔵 Réaliste", "pessimistic": "🟠 Pessimiste"}
    dashes = {"optimistic": "dash", "realistic": "solid", "pessimistic": "dot"}

    for col in ["optimistic", "realistic", "pessimistic"]:
        fig_scenario.add_trace(go.Scatter(
            x=scenario_df.index,
            y=scenario_df[col],
            name=labels[col],
            line=dict(color=colors[col], width=2.5, dash=dashes[col]),
            mode="lines+markers",
            marker=dict(size=6)
        ))

    # Fan area
    fig_scenario.add_trace(go.Scatter(
        x=list(scenario_df.index) + list(scenario_df.index[::-1]),
        y=list(scenario_df["optimistic"]) + list(scenario_df["pessimistic"][::-1]),
        fill="toself",
        fillcolor="rgba(13,148,136,0.08)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
        name="Zone d'incertitude"
    ))

    fig_scenario = style_plotly(fig_scenario, "Fan Chart — Scénarios de Projection")
    fig_scenario.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig_scenario, use_container_width=True)

    # Projected table
    st.markdown('<div class="section-header">Détail des Projections</div>', unsafe_allow_html=True)
    st.dataframe(
        scenario_df.style.format("{:.0f}").background_gradient(cmap="BuGn", axis=None),
        use_container_width=True
    )

# ═══════════════════════════════════════════════
# TAB 5 — EXPORT
# ═══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Exporter les Données Traitées</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
      <h4>📥 Export CSV</h4>
      <p>Téléchargez le dataset complet avec toutes les features ingéniées : <b>tenure_days</b>, <b>days_inactive</b>, <b>activity_rate</b>, <b>churn</b> et les cohortes.</p>
    </div>
    """, unsafe_allow_html=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Télécharger le CSV traité",
        data=csv,
        file_name="saas_analytics_processed.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown("""
    <div class="info-card accent" style="margin-top:1.5rem">
      <h4>ℹ️ À propos du projet</h4>
      <p>Ce dashboard analytique a été conçu dans le cadre d'un projet de data analytics SaaS, entièrement sans machine learning. 
         Il couvre les 7 parties du projet : prétraitement, rétention, churn, prédiction, classification des motifs, dashboard et recommandations business.</p>
    </div>
    """, unsafe_allow_html=True)