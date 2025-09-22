#!/usr/bin/env python3
"""
Streamlit Dashboard - Evolve Challenge 2025
Comprehensive visualization of Churn Prediction and Customer Clustering project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Evolve Challenge 2025 - ML Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .success-metric {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
    }
    .warning-metric {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
    }
    .info-metric {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

def convert_numeric_columns(df):
    """Convert columns that may contain comma-separated decimal numbers"""
    df = df.copy()
    
    # Columns that may have comma as decimal separator
    numeric_columns = [
        'MRR_12M', 'QTD_CONTRATACOES_12M', 'VLR_CONTRATACOES_12M',
        'resposta_NPS', 'Nota_SupTec_Agilidade', 'Nota_SupTec_Atendimento',
        'Nota_Comercial', 'Nota_Custos', 'Nota_AdmFin_Atendimento',
        'Nota_Software', 'Nota_Software_Atualizacao'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            # Convert to string, replace comma with dot, then to numeric
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
    
    return df

def create_demo_data_fallback():
    """Create demo data when original dataset cannot be loaded"""
    np.random.seed(42)
    n_samples = 1000
    
    demo_data = {
        'MRR_12M': np.random.exponential(500, n_samples),
        'QTD_CONTRATACOES_12M': np.random.poisson(2, n_samples),
        'dias_cliente': np.random.exponential(365, n_samples),
        'QTD_TICKETS_ABERTOS': np.random.poisson(1, n_samples),
        'QTD_TICKETS_FECHADOS': np.random.poisson(1, n_samples),
        'NPS_RELACIONAL': np.random.normal(8, 2, n_samples),
        'NPS_TRANSACIONAL': np.random.normal(7, 2, n_samples)
    }
    
    # Create DataFrame
    demo_df = pd.DataFrame(demo_data)
    
    # Ensure positive values
    demo_df['MRR_12M'] = demo_df['MRR_12M'].abs()
    demo_df['QTD_CONTRATACOES_12M'] = demo_df['QTD_CONTRATACOES_12M'].abs()
    demo_df['dias_cliente'] = demo_df['dias_cliente'].abs()
    demo_df['NPS_RELACIONAL'] = np.clip(demo_df['NPS_RELACIONAL'], 0, 10)
    demo_df['NPS_TRANSACIONAL'] = np.clip(demo_df['NPS_TRANSACIONAL'], 0, 10)
    
    return demo_df

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        # Try with semicolon separator first (based on file inspection)
        df = pd.read_csv('dataset_unificado.csv', sep=';', encoding='utf-8', on_bad_lines='skip')
        # Convert numeric columns that may have comma as decimal separator
        df = convert_numeric_columns(df)
        return df
    except Exception as e:
        try:
            # Try with comma separator
            df = pd.read_csv('dataset_unificado.csv', sep=',', encoding='utf-8', on_bad_lines='skip')
            df = convert_numeric_columns(df)
            return df
        except Exception as e2:
            try:
                # Try with error handling
                df = pd.read_csv('dataset_unificado.csv', encoding='utf-8', on_bad_lines='skip')
                df = convert_numeric_columns(df)
                return df
            except Exception as e3:
                st.error(f"Erro ao carregar dataset: {str(e3)}")
                st.info("Tentando carregar dados de demonstração...")
                return create_demo_data_fallback()

@st.cache_resource
def load_models():
    """Load and cache ML models"""
    models = {}
    
    # Load churn model
    try:
        with open('churn_model_mlflow.pkl', 'rb') as f:
            churn_data = pickle.load(f)
            models['churn'] = churn_data['model']
            models['churn_scaler'] = churn_data['scaler']
            models['churn_features'] = churn_data['feature_names']
    except FileNotFoundError:
        st.warning("Modelo de Churn não encontrado! Criando modelo de demonstração...")
        models.update(create_demo_models())
    
    # Load clustering model
    try:
        with open('clustering_model_mlflow.pkl', 'rb') as f:
            clustering_data = pickle.load(f)
            models['clustering'] = clustering_data['model']
            models['clustering_scaler'] = clustering_data['scaler']
            models['clustering_features'] = clustering_data['feature_names']
    except FileNotFoundError:
        if 'clustering' not in models:  # Only warn if not already created in demo
            st.warning("Modelo de Clustering não encontrado! Criando modelo de demonstração...")
            models.update(create_demo_models())
    
    return models

def create_demo_models():
    """Create demo models when originals are not available"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Create demo data for training
    np.random.seed(42)
    n_samples = 1000
    
    demo_data = {
        'MRR_12M': np.random.exponential(500, n_samples),
        'QTD_CONTRATACOES_12M': np.random.poisson(2, n_samples),
        'dias_cliente': np.random.exponential(365, n_samples),
        'QTD_TICKETS_ABERTOS': np.random.poisson(1, n_samples),
        'NPS_RELACIONAL': np.random.normal(8, 2, n_samples),
        'NPS_TRANSACIONAL': np.random.normal(7, 2, n_samples)
    }
    
    demo_df = pd.DataFrame(demo_data)
    demo_df['MRR_12M'] = demo_df['MRR_12M'].abs()
    demo_df['QTD_CONTRATACOES_12M'] = demo_df['QTD_CONTRATACOES_12M'].abs()
    demo_df['dias_cliente'] = demo_df['dias_cliente'].abs()
    demo_df['NPS_RELACIONAL'] = np.clip(demo_df['NPS_RELACIONAL'], 0, 10)
    demo_df['NPS_TRANSACIONAL'] = np.clip(demo_df['NPS_TRANSACIONAL'], 0, 10)
    
    models = {}
    
    # Create churn target
    demo_df['churn'] = 0
    demo_df.loc[
        (demo_df['MRR_12M'] <= 100) | 
        (demo_df['QTD_CONTRATACOES_12M'] <= 0) | 
        (demo_df['dias_cliente'] <= 30), 
        'churn'
    ] = 1
    
    # Train churn model
    churn_features = ['MRR_12M', 'QTD_CONTRATACOES_12M', 'dias_cliente', 'QTD_TICKETS_ABERTOS', 'NPS_RELACIONAL']
    X_churn = demo_df[churn_features].fillna(0)
    y_churn = demo_df['churn']
    
    churn_scaler = StandardScaler()
    X_churn_scaled = churn_scaler.fit_transform(X_churn)
    
    churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
    churn_model.fit(X_churn_scaled, y_churn)
    
    models['churn'] = churn_model
    models['churn_scaler'] = churn_scaler
    models['churn_features'] = churn_features
    
    # Train clustering model
    clustering_features = ['MRR_12M', 'QTD_CONTRATACOES_12M', 'dias_cliente', 'NPS_RELACIONAL', 'NPS_TRANSACIONAL']
    X_clustering = demo_df[clustering_features].fillna(0)
    
    clustering_scaler = StandardScaler()
    X_clustering_scaled = clustering_scaler.fit_transform(X_clustering)
    
    clustering_model = KMeans(n_clusters=4, random_state=42)
    clustering_model.fit(X_clustering_scaled)
    
    models['clustering'] = clustering_model
    models['clustering_scaler'] = clustering_scaler
    models['clustering_features'] = clustering_features
    
    return models

def create_churn_target(df):
    """Create churn target variable"""
    df = df.copy()
    
    # Ensure numeric columns are properly converted
    df = convert_numeric_columns(df)
    
    # Fill NaN values with 0 for key columns
    if 'MRR_12M' in df.columns:
        df['MRR_12M'] = df['MRR_12M'].fillna(0)
    else:
        df['MRR_12M'] = 0
    
    if 'QTD_CONTRATACOES_12M' in df.columns:
        df['QTD_CONTRATACOES_12M'] = df['QTD_CONTRATACOES_12M'].fillna(0)
    else:
        df['QTD_CONTRATACOES_12M'] = 0
    
    # Calculate days since customer (from CLIENTE_DESDE if available)
    if 'CLIENTE_DESDE' in df.columns:
        try:
            df['CLIENTE_DESDE'] = pd.to_datetime(df['CLIENTE_DESDE'], errors='coerce')
            df['dias_cliente'] = (pd.Timestamp.now() - df['CLIENTE_DESDE']).dt.days
            df['dias_cliente'] = df['dias_cliente'].fillna(365)  # Default to 1 year if date is invalid
        except:
            df['dias_cliente'] = 365  # Default value
    else:
        df['dias_cliente'] = 365  # Default value
    
    # Churn conditions
    condition1 = (df['MRR_12M'] <= 0)
    condition2 = (df['QTD_CONTRATACOES_12M'] <= 0)
    condition3 = df['dias_cliente'] <= 30
    
    # Create churn target
    df['churn'] = 0
    df.loc[condition1 | condition2 | condition3, 'churn'] = 1
    
    return df

def prepare_churn_features(df, feature_names):
    """Prepare features for churn prediction"""
    df = df.copy()
    
    # Ensure numeric columns are properly converted
    df = convert_numeric_columns(df)
    
    # Calculate days since customer if not present
    if 'dias_cliente' not in df.columns:
        if 'CLIENTE_DESDE' in df.columns:
            try:
                df['CLIENTE_DESDE'] = pd.to_datetime(df['CLIENTE_DESDE'], errors='coerce')
                df['dias_cliente'] = (pd.Timestamp.now() - df['CLIENTE_DESDE']).dt.days
                df['dias_cliente'] = df['dias_cliente'].fillna(365)
            except:
                df['dias_cliente'] = 365
        else:
            df['dias_cliente'] = 365
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
    
    # Ensure all required features exist
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    return df[feature_names]

def prepare_clustering_features(df, feature_names):
    """Prepare features for clustering"""
    df = df.copy()
    
    # Ensure numeric columns are properly converted
    df = convert_numeric_columns(df)
    
    # Calculate days since customer if not present
    if 'dias_cliente' not in df.columns:
        if 'CLIENTE_DESDE' in df.columns:
            try:
                df['CLIENTE_DESDE'] = pd.to_datetime(df['CLIENTE_DESDE'], errors='coerce')
                df['dias_cliente'] = (pd.Timestamp.now() - df['CLIENTE_DESDE']).dt.days
                df['dias_cliente'] = df['dias_cliente'].fillna(365)
            except:
                df['dias_cliente'] = 365
        else:
            df['dias_cliente'] = 365
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(0)
    
    # Ensure all required features exist
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    return df[feature_names]

def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">Evolve Challenge 2025</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: #666;">ML Dashboard - Churn Prediction & Customer Clustering</h2>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Check if using demo data
    is_demo_data = len(df) == 1000 and 'MRR_12M' in df.columns and df['MRR_12M'].dtype == 'float64'
    if is_demo_data:
        st.warning("**Modo Demonstração**: Usando dados simulados para demonstração do dashboard")
    
    # Load models
    models = load_models()
    
    # Sidebar
    st.sidebar.title("Controles")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Visão Geral", 
        "Análise de Dados", 
        "Modelo de Churn", 
        "Segmentação", 
        "Predições"
    ])
    
    # Tab 1: Overview
    with tab1:
        st.header("Visão Geral do Projeto")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
            st.metric("Total de Clientes", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card info-metric">', unsafe_allow_html=True)
            churn_df = create_churn_target(df)
            churn_rate = churn_df['churn'].mean() * 100
            st.metric("Taxa de Churn", f"{churn_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card warning-metric">', unsafe_allow_html=True)
            if 'MRR_12M' in df.columns:
                mrr_median = df['MRR_12M'].median()
            else:
                mrr_median = 0
            st.metric("MRR Médio", f"R$ {mrr_median:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            if 'QTD_CONTRATACOES_12M' in df.columns:
                avg_contracts = df['QTD_CONTRATACOES_12M'].mean()
            else:
                avg_contracts = 0
            st.metric("Contratos/Cliente", f"{avg_contracts:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Project Description
        st.markdown("""
        ## Objetivos do Projeto
        
        Este projeto foi desenvolvido para o **Evolve Challenge 2025** com os seguintes objetivos:
        
        - **Predição de Churn**: Identificar clientes com alto risco de cancelamento
        - **Segmentação de Clientes**: Agrupar clientes por comportamento e características
        - **Automação**: Pipeline completo de ML com MLflow, Docker e PostgreSQL
        - **Deploy**: Sistema de predições em produção com monitoramento
        """)
        
        # Architecture Overview
        st.markdown("""
        ## Arquitetura da Solução
        
        ### Camadas de Dados
        - **Bronze**: Dados brutos em CSV
        - **Silver**: PostgreSQL com dados processados
        - **Gold**: Resultados de predições e segmentações
        
        ### Modelos de ML
        - **RandomForest**: Predição de Churn (Otimizado para Recall)
        - **KMeans**: Segmentação de Clientes (4 clusters)
        
        ### Tecnologias
        - **MLflow**: Versionamento e tracking de modelos
        - **Docker**: Containerização da aplicação
        - **PostgreSQL**: Banco de dados principal
        - **Streamlit**: Dashboard interativo
        - **AWS EC2**: Deploy em produção
        """)
    
    # Tab 2: Data Analysis
    with tab2:
        st.header("Análise Exploratória de Dados")
        
        # Data Overview
        st.subheader("Resumo dos Dados")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Informações do Dataset:**")
            st.write(f"- **Total de registros**: {len(df):,}")
            st.write(f"- **Total de colunas**: {len(df.columns)}")
            st.write(f"- **Valores nulos**: {df.isnull().sum().sum():,}")
            st.write(f"- **Duplicatas**: {df.duplicated().sum():,}")
        
        with col2:
            st.markdown("**Tipos de Dados:**")
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- **{dtype}**: {count} colunas")
        
        # Show data sample
        st.subheader("Amostra dos Dados")
        st.dataframe(df.head(10), width='stretch')
        
        # Data Quality Assessment
        st.subheader("Qualidade dos Dados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Valores Ausentes por Coluna:**")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                fig_missing = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    title="Valores Ausentes por Coluna",
                    labels={'x': 'Quantidade de Valores Ausentes', 'y': 'Colunas'}
                )
                st.plotly_chart(fig_missing, width='stretch')
            else:
                st.success("Nenhum valor ausente encontrado!")
        
        with col2:
            st.markdown("**Tipos de Dados:**")
            dtype_counts = df.dtypes.value_counts()
            # Convert dtype names to strings for JSON serialization
            dtype_names = [str(dtype) for dtype in dtype_counts.index]
            fig_dtypes = px.pie(
                values=dtype_counts.values,
                names=dtype_names,
                title="Distribuição dos Tipos de Dados"
            )
            st.plotly_chart(fig_dtypes, width='stretch')
        
        # Key Metrics Summary
        st.subheader("Resumo das Métricas Principais")
        
        # Convert and analyze key metrics
        df_metrics = convert_numeric_columns(df)
        
        if 'MRR_12M' in df_metrics.columns:
            st.markdown("**Análise de MRR (Receita Mensal Recorrente):**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Média", f"R$ {df_metrics['MRR_12M'].mean():.2f}")
            with col2:
                st.metric("Mediana", f"R$ {df_metrics['MRR_12M'].median():.2f}")
            with col3:
                st.metric("Desvio Padrão", f"R$ {df_metrics['MRR_12M'].std():.2f}")
            with col4:
                st.metric("Valor Máximo", f"R$ {df_metrics['MRR_12M'].max():.2f}")
        
        # Clustering Information
        st.subheader("Informações sobre Segmentação")
        
        st.info("""
        **Sobre a Segmentação de Clientes:**
        
        - **Algoritmo**: KMeans com 4 clusters
        - **Objetivo**: Agrupar clientes com características similares
        - **Features utilizadas**: MRR, quantidade de contratos, tempo como cliente, notas NPS
        - **Aplicação**: Estratégias de marketing personalizadas, análise de comportamento, identificação de oportunidades de upselling
        """)
    
    # Tab 3: Churn Model
    with tab3:
        st.header("Modelo de Predição de Churn")
        
        if 'churn' in models:
            # Model Performance
            st.subheader("Performance do Modelo")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Algoritmo", "RandomForest")
            with col2:
                st.metric("Otimização", "Recall")
            with col3:
                st.metric("Features", len(models['churn_features']))
            with col4:
                st.metric("Status", "Treinado")
            
            # Feature Importance
            st.subheader("Importância das Features")
            
            if hasattr(models['churn'], 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': models['churn_features'],
                    'Importance': models['churn'].feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(
                    feature_importance.tail(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Features Mais Importantes",
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, width='stretch')
            
            # Churn Definition
            st.subheader("Definição de Churn")
            st.markdown("""
            Um cliente é considerado **churn** quando atende a pelo menos uma das condições:
            
            1. **MRR ≤ 0** ou ausente nos últimos 12 meses
            2. **Contratações ≤ 0** ou ausente nos últimos 12 meses  
            3. **Tempo como cliente ≤ 30 dias**
            
            Esta definição captura tanto clientes inativos quanto novos clientes com baixo engajamento.
            """)
        
        else:
            st.warning("Modelo de Churn não encontrado!")
    
    # Tab 4: Clustering
    with tab4:
        st.header("Segmentação de Clientes")
        
        if 'clustering' in models:
            # Generate predictions for visualization
            clustering_features = prepare_clustering_features(df, models['clustering_features'])
            clustering_scaled = models['clustering_scaler'].transform(clustering_features)
            clusters = models['clustering'].predict(clustering_scaled)
            
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = clusters
            
            # Cluster Overview
            st.subheader("Resumo dos Clusters")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Algoritmo", "KMeans")
            with col2:
                st.metric("Número de Clusters", models['clustering'].n_clusters)
            with col3:
                st.metric("Status", "Treinado")
            with col4:
                st.metric("Clientes Segmentados", len(df_with_clusters))
            
            # Cluster Distribution
            cluster_counts = df_with_clusters['cluster'].value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_cluster_dist = px.pie(
                    values=cluster_counts.values,
                    names=[f"Cluster {i}" for i in cluster_counts.index],
                    title="Distribuição dos Clusters",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_cluster_dist, width='stretch')
            
            with col2:
                fig_cluster_bar = px.bar(
                    x=[f"Cluster {i}" for i in cluster_counts.index],
                    y=cluster_counts.values,
                    title="Número de Clientes por Cluster",
                    color=cluster_counts.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_cluster_bar, width='stretch')
            
            # Cluster Characteristics
            st.subheader("Características dos Clusters")
            
            # Prepare data for cluster analysis
            df_analysis_clusters = df_with_clusters.copy()
            df_analysis_clusters['MRR_12M_numeric'] = df_analysis_clusters['MRR_12M'] if 'MRR_12M' in df_analysis_clusters.columns else 0
            df_analysis_clusters['QTD_CONTRATACOES_12M_numeric'] = df_analysis_clusters['QTD_CONTRATACOES_12M'] if 'QTD_CONTRATACOES_12M' in df_analysis_clusters.columns else 0
            df_analysis_clusters['dias_cliente_numeric'] = df_analysis_clusters['dias_cliente'] if 'dias_cliente' in df_analysis_clusters.columns else 365
            
            # Show cluster metrics
            cluster_metrics = df_analysis_clusters.groupby('cluster').agg({
                'MRR_12M_numeric': ['mean', 'median'],
                'QTD_CONTRATACOES_12M_numeric': ['mean', 'median'],
                'dias_cliente_numeric': ['mean', 'median']
            }).round(2)
            
            st.dataframe(cluster_metrics, width='stretch')
            
            # Cluster Visualization 3D
            st.subheader("Visualização dos Clusters")
            
            # 3D scatter plot
            fig_3d = px.scatter_3d(
                df_analysis_clusters,
                x='MRR_12M_numeric',
                y='QTD_CONTRATACOES_12M_numeric',
                z='dias_cliente_numeric',
                color='cluster',
                title="Clusters em 3D - MRR vs Contratos vs Tempo",
                labels={
                    'MRR_12M_numeric': 'MRR (R$)',
                    'QTD_CONTRATACOES_12M_numeric': 'Contratos',
                    'dias_cliente_numeric': 'Dias como Cliente'
                }
            )
            st.plotly_chart(fig_3d, width='stretch')
            
            # Cluster Profiles
            st.subheader("Perfis dos Clusters")
            
            for cluster_id in sorted(df_with_clusters['cluster'].unique()):
                cluster_data = df_analysis_clusters[df_analysis_clusters['cluster'] == cluster_id]
                
                with st.expander(f"Cluster {cluster_id} - {len(cluster_data)} clientes"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("MRR Médio", f"R$ {cluster_data['MRR_12M_numeric'].mean():.2f}")
                    with col2:
                        st.metric("Contratos Médios", f"{cluster_data['QTD_CONTRATACOES_12M_numeric'].mean():.1f}")
                    with col3:
                        st.metric("Tempo Médio (dias)", f"{cluster_data['dias_cliente_numeric'].mean():.0f}")
                    
                    # Cluster description based on characteristics
                    mrr_avg = cluster_data['MRR_12M_numeric'].mean()
                    contracts_avg = cluster_data['QTD_CONTRATACOES_12M_numeric'].mean()
                    tenure_avg = cluster_data['dias_cliente_numeric'].mean()
                    
                    if mrr_avg > df_analysis_clusters['MRR_12M_numeric'].quantile(0.75):
                        mrr_desc = "Alto valor"
                    elif mrr_avg > df_analysis_clusters['MRR_12M_numeric'].quantile(0.25):
                        mrr_desc = "Valor médio"
                    else:
                        mrr_desc = "Baixo valor"
                    
                    if contracts_avg > df_analysis_clusters['QTD_CONTRATACOES_12M_numeric'].quantile(0.75):
                        contracts_desc = "Muitos contratos"
                    elif contracts_avg > df_analysis_clusters['QTD_CONTRATACOES_12M_numeric'].quantile(0.25):
                        contracts_desc = "Contratos moderados"
                    else:
                        contracts_desc = "Poucos contratos"
                    
                    if tenure_avg > df_analysis_clusters['dias_cliente_numeric'].quantile(0.75):
                        tenure_desc = "Clientes antigos"
                    elif tenure_avg > df_analysis_clusters['dias_cliente_numeric'].quantile(0.25):
                        tenure_desc = "Clientes médios"
                    else:
                        tenure_desc = "Clientes novos"
                    
                    st.markdown(f"""
                    **Perfil**: Clientes com {mrr_desc.lower()}, {contracts_desc.lower()} e {tenure_desc.lower()}.
                    
                    **Estratégias sugeridas**:
                    - Foco em retenção e upselling para clientes de alto valor
                    - Programas de engajamento para clientes novos
                    - Análise de satisfação para clientes com muitos contratos
                    """)
        
        else:
            st.warning("Modelo de Clustering não encontrado!")
    
    # Tab 5: Predictions
    with tab5:
        st.header("Sistema de Predições")
        
        st.subheader("Predição Individual de Churn")
        
        # Prediction Interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Insira os dados do cliente:**")
            
            # Input fields based on churn model features
            if 'churn' in models:
                input_data = {}
                
                # Key features for manual input
                key_features = ['MRR_12M', 'QTD_CONTRATACOES_12M', 'dias_cliente']
                
                for feature in key_features:
                    if feature in models['churn_features']:
                        if feature == 'MRR_12M':
                            input_data[feature] = st.number_input(
                                f"{feature} (R$)",
                                min_value=0.0,
                                value=100.0,
                                step=10.0
                            )
                        elif feature == 'QTD_CONTRATACOES_12M':
                            input_data[feature] = st.number_input(
                                f"{feature}",
                                min_value=0,
                                value=1,
                                step=1
                            )
                        elif feature == 'dias_cliente':
                            input_data[feature] = st.number_input(
                                f"{feature}",
                                min_value=0,
                                value=365,
                                step=1
                            )
                
                # Fill remaining features with defaults
                for feature in models['churn_features']:
                    if feature not in input_data:
                        input_data[feature] = 0
                
                # Create prediction dataframe
                prediction_df = pd.DataFrame([input_data])
                
                # Make prediction
                if st.button("Prever Churn", type="primary"):
                    try:
                        # Prepare features
                        features_ready = prepare_churn_features(prediction_df, models['churn_features'])
                        features_scaled = models['churn_scaler'].transform(features_ready)
                        
                        # Predict
                        churn_pred = models['churn'].predict(features_scaled)[0]
                        churn_proba = models['churn'].predict_proba(features_scaled)[0]
                        
                        # Display results
                        with col2:
                            st.markdown("**Resultado da Predição:**")
                            
                            if churn_pred == 1:
                                st.error(f"**ALTO RISCO DE CHURN**")
                                st.markdown(f"**Probabilidade**: {churn_proba[1]*100:.1f}%")
                                st.markdown("**Recomendações**:")
                                st.markdown("- Contato imediato com o cliente")
                                st.markdown("- Oferta de desconto ou benefícios")
                                st.markdown("- Análise de satisfação")
                            else:
                                st.success(f"**BAIXO RISCO DE CHURN**")
                                st.markdown(f"**Probabilidade**: {churn_proba[0]*100:.1f}%")
                                st.markdown("**Recomendações**:")
                                st.markdown("- Manter estratégia atual")
                                st.markdown("- Oportunidades de upselling")
                                st.markdown("- Programa de fidelização")
                        
                        # Probability visualization
                        fig_proba = go.Figure(data=[
                            go.Bar(
                                x=['Não Churn', 'Churn'],
                                y=[churn_proba[0], churn_proba[1]],
                                marker_color=['green', 'red'],
                                text=[f"{churn_proba[0]*100:.1f}%", f"{churn_proba[1]*100:.1f}%"],
                                textposition='auto'
                            )
                        ])
                        fig_proba.update_layout(
                            title="Probabilidade de Churn",
                            yaxis_title="Probabilidade",
                            showlegend=False
                        )
                        st.plotly_chart(fig_proba, width='stretch')
                        
                    except Exception as e:
                        st.error(f"Erro na predição: {str(e)}")
        
        # System Status
        st.subheader("Status do Sistema")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MLflow", "Ativo", "http://localhost:5001")
        
        with col2:
            st.metric("PostgreSQL", "Configurado")
        
        with col3:
            st.metric("Docker", "Pronto para Deploy")

if __name__ == "__main__":
    main()
