# Evolve Challenge — Pipeline de Dados, Segmentação & Predição de Churn (MLOps)

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![SQL Server](https://img.shields.io/badge/SQL-SQLAlchemy_%2F_SQL_Server-CC2927?logo=microsoftsqlserver&logoColor=white)](https://www.sqlalchemy.org/)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

Pipeline ponta a ponta de **Engenharia de Dados, Machine Learning e MLOps** desenvolvido para unificar bases operacionais (MRR, histórico contratual e pesquisas de NPS), segmentar perfis de clientes via clusterização e prever o risco de cancelamento (**Churn**) com rastreamento de experimentos em **MLflow** e interface interativa em **Streamlit**.

---

## Sumário Executivo & Resultados

- **Problema de Negócio:** Identificar preventivamente clientes B2B com alta probabilidade de cancelamento (churn) e compreender como indicadores financeiros (**MRR nos últimos 12 meses**) e de satisfação (**NPS Relacional**, suporte técnico e atendimento comercial) impactam a retenção.
- **Solução Implementada:**
  1. **Pipeline ETL (`build_dataset.py`):** Extração de 4 fontes brutas (`mrr.csv`, `clientes_desde.csv`, `contratacoes_ultimos_12_meses.csv`, `nps_relacional.csv`), padronização de chaves (`id_cliente`), tratamento de duplicidades/ausentes, agregação histórica de notas de NPS e carga dupla em camada Silver (`dataset_unificado.csv` e tabela `clientes_unificados` via **SQLAlchemy / SQL Server**).
  2. **Engenharia de Atributos (`feature_engineering.py`):** Cálculo de *tenure* em meses (`tempo_cliente_meses`), categorização de NPS (*Detrator*, *Neutro*, *Promotor*), discretização em quartis de receita (`faixa_mrr`) e pipeline `ColumnTransformer` (`MinMaxScaler` + `OneHotEncoder`) sem vazamento de IDs.
  3. **Modelagem & MLOps (`03_clustering_training.ipynb`, `04_churn_training.ipynb`, `MLflow/`):** Treinamento de modelos de clusterização de clientes e classificação de churn com versionamento de hiperparâmetros, métricas e artefatos no **MLflow** (`sqlite:///mlflow.db`).
  4. **Inferência & Aplicação Web (`predict.py` e `streamlit_app.py`):** Módulo de predição em lote/individual e dashboard interativo em **Streamlit** conteinerizado via **Docker**.

---

## Arquitetura do Projeto

```text
evolve_challenge_data/
├── build_dataset.py              # Pipeline ETL (CSV -> Transformação -> CSV Silver & SQL Server)
├── feature_engineering.py        # Criação de features (tenure, faixa MRR, categoria NPS) e ColumnTransformer
├── 02_eda.ipynb                  # Análise Exploratória de Dados (EDA) detalhada
├── 03_clustering_training.ipynb  # Segmentação de clientes (Clustering) + registro no MLflow
├── 04_churn_training.ipynb       # Treinamento e avaliação do modelo de Churn + registro no MLflow
├── predict.py                    # Script de inferência em produção
├── streamlit_app.py              # Aplicação web interativa para simulação e visualização de risco
├── Dockerfile                    # Imagem Docker (Python 3.9-slim, usuário não-root e healthcheck)
├── requirements.txt              # Dependências do projeto
└── MLflow/                       # Artefatos e configurações de rastreamento de experimentos
```

---

## Como Executar

### 1. Ambiente Local

```bash
# Clone o repositório
git clone https://github.com/Landoftha/evolve_challenge_data.git
cd evolve_challenge_data

# Crie e ative o ambiente virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

### 2. Executar Pipeline ETL e Predição

```bash
# Construir base unificada (Camada Silver)
python build_dataset.py

# Executar inferência
python predict.py
```

### 3. Iniciar o Dashboard Streamlit

```bash
streamlit run streamlit_app.py
```

### 4. Executar via Docker

```bash
docker build -t evolve-churn-mlops .
docker run --rm -p 8501:8501 evolve-churn-mlops
```

---

## Autora

**Thays Yohana Costa** — [Portfólio](https://thays-portfolio-six.vercel.app) · [LinkedIn](https://www.linkedin.com/in/thays-yohana-46260527a/) · [GitHub](https://github.com/Landoftha)
