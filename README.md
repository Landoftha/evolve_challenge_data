# 🚀 Projeto Evolve 

Plataforma inteligente de segmentação de clientes B2B, desenvolvida para a TOTVS, com foco em retenção, personalização e eficiência operacional. 

---

## 🏗️ Arquitetura do Projeto 
O projeto está dividido em cinco fases: 
1. **Engenharia de Dados**: Diagnóstico, limpeza, unificação e carga no SQL
2. **Análise e Modelagem**: EDA, feature engineering, clusterização e predição
3. **Implantação (MLOps)**: Inferência, containerização e deploy automatizado
4. **Validação**: Views para Power BI e validação cruzada
5. **Manutenção**: Monitoramento de drift e estratégia de retreinamento

## 🛠️ Tecnologias Utilizadas
- Python 3.10 - Pandas / NumPy / Scikit-learn
- MLflow (gestão de modelos)
- SQL Server (camadas Silver e Gold)
- Docker (containerização)
- AWS EC2 + cron (deploy e agendamento)
- Power BI (visualização de insights)

## ▶️ Como Executar Localmente 
1. Clonar o repositório
Se ainda não tiver feito isso:
```bash
git clone https://github.com/Landoftha/evolve_challenge_data
```

2. Criar e ativar o ambiente virtual (opcional, mas recomendado)
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Instale as dependências: 
```bash 
pip install -r requirements.txt
```

4. Executar o script principal
```bash
python src/build_dataset.py
```

5. Rode os notebooks na pasta notebooks/ para análise e modelagem na seguinte ordem:
- 01_data_profiling
- 02_eda.ipynb
- 03_clustering_training.ipynb
- 04_churn_training.ipynb
- 05_data_drift.ipynb

6. Execute o script de inferência:
```bash
python src/predict.py
```

7. Para containerizar:
```bash
docker build -t evolve-infer .
docker run evolve-infer 
```

## 📊 Dashboard e Visualização

Este projeto inclui um dashboard desenvolvido em Power BI para monitoramento de churn e performance do modelo.

- O dashboard consome os dados gerados pelo pipeline automatizado.
- Alertas e KPIs são atualizados conforme o modelo detecta risco de churn.
- A visualização está integrada à jornada do cliente e apoia decisões de retenção.

---

> ⚠️ Por questões de licença e acesso, o arquivo `.pbix` não está incluído no repositório. Para visualizar, entre em contato ou utilize os dados exportados via Docker.

---

## 👥 Autoras 
- Aline Manente: @linemanente
- Caroline Maia: @MaiaCaroline
- Isabelle Manzano: @IsaManza
- Luana Segal: @luanasegal
- Thays Costa: @Landoftha
