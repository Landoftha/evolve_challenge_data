# build_dataset.py

import pandas as pd
import os
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings('ignore')

# Importa as funções da nossa "caixa de ferramentas"
from src.data_processing import (
    padronizar_id_cliente,
    tratar_tipos_dados,
    remover_duplicatas,
    tratar_valores_faltantes_nps
)

def main():
    """
    Função principal que orquestra o pipeline de ETL completo:
    1. Extração: Carrega os dados dos arquivos CSV.
    2. Transformação: Limpa, padroniza e unifica os dados.
    3. Carga: Salva o resultado em um CSV e em uma tabela no SQL Server.
    """
    print("--- INICIANDO PIPELINE DE ENGENHARIA DE DADOS ---")

    # --- 1. EXTRAÇÃO (Extraction) ---
    print("\n[ETAPA 1/3] Carregando dados das fontes...")
    try:
        mrr_df = pd.read_csv("data/raw/mrr.csv", sep=';')
        clientes_df = pd.read_csv("data/raw/clientes_desde.csv", sep=';')
        contratacoes_df = pd.read_csv("data/raw/contratacoes_ultimos_12_meses.csv", sep=';')
        nps_df = pd.read_csv("data/raw/nps_relacional.csv", sep=';')
        print("   - Dados carregados com sucesso.")
    except FileNotFoundError as e:
        print(f"   - ERRO: Arquivo não encontrado. Verifique o caminho: {e}")
        return

    # --- 2. TRANSFORMAÇÃO (Transformation) ---
    print("\n[ETAPA 2/3] Aplicando limpeza e transformação nos dados...")

    # Padroniza os IDs de cliente em todos os dataframes
    print(" - Padronizando IDs de cliente...")
    mrr_df = padronizar_id_cliente(mrr_df)
    clientes_df = padronizar_id_cliente(clientes_df)
    contratacoes_df = padronizar_id_cliente(contratacoes_df)
    nps_df = padronizar_id_cliente(nps_df)

    # Aplica tratamentos de tipo específicos
    print(" - Corrigindo tipos de dados...")
    clientes_df = tratar_tipos_dados(clientes_df)
    contratacoes_df = tratar_tipos_dados(contratacoes_df)

    # Aplica limpeza específica para o dataset de NPS
    print(" - Limpando dataset de NPS...")
    nps_df = remover_duplicatas(nps_df)
    nps_df = tratar_valores_faltantes_nps(nps_df)

    # Unificação dos dados
    print(" - Unificando os datasets...")
    # Começamos com a lista de clientes como base para não perder ninguém
    df_unificado = pd.merge(clientes_df, mrr_df, on='id_cliente', how='left')
    df_unificado = pd.merge(df_unificado, contratacoes_df, on='id_cliente', how='left')
    
    # Para o NPS, podemos ter múltiplas respostas por cliente.
    # Vamos agregar, pegando a média das notas e a resposta mais recente.
    nps_agg = nps_df.groupby('id_cliente').agg({
        'resposta_NPS': 'mean',
        'Nota_SupTec_Agilidade': 'mean',
        'Nota_SupTec_Atendimento': 'mean',
        'Nota_Comercial': 'mean',
        'Nota_Custos': 'mean',
        'Nota_AdmFin_Atendimento': 'mean',
        'Nota_Software': 'mean',
        'Nota_Software_Atualizacao': 'mean',
        'respondedAt': 'max' # Pega a data da última resposta
    }).reset_index()
    nps_agg.rename(columns={'respondedAt': 'data_ultima_resposta_nps'}, inplace=True)
    
    df_unificado = pd.merge(df_unificado, nps_agg, on='id_cliente', how='left')
    print(f"   - Dataset unificado criado com {len(df_unificado)} linhas.")
    
    # --- 3. CARGA (Load) ---
    print("\n[ETAPA 3/3] Carregando dados para os destinos...")
    
    # Destino 1: Arquivo CSV (camada Silver)
    try:
        output_path = "data/dataset_unificado.csv"
        df_unificado.to_csv(output_path, index=False, sep=';', decimal=',')
        print(f"   - Arquivo '{output_path}' salvo com sucesso.")
    except Exception as e:
        print(f"   - ERRO ao salvar CSV: {e}")

    # Destino 2: Tabela no SQL Server (camada Silver)
    try:
        db_server = os.getenv('DB_SERVER')
        db_database = os.getenv('DB_DATABASE')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')

        if not all([db_server, db_database, db_user, db_password]):
            raise ValueError("Variáveis de ambiente do banco de dados não estão configuradas.")

        connection_string = f"mssql+pyodbc://{db_user}:{db_password}@{db_server}/{db_database}?driver=ODBC+Driver+17+for+SQL+Server"
        engine = create_engine(connection_string)
        
        table_name = 'clientes_unificados'
        df_unificado.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"   - Tabela '{table_name}' carregada no SQL Server com sucesso.")
    
    except Exception as e:
        print(f"   - ERRO na carga para o SQL Server: {e}")

    print("\n--- PIPELINE CONCLUÍDO ---")

if __name__ == '__main__':
    main()