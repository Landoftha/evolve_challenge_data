# src/feature_engineering.py (VERSÃO CORRIGIDA)

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def processar_features_para_modelagem(df: pd.DataFrame, target_col: str = None):
    """
    Prepara o dataset unificado para modelagem, aplicando feature engineering,
    scaling e encoding.
    """
    df_eng = df.copy()
    
    # 1. Criação de Atributos
    df_eng['CLIENTE_DESDE'] = pd.to_datetime(df_eng['CLIENTE_DESDE'], errors='coerce')
    hoje = pd.Timestamp('today').normalize()
    df_eng['tempo_cliente_meses'] = ((hoje - df_eng['CLIENTE_DESDE']).dt.days / 30).round().astype('Int64', errors='ignore')
    df_eng['categoria_nps'] = pd.cut(df_eng['resposta_NPS'], bins=[-1, 6, 8, 10], labels=['Detrator', 'Neutro', 'Promotor'])
    df_eng['faixa_mrr'] = pd.qcut(df_eng['MRR_12M'], 4, labels=['Q1 (Baixo)', 'Q2 (Médio-Baixo)', 'Q3 (Médio-Alto)', 'Q4 (Alto)'])

    # 2. Definição da Variável Alvo e separação de IDs
    y = None
    if target_col and target_col in df_eng.columns:
        y = df_eng[target_col]
        df_eng = df_eng.drop(columns=[target_col])
    
    ids = df_eng['id_cliente'] # Guarda os IDs
    df_features = df_eng.drop(columns=['id_cliente']) # **CORREÇÃO: Remove o ID antes de processar**

    # 3. Seleção de Features e Definição de Tipos
    features_numericas = df_features.select_dtypes(include=np.number).columns.tolist()
    features_categoricas = df_features.select_dtypes(exclude=np.number).columns.tolist()

    # 4. Criação dos Pipelines de Pré-processamento
    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))])

    # 5. Combinação dos Pipelines com ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features_numericas),
            ('cat', categorical_transformer, features_categoricas)
        ],
        remainder='drop' # Ignora colunas não especificadas
    )

    # 6. Aplicação do Pré-processamento
    X_processed = preprocessor.fit_transform(df_features)

    # 7. Reconstrução do DataFrame
    ohe_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(features_categoricas)
    final_feature_names = features_numericas + list(ohe_feature_names)
    
    df_final = pd.DataFrame(X_processed, columns=final_feature_names, index=ids)
    
    print("Engenharia de atributos concluída. IDs separados das features.")
    
    if y is not None:
        return df_final, y
    return df_final