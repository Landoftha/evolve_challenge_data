import pandas as pd
import numpy as np

def padronizar_id_cliente(df: pd.DataFrame) -> pd.DataFrame:
    """
    Busca por colunas de identificação do cliente com nomes comuns
    e as renomeia para o padrão 'id_cliente'.

    Args:
        df (pd.DataFrame): O DataFrame de entrada.

    Returns:
        pd.DataFrame: O DataFrame com a coluna de ID padronizada.
    """
    colunas_possiveis = ['CLIENTE', 'CD_CLIENTE', 'metadata_codcliente']
    for col in colunas_possiveis:
        if col in df.columns:
            df.rename(columns={col: 'id_cliente'}, inplace=True)
            print(f"   - Coluna '{col}' renomeada para 'id_cliente'.")
            break
    return df

def tratar_tipos_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas para os tipos de dados corretos, conforme identificado
    na fase de profiling.

    Args:
        df (pd.DataFrame): O DataFrame de entrada.

    Returns:
        pd.DataFrame: O DataFrame com os tipos de dados corrigidos.
    """
    if 'VLR_CONTRATACOES_12M' in df.columns:
        # Substitui vírgula por ponto e converte para float
        df['VLR_CONTRATACOES_12M'] = df['VLR_CONTRATACOES_12M'].astype(str).str.replace(',', '.').astype(float)
        print("   - Coluna 'VLR_CONTRATACOES_12M' convertida para float.")

    if 'CLIENTE_DESDE' in df.columns:
        # Converte para datetime. 'coerce' transforma erros em NaT (Not a Time)
        df['CLIENTE_DESDE'] = pd.to_datetime(df['CLIENTE_DESDE'], errors='coerce')
        print("   - Coluna 'CLIENTE_DESDE' convertida para datetime.")
    
    return df

def remover_duplicatas(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Remove linhas duplicadas de um DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        subset (list, optional): Lista de colunas para considerar na
                                  identificação de duplicatas. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame sem linhas duplicadas.
    """
    linhas_antes = len(df)
    df.drop_duplicates(subset=subset, inplace=True)
    linhas_depois = len(df)
    if (linhas_antes - linhas_depois) > 0:
        print(f"   - {linhas_antes - linhas_depois} linhas duplicadas foram removidas.")
    return df

def tratar_valores_faltantes_nps(df_nps: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche valores nulos especificamente no dataset de NPS usando a mediana.

    Args:
        df_nps (pd.DataFrame): O DataFrame de NPS.

    Returns:
        pd.DataFrame: O DataFrame de NPS com valores nulos tratados.
    """
    # Identifica todas as colunas que representam uma nota
    colunas_notas = [col for col in df_nps.columns if 'Nota_' in col or 'resposta_' in col]
    
    for col in colunas_notas:
        if df_nps[col].isnull().sum() > 0:
            mediana = df_nps[col].median()
            df_nps[col].fillna(mediana, inplace=True)
            print(f"   - Nulos na coluna '{col}' preenchidos com a mediana ({mediana}).")
    return df_nps

# Bloco de teste: permite testar este script de forma independente
if __name__ == '__main__':
    print("Testando as funções de data_processing...")
    # Crie um DataFrame de exemplo para testar
    data_teste = {'CLIENTE': ['A', 'B', 'A'], 'VLR_CONTRATACOES_12M': ['100,50', '200,00', '100,50']}
    df_teste = pd.DataFrame(data_teste)
    
    print("\nTestando padronizar_id_cliente...")
    df_teste = padronizar_id_cliente(df_teste)
    assert 'id_cliente' in df_teste.columns
    
    print("\nTestando tratar_tipos_dados...")
    df_teste = tratar_tipos_dados(df_teste)
    assert df_teste['VLR_CONTRATACOES_12M'].dtype == 'float64'
    
    print("\nTestando remover_duplicatas...")
    df_teste = remover_duplicatas(df_teste)
    assert len(df_teste) == 2
    
    print("\nTestes concluídos com sucesso!")