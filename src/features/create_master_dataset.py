import pandas as pd
import os
import logging
from pathlib import Path

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_processed_abate_data(processed_path: Path) -> pd.DataFrame:
    """Carrega a série temporal de abate de animais já processada."""
    file_path = processed_path / 'abate_base_para_analise.parquet'
    logging.info(f"Carregando dados de abate de: {file_path}")
    if not file_path.exists():
        logging.error("Arquivo de abate para análise não encontrado. Execute o script de features primeiro.")
        return None
    df = pd.read_parquet(file_path)
    # Pivotar para ter espécies como colunas e o ano como índice
    df_pivot = df.pivot(index='ano', columns='especie', values='quantidade_cabecas')
    # Transformar o índice de ano para um datetime no início de cada ano
    df_pivot.index = pd.to_datetime(df_pivot.index.astype(str) + '-01-01')
    return df_pivot


# --- Funções Placeholder para carregar dados externos ---
# Você precisará implementar a lógica de download e limpeza para cada fonte.
def load_economic_data(external_path: Path) -> pd.DataFrame:
    """
    Carrega dados econômicos (ex: Dólar, IPCA).
    Esta é uma função placeholder. Você precisa baixar e limpar os dados reais.
    """
    logging.warning("Usando dados econômicos de exemplo. Substitua pela sua implementação real.")
    # Exemplo: Criando dados fictícios mensais
    dates = pd.to_datetime(pd.date_range(start='2005-01-01', end='2024-12-31', freq='MS'))
    data = {
        'cambio_dolar': 2.5 + (dates.year - 2005) * 0.15 + (dates.month * 0.05),
        'ipca': 0.5 + (dates.month % 6) * 0.08
    }
    df = pd.DataFrame(data, index=dates)
    return df


def load_commodities_data(external_path: Path) -> pd.DataFrame:
    """
    Carrega dados de commodities (ex: Preço do Milho).
    Esta é uma função placeholder.
    """
    logging.warning("Usando dados de commodities de exemplo. Substitua pela sua implementação real.")
    dates = pd.to_datetime(pd.date_range(start='2005-01-01', end='2024-12-31', freq='MS'))
    data = {
        'preco_milho': 30 + (dates.year - 2005) * 2.5 - (dates.month * 0.5)
    }
    df = pd.DataFrame(data, index=dates)
    return df


def align_and_create_features(df_target: pd.DataFrame, list_of_feature_dfs: list) -> pd.DataFrame:
    """
    Alinha todos os DataFrames de features com o DataFrame alvo e cria features defasadas (lags).
    """
    logging.info("Alinhando DataFrames e criando features defasadas...")
    df_master = df_target.copy()

    # Junta todos os dataframes de features
    for df_feature in list_of_feature_dfs:
        df_master = df_master.join(df_feature, how='left')

    # Após juntar, precisamos reamostrar os dados anuais de abate para mensal
    # e preencher os valores para frente, já que o dado é anual.
    df_master = df_master.resample('MS').ffill()

    # Preencher quaisquer outros valores nulos (ex: no início da série)
    df_master.ffill(inplace=True)
    df_master.bfill(inplace=True)

    # --- Criação de Features Defasadas (Lags) ---
    # Esta é a etapa mais importante para modelos preditivos.
    feature_cols = ['cambio_dolar', 'ipca', 'preco_milho']
    for col in feature_cols:
        for lag in [1, 2, 3, 6, 12]:  # Defasagens de 1, 2, 3, 6 e 12 meses
            df_master[f'{col}_lag_{lag}m'] = df_master[col].shift(lag)

    # Remove linhas com valores nulos resultantes da criação dos lags
    df_master.dropna(inplace=True)

    logging.info("Base de dados mestre criada com sucesso.")
    return df_master


def main():
    """Orquestra a criação do dataset mestre para modelagem avançada."""
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_path = project_root / "data" / "processed"
    external_path = project_root / "data" / "external"  # Crie esta pasta para seus novos dados
    os.makedirs(external_path, exist_ok=True)

    # 1. Carregar o alvo (dados de abate)
    df_abate = load_processed_abate_data(processed_path)
    if df_abate is None:
        return

    # 2. Carregar todas as fontes de dados externas
    df_economia = load_economic_data(external_path)
    df_commodities = load_commodities_data(external_path)

    # 3. Alinhar, juntar e criar features
    df_master = align_and_create_features(df_abate, [df_economia, df_commodities])

    # 4. Salvar a base robusta final
    output_file = processed_path / 'master_dataset_para_modelagem_avancada.parquet'
    df_master.to_parquet(output_file)

    logging.info(f"Dataset mestre salvo em: {output_file}")
    logging.info("Colunas da base final:")
    print(df_master.info())
    logging.info("Amostra da base final:")
    print(df_master.head())


if __name__ == '__main__':
    main()
