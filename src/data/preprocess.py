# src/data/master_preprocess.py

import pandas as pd
import logging
from pathlib import Path

# --- Configuração de Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)

# --- Definição de Caminhos ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(exist_ok=True)  # Garante que o diretório exista

# --- Dicionário de Configuração de Datasets (CORRIGIDO) ---
# Com base nos resultados da ferramenta de diagnóstico, ajustamos a configuração.
DATASET_CONFIG = {
    "abates": {
        "filename": "sigsifrelatorioabatesporanouf.csv",
        "is_primary": True,
        "cols_rename": {
            'MES_ANO': 'data',
            'UF_PROCEDENCIA': 'uf',
            'CATEGORIA': 'especie_original',
            'QTD': 'quantidade_cabecas'
        },
        "date_cols": {"data": "%m/%Y"},
        "join_keys": ['ano', 'uf']
    },
    "estabelecimentos": {
        "filename": "sigsifestabelecimentosnacionais.csv",
        # Como não há 'Status', não podemos filtrar por 'Ativo'.
        # Vamos apenas contar o número de registros por UF.
        "cols_rename": {'UF': 'uf'},
        "processing_func": "process_estabelecimentos",
        "join_keys": ['uf']
    },
    "fertilizantes_estab": {  # Renomeado para refletir que são estabelecimentos
        "filename": "sipeagrofertilizante.csv",
        # Usando as colunas reais encontradas pelo diagnóstico
        "cols_rename": {
            'UNIDADE_DA_FEDERACAO': 'uf',
            'STATUS_DO_REGISTRO': 'status'
        },
        "processing_func": "process_fertilizantes_estab",
        "join_keys": ['uf']
    },
    # "agrofit_formulados": {
    #     # ESTE DATASET FOI DESATIVADO.
    #     # Motivo: O diagnóstico mostrou que ele não possui uma coluna de 'UF',
    #     # impedindo a correlação geográfica com os dados de abate.
    #     "filename": "agrofitprodutosformulados.csv",
    #     "cols_rename": {},
    #     "processing_func": "process_agrofit",
    #     "join_keys": ['uf']
    # }
}


# ==============================================================================
# Funções de Processamento Específicas (ADAPTADAS)
# ==============================================================================

def normalize_species(df):
    """Normaliza as categorias de espécies."""
    if 'especie_original' not in df.columns: return df
    logging.info("Normalizing species categories...")
    especie_map = {
        'BOVINOS': ['BOVINO', 'NOVILHO', 'VACA', 'VITELO'],
        'SUÍNOS': ['SUINO', 'LEITAO', 'SUÍNA'],
        'AVES': ['AVE', 'FRANGO', 'GALINHA', 'PERU', 'AVESTRUZ']
    }
    df['especie_normalizada'] = None
    for categoria, termos in especie_map.items():
        for termo in termos:
            df.loc[df['especie_original'].str.contains(termo, case=False, na=False), 'especie_normalizada'] = categoria
    return df


def process_estabelecimentos(df):
    """Processa dados de estabelecimentos SIF (contagem por UF)."""
    if 'uf' not in df.columns: return None
    df_agg = df.groupby('uf').size().reset_index(name='n_estabelecimentos_sif')
    return df_agg


def process_fertilizantes_estab(df):
    """Processa dados de estabelecimentos de fertilizantes."""
    if 'status' not in df.columns or 'uf' not in df.columns: return None
    df_ativos = df[df['status'] == 'ATIVO'].copy()  # O status real é 'ATIVO'
    df_agg = df_ativos.groupby('uf').size().reset_index(name='n_estab_fert_ativos')
    return df_agg


# ==============================================================================
# Orquestrador Principal do Pipeline
# ==============================================================================

def run_master_preprocessing_pipeline():
    logging.info("--- INICIANDO PIPELINE MESTRE DE PRÉ-PROCESSAMENTO (VERSÃO FINAL) ---")
    processed_dfs = {}
    primary_df_key = None

    # Etapa 1: Carregamento e Limpeza Individual
    for key, config in DATASET_CONFIG.items():
        file_path = RAW_DATA_DIR / config['filename']
        if not file_path.exists():
            logging.warning(f"Arquivo '{config['filename']}' não encontrado. Pulando.")
            continue

        logging.info(f"Processando arquivo: {config['filename']}")
        df = pd.read_csv(file_path, sep=';', encoding='latin1', low_memory=False, on_bad_lines='skip')

        # Renomeação robusta de colunas
        if 'cols_rename' in config:
            df.rename(columns=lambda c: c.strip(), inplace=True)  # Remove espaços em branco
            rename_map = {k: v for k, v in config['cols_rename'].items() if k in df.columns}
            df.rename(columns=rename_map, inplace=True)
            logging.info(f"Colunas renomeadas para '{key}': {rename_map}")

        if 'date_cols' in config:
            for col, fmt in config['date_cols'].items():
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors='coerce')
            df.dropna(subset=config['date_cols'].keys(), inplace=True)

        if 'processing_func' in config:
            df_result = globals()[config['processing_func']](df.copy())
            if df_result is None:
                logging.warning(f"Processamento para '{key}' falhou ou foi pulado.")
                continue
            df = df_result

        processed_dfs[key] = df
        if config.get('is_primary', False): primary_df_key = key

    if not primary_df_key or primary_df_key not in processed_dfs:
        logging.error("Tabela primária não foi processada com sucesso. Abortando.")
        return

    # Etapa 2: Preparação da Tabela Fato
    df_master = processed_dfs[primary_df_key]
    df_master['ano'] = df_master['data'].dt.year
    df_master = normalize_species(df_master)
    if 'especie_normalizada' in df_master.columns: df_master.dropna(subset=['especie_normalizada'], inplace=True)
    df_master = df_master.groupby(['data', 'ano', 'uf', 'especie_normalizada'])[
        'quantidade_cabecas'].sum().reset_index()
    logging.info(f"Tabela fato principal '{primary_df_key}' processada. Shape: {df_master.shape}")

    # Etapa 3: Junção (Merge) Iterativa
    for key, df_to_merge in processed_dfs.items():
        if key == primary_df_key: continue
        logging.info(f"Realizando merge com '{key}' usando as chaves: {DATASET_CONFIG[key]['join_keys']}")
        df_master = pd.merge(df_master, df_to_merge, on=DATASET_CONFIG[key]['join_keys'], how='left')
        logging.info(f"Shape após merge: {df_master.shape}")

    # Etapa 4: Tratamento Pós-Merge
    logging.info("Realizando tratamento pós-merge (imputação de nulos)...")
    count_cols = [col for col in df_master.columns if col.startswith('n_')]
    df_master[count_cols] = df_master[count_cols].fillna(0).astype(int)

    # Etapa 5: Salvando o Artefato Final
    output_file = PROCESSED_DATA_DIR / 'master_analytical_table.parquet'
    df_master.to_parquet(output_file, index=False)

    logging.info(f"--- SUCESSO! Tabela analítica mestre salva em: {output_file} ---")
    logging.info("Amostra da Tabela Final:")
    print(df_master.head())
    logging.info(f"Shape final: {df_master.shape}")


if __name__ == "__main__":
    run_master_preprocessing_pipeline()
