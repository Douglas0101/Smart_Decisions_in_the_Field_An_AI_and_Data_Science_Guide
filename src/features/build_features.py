import pandas as pd
import os
import logging
from pathlib import Path

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Definição de Caminhos ---
# Usamos pathlib para uma manipulação de caminhos mais robusta e independente de sistema operacional.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEAN_ABATE_FILE = PROCESSED_DATA_DIR / 'abate_animal_limpo.csv'
ANALYSIS_READY_FILE = PROCESSED_DATA_DIR / 'abate_base_para_analise.parquet'

def create_aggregated_analysis_file(clean_file_path: Path, output_file_path: Path):
    """
    Carrega os dados de abate limpos, agrega por ano e espécie, e salva o
    resultado em formato Parquet, criando a base para as próximas etapas.

    Args:
        clean_file_path (Path): Caminho para o arquivo CSV limpo.
        output_file_path (Path): Caminho onde o arquivo Parquet agregado será salvo.
    """
    logging.info(f"Iniciando a criação do arquivo de análise a partir de: {clean_file_path.name}")
    try:
        # 1. Carregar os dados limpos
        df = pd.read_csv(clean_file_path, parse_dates=['data'])
        logging.info("Dados limpos carregados com sucesso.")

        # 2. Agregar os dados por ano e pela espécie normalizada.
        # Esta é a etapa crucial que cria a base para análise.
        df_agg = df.groupby(
            [df['data'].dt.year, 'especie_normalizada']
        )['quantidade_cabecas'].sum().reset_index()

        # Renomeia as colunas para o formato final
        df_agg.rename(columns={'data': 'ano', 'especie_normalizada': 'especie'}, inplace=True)
        logging.info("Dados agregados por ano e espécie.")

        # 3. Salvar o arquivo de análise em formato Parquet
        # Parquet é mais eficiente para armazenamento e leitura do que CSV.
        df_agg.to_parquet(output_file_path, index=False)
        logging.info(f"Arquivo de análise salvo com sucesso em: {output_file_path}")
        logging.info("Amostra dos dados agregados:")
        print(df_agg.head())

    except FileNotFoundError:
        logging.error(f"ERRO: Arquivo de entrada não encontrado em {clean_file_path}. "
                      "Execute o script de pré-processamento ('preprocess.py') primeiro.")
    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado durante a agregação: {e}")

# --- Bloco de Execução Principal ---
# Agora, quando você executa 'python src/features/build_features.py',
# ele executa a tarefa principal de criar o arquivo de análise.
if __name__ == '__main__':
    logging.info("--- Executando o Módulo de Construção de Features: Agregação Final ---")
    create_aggregated_analysis_file(CLEAN_ABATE_FILE, ANALYSIS_READY_FILE)
    logging.info("--- Módulo de Construção de Features Concluído ---")
