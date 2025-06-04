# src/data/diagnose_columns.py
import pandas as pd
import logging
from pathlib import Path

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Lista dos arquivos que falharam no log anterior
FILES_TO_DIAGNOSE = [
    "sigsifestabelecimentosnacionais.csv",
    "sipeagrofertilizante.csv",
    "agrofitprodutosformulados.csv"
]


def diagnose_csv_columns(file_list):
    """
    Lê o cabeçalho de cada CSV e imprime a lista de colunas para diagnóstico.
    """
    logging.info("--- Iniciando Ferramenta de Diagnóstico de Colunas ---")

    for filename in file_list:
        file_path = RAW_DATA_DIR / filename

        print("\n" + "=" * 60)
        logging.info(f"Analisando arquivo: {filename}")
        print("=" * 60)

        if not file_path.exists():
            logging.error(f"Arquivo não encontrado: {file_path}")
            continue

        try:
            # Lê apenas as primeiras 5 linhas para obter as colunas sem carregar o arquivo inteiro
            df = pd.read_csv(file_path, sep=';', encoding='latin1', on_bad_lines='skip', nrows=5)

            print("Nomes exatos das colunas encontradas no arquivo:")
            # Imprime a lista de colunas para fácil cópia e cola
            print(df.columns.tolist())

            print("\nAmostra dos dados:")
            print(df.head())

        except Exception as e:
            logging.error(f"Não foi possível ler o arquivo '{filename}'. Erro: {e}")


if __name__ == "__main__":
    diagnose_csv_columns(FILES_TO_DIAGNOSE)

