import os
import requests
from urllib.parse import urlparse
import logging
import gzip
import shutil

# Configuração do logging para rastrear o carregamento, conforme o roadmap
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Lista de todas as URLs dos arquivos CSV (AGORA CORRIGIDA)
DATA_URLS = [
    "https://dados.agricultura.gov.br/dataset/6d3d141c-885e-41a4-ab7f-dc8ff323b96f/resource/a8875ff8-fe4d-4c3c-b1a1-3b19c32916f1/download/dados-abertos-tabua-de-risco.csv",
    "https://dados.agricultura.gov.br/dataset/6c913699-e82e-4da3-a0a1-fb6c431e367f/resource/d30b30d7-e256-484e-9ab8-cd40974e1238/download/agrofitprodutosformulados.csv",
    "https://dados.agricultura.gov.br/dataset/6c913699-e82e-4da3-a0a1-fb6c431e367f/resource/a200c70b-e025-4a9a-be1b-ec7275d7921f/download/agrofitprodutostecnicos.csv",
    "https://dados.agricultura.gov.br/dataset/062166e3-b515-4274-8e7d-68aadd64b820/resource/239eaa90-35cd-4b67-8902-d34eda3dca53/download/sigsifquantitativoanimaisabatidoscategoriauf.csv",
    "https://dados.agricultura.gov.br/dataset/062166e3-b515-4274-8e7d-68aadd64b820/resource/8c2cc427-bb38-4341-8b6f-a397a5f2da5c/download/sigsifcondenacaoanimaisporespecie.csv",
    "https://dados.agricultura.gov.br/dataset/062166e3-b515-4274-8e7d-68aadd64b820/resource/fcb7f87d-0092-4a52-a44b-b3550747b4c2/download/sigsifestabelecimentosnacionais.csv",
    "https://dados.agricultura.gov.br/dataset/062166e3-b515-4274-8e7d-68aadd64b820/resource/341dc717-4716-42ab-b189-c8d7a9d2a1ba/download/sigsifrelatorioabates.csv",
    "https://dados.agricultura.gov.br/dataset/062166e3-b515-4274-8e7d-68aadd64b820/resource/c6f5abb0-3b26-4c93-81c3-b3c88755baec/download/sigsifquantitativodoencasporprocedencia.csv",
    "https://dados.agricultura.gov.br/dataset/062166e3-b515-4274-8e7d-68aadd64b820/resource/7d02af92-e3cf-4ae4-af8a-0dad334ffdfa/download/sigsifrelatorioestabelecimentos.csv",
    "https://dados.agricultura.gov.br/dataset/062166e3-b515-4274-8e7d-68aadd64b820/resource/6b28acd9-2d55-4d0a-a9d1-06cb49952812/download/sigsifrelatoriodoencasporprocedencia.csv",
    "https://dados.agricultura.gov.br/dataset/062166e3-b515-4274-8e7d-68aadd64b820/resource/211df9dd-5242-4be2-8f65-56b893d8479a/download/sigsifestabelecimentosestrangeiros.csv",
    "https://dados.agricultura.gov.br/dataset/062166e3-b515-4274-8e7d-68aadd64b820/resource/09043a33-680c-4411-ae62-95f0bf522cf5/download/sigsifrelatorioabatesporanouf.csv",
    "https://dados.agricultura.gov.br/dataset/062166e3-b515-4274-8e7d-68aadd64b820/resource/97277e92-264a-4dc0-9aea-f87b8ea93798/download/sigsifestabelecimentosregistradosnosif.csv",
    "https://dados.agricultura.gov.br/dataset/062166e3-b515-4274-8e7d-68aadd64b820/resource/6003a82a-1415-4b0f-964e-d74869ed578e/download/sigsifrelatoriocondenacao.csv",
    "https://dados.agricultura.gov.br/dataset/c7784a6e-f0ec-4196-a1ce-1d2d4784a58e/resource/6ab20c11-73a0-4ab0-8e13-2420d48dd6f5/download/sigefcamposproducaodesementes.csv",
    "https://dados.agricultura.gov.br/dataset/c7784a6e-f0ec-4196-a1ce-1d2d4784a58e/resource/3fc8e266-ec41-40b0-8d62-157b91b36b2c/download/sigefdeclaracaoareaproducao.csv",
    "https://dados.agricultura.gov.br/dataset/52a01565-72d6-410e-b21b-64035831a7be/resource/e0bbc9d5-f161-448b-a6d4-c7beb312ec33/download/sipeagrofertilizante.csv",
    "https://dados.agricultura.gov.br/dataset/52a01565-72d6-410e-b21b-64035831a7be/resource/e9368a44-4c15-4218-ad96-1a62052ff2c6/download/sipeagroqualidadevegetal.csv",
    # LINHA CORRIGIDA ABAIXO
    "https://dados.agricultura.gov.br/dataset/52a01565-72d6-410e-b21b-64035831a7be/resource/7ce5fac0-9c8f-4e14-82d9-6deab9b5e2e9/download/sipeagroprodutoveterinario.csv",
    "https://dados.agricultura.gov.br/dataset/52a01565-72d6-410e-b21b-64035831a7be/resource/8ef7a4fc-f9d9-495b-b3ae-a2ffe931ff82/download/sipeagrovinhosebebidas.csv",
    "https://dados.agricultura.gov.br/dataset/52a01565-72d6-410e-b21b-64035831a7be/resource/378b184b-67bd-48fa-9901-17bb7a0700fb/download/sipeagroalimentacaoanimal.csv",
    "https://dados.agricultura.gov.br/dataset/52a01565-72d6-410e-b21b-64035831a7be/resource/837ed03c-e030-4304-8028-359224d6811b/download/sipeagroavesreproducao.csv",
    "https://dados.agricultura.gov.br/dataset/52a01565-72d6-410e-b21b-64035831a7be/resource/fac50de6-d4c4-4b47-bac9-5b144de448c9/download/sipeagroaviacaoagricolaregistro.csv",
    "https://dados.agricultura.gov.br/dataset/52a01565-72d6-410e-b21b-64035831a7be/resource/0626b768-7a83-49ba-a0c7-2365880e383d/download/sipeagroaviacaoagricolaautorizacao.csv",
    "https://dados.agricultura.gov.br/dataset/d68e269e-dbe5-44d9-83ec-1f0871427773/resource/97038867-7afc-4f93-85ef-f39cf8368581/download/siszarc_cronograma.csv.gz",
]

RAW_DATA_DIR = '../data/raw'  # Alterado para ser relativo à localização do script em src/data


def download_and_extract_data(url_list, save_dir):
    # Garante que o caminho do diretório seja relativo ao diretório raiz do projeto
    # __file__ é o caminho do script atual (download_data.py)
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    full_save_dir = os.path.join(project_root, save_dir)

    os.makedirs(full_save_dir, exist_ok=True)
    logging.info(f"Diretório '{full_save_dir}' pronto para receber os dados.")

    for url in url_list:
        try:
            file_name = os.path.basename(urlparse(url).path)
            file_path = os.path.join(full_save_dir, file_name)

            logging.info(f"Iniciando download de: {url}")
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logging.info(f"SUCESSO - Arquivo salvo em: {file_path}")

            if file_path.endswith('.gz'):
                unzipped_file_path = file_path[:-3]
                with gzip.open(file_path, 'rb') as f_in:
                    with open(unzipped_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(file_path)
                logging.info(f"SUCESSO - Arquivo descompactado para: {unzipped_file_path}")

        except requests.exceptions.RequestException as e:
            logging.error(f"FALHA - Erro de rede ao baixar {url}: {e}")
        except Exception as e:
            logging.error(f"FALHA - Erro inesperado ao processar {url}: {e}")


if __name__ == '__main__':
    logging.info("--- INICIANDO PIPELINE: Etapa 1.1 - Carregamento dos Dados ---")
    download_and_extract_data(DATA_URLS, 'data/raw')
    logging.info("--- PROCESSO DE DOWNLOAD CONCLUÍDO ---")