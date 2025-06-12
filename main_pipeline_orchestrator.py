# main_pipeline_orchestrator.py

import json
from pathlib import Path
import logging

# Importa as classes dos nossos módulos
from src.pipeline import PredictionPipeline
from src.models.evaluate_model import ModelEvaluator
from src.models.tune_xgboost_model import HyperparameterTuner

# --- Configuração Global ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
REPORTS_DIR = Path("reports")


def run_optimization_pipeline(config, scenarios, n_trials=50):
    """
    Executa o pipeline completo de avaliação e otimização.
    """
    pipeline = PredictionPipeline(config)
    df_full = pipeline.load_and_preprocess_data()
    if df_full is None:
        logging.error("Falha ao carregar dados. A abortar a otimização.")
        return

    evaluator = ModelEvaluator(validation_start_date=config['validation_start_date'])
    tuning_results = {config['name']: {}}

    for scenario in scenarios:
        scenario_key = "_".join(str(v) for v in scenario.values())
        logging.info(f"\n[{config['name']}] ### OTIMIZANDO CENÁRIO: {scenario_key} ###")

        query = " & ".join([f"`{col}` == '{val}'" for col, val in scenario.items()])
        df_scenario = df_full.query(query)

        if df_scenario.empty:
            logging.warning(f"Cenário '{scenario_key}' sem dados. A pular.")
            continue

        df_train_raw, df_val_raw = evaluator.split_data(df_scenario, config['date_column'])

        if df_val_raw.empty:
            logging.warning(f"Cenário '{scenario_key}' sem dados de validação. A pular otimização.")
            continue

        df_train_featured = pipeline._create_features(df_train_raw)
        df_val_featured = pipeline._create_features(df_val_raw)
        df_val_featured = df_val_featured.reindex(columns=df_train_featured.columns, fill_value=0)

        tuner = HyperparameterTuner(df_train_featured, df_val_featured, config['target_column'])
        scenario_tuning_result = tuner.tune(n_trials=n_trials)
        tuning_results[config['name']][scenario_key] = scenario_tuning_result

    # Salva o relatório de otimização
    report_path = REPORTS_DIR / "hyperparameter_tuning_results.json"
    logging.info(f"A guardar relatório de otimização em: {report_path}")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(tuning_results, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    abate_config = {
        'name': 'abate_animais', 'filename': 'sigsifrelatorioabatesporanouf.csv',
        'target_column': 'quantidade_cabecas', 'date_column': 'data', 'date_format': '%m/%Y',
        'group_by_columns': ['uf', 'especie_normalizada'],
        'column_mapping': {'data': ['MÊS/ANO', 'MES_ANO'], 'uf': ['UF', 'UF_PROCEDENCIA'],
                           'especie_original': ['ANIMAL', 'CATEGORIA'], 'quantidade_cabecas': ['QUANTIDADE', 'QTD']},
        'processing_func': 'process_abate_data', 'validation_start_date': '2023-01-01'
    }

    scenarios = [
        {'uf': 'MT', 'especie_normalizada': 'BOVINOS'},
        {'uf': 'PR', 'especie_normalizada': 'AVES'},
        {'uf': 'SC', 'especie_normalizada': 'SUÍNOS'}
    ]

    # Executa a otimização (pode ser executado com menos frequência, apenas para atualizar os parâmetros)
    run_optimization_pipeline(abate_config, scenarios, n_trials=20)
