# src/main_pipeline_orchestrator.py

import pandas as pd
import xgboost as xgb
import logging
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Importar as classes dos nossos módulos
from models.evaluate_model import ModelEvaluator
from models.tune_xgboost_model import HyperparameterTuner

# --- Configuração Global ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT_PATH / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT_PATH / "reports"
MODELS_DIR = PROJECT_ROOT_PATH / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --- Funções de Processamento ---
def process_abate_data(df):
    if 'especie_original' not in df.columns:
        logging.error("[abate_animais] Coluna 'especie_original' não encontrada.")
        return None
    especie_map = {'BOVINOS': ['BOVINO', 'NOVILHO', 'VACA', 'VITELO'], 'SUÍNOS': ['SUINO', 'LEITAO', 'SUÍNA'],
                   'AVES': ['AVE', 'FRANGO', 'GALINHA', 'PERU', 'AVESTRUZ']}
    df['especie_normalizada'] = None
    for categoria, termos in especie_map.items():
        mask = df['especie_original'].astype(str).str.contains('|'.join(termos), case=False, na=False)
        df.loc[mask, 'especie_normalizada'] = categoria
    df.dropna(subset=['especie_normalizada'], inplace=True)
    if 'especie_original' in df.columns: df.drop(columns=['especie_original'], inplace=True)
    if 'uf' in df.columns: df['uf'] = df['uf'].astype(str).str.upper().str.strip()
    return df


class PredictionPipeline:
    def __init__(self, task_config):
        self.config = task_config
        self.task_name = task_config['name']
        self.target_col = task_config['target_column']
        self.date_col = task_config['date_column']
        self.group_by_cols = task_config['group_by_columns']
        self.default_params = {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5, 'colsample_bytree': 0.7,
                               'subsample': 0.8, 'n_jobs': -1, 'random_state': 42}
        self.quantiles = [0.05, 0.5, 0.95]
        logging.info(f"Pipeline inicializado para a tarefa: '{self.task_name}'")

    def _map_and_validate_columns(self, df):
        rename_map = {}
        original_cols_stripped = {c.strip().upper(): c for c in df.columns}
        for final_name, possible_names in self.config['column_mapping'].items():
            found = False
            for p_name in possible_names:
                p_name_upper = p_name.strip().upper()
                if p_name_upper in original_cols_stripped:
                    rename_map[original_cols_stripped[p_name_upper]] = final_name
                    found = True
                    break
            if not found: logging.warning(
                f"[{self.task_name}] Coluna de mapeamento '{final_name}' não encontrada: {possible_names}")
        df.rename(columns=rename_map, inplace=True)
        return df

    def _load_and_preprocess_data(self):
        file_path = RAW_DATA_DIR / self.config['filename']
        if not file_path.exists():
            logging.error(f"[{self.task_name}] Ficheiro não encontrado: {file_path}")
            return None
        df = pd.read_csv(file_path, sep=';', encoding='latin1', on_bad_lines='skip', low_memory=False)
        df = self._map_and_validate_columns(df)
        if 'processing_func' in self.config: df = globals()[self.config['processing_func']](df)
        required_cols = [self.date_col, self.target_col] + self.group_by_cols
        if not all(col in df.columns for col in required_cols): return None
        df[self.date_col] = pd.to_datetime(df[self.date_col], format=self.config.get('date_format'), errors='coerce')
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')
        df.dropna(subset=[self.date_col, self.target_col], inplace=True)
        return df

    def _create_features(self, df):
        df_copy = df.copy()
        df_copy['ano'] = df_copy[self.date_col].dt.year
        df_copy['mes'] = df_copy[self.date_col].dt.month
        df_copy = df_copy.sort_values(by=self.group_by_cols + [self.date_col])
        for lag in [1, 3, 12]:
            if self.config.get('is_annual', False) and lag > 1: continue
            df_copy[f'{self.target_col}_lag_{lag}'] = df_copy.groupby(self.group_by_cols)[self.target_col].shift(lag)
        df_copy.dropna(inplace=True)
        df_copy = pd.get_dummies(df_copy, columns=self.group_by_cols, prefix=self.group_by_cols)
        return df_copy

    def _train_and_forecast(self, df_history, params, horizon_months):
        """Treina modelos para cada quantil e gera a previsão futura."""
        df_history_featured = self._create_features(df_history)
        feature_cols = [col for col in df_history_featured.columns if col not in [self.target_col, self.date_col]]

        forecasts = {}
        for q in self.quantiles:
            model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=q, **params)
            model.fit(df_history_featured[feature_cols], df_history_featured[self.target_col])

            # Previsão iterativa OTIMIZADA
            # Os lags usados para previsão são fixados a partir do último ponto do histórico.
            # Apenas as features de data (ano, mês) são atualizadas a cada passo da previsão.
            last_historical_features_dict = df_history_featured[feature_cols].iloc[-1].to_dict()

            future_dates = pd.date_range(
                start=df_history_featured[self.date_col].iloc[-1], # Baseado no último timestamp do histórico
                periods=horizon_months + 1, # +1 porque a primeira data é o start, queremos N períodos *após*
                freq='MS'
            )[1:] # [1:] para excluir a data de início, que já está no histórico

            predictions = []
            current_features_for_step = last_historical_features_dict.copy()

            for current_pred_date in future_dates:
                # Atualizar features de data para o passo de previsão atual
                current_features_for_step['ano'] = current_pred_date.year
                current_features_for_step['mes'] = current_pred_date.month
                # Se houver outras features baseadas em data (ex: trimestre), atualize-as aqui também.
                # Ex: if 'trimestre' in current_features_for_step:
                # current_features_for_step['trimestre'] = current_pred_date.quarter

                pred_input_df = pd.DataFrame([current_features_for_step], columns=feature_cols)
                prediction = model.predict(pred_input_df)[0]
                predictions.append(prediction)

                # Nesta estratégia, as features de lag em current_features_for_step
                # não são atualizadas com a 'prediction' atual. Elas permanecem as mesmas
                # do último ponto do histórico. As colunas dummy também não mudam.

            forecasts[q] = pd.Series(predictions, index=future_dates)

        return pd.DataFrame(forecasts)

    def run_pipeline(self, scenarios: list, horizon_months: int, n_trials: int = 50):
        df_full = self._load_and_preprocess_data()
        if df_full is None or df_full.empty:
            logging.error(f"[{self.task_name}] Não foi possível carregar os dados. A abortar.")
            return

        evaluator = ModelEvaluator(validation_start_date=self.config['validation_start_date'])
        tuning_results = {self.task_name: {}}

        for scenario in scenarios:
            scenario_key = "_".join(str(v) for v in scenario.values())
            logging.info(f"\n[{self.task_name}] ### PROCESSANDO CENÁRIO: {scenario_key} ###")

            query = " & ".join([f"`{col}` == '{val}'" for col, val in scenario.items()])
            df_scenario_full = df_full.query(query).copy()

            if df_scenario_full.empty:
                logging.warning(f"Cenário '{scenario_key}' sem dados históricos. A pular.")
                continue

            df_train_raw, df_val_raw = evaluator.split_data(df_scenario_full, self.date_col)
            best_params = self.default_params.copy()

            if not df_val_raw.empty:
                df_train_featured = self._create_features(df_train_raw)
                df_val_featured = self._create_features(df_val_raw)
                df_val_featured = df_val_featured.reindex(columns=df_train_featured.columns, fill_value=0)

                tuner = HyperparameterTuner(df_train_featured, df_val_featured, self.target_col)
                scenario_tuning_result = tuner.tune(n_trials=n_trials)
                tuning_results[self.task_name][scenario_key] = scenario_tuning_result
                best_params.update(scenario_tuning_result['best_params'])
            else:
                logging.warning(f"Cenário '{scenario_key}' sem dados de validação. A usar parâmetros padrão.")

            logging.info(f"A treinar modelo final e a gerar forecast para '{scenario_key}'...")
            forecast_df = self._train_and_forecast(df_scenario_full, best_params, horizon_months)

            # Plotagem
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(18, 8))
            ax.plot(df_scenario_full[self.date_col], df_scenario_full[self.target_col], label='Dados Históricos',
                    color='royalblue')
            if not forecast_df.empty:
                ax.plot(forecast_df.index, forecast_df[0.5], label='Projeção Otimizada (Mediana)', color='darkorange',
                        linestyle='--')
                ax.fill_between(forecast_df.index, forecast_df[0.05], forecast_df[0.95], color='orange', alpha=0.2,
                                label='Intervalo de Confiança (90%)')

            title = f"Projeção para {scenario_key.replace('_', ' ')}"
            ax.set_title(title, fontsize=16, weight='bold')
            ax.legend()
            fig_path = REPORTS_DIR / "figures" / f"forecast_{self.task_name}_{scenario_key}.png"
            fig.savefig(fig_path)
            logging.info(f"Gráfico de projeção guardado em: {fig_path}")
            plt.close(fig)

        report_path = REPORTS_DIR / "hyperparameter_tuning_results.json"
        logging.info(f"A guardar relatório de otimização em: {report_path}")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(tuning_results, f, ensure_ascii=False, indent=4)


# ==============================================================================
# DEFINIÇÃO E EXECUÇÃO DAS TAREFAS
# ==============================================================================
if __name__ == '__main__':
    abate_task_config = {
        'name': 'abate_animais', 'filename': 'sigsifrelatorioabatesporanouf.csv',
        'target_column': 'quantidade_cabecas', 'date_column': 'data', 'date_format': '%m/%Y',
        'is_annual': False, 'group_by_columns': ['uf', 'especie_normalizada'],
        'column_mapping': {'data': ['MÊS/ANO', 'MES_ANO'], 'uf': ['UF', 'UF_PROCEDENCIA'],
                           'especie_original': ['ANIMAL', 'CATEGORIA'], 'quantidade_cabecas': ['QUANTIDADE', 'QTD']},
        'processing_func': 'process_abate_data', 'validation_start_date': '2023-01-01'
    }

    pipeline = PredictionPipeline(abate_task_config)
    scenarios = [
        {'uf': 'MT', 'especie_normalizada': 'BOVINOS'},
        {'uf': 'PR', 'especie_normalizada': 'AVES'},
        {'uf': 'SC', 'especie_normalizada': 'SUÍNOS'}
    ]

    pipeline.run_pipeline(scenarios, horizon_months=36, n_trials=20)
