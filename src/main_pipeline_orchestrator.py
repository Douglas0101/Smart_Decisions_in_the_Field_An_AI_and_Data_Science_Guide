# src/main_pipeline_orchestrator.py

import pandas as pd
import xgboost as xgb
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# --- Configuração Global ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT_PATH / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT_PATH / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT_PATH / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT_PATH / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Funções de Processamento Específicas
# ==============================================================================
def process_abate_data(df):
    """Função dedicada para normalizar espécies nos dados de abate."""
    if 'especie_original' not in df.columns:
        logging.error("[abate_animais] Coluna 'especie_original' não encontrada para normalização.")
        return None

    especie_map = {
        'BOVINOS': ['BOVINO', 'NOVILHO', 'VACA', 'VITELO'],
        'SUÍNOS': ['SUINO', 'LEITAO', 'SUÍNA'],
        'AVES': ['AVE', 'FRANGO', 'GALINHA', 'PERU', 'AVESTRUZ']
    }
    df['especie_normalizada'] = None
    for categoria, termos in especie_map.items():
        mask = df['especie_original'].astype(str).str.contains('|'.join(termos), case=False, na=False)
        df.loc[mask, 'especie_normalizada'] = categoria

    df.dropna(subset=['especie_normalizada'], inplace=True)
    return df


def process_sementes_data(df):
    """Função dedicada para limpar e preparar os dados de produção de sementes."""
    required_cols = ['safra_ano', 'area_ha']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"[producao_sementes] Colunas necessárias {required_cols} não encontradas no DataFrame.")
        return None

    # Limpeza da coluna de data (safra)
    # Pega apenas os 4 primeiros dígitos (o primeiro ano da safra 'YYYY/YYYY')
    df['safra_ano'] = df['safra_ano'].astype(str).str.slice(0, 4)

    # Limpeza da coluna de área (alvo)
    # Substitui vírgula por ponto e converte para numérico
    if df['area_ha'].dtype == 'object':
        df['area_ha'] = df['area_ha'].str.replace(',', '.', regex=False).astype(float)

    return df


# ==============================================================================
# CLASSE DO PIPELINE DE PREVISÃO GENÉRICO
# ==============================================================================
class PredictionPipeline:
    def __init__(self, task_config):
        self.config = task_config
        self.task_name = task_config['name']
        self.target_col = task_config['target_column']
        self.date_col = task_config['date_column']
        self.group_by_cols = task_config['group_by_columns']

        self.best_params = {
            'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5,
            'colsample_bytree': 0.7, 'subsample': 0.8,
            'n_jobs': -1, 'random_state': 42
        }
        self.quantiles = [0.05, 0.5, 0.95]
        logging.info(f"Pipeline inicializado para a tarefa: '{self.task_name}'")

    def _map_and_validate_columns(self, df):
        """Mapeia dinamicamente as colunas e valida o esquema."""
        rename_map = {}
        original_cols_stripped = {c.strip().upper(): c for c in df.columns}

        for final_name, possible_names in self.config['column_mapping'].items():
            found = False
            for p_name in possible_names:
                if p_name.strip().upper() in original_cols_stripped:
                    rename_map[original_cols_stripped[p_name.strip().upper()]] = final_name
                    found = True
                    break

        df.rename(columns=rename_map, inplace=True)
        return df

    def _load_and_preprocess_data(self):
        file_path = RAW_DATA_DIR / self.config['filename']
        if not file_path.exists():
            logging.error(f"[{self.task_name}] Arquivo não encontrado: {file_path}")
            return None

        df = pd.read_csv(file_path, sep=';', encoding='latin1', on_bad_lines='skip', low_memory=False)

        df = self._map_and_validate_columns(df)
        if df is None: return None

        if 'processing_func' in self.config:
            df = globals()[self.config['processing_func']](df)
            if df is None: return None

        required_cols = [self.date_col, self.target_col] + self.group_by_cols
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logging.error(
                f"[{self.task_name}] Colunas necessárias ausentes: {missing}. Verifique a configuração e o arquivo de origem.")
            logging.info(f"[{self.task_name}] DIAGNÓSTICO: Colunas disponíveis são: {df.columns.tolist()}")
            return None

        date_format = self.config.get('date_format')
        df[self.date_col] = pd.to_datetime(df[self.date_col], format=date_format, errors='coerce')
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors='coerce')
        df.dropna(subset=[self.date_col, self.target_col], inplace=True)

        return df

    def _create_features(self, df):
        df['ano'] = df[self.date_col].dt.year
        df['mes'] = df[self.date_col].dt.month
        df = df.sort_values(by=self.group_by_cols + [self.date_col])
        for lag in [1, 3, 12]:
            if self.config.get('is_annual', False) and lag > 1: continue
            df[f'{self.target_col}_lag_{lag}'] = df.groupby(self.group_by_cols)[self.target_col].shift(lag)

        lag_cols = [col for col in df.columns if '_lag_' in col]
        df[lag_cols] = df[lag_cols].fillna(0)

        df = pd.get_dummies(df, columns=self.group_by_cols, prefix=self.group_by_cols)
        return df

    def train_models(self, force_retrain=False):
        model_paths = [MODELS_DIR / f"xgboost_quantile_{self.task_name}_{q}.json" for q in self.quantiles]
        if all(p.exists() for p in model_paths) and not force_retrain:
            logging.info(f"[{self.task_name}] Todos os modelos já existem. Pulando.")
            return

        df_raw = self._load_and_preprocess_data()
        if df_raw is None: return

        df_featured = self._create_features(df_raw)
        if df_featured.empty:
            logging.error(
                f"[{self.task_name}] DataFrame vazio após criação de features. Verifique a lógica de pré-processamento.")
            return

        X = df_featured.drop(columns=[self.target_col, self.date_col, 'ano', 'mes'])
        y = df_featured[self.target_col]

        for i, q in enumerate(self.quantiles):
            model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=q, **self.best_params)
            model.fit(X, y)
            model.save_model(model_paths[i])

    def generate_forecast(self, history_df, horizon_months):
        forecasts = {}
        for q in self.quantiles:
            model_path = MODELS_DIR / f"xgboost_quantile_{self.task_name}_{q}.json"
            model = xgb.XGBRegressor();
            model.load_model(model_path)

            df_dynamic = history_df.copy().sort_values(self.date_col).reset_index(drop=True)
            offset = pd.DateOffset(years=1) if self.config.get('is_annual', False) else pd.DateOffset(months=1)
            freq = 'AS-JAN' if self.config.get('is_annual', False) else 'MS'
            future_dates = pd.date_range(start=df_dynamic[self.date_col].iloc[-1] + offset, periods=horizon_months,
                                         freq=freq)

            predictions = []
            for date in future_dates:
                last_known = df_dynamic.iloc[-1]
                current_features = pd.DataFrame([last_known.to_dict()])
                current_features[self.date_col] = date
                current_features['ano'], current_features['mes'] = date.year, date.month

                for lag in [1, 3, 12]:
                    if self.config.get('is_annual', False) and lag > 1: continue
                    current_features[f'{self.target_col}_lag_{lag}'] = df_dynamic.iloc[-lag][self.target_col]

                X_pred_pre = pd.get_dummies(current_features, columns=self.group_by_cols, prefix=self.group_by_cols)
                X_pred = X_pred_pre.reindex(columns=model.get_booster().feature_names, fill_value=0)

                prediction = model.predict(X_pred)[0]
                predictions.append(prediction)

                new_row = current_features.to_dict('records')[0]
                new_row[self.target_col] = prediction
                df_dynamic = pd.concat([df_dynamic, pd.DataFrame([new_row])], ignore_index=True)

            forecasts[q] = pd.Series(predictions, index=future_dates)

        return pd.DataFrame(forecasts)

    def run(self, scenarios, horizon_months=36):
        df_full = self._load_and_preprocess_data()
        if df_full is None: return

        for scenario in scenarios:
            logging.info(f"\n[{self.task_name}] ### EXECUTANDO CENÁRIO: {scenario} ###")

            query_parts = [f"`{col}` == '{val}'" for col, val in scenario.items()]
            history_query = " & ".join(query_parts)
            df_history = df_full.query(history_query).copy()

            if df_history.empty:
                logging.warning(f"Não há dados históricos para o cenário {scenario}. Pulando.")
                continue

            forecast_df = self.generate_forecast(df_history, horizon_months)

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(18, 8))
            ax.plot(df_history[self.date_col], df_history[self.target_col], label='Dados Históricos', color='royalblue')
            ax.plot(forecast_df.index, forecast_df[0.5], label='Projeção (Mediana)', color='darkorange', linestyle='--')
            ax.fill_between(forecast_df.index, forecast_df[0.05], forecast_df[0.95], color='orange', alpha=0.2,
                            label='Intervalo de Confiança (90%)')

            title = f"Projeção para '{self.task_name}': {scenario}"
            ax.set_title(title, fontsize=16, weight='bold')
            ax.legend()

            filename = f"forecast_{self.task_name}_{'_'.join(str(v).replace(' ', '_') for v in scenario.values())}.png"
            output_path = REPORTS_DIR / filename
            fig.savefig(output_path)
            logging.info(f"Gráfico de projeção salvo em: {output_path}")
            plt.show()


# ==============================================================================
# DEFINIÇÃO E EXECUÇÃO DAS TAREFAS
# ==============================================================================
if __name__ == '__main__':

    abate_task_config = {
        'name': 'abate_animais', 'filename': 'sigsifrelatorioabatesporanouf.csv',
        'target_column': 'quantidade_cabecas', 'date_column': 'data',
        'date_format': '%m/%Y', 'is_annual': False,
        'group_by_columns': ['uf', 'especie_normalizada'],
        'column_mapping': {
            'data': ['MES_ANO'], 'uf': ['UF_PROCEDENCIA'],
            'especie_original': ['CATEGORIA'], 'quantidade_cabecas': ['QTD']
        },
        'processing_func': 'process_abate_data'
    }

    sementes_task_config = {
        'name': 'producao_sementes', 'filename': 'sigefcamposproducaodesementes.csv',
        'target_column': 'area_ha', 'date_column': 'safra_ano',
        'date_format': None, 'is_annual': True,
        'group_by_columns': ['uf', 'cultura_nome'],
        'column_mapping': {  # Mapeamento corrigido com base no diagnóstico do log anterior
            'safra_ano': ['Safra'],
            'uf': ['uf'],
            'cultura_nome': ['Especie'],
            'area_ha': ['Area']
        },
        'processing_func': 'process_sementes_data'  # Função dedicada para esta tarefa
    }

    prediction_tasks = [abate_task_config, sementes_task_config]

    for task_conf in prediction_tasks:
        pipeline = PredictionPipeline(task_conf)
        pipeline.train_models(force_retrain=True)

        if task_conf['name'] == 'abate_animais':
            scenarios = [
                {'uf': 'PR', 'especie_normalizada': 'AVES'},
                {'uf': 'SC', 'especie_normalizada': 'SUÍNOS'},
                {'uf': 'MT', 'especie_normalizada': 'BOVINOS'}
            ]
            pipeline.run(scenarios, horizon_months=36)

        elif task_conf['name'] == 'producao_sementes':
            scenarios = [{'uf': 'MT', 'cultura_nome': 'SOJA'}, {'uf': 'RS', 'cultura_nome': 'MILHO'}]
            pipeline.run(scenarios, horizon_months=5)

