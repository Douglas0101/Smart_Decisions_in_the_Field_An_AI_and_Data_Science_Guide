# src/pipeline.py

import pandas as pd
import xgboost as xgb
import logging
from pathlib import Path

# --- Configuração de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


def process_abate_data(df):
    """Função dedicada para normalizar espécies nos dados de abate."""
    if 'especie_original' not in df.columns:
        logging.error("[abate_animais] Coluna 'especie_original' não encontrada.")
        return None
    especie_map = {'BOVINOS': ['BOVINO', 'NOVILHO', 'VACA', 'VITELO'], 'SUÍNOS': ['SUINO', 'LEITAO', 'SUÍNA'],
                   'AVES': ['AVE', 'FRANGO', 'GALINHA', 'PERU', 'AVESTRUZ']}
    df['especie_normalizada'] = None
    # [CORREÇÃO] A sintaxe do loop foi corrigida para iterar sobre os itens do dicionário.
    for categoria, termos in especie_map.items():
        mask = df['especie_original'].astype(str).str.contains('|'.join(termos), case=False, na=False)
        df.loc[mask, 'especie_normalizada'] = categoria
    df.dropna(subset=['especie_normalizada'], inplace=True)
    if 'especie_original' in df.columns: df.drop(columns=['especie_original'], inplace=True)
    if 'uf' in df.columns: df['uf'] = df['uf'].astype(str).str.upper().str.strip()
    return df


class PredictionPipeline:
    """
    Classe que encapsula o pipeline de ponta a ponta para treino e previsão.
    """

    def __init__(self, task_config):
        self.config = task_config
        self.task_name = task_config['name']
        self.target_col = task_config['target_column']
        self.date_col = task_config['date_column']
        self.group_by_cols = task_config['group_by_columns']
        self.quantiles = [0.05, 0.5, 0.95]
        logging.info(f"Instância do Pipeline criada para a tarefa: '{self.task_name}'")

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

    def load_and_preprocess_data(self):
        """Carrega e pré-processa os dados brutos a partir do ficheiro de origem."""
        project_root = Path(__file__).resolve().parent.parent
        file_path = project_root / "data" / "raw" / self.config['filename']
        if not file_path.exists():
            logging.error(f"[{self.task_name}] Ficheiro não encontrado: {file_path}")
            return None

        df = pd.read_csv(file_path, sep=';', encoding='latin1', on_bad_lines='skip', low_memory=False)
        df = self._map_and_validate_columns(df)
        if 'processing_func' in self.config:
            # Assumimos que a função de processamento está disponível globalmente neste módulo
            process_func = globals().get(self.config['processing_func'])
            if process_func:
                df = process_func(df)

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

    def train_and_forecast(self, df_history, params, horizon_months):
        """Treina modelos para cada quantil e gera a previsão futura."""
        df_history_featured = self._create_features(df_history)
        feature_cols = [col for col in df_history_featured.columns if col not in [self.target_col, self.date_col]]

        forecasts = {}
        for q in self.quantiles:
            # Adiciona o quantile_alpha aos parâmetros
            model_params = params.copy()
            model_params['objective'] = 'reg:quantileerror'
            model_params['quantile_alpha'] = q

            model = xgb.XGBRegressor(**model_params)
            model.fit(df_history_featured[feature_cols], df_history_featured[self.target_col])

            # Previsão iterativa
            df_dynamic = df_history_featured.copy()
            last_date = df_dynamic[self.date_col].iloc[-1]
            future_dates = pd.date_range(start=last_date, periods=horizon_months + 1, freq='MS')[1:]

            predictions = []
            for date in future_dates:
                last_known_features = df_dynamic[feature_cols].iloc[-1].to_dict()
                pred_input = pd.DataFrame([last_known_features], columns=feature_cols)

                prediction = model.predict(pred_input)[0]
                predictions.append(prediction)

                new_row = pred_input.iloc[0].to_dict()
                new_row[self.date_col] = date
                new_row[self.target_col] = prediction
                df_dynamic = pd.concat([df_dynamic, pd.DataFrame([new_row])], ignore_index=True)

            forecasts[q] = pd.Series(predictions, index=future_dates)

        return pd.DataFrame(forecasts)
