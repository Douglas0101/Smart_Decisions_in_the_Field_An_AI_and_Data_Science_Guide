# src/models/pipeline_forecast_with_intervals.py

import pandas as pd
import xgboost as xgb
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import joblib

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT_PATH / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT_PATH / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT_PATH / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# --- Funções de Preparação de Dados (Reutilizadas) ---
def create_time_series_features(df, target_col):
    """Cria features temporais avançadas (lags e janelas móveis)."""
    logging.info("Criando features temporais avançadas...")
    df = df.sort_values(by=['uf', 'especie_normalizada', 'data'])
    for lag in [1, 3, 12]:
        df[f'{target_col}_lag_{lag}'] = df.groupby(['uf', 'especie_normalizada'])[target_col].shift(lag)
    for window in [3, 6]:
        df[f'{target_col}_rolling_mean_{window}'] = df.groupby(['uf', 'especie_normalizada'])[target_col].transform(
            lambda x: x.shift(1).rolling(window=window).mean())
    df.dropna(inplace=True)
    return df


# ==============================================================================
# ESTÁGIO 1: TREINAMENTO AVANÇADO (REGRESSÃO QUANTÍLICA)
# ==============================================================================
def train_quantile_models(quantiles, df_full, best_params, force_retrain=False):
    """
    Treina e salva um modelo XGBoost para cada quantil especificado.
    Adicionado 'force_retrain' para evitar retreinamentos desnecessários.
    """
    logging.info(f"Iniciando verificação/treinamento de modelos quantílicos para: {quantiles}")

    # Verifica se todos os modelos já existem
    all_models_exist = all([(MODELS_DIR / f"xgboost_quantile_{q}.json").exists() for q in quantiles])
    if all_models_exist and not force_retrain:
        logging.info("Todos os modelos quantílicos já existem e 'force_retrain' é False. Pulando treinamento.")
        return

    # --- Engenharia de Features ---
    target_col = 'quantidade_cabecas'
    df_full['mes'] = df_full['data'].dt.month
    df_full['trimestre'] = df_full['data'].dt.quarter
    df_featured = create_time_series_features(df_full, target_col)
    df_final = pd.get_dummies(df_featured, columns=['uf', 'especie_normalizada'], prefix=['uf', 'especie'])

    X = df_final.drop(columns=[target_col, 'data', 'ano'])
    y = df_final[target_col]

    # Treina e salva um modelo para cada quantil
    for q in quantiles:
        logging.info(f"Treinando modelo para o quantil {q * 100:.0f}%...")
        model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=q, **best_params)
        model.fit(X, y)

        model_path = MODELS_DIR / f"xgboost_quantile_{q}.json"
        model.save_model(model_path)
        logging.info(f"Modelo para o quantil {q} salvo em {model_path}")


# ==============================================================================
# ESTÁGIO 2: PROJEÇÃO FUTURA COM INTERVALO DE CONFIANÇA
# ==============================================================================
def generate_forecast_with_intervals(df_history, horizon_months, quantiles):
    """
    Gera previsões futuras usando os modelos quantílicos salvos.
    """
    logging.info("Iniciando projeção futura com intervalos de confiança...")

    forecasts = {}
    models = {}
    for q in quantiles:
        model_path = MODELS_DIR / f"xgboost_quantile_{q}.json"
        if not model_path.exists():
            logging.error(f"Modelo para o quantil {q} não encontrado. Execute o treinamento primeiro.")
            return None
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        models[q] = model

    for q, model in models.items():
        logging.info(f"Gerando projeção para o quantil {q * 100:.0f}%...")
        df_dynamic = df_history.copy().sort_values('data').reset_index(drop=True)
        target_col = 'quantidade_cabecas'
        future_dates = pd.to_datetime(
            pd.date_range(start=df_dynamic['data'].iloc[-1] + pd.DateOffset(months=1), periods=horizon_months,
                          freq='MS'))

        predictions = []
        for date in future_dates:
            last_row = df_dynamic.iloc[[-1]].copy()
            X_pred_features = last_row.drop(columns=[target_col, 'data', 'ano'])

            X_pred_features['mes'] = date.month
            X_pred_features['trimestre'] = date.quarter

            for lag in [1, 3, 12]:
                X_pred_features[f'{target_col}_lag_{lag}'] = df_dynamic.iloc[-lag][target_col]
            for window in [3, 6]:
                X_pred_features[f'{target_col}_rolling_mean_{window}'] = df_dynamic.tail(window)[target_col].mean()

            model_cols = model.get_booster().feature_names
            X_pred = X_pred_features.reindex(columns=model_cols, fill_value=0)
            prediction = model.predict(X_pred)[0]
            predictions.append(prediction)

            new_row = last_row.copy()
            new_row['data'] = date
            new_row[target_col] = prediction
            df_dynamic = pd.concat([df_dynamic, new_row], ignore_index=True)

        forecasts[q] = pd.Series(predictions, index=future_dates)

    return pd.DataFrame(forecasts)


# ==============================================================================
# ORQUESTRADOR PRINCIPAL
# ==============================================================================
def run_full_advanced_pipeline(scenarios, horizon_months=36, force_retrain=False):
    """
    Orquestra todo o pipeline: treina os modelos (se necessário) e depois
    gera as projeções para uma lista de cenários.
    """
    # Parâmetros otimizados da etapa anterior
    best_params = {
        'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5,
        'colsample_bytree': 0.7, 'subsample': 0.8,
        'n_jobs': -1, 'random_state': 42
    }
    quantiles_to_train = [0.05, 0.5, 0.95]  # Pessimista, Mediana, Otimista

    # --- Etapa de Treinamento ---
    logging.info("### INICIANDO ETAPA DE TREINAMENTO DOS MODELOS QUANTÍLICOS ###")
    data_path = PROCESSED_DATA_DIR / 'master_analytical_table.parquet'
    if not data_path.exists():
        logging.error(
            f"Arquivo de dados mestre não encontrado em {data_path}. Execute os scripts de pré-processamento.")
        return
    df_full = pd.read_parquet(data_path)
    train_quantile_models(quantiles_to_train, df_full, best_params, force_retrain)

    # --- Etapa de Projeção para Múltiplos Cenários ---
    for scenario in scenarios:
        target_especie = scenario['especie']
        target_uf = scenario['uf']

        logging.info(f"\n### INICIANDO ETAPA DE PROJEÇÃO PARA {target_especie} EM {target_uf} ###")
        df_history = df_full[(df_full['especie_normalizada'] == target_especie) & (df_full['uf'] == target_uf)].copy()

        if df_history.empty:
            logging.warning(f"Não há dados históricos para a combinação {target_especie}/{target_uf}. Pulando cenário.")
            continue

        forecast_df = generate_forecast_with_intervals(df_history, horizon_months, quantiles_to_train)
        if forecast_df is None: continue

        # --- Visualização Final ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.plot(df_history['data'], df_history['quantidade_cabecas'], label='Dados Históricos', color='royalblue',
                linewidth=2)
        ax.plot(forecast_df.index, forecast_df[0.5], label='Projeção (Mediana)', color='darkorange', linestyle='--')
        ax.fill_between(forecast_df.index, forecast_df[0.05], forecast_df[0.95], color='orange', alpha=0.2,
                        label='Intervalo de Confiança (90%)')

        ax.set_title(f'Projeção com Intervalo de Confiança para {target_especie} em {target_uf}', fontsize=16,
                     weight='bold')
        ax.set_xlabel('Data', fontsize=12)
        ax.set_ylabel('Quantidade de Cabeças', fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        output_path = REPORTS_DIR / f"forecast_intervals_{target_especie.lower()}_{target_uf.lower()}.png"
        fig.savefig(output_path)
        logging.info(f"Gráfico final de projeção salvo em: {output_path}")
        plt.show()


if __name__ == '__main__':
    # --- Defina aqui os cenários que você deseja analisar ---
    scenarios_to_run = [
        {'especie': 'AVES', 'uf': 'PR'},
        {'especie': 'SUÍNOS', 'uf': 'SC'},
        {'especie': 'BOVINOS', 'uf': 'MT'},
        # Adicione outros cenários de interesse aqui
        # {'especie': 'AVES', 'uf': 'SC'},
        # {'especie': 'BOVINOS', 'uf': 'GO'},
    ]

    # Execute o pipeline completo para todos os cenários definidos
    # Mude force_retrain=True se quiser forçar o retreinamento dos modelos
    run_full_advanced_pipeline(scenarios_to_run, horizon_months=36, force_retrain=False)
