# src/models/forecast_future.py (Versão Final com Projeção Recursiva Dinâmica)

import pandas as pd
import xgboost as xgb
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# --- Configuração ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
PROJECT_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT_PATH / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT_PATH / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT_PATH / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_artifacts():
    """Carrega o modelo treinado e os dados processados."""
    logging.info("Carregando artefatos: modelo XGBoost treinado e dados mestre...")

    model_path = MODELS_DIR / "xgboost_optimized_model.json"
    if not model_path.exists():
        logging.error(f"Modelo não encontrado em {model_path}. Execute o script de treinamento primeiro.")
        return None, None

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    data_path = PROCESSED_DATA_DIR / 'master_analytical_table.parquet'
    if not data_path.exists():
        logging.error(f"Tabela analítica não encontrada em {data_path}. Execute o pré-processamento.")
        return None, None

    df = pd.read_parquet(data_path)

    return model, df


def create_features_for_single_step(df, date, target_col='quantidade_cabecas'):
    """
    Cria um DataFrame de uma linha com todas as features para um único passo de tempo.
    """
    row = {}
    row['data'] = date
    row['mes'] = date.month
    row['trimestre'] = date.quarter

    # Propaga features estáticas do último registro conhecido
    last_known = df.iloc[-1]
    for col in ['uf', 'especie_normalizada', 'n_estabelecimentos_sif', 'n_estab_fert_ativos']:
        if col in last_known.index:
            row[col] = last_known[col]

    # Calcula features de lag dinamicamente
    for lag in [1, 3, 12]:
        lag_date = date - pd.DateOffset(months=lag)
        if lag_date in df['data'].values:
            row[f'{target_col}_lag_{lag}'] = df.loc[df['data'] == lag_date, target_col].iloc[0]
        else:
            # Se o lag não for encontrado (início da série), usa o valor mais antigo
            row[f'{target_col}_lag_{lag}'] = df[target_col].iloc[0]

    # Calcula features de janela móvel dinamicamente
    rolling_window = 3
    recent_data = df[df['data'] < date].tail(rolling_window)
    row[f'{target_col}_rolling_mean_{rolling_window}'] = recent_data[target_col].mean()
    row[f'{target_col}_rolling_std_{rolling_window}'] = recent_data[target_col].std()

    return pd.DataFrame([row])


def generate_future_forecast(model, df_history, horizon_months):
    """
    Gera previsões futuras de forma recursiva e dinâmica, recalculando
    as features a cada passo.
    """
    logging.info(f"Iniciando projeção futura para {horizon_months} meses...")

    target_col = 'quantidade_cabecas'

    # DataFrame para armazenar o histórico e as novas previsões
    df_dynamic = df_history.copy().sort_values('data').reset_index(drop=True)

    future_dates = pd.to_datetime(
        pd.date_range(start=df_dynamic['data'].iloc[-1] + pd.DateOffset(months=1), periods=horizon_months, freq='MS'))

    predictions = []

    for date in future_dates:
        # 1. Criar as features para o passo de tempo atual com base no histórico dinâmico
        X_pred_features = create_features_for_single_step(df_dynamic, date, target_col)

        # Preencher NaNs em std (ocorre se a janela tem 1 valor)
        X_pred_features.fillna(0, inplace=True)

        # 2. One-Hot Encoding e alinhamento de colunas
        X_pred_pre = pd.get_dummies(X_pred_features, columns=['uf', 'especie_normalizada'], prefix=['uf', 'especie'])
        model_cols = model.get_booster().feature_names
        X_pred = X_pred_pre.reindex(columns=model_cols, fill_value=0)

        # 3. Fazer a previsão
        prediction = model.predict(X_pred)[0]
        predictions.append(prediction)

        # 4. Adicionar a previsão ao histórico dinâmico para a próxima iteração
        new_row = X_pred_features.copy()
        new_row[target_col] = prediction
        df_dynamic = pd.concat([df_dynamic, new_row], ignore_index=True)

    return pd.Series(predictions, index=future_dates)


def run_forecasting_pipeline(target_especie='AVES', target_uf='PR', horizon_months=60):
    """Orquestra o pipeline de projeção futura para uma combinação específica."""
    logging.info(f"--- Gerando Projeção para {target_especie} em {target_uf} ---")

    model, df_full = load_artifacts()
    if model is None: return

    # Filtra o histórico para a série desejada
    df_history = df_full[(df_full['especie_normalizada'] == target_especie) & (df_full['uf'] == target_uf)].copy()
    if df_history.empty:
        logging.error(f"Não há dados históricos para a combinação {target_especie}/{target_uf}.")
        return

    # Gera a projeção
    forecast = generate_future_forecast(model, df_history, horizon_months)

    # --- Visualização ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(df_history['data'], df_history['quantidade_cabecas'], label='Dados Históricos', color='royalblue',
            linewidth=2)
    ax.plot(forecast.index, forecast.values, label='Projeção Futura (XGBoost)', color='darkorange', linestyle='--')

    ax.set_title(f'Projeção Dinâmica de Abate de {target_especie} em {target_uf} (Horizonte de {horizon_months} Meses)',
                 fontsize=16, weight='bold')
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Quantidade de Cabeças', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.axvspan(df_history['data'].iloc[-1], forecast.index[-1], color='orange', alpha=0.1, label='Período de Projeção')
    plt.tight_layout()

    output_path = REPORTS_DIR / f"forecast_dynamic_{target_especie.lower()}_{target_uf.lower()}_{horizon_months}m.png"
    fig.savefig(output_path)
    logging.info(f"Gráfico de projeção futura salvo em: {output_path}")
    plt.show()


if __name__ == '__main__':
    # Gere a projeção final até 2030
    run_forecasting_pipeline(target_especie='AVES', target_uf='PR', horizon_months=60)
    run_forecasting_pipeline(target_especie='SUÍNOS', target_uf='SC', horizon_months=60)
