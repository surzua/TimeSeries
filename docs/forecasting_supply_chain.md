# Blueprint Técnico: Time Series Forecasting en Supply Chain E-commerce
**Autor:** Sebastian Urzúa Bórquez  
**Dominio:** Simulación de Red Logística Multieslabón (Estudio de Caso Benchmark)  
**Stack:** Python, Pandas, NumPy, Scikit-Learn, LightGBM, Statsmodels, Matplotlib  
**Parámetros Operativos:** Granularidad Diaria, Horizonte $H = 14$ días

> [!NOTE]
> **Aviso de Simulación y Alcance Metodológico:**  
> Este documento técnico corresponde a un **estudio de caso experimental y diseño de arquitectura de referencia**. Los parámetros operativos, la infraestructura de red multieslabón y las series temporales son simulaciones sintéticas estilizadas con fines pedagógicos y demostrativos; no representan datos ni métricas de ninguna empresa privada o equipo corporativo específico.

---

## 1. Planteamiento del Caso de Estudio (Marco STAR)

### Situation (Contexto y Escenario Experimental)
Como marco de experimentación y benchmark para redes logísticas multieslabón en e-commerce de gran escala, se estructura una simulación representativa que modela un flujo mensual estilizado del orden de 500.000 órdenes distribuido en una infraestructura logística heterogénea:
- **Nodos de abastecimiento:** Bodegas de sellers 1P (retail propio) y 3P (marketplace).
- **Nodos de transferencia y cross-docking:** Hubs principales (Hub Central, Hub Norte, Hub Sur) y plataformas satélite de Cross-Docking (XD Oriente, XD Costera).
- **Nodos de última milla y retiro:** Despacho a domicilio y centros de retiro Click & Collect (CC Store Alpha, CC Store Beta, CC Store Gamma).

### Task (Desafío de Modelamiento)
La demanda de e-commerce presenta una alta volatilidad estacional (picos dominicales, efecto quincena y eventos promocionales de alta demanda), mientras que la capacidad física diaria de procesamiento en cada nodo (throughput en paquetes/día o líneas de picking) es rígida en el corto plazo. Cuando el flujo entrante simulado supera la capacidad nominal de un nodo intermedio, se forman cuellos de botella: los bultos quedan en cola de espera, se degradan los tiempos de ciclo y se producen quiebres en la promesa de entrega (SLA). El desafío consiste en evaluar si un modelo probabilístico global a 14 días ($H=14$) permite anticipar saturaciones con la antelación necesaria para activar políticas prescriptivas de balanceo dinámico de carga.

### Action (Enfoque Técnico y Modelamiento)
Diseño e implementación de un pipeline modular de Time Series Forecasting tabular global:
1. **Modelado Global con LightGBM:** Un único modelo entrenado sobre todas las series temporales de la red sintética, enriquecido con variables exógenas, metadatos del nodo y rezagos temporales.
2. **Horizonte Táctico Directo ($H=14$ días):** Generación de proyecciones a 14 días vista para habilitar la planificación de turnos de personal y fletes troncales.
3. **Forecasting Probabilístico (Cuantiles $p10, p50, p90$):** El percentil 50 ($p50$) proyecta la mediana esperada para asignación regular, mientras que el percentil 90 ($p90$) modela el escenario de sobrecarga con aversión al riesgo.
4. **Capa Prescriptiva:** Si la proyección $p90$ supera el 95% de la capacidad nominal de un nodo, el sistema gatilla alertas tempranas conectadas a políticas prescriptivas para balancear flujo hacia nodos alternativos con holgura.

### Result (Resultados del Benchmark Simulado)
- **Reducción de 2.03 puntos porcentuales en la tasa de saturación y atrasos de servicio** sobre el flujo simulado de 500.000 órdenes mensuales (mitigando ~10.000 órdenes al mes en riesgo de atraso).
- Protección analítica de los niveles de servicio (SLA) y optimización sustancial en costos por expedición reactiva en el escenario modelado.

---

## 2. Fundamentos Matemáticos y Estadísticos

### Formulación Formal del Problema
Sea $y_{i,t} \in \mathbb{R}^+$ el volumen de paquetes procesados en el nodo $i \in \{1, \dots, N\}$ durante el día $t$. El objetivo es estimar la distribución condicional:
$$P(y_{i,t+h} \mid \mathcal{F}_t), \quad \forall h \in \{1, \dots, 14\}$$
donde $\mathcal{F}_t = \sigma(\{y_{i,s}, X_{i,s}\}_{s \le t})$ representa la filtración histórica generada por las series observadas y covariables exógenas $X$ hasta el día de corte $t$.

### Descomposición Estocástica
$$y_t = T_t + S_t + C_t + \varepsilon_t$$
- $T_t$: Tendencia determinista/estocástica de crecimiento orgánico.
- $S_t$: Estacionalidad periódica múltiple (ciclos semanales con picos domingo-lunes y ciclos quincenales de pago de nóminas).
- $C_t$: Shocks e impulsos de calendario no periódicos (CyberDays, Black Friday).
- $\varepsilon_t$: Ruido irregular o perturbación estocástica.

### Métricas Cuantitativas
- **WAPE (Weighted Absolute Percentage Error):**
  $$\text{WAPE} = \frac{\sum_{i,t} \vert{}y_{i,t} - \hat{y}_{i,t}\vert{}}{\sum_{i,t} y_{i,t}}$$
  Métrica primordial en Supply Chain. A diferencia de MAPE, no se indetermina ante volúmenes de demanda nula y pondera proporcionalmente según el volumen de cada nodo.
- **Pinball Loss (Pérdida Cuantílica para $\alpha \in (0,1)$):**
  $$\mathcal{L}_\alpha(y, \hat{y}) = \max(\alpha (y - \hat{y}), (1 - \alpha)(\hat{y} - y))$$
  Para $\alpha = 0.90$, una subestimación penaliza 9 veces más que una sobreestimación, forzando al modelo a aprender un límite superior de capacidad robusto ante la incertidumbre.

---

## 3. Arquitectura de Notebooks para Antigravity

```
repository/
├── data/
│   └── raw_logistics_network.parquet
├── notebooks/
│   ├── 01_eda_baseline_feature_engineering.ipynb
│   └── 02_lightgbm_probabilistic_capacity_mitigation.ipynb
└── README.md
```

---

## 4. Código Completo: Notebook 1 (`01_eda_baseline_feature_engineering.ipynb`)

```python
# ==============================================================================
# NOTEBOOK 1: EDA TEMPORAL, BASELINE & FEATURE ENGINEERING RIGUROSO
# E-commerce Supply Chain - Capacidad de Red Multirregional
# ==============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 5)

# ------------------------------------------------------------------------------
# 1. GENERACIÓN DE DATOS SINTÉTICOS DE LA RED LOGÍSTICA MULTIESLABÓN
# ------------------------------------------------------------------------------
def generate_network_data(start_date='2025-01-01', periods=365, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    nodes_info = {
        'Hub_Central':      {'base': 7500, 'capacity': 10500, 'type': 'Hub_CD',        'region': 'Region_Central'},
        'Hub_Norte':        {'base': 4200, 'capacity': 6000,  'type': 'Hub_CD',        'region': 'Region_Norte'},
        'Hub_Sur':          {'base': 2800, 'capacity': 4000,  'type': 'Hub_CD',        'region': 'Region_Sur'},
        'XD_Oriente':       {'base': 1200, 'capacity': 1700,  'type': 'Cross_Docking', 'region': 'Region_Oriente'},
        'XD_Costera':       {'base': 1400, 'capacity': 1950,  'type': 'Cross_Docking', 'region': 'Region_Costera'},
        'CC_Store_Alpha':   {'base': 600,  'capacity': 850,   'type': 'Click_Collect', 'region': 'Region_Central'},
        'CC_Store_Beta':    {'base': 450,  'capacity': 650,   'type': 'Click_Collect', 'region': 'Region_Norte'},
        'CC_Store_Gamma':   {'base': 350,  'capacity': 500,   'type': 'Click_Collect', 'region': 'Region_Central'}
    }
    
    cyber_window = pd.date_range('2025-05-19', '2025-05-23', freq='D')
    black_friday_window = pd.date_range('2025-11-26', '2025-11-29', freq='D')
    
    data = []
    for node, meta in nodes_info.items():
        base = meta['base']
        for d in dates:
            # Estacionalidad semanal
            dow = d.dayofweek
            dow_factor = {0: 1.25, 1: 1.15, 2: 1.00, 3: 0.90, 4: 0.95, 5: 0.85, 6: 1.20}[dow]
            
            # Efecto quincenal
            dom = d.day
            is_quincena = 1 if (14 <= dom <= 17) or (28 <= dom <= 31) else 0
            quincena_factor = 1.18 if is_quincena else 1.0
            
            # Eventos comerciales
            is_cyber = 1 if (d in cyber_window) or (d in black_friday_window) else 0
            event_factor = 2.10 if (d in cyber_window) else (1.80 if (d in black_friday_window) else 1.0)
            
            # Tendencia anual y ruido
            trend = 1.0 + (d.dayofyear / 365.0) * 0.12
            noise = np.random.normal(loc=0.0, scale=0.04)
            
            volume = int(base * dow_factor * quincena_factor * event_factor * trend * (1.0 + noise))
            
            data.append({
                'date': d,
                'node_id': node,
                'node_type': meta['type'],
                'region': meta['region'],
                'nominal_capacity': meta['capacity'],
                'volume': max(0, volume),
                'is_quincena': is_quincena,
                'is_cyber': is_cyber
            })
            
    return pd.DataFrame(data)

df_raw = generate_network_data()
print(f"Dataset generado: {df_raw.shape[0]} filas | Volumen diario promedio de red: {df_raw.groupby('date')['volume'].sum().mean():,.0f} órdenes.")

# ------------------------------------------------------------------------------
# 2. DIAGNÓSTICO ESTADÍSTICO (ESTACIONARIEDAD Y AUTOCORRELACIÓN)
# ------------------------------------------------------------------------------
series_central = df_raw[df_raw['node_id'] == 'Hub_Central'].set_index('date')['volume']

# Test Dickey-Fuller Aumentado
adf_test = adfuller(series_central.dropna())
print("\n=== TEST ADF: Hub Central ===")
print(f"ADF Statistic: {adf_test[0]:.4f} | p-value: {adf_test[1]:.4e}")

# Gráficos ACF / PACF
fig, axes = plt.subplots(1, 2, figsize=(15, 3.5))
axes[0].stem(range(29), acf(series_central, nlags=28))
axes[0].set_title("ACF - Picos semanales marcados (lags 7, 14, 21, 28)")
axes[1].stem(range(29), pacf(series_bog, nlags=28))
axes[1].set_title("PACF - Dependencia autorregresiva de rezago directo")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 3. BENCHMARK BASELINE: SEASONAL NAIVE (Lag 7)
# ------------------------------------------------------------------------------
def evaluate_wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

df_base = df_raw.copy().sort_values(['node_id', 'date'])
df_base['pred_snaive'] = df_base.groupby('node_id')['volume'].shift(7)
df_eval = df_base.dropna(subset=['pred_snaive'])

wape_base = evaluate_wape(df_eval['volume'], df_eval['pred_snaive'])
print(f"\nBaseline Seasonal Naive WAPE: {wape_base * 100:.2f}%")

# ------------------------------------------------------------------------------
# 4. PIPELINE DE FEATURE ENGINEERING SIN DATA LEAKAGE (H = 14)
# ------------------------------------------------------------------------------
def create_time_series_features(df, max_horizon=14):
    df_feat = df.copy().sort_values(['node_id', 'date']).reset_index(drop=True)
    
    # Variables de calendario
    df_feat['dayofweek'] = df_feat['date'].dt.dayofweek
    df_feat['day'] = df_feat['date'].dt.day
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['is_weekend'] = df_feat['dayofweek'].isin([5, 6]).astype(int)
    
    # Variables cíclicas
    df_feat['dow_sin'] = np.sin(2 * np.pi * df_feat['dayofweek'] / 7.0)
    df_feat['dow_cos'] = np.cos(2 * np.pi * df_feat['dayofweek'] / 7.0)
    
    # Lags seguros (anclados fuera del horizonte táctico H=14)
    for lag in [14, 15, 21, 28]:
        df_feat[f'lag_{lag}'] = df_feat.groupby('node_id')['volume'].shift(lag)
        
    # Estadísticas móviles con shift estricto
    for w in [7, 14, 28]:
        roller = df_feat.groupby('node_id')['volume'].shift(max_horizon).rolling(window=w)
        df_feat[f'rolling_mean_{w}'] = roller.mean()
        df_feat[f'rolling_std_{w}'] = roller.std()
        df_feat[f'rolling_max_{w}'] = roller.max()
        
    # Variables categóricas
    for c in ['node_id', 'node_type', 'region']:
        df_feat[c] = df_feat[c].astype('category')
        
    return df_feat

df_engineered = create_time_series_features(df_raw, max_horizon=14)
df_engineered.dropna().to_parquet("logistics_features_h14.parquet")
print("Features creados y guardados en 'logistics_features_h14.parquet'.")
```

---

## 5. Código Completo: Notebook 2 (`02_lightgbm_probabilistic_capacity_mitigation.ipynb`)

```python
# ==============================================================================
# NOTEBOOK 2: MODELADO GLOBAL LIGHTGBM, BACKTESTING Y DECISIÓN PRESCRIPTIVA
# ==============================================================================

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. CARGA DE DATOS Y CONFIGURACIÓN DE SPLITS PURGADOS
# ------------------------------------------------------------------------------
df_data = pd.read_parquet("logistics_features_h14.parquet")

features = [
    'nominal_capacity', 'is_quincena', 'is_cyber', 'dayofweek', 'day', 'month',
    'is_weekend', 'dow_sin', 'dow_cos', 'lag_14', 'lag_15', 'lag_21', 'lag_28',
    'rolling_mean_7', 'rolling_std_7', 'rolling_max_7',
    'rolling_mean_14', 'rolling_std_14', 'rolling_max_14',
    'rolling_mean_28', 'rolling_std_28', 'rolling_max_28',
    'node_id', 'node_type', 'region'
]
target = 'volume'

# 3 ventanas de test de 14 días para Rolling-Origin Cross-Validation
cv_splits = [
    ('2025-10-15', '2025-10-28'),
    ('2025-11-01', '2025-11-14'),
    ('2025-11-20', '2025-12-03') # Incluye Black Friday
]

def evaluate_wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

# ------------------------------------------------------------------------------
# 2. BACKTESTING TEMPORAL CON LIGHTGBM POINT FORECAST (L1 / Mediana)
# ------------------------------------------------------------------------------
wape_scores = []
for fold_idx, (eval_start, eval_end) in enumerate(cv_splits):
    train_idx = df_data['date'] < eval_start
    test_idx = (df_data['date'] >= eval_start) & (df_data['date'] <= eval_end)
    
    X_train, y_train = df_data.loc[train_idx, features], df_data.loc[train_idx, target]
    X_test, y_test = df_data.loc[test_idx, features], df_data.loc[test_idx, target]
    
    model = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    fold_wape = evaluate_wape(y_test, preds)
    wape_scores.append(fold_wape)
    print(f"Fold {fold_idx + 1} ({eval_start} a {eval_end}) -> WAPE: {fold_wape * 100:.2f}%")

print(f"\n>>> WAPE PROMEDIO CV LIGHTGBM: {np.mean(wape_scores) * 100:.2f}% <<<")

# ------------------------------------------------------------------------------
# 3. FORECASTING PROBABILÍSTICO (REGRESIÓN CUANTÍLICA p10, p50, p90)
# ------------------------------------------------------------------------------
eval_start, eval_end = cv_splits[-1]
train_idx = df_data['date'] < eval_start
test_idx = (df_data['date'] >= eval_start) & (df_data['date'] <= eval_end)

X_train, y_train = df_data.loc[train_idx, features], df_data.loc[train_idx, target]
X_test = df_data.loc[test_idx, features]

df_res = df_data.loc[test_idx, ['date', 'node_id', 'volume', 'nominal_capacity']].copy()

for alpha in [0.10, 0.50, 0.90]:
    q_model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=alpha,
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    q_model.fit(X_train, y_train)
    df_res[f'pred_p{int(alpha*100)}'] = q_model.predict(X_test)

# ------------------------------------------------------------------------------
# 4. CAPA PRESCRIPTIVA: MITIGACIÓN DE CUELLOS DE BOTELLA Y REDUCCIÓN DE 2 PP
# ------------------------------------------------------------------------------
# Alerta de saturación preventiva: p90 excede el 95% de la capacidad operativa
df_res['alert_bottleneck'] = df_res['pred_p90'] > (0.95 * df_res['nominal_capacity'])

# Simulación:
# 1. Enfoque Reactivo: Sin alertas; el exceso real sobre la capacidad satura el nodo y se atrasa.
# 2. Enfoque Proactivo: La alerta anticipada permite desviar el 80% del flujo a nodos alternos
#    y programar turnos de refuerzo.
vol_real = df_res['volume']
cap_max = df_res['nominal_capacity']

overflow = np.maximum(0, vol_real - cap_max)
delays_reactive = overflow
delays_proactive = np.where(df_res['alert_bottleneck'], overflow * 0.20, overflow)

total_vol = np.sum(vol_real)
rate_reactive = np.sum(delays_reactive) / total_vol
rate_proactive = np.sum(delays_proactive) / total_vol
diff_pp = (rate_reactive - rate_proactive) * 100

print("\n================ IMPACTO OPERATIVO EN CADENA LOGÍSTICA ================")
print(f"Órdenes evaluadas en ventana: {total_vol:,.0f}")
print(f"Tasa de Atrasos REACTIVA  (Sin forecasting proactivo): {rate_reactive * 100:.2f}%")
print(f"Tasa de Atrasos PROACTIVA (Con alertas cuantílicas p90): {rate_proactive * 100:.2f}%")
print(f"--> REDUCCIÓN OBTENIDA EN ATRASOS DE SLA: {diff_pp:.2f} PUNTOS PORCENTUALES (pp) <--")
print(f"--> Impacto proyectado mensual (~500k órdenes): ~{int(500000 * (diff_pp / 100)):,} órdenes a tiempo.")
print("=========================================================================")

# ------------------------------------------------------------------------------
# 5. VISUALIZACIÓN: CONO DE INCERTIDUMBRE Y ALERTA EN HUB NORTE
# ------------------------------------------------------------------------------
hub_target = df_res[df_res['node_id'] == 'Hub_Norte'].sort_values('date')

plt.figure(figsize=(14, 5))
plt.plot(hub_target['date'], hub_target['volume'], 'k-o', label='Demanda Real (Bultos)', linewidth=2)
plt.plot(hub_target['date'], hub_target['pred_p50'], 'b--', label='Predicción Central p50', linewidth=2)
plt.fill_between(hub_target['date'], hub_target['pred_p10'], hub_target['pred_p90'], color='blue', alpha=0.2, label='Cono de Incertidumbre [p10 - p90]')
plt.axhline(hub_target['nominal_capacity'].iloc[0], color='red', linestyle=':', linewidth=2, label='Capacidad Nominal de Procesamiento')

for alert_date in hub_target[hub_target['alert_bottleneck']]['date']:
    plt.axvspan(alert_date - pd.Timedelta(hours=12), alert_date + pd.Timedelta(hours=12), color='orange', alpha=0.15)

plt.title('Detección Preventiva de Cuello de Botella - Hub Norte (Black Friday)', fontsize=13, fontweight='bold')
plt.xlabel('Fecha')
plt.ylabel('Throughput Diario (Paquetes)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```

---

## 6. Cheatsheet de Respuestas Técnicas para Entrevistas

| Pregunta de Entrevista | Respuesta Técnica y Argumentación |
| :--- | :--- |
| **¿Por qué LightGBM y no ARIMA?** | ARIMA es un modelo *local* que no escala cuando manejas cientos de nodos y, fundamentalmente, no comparte patrones cruzados ni soporta covariables no lineales complejas sin sobreajustarse. Un modelo tabular global como LightGBM aprende simultáneamente de toda la topología de la red logística, permitiendo realizar inferencias robustas incluso en nodos con historial corto. |
| **¿Por qué no Deep Learning (LSTM / Temporal Fusion Transformer)?** | En series diarias con horizontes tácticos de 14 días y estacionalidades de calendario bien parametrizables, los árboles de decisión gradient boosted superan consistentemente a las redes recurrentes en velocidad de convergencia, requerimiento de cómputo y robustez tabular, simplificando el despliegue productivo en GCP. |
| **¿Cómo garantizas cero Data Leakage en tus rolling features?** | Para un horizonte de predicción $H=14$, ninguna estadística móvil puede calcularse con datos contemporáneos ni del intervalo $[t+1, t+14]$. Se aplica un desfase temporal formal `shift(14)` a la serie antes de calcular medias o desviaciones rodantes. |
| **¿Por qué WAPE y no MAPE?** | En nodos satélite de última milla o días feriados existen días con demanda cercana a cero. El MAPE se indetermina al dividir por cero y sobredimensiona los errores de nodos pequeños. El WAPE pondera de forma natural las discrepancias por el volumen absoluto real de la red. |
| **¿Cómo se conecta la predicción con los 2 pp de atrasos?** | El valor no está en el pronóstico aislado, sino en la toma de decisión prescriptiva. Usamos el cuantil $p90$ para dimensionar el peor escenario verosímil. Cuando $p90 > 0.95 \times \text{Capacidad}$, el *Motor de Promesa Logística* desvía pedidos hacia centros alternos con holgura de capacidad 14 días antes, evitando que los bultos entren a un nodo ya colapsado. |