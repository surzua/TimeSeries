# 📦 Time Series Forecasting en Supply Chain: Red Logística E-commerce Multirregional

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/Environment-uv-brightgreen.svg)](https://github.com/astral-sh/uv)
[![ML Framework: LightGBM](https://img.shields.io/badge/ML-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)
![Status: Production Ready](https://img.shields.io/badge/Status-Complete-success.svg)

> **Pipeline Integral de Predicción de Demanda Tabular Global, Validación Purgada y Mitigación Prescriptiva de Cuellos de Botella ($H = 14$ días) en Redes Logísticas Multieslabón.**

> [!NOTE]
> **Aviso de Simulación y Alcance Metodológico:**  
> Este proyecto constituye un **marco de experimentación sintético, estudio de caso benchmark y diseño de arquitectura técnica** desarrollado para explorar y validar algoritmos de Machine Learning e Investigación de Operaciones en logística de e-commerce.  
> **La topología de red multieslabón, las capacidades operativas, los volúmenes de demanda y las métricas de mitigación son el producto de un entorno de simulación estilizado con fines pedagógicos y de portafolio profesional; no constituyen, utilizan ni reflejan datos reales, métricas privadas ni operaciones propietarias de ninguna entidad comercial o equipo corporativo específico.**

---

## 📑 Tabla de Contenidos
1. [Planteamiento del Caso de Estudio (Marco STAR)](#1-planteamiento-del-caso-de-estudio-marco-star)
2. [Arquitectura del Repositorio](#2-arquitectura-del-repositorio)
3. [Fundamentos Teóricos y Matemáticos](#3-fundamentos-teóricos-y-matemáticos)
4. [Estructura y Contenido de los Notebooks](#4-estructura-y-contenido-de-los-notebooks)
5. [Guía de Instalación y Uso con `uv`](#5-guía-de-instalación-y-uso-con-uv)
6. [Resultados e Impacto Operativo](#6-resultados-e-impacto-operativo)
7. [Cheatsheet de Respuestas Técnicas para Entrevistas](#7-cheatsheet-de-respuestas-técnicas-para-entrevistas)

---

## 1. Planteamiento del Caso de Estudio (Marco STAR)

### 📌 Situation (Contexto y Escenario Experimental)
Como escenario de experimentación y benchmark para redes logísticas multieslabón en e-commerce de gran escala, se estructura una simulación representativa que modela un flujo mensual estilizado del orden de **500.000 órdenes** distribuido en una infraestructura logística heterogénea:
- **Nodos de abastecimiento:** Bodegas de sellers 1P (retail propio) y 3P (marketplace).
- **Nodos de transferencia y cross-docking:** Hubs principales de distribución (Hub Central, Hub Norte, Hub Sur) y plataformas satélite de Cross-Docking (XD Oriente, XD Costera).
- **Nodos de última milla y retiro:** Despacho a domicilio y centros de retiro Click & Collect (CC Store Alpha, CC Store Beta, CC Store Gamma).

### ⚡ Task (Desafío de Modelamiento)
La demanda e-commerce exhibe una **alta volatilidad estacional** (picos dominicales y de lunes, incremento del 18% en ciclos quincenales de pago y shocks de hasta +110% en eventos promocionales como CyberDays y Black Friday). En contraste, la capacidad física de procesamiento diario (throughput de líneas de picking y empaque en paquetes/día) es rígida en el corto plazo.

Cuando el flujo entrante simulado supera la capacidad nominal de un nodo intermedio, se generan cuellos de botella: los bultos quedan atrapados en colas de espera, aumentan los tiempos de ciclo y se producen quiebres en la promesa de entrega comprometida (SLA). El desafío consiste en **evaluar si un modelo probabilístico global a 14 días ($H=14$) permite anticipar saturaciones con la antelación necesaria para activar políticas prescriptivas de balanceo dinámico de carga.**

### 🛠️ Action (Enfoque Técnico y Machine Learning)
Diseño e implementación de un pipeline modular de Time Series Forecasting tabular global:
1. **Modelado Global con LightGBM:** Un único modelo entrenado simultáneamente sobre todas las series temporales de la red sintética, enriquecido con metadatos de los nodos, variables de calendario y covariables exógenas.
2. **Horizonte Táctico Directo ($H=14$ días):** Generación de proyecciones a 14 días vista para habilitar la planificación oportuna de turnos de personal y contratación de transporte troncal.
3. **Forecasting Probabilístico Asimétrico (Cuantiles $p_{10}, p_{50}, p_{90}$):** Estimación mediante la función de pérdida de Pinball. El percentil 50 ($p_{50}$) proyecta la mediana esperada para planificación regular, mientras que el percentil 90 ($p_{90}$) modela el escenario de sobrecarga con aversión al riesgo (donde subestimar la demanda penaliza 9 veces más que sobrestimarla).
4. **Capa Prescriptiva Acoplada:** Si la proyección $p_{90}$ supera el 95% de la capacidad nominal de un nodo, el sistema dispara alertas preventivas para desviar flujo hacia nodos alternos con holgura y solicitar refuerzos operativos.

### 🏆 Result (Resultados del Benchmark Simulado)
- **Reducción de 2.03 puntos porcentuales en la tasa de saturación y atrasos operativos** sobre el flujo simulado de 500.000 órdenes mensuales.
- **Mitigación proactiva de ~10.150 órdenes en riesgo de atraso por mes** mediante alertas tempranas y rebalanceo dinámico de capacidad.
- Demostración de viabilidad técnica para integrar pronósticos probabilísticos asimétricos con políticas de control operativo en cadenas de suministro.
- Reducción sustancial en costos de expedición reactiva y fletes de urgencia no planificados en el entorno simulado.

---

## 2. Arquitectura del Repositorio

```text
TimeSeries/
├── .venv/                                      # Entorno virtual aislado gestionado por uv
├── data/                                       # Almacenamiento de datasets en formato Parquet
│   ├── raw_logistics_network.parquet           # Datos crudos generados de la red
│   └── logistics_features_h14.parquet          # Matriz de features engineered (H=14)
├── docs/                                       # Documentación técnica, blog y recursos visuales
│   ├── blog_supply_chain_forecasting.md        # 📖 Artículo exhaustivo para blog personal (pedagogía, gráficos e impacto)
│   ├── forecasting_supply_chain.md             # Blueprint técnico y especificaciones operacionales
│   └── images/                                 # Gráficos de alta resolución extraídos de los experimentos
├── notebooks/                                  # Cuadernos de aprendizaje y experimentación
│   ├── 01_eda_baseline_feature_engineering.ipynb
│   └── 02_lightgbm_probabilistic_capacity_mitigation.ipynb
├── scripts/                                    # Scripts auxiliares de automatización
│   └── generate_notebooks.py
├── pyproject.toml                              # Configuración del proyecto y dependencias
├── uv.lock                                     # Archivo de bloqueo determinista de librerías
├── .gitignore                                  # Patrones de exclusión de git
└── README.md                                   # Documentación técnica principal
```

> [!TIP]
> **Lectura recomendada:** Para una explicación exhaustiva, narrativa pedagógica paso a paso y visualización de todos los gráficos con su interpretación de negocio (pensado para blog de GitHub), consulta el artículo en [`docs/blog_supply_chain_forecasting.md`](docs/blog_supply_chain_forecasting.md).


---

## 3. Fundamentos Teóricos y Matemáticos

### 3.1 Formulación Matemática del Problema

Sea $y_{i,t} \in \mathbb{R}^+$ el volumen de paquetes procesados en el nodo $i \in \lbrace 1, \dots, N \rbrace$ durante el día $t$. El objetivo es estimar la distribución condicional de la demanda futura:

$$
P(y_{i,t+h} \mid \mathcal{F}_t), \quad \forall h \in \lbrace 1, \dots, 14 \rbrace
$$

donde la filtración histórica $\mathcal{F}_t$, generada por las series observadas y covariables exógenas $X$ hasta el día de corte $t$, se define formalmente como:

$$
\mathcal{F}_t = \sigma\left(\lbrace y_{i,s}, X_{i,s} \rbrace_{s \le t}\right)
$$

### 3.2 Descomposición Estocástica

$$
y_t = T_t + S_t + C_t + \varepsilon_t
$$

- **$T_t$ (Tendencia):** Crecimiento orgánico de penetración e-commerce.
- **$S_t$ (Estacionalidad):** Armónicos periódicos dobles:
  - Ciclo semanal ($s = 7$ días) con picos en domingo/lunes.
  - Ciclo quincenal de pago de nóminas (días 14-17 y 28-31).
- **$C_t$ (Shocks Exógenos):** Impulsos comerciales extraordinarios (CyberDays y Black Friday).
- **$\varepsilon_t$ (Perturbación):** Ruido blanco no correlacionado.

### 3.3 Test Dickey-Fuller Aumentado (ADF)

Para contrastar formalmente la presencia de raíces unitarias:

$$
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{j=1}^p \delta_j \Delta y_{t-j} + e_t
$$

- $H_0: \gamma = 0$ (Serie no estacionaria con raíz unitaria).
- $H_1: \gamma < 0$ (Serie estacionaria).

### 3.4 Métrica de Error: WAPE vs MAPE

En retail y logística, la métrica estándar es el **WAPE (Weighted Absolute Percentage Error)**:

$$
\text{WAPE} = \frac{\sum_{i,t} |y_{i,t} - \hat{y}_{i,t}|}{\sum_{i,t} y_{i,t}}
$$

- **Por qué supera a MAPE:** MAPE divide por $y_{i,t}$, lo que resulta en división por cero si la demanda es nula, y sobrepenaliza desproporcionadamente a los nodos pequeños de última milla (un error de 5 bultos en Click & Collect tendría un impacto en MAPE idéntico a 5.000 bultos en un Hub principal). WAPE pondera naturalmente por volumen absoluto.

### 3.5 Regresión Cuántica y Función de Pérdida de Pinball

Para un cuantil $\alpha \in (0, 1)$, la función de pérdida asimétrica de Pinball se define como:

$$
\mathcal{L}_\alpha(y, \hat{y}) = \max\left(\alpha(y - \hat{y}), (1 - \alpha)(\hat{y} - y)\right) = \begin{cases} \alpha(y - \hat{y}) & \text{si } y \ge \hat{y} \\ (1 - \alpha)(\hat{y} - y) & \text{si } y < \hat{y} \end{cases}
$$

Para $\alpha = 0.90$, subestimar la demanda ($y > \hat{y}$) es penalizado con factor $0.90$, mientras que sobreestimar ($y < \hat{y}$) solo con $0.10$. La relación de penalización es:

$$
\frac{\text{Costo Subestimación}}{\text{Costo Sobreestimación}} = \frac{0.90}{0.10} = 9
$$

Esto garantiza que la proyección $p_{90}$ represente una cota superior con aversión al riesgo operativa.

### 3.6 Regla Prescriptiva de Capacidad

$$
\text{Alerta de Cuello de Botella} = \mathbb{I}(\hat{y}_{p_{90}} > 0.95 \times \text{Capacidad Nominal})
$$

- **Atraso Reactivo (sin forecast):** $\text{Atraso} = \max(0, y - \text{Capacidad})$.
- **Atraso Proactivo (con alerta a 14 días):** Alerta activa permite desviar el 80% del exceso hacia nodos alternos y programar sobreturnos de personal: $\text{Atraso} = 0.20 \times \max(0, y - \text{Capacidad})$.

---

## 4. Estructura y Contenido de los Notebooks

### 📓 [Notebook 1: 01_eda_baseline_feature_engineering.ipynb](notebooks/01_eda_baseline_feature_engineering.ipynb)
- **Generación y Guardado de Datos:** Creación de las series para los 8 nodos de la red logística multieslabón y persistencia en `data/raw_logistics_network.parquet`.
- **Diagnóstico Estadístico:**
  - Test ADF aplicado a cada serie temporal.
  - Gráficos de Autocorrelación (ACF) y Autocorrelación Parcial (PACF) para 28 rezagos con bandas de confianza al 95%.
- **Benchmark Baseline:** Cálculo del WAPE para el Seasonal Naive ($s=7$), alcanzando un **12.50% de error base**.
- **Pipeline de Feature Engineering ($H=14$):**
  - Codificación cíclica (seno y coseno) para día de semana.
  - Rezagos temporales anclados ($k \in \{14, 15, 21, 28\}$).
  - Medias, desviaciones estándar y máximos rodantes desfasados estrictamente con `.shift(14)`.
  - Guardado en `data/logistics_features_h14.parquet`.

### 📓 [Notebook 2: 02_lightgbm_probabilistic_capacity_mitigation.ipynb](notebooks/02_lightgbm_probabilistic_capacity_mitigation.ipynb)
- **Rolling-Origin Cross-Validation:** 3 ventanas de test de 14 días sin fuga de datos.
- **Modelado Global LightGBM (L1 / Mediana):**
  - Optimización sobre función de pérdida `regression_l1`.
  - Obtención de un **WAPE promedio de 6.42%** (reducción relativa del ~50% frente al baseline).
  - Análisis de Feature Importance por Ganancia (Gain) y Cortes (Split).
- **Forecasting Probabilístico:**
  - Entrenamiento de 3 regresores cuantílicos ($p_{10}, p_{50}, p_{90}$) para Black Friday.
  - Verificación de no-cruce de cuantiles (monotonía estocástica 100% verificada).
- **Simulación Prescriptiva de Capacidad:**
  - Demostración empírica de la **reducción de 2.03 puntos porcentuales** en la tasa de atrasos de SLA.
  - Extrapolación a nivel red: rescate de más de 10.000 órdenes al mes dentro de la promesa.
- **Visualización del Cono de Incertidumbre:** Monitoreo predictivo y zonas de alarma en Hub Norte y visión consolidada de la red.

---

## 5. Guía de Instalación y Uso con `uv`

Este proyecto utiliza [`uv`](https://github.com/astral-sh/uv), el gestor de paquetes y entornos virtuales de Python ultrarrápido desarrollado en Rust.

### Paso 1: Clonar o ingresar al repositorio
```bash
cd /Users/sebastianfelipeurzuaborquez/Proyectos/TimeSeries
```

### Paso 2: Sincronizar el entorno virtual
El archivo `uv.lock` bloquea exactamente las versiones de las librerías. Para instalar el entorno:
```bash
uv sync
```
Esto crea automáticamente la carpeta `.venv/` con todas las dependencias instaladas.

### Paso 3: Registrar el kernel para Jupyter
El kernel ya ha sido registrado en el sistema. Si deseas registrarlo manualmente:
```bash
uv run python -m ipykernel install --user --name timeseries --display-name "Python (TimeSeries - uv)"
```

### Paso 4: Abrir y ejecutar los notebooks
Puedes abrir los notebooks en VS Code, Cursor o JupyterLab seleccionando el kernel **`Python (TimeSeries - uv)`**:
```bash
uv run jupyter lab
```
O ejecutar los scripts directamente en el entorno de `uv`:
```bash
uv run python scripts/generate_notebooks.py
```

---

## 6. Resultados e Impacto Operativo

| Métrica / Parámetro | Baseline Heurístico (Seasonal Naive) | Modelo Tabular Global (LightGBM L1) | Impacto / Mejora |
| :--- | :---: | :---: | :---: |
| **WAPE Global Red** | 12.50% | **6.42%** | **-48.6% error relativo** |
| **Horizonte Operativo** | 7 días (reactivo) | **14 días (directo)** | **+7 días de antelación** |
| **Incertidumbre** | No cuantificada | Cono $[p_{10}, p_{90}]$ | Cobertura probabilística total |
| **Tasa Atrasos Red (Black Friday)** | 2.54% (Reactiva) | **0.51% (Proactiva)** | **-2.03 pp en atrasos** |
| **Órdenes Rescatadas a Tiempo** | 0 órdenes | **~10.150 órdenes / mes** | Reducción de reclamos y fletes |

---

## 7. Cheatsheet de Respuestas Técnicas para Entrevistas

| Pregunta de Entrevista | Respuesta Técnica y Argumentación de Alto Nivel |
| :--- | :--- |
| **¿Por qué LightGBM y no ARIMA?** | ARIMA es un modelo *local* univariado: exige ajustar y mantener un modelo por cada nodo, no escala cuando manejas cientos de series y es incapaz de transferir conocimiento cruzado o soportar covariables exógenas no lineales sin sobreajustarse. Un modelo tabular global como LightGBM aprende simultáneamente de toda la topología de la red logística, permitiendo realizar inferencias robustas incluso en nodos con historial corto (*cold-start*). |
| **¿Por qué no Deep Learning (LSTM / Temporal Fusion Transformer)?** | En series diarias con horizontes tácticos de 14 días y estacionalidades de calendario bien parametrizables, los árboles de decisión gradient boosted superan consistentemente a las redes recurrentes en velocidad de convergencia, menor requerimiento de cómputo y robustez tabular, simplificando el despliegue productivo y el monitoreo en GCP / Vertex AI. |
| **¿Cómo garantizas cero Data Leakage en tus rolling features?** | Para un horizonte de predicción $H=14$, ninguna estadística móvil puede calcularse con datos contemporáneos ni del intervalo $[t+1, t+14]$. Se aplica un desfase temporal formal `shift(14)` a la serie antes de calcular medias o desviaciones rodantes, garantizando que en el momento $T$ solo se consuman datos efectivamente observados. |
| **¿Por qué WAPE y no MAPE?** | En nodos satélite de última milla o días feriados existen días con demanda cercana a cero. El MAPE se indetermina al dividir por cero y sobredimensiona los errores de nodos pequeños. El WAPE pondera de forma natural las discrepancias por el volumen absoluto real de la red. |
| **¿Cómo se conecta la predicción con los 2 pp de atrasos?** | El valor no está en el pronóstico aislado, sino en la toma de decisión prescriptiva. Usamos el cuantil $p_{90}$ para dimensionar el peor escenario verosímil. Cuando $p_{90} > 0.95 \times \text{Capacidad}$, el *Motor de Promesa Logística* desvía pedidos hacia centros alternos con holgura de capacidad 14 días antes, evitando que los bultos entren a un nodo ya colapsado. |

---

**Autor:** Sebastian Urzúa Bórquez  
**Especialidad:** Advanced Analytics & Data Science en Supply Chain / E-commerce
