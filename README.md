# ⚽ MUNDIAL 2026 — Modelo Predictivo XGBoost

> *"El fútbol no se gana en el campo. Se gana en los datos."*

**¿Quién levantará la copa en julio de 2026?** Este proyecto combina machine learning, datos reales de mercado, historial mundialista y rendimiento en eliminatorias para predecir — con fundamento matemático — el destino de las **48 selecciones** que disputarán el primer Mundial de la historia con este formato ampliado.

---

## 🏟️ ¿Por qué este modelo es diferente?

La mayoría de predicciones del Mundial se basan en el ranking FIFA o en intuición. Este modelo va mucho más allá: entrena un **XGBoost Regresor** sobre los resultados reales de Qatar 2022 y Rusia 2018 (con pesos diferenciados), y un **XGBoost Clasificador Binario** para las eliminatorias directas. Cada predicción es el resultado de 7 variables cuidadosamente seleccionadas y ponderadas, desde el valor de mercado de los jugadores hasta la dificultad real del camino clasificatorio.

**El resultado no es una opinión. Es una probabilidad calculada.**

---

## 📁 Estructura del Dataset

El dataset `dataset_mundial_2026_v6.csv` contiene **48 filas y 23 columnas**, una por cada selección clasificada al Mundial 2026 distribuida en 12 grupos (A–L).

### 🗺️ Bloque 1 — Identificación

| Columna | Descripción |
|---------|-------------|
| `equipo` | Nombre de la selección |
| `grupo` | Grupo del sorteo oficial FIFA (A a L) |
| `conf` | Confederación: UEFA · CONMEBOL · CAF · AFC · CONCACAF · OFC |
| `primera_vez` | **1** si la selección debuta en un Mundial — penaliza automáticamente al 4to de su grupo |

Son **5 debutantes absolutos** en este Mundial: Haití, Curazao, Cabo Verde, Jordania y Uzbekistán. Ninguno puede clasificar a la fase eliminatoria por modelo.

---

### 📊 Bloque 2 — Ranking FIFA y Valor de Mercado

| Columna | Descripción |
|---------|-------------|
| `ranking_fifa` | Posición oficial FIFA (abril 2026) |
| `ranking_score` | `210 − ranking_fifa` → mayor = mejor |
| `tm_value_m` | Valor de mercado total del plantel en millones de euros (Transfermarkt, dic 2025) |
| `tm_cat` | Categoría ordinal: **0** (<100M) · **1** (100-400M) · **2** (400-650M) · **3** (>650M) |

**Dato de impacto**: la brecha de valor de mercado entre selecciones es abismal. **Inglaterra** encabeza con **€1,400M** de plantel; **Haití** cierra con apenas **€8M** — una diferencia de 175 veces. En total, los 48 planteles suman **€14,135 millones** sobre el césped.

Las 7 selecciones con presupuesto Gamma (>€650M): Brasil, Alemania, Países Bajos, España, Francia, Portugal e Inglaterra. Las 20 selecciones con presupuesto sumamente bajo (<€100M) incluyen desde Qatar hasta Ghana.

---

### 🏆 Bloque 3 — Historial en Mundiales recientes

| Columna | Descripción |
|---------|-------------|
| `pos_wc22` | Posición final en Qatar 2022 (1=campeón · 99=no participó) |
| `pos_wc18` | Posición final en Rusia 2018 (1=campeón · 99=no participó) |
| `wc_score` | Score ponderado: `WC2022 × 2.0 + WC2018 × 1.5` (posición invertida, mayor = mejor) |

Esta es la variable de **mayor poder predictivo** en el modelo (gain=0.567). La lógica es simple: los equipos que llegaron lejos en los últimos dos Mundiales tienden a repetir. Se penaliza más a quien falló en el más reciente.

**Top 5 por wc_score**:
1. 🇫🇷 Francia — 3.46 (finalista 2022 · campeón 2018)
2. 🇭🇷 Croacia — 3.37 (3er lugar 2022 · finalista 2018)
3. 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Inglaterra — 3.01 (cuartos 2022 · semis 2018)
4. 🇦🇷 Argentina — 2.73 (campeón 2022)
5. 🇲🇦 Marruecos — 2.60 (semis 2022)

---

### 🛣️ Bloque 4 — Eliminatorias 2026

| Columna | Descripción |
|---------|-------------|
| `elim_pos` | Posición final en la tabla de eliminatorias de su confederación |
| `elim_total` | Total de equipos en competencia dentro de su zona |
| `elim_raw` | `1 − (elim_pos / elim_total)` → 1.0 si terminó primero, 0.0 si último |

No todas las eliminatorias son iguales. Terminar **primero en CONMEBOL** (10 equipos incluyendo Argentina, Brasil y Uruguay) no es lo mismo que terminar primero en la OFC (donde la competencia es sensiblemente menor). Por eso `elim_raw` se usa en conjunto con el `copa_score` ponderado por confederación.

---

### 🌍 Bloque 5 — Copa Continental (últimos 2 torneos)

| Columna | Descripción |
|---------|-------------|
| `tit_rec` | Títulos en el torneo continental más reciente (Eurocopa, Copa América, etc.) |
| `pos_rec` | Mejor posición en ese torneo (1=campeón · 13=grupos) |
| `tit_ant` | Títulos en el torneo anterior |
| `pos_ant` | Mejor posición en ese torneo anterior |
| `copa_score` | Score combinado: `(torneo_rec × 2.0 + torneo_ant × 1.0) × w_conf` |

El multiplicador `w_conf` refleja la dificultad real de cada confederación:

| Confederación | Torneo | Peso |
|--------------|--------|------|
| UEFA | Eurocopa | **0.96** — el "Mundial sin Sudamérica" |
| CONMEBOL | Copa América | **0.91** — el promedio FIFA más alto del planeta |
| CAF | AFCON | **0.71** — alta competitividad interna, terreno impredecible |
| AFC | Copa Asia | **0.55** — cúspide fuerte, cola muy larga |
| CONCACAF | Gold Cup | **0.48** — competitivo solo desde semis |
| OFC | OFC Nations | **0.11** — monólogo de Nueva Zelanda desde que Australia emigró a la AFC |

**Argentina lidera copa_score** con 1.479 — ganó la Copa América 2021 y 2024 consecutivamente en la confederación más exigente del mundo.

---

### 🌐 Bloque 6 — Rendimiento en Torneos Internacionales de Clubes

| Columna | Descripción |
|---------|-------------|
| `cwc_raw` | Rendimiento del/los club(es) del país en el **FIFA Club World Cup 2025** (0.0–1.0) |
| `fic_raw` | Rendimiento en la **FIFA Intercontinental Cup 2025** (0.0–1.0) |
| `torneo_score` | Score combinado: `CWC × 2.8 + FIC × 2.2 + Elim × 1.8` |

Escala de rendimiento:

| Resultado | Valor raw |
|-----------|-----------|
| Campeón | 1.00 |
| Finalista | 0.85 |
| Semifinal | 0.65 |
| Cuartos | 0.45 |
| Round of 16 | 0.25 |
| Fase de grupos | 0.10 |
| No participó | 0.00 |

**Resultados reales usados:**
- 🏆 **CWC 2025**: Chelsea campeón 3–0 PSG. Semis: Chelsea 2–0 Fluminense · PSG 4–0 Real Madrid
- 🏆 **FIC 2025**: PSG campeón vs Flamengo (pen. 2–1, Ahmad bin Ali Stadium, 17 dic 2025)

Por eso **Francia lidera torneo_score** (6.02): PSG fue finalista del CWC y campeón del FIC. **Brasil es segunda** (5.13): Fluminense llegó a semis del CWC y Flamengo fue finalista del FIC.

---

## 🤖 Arquitectura del Modelo

### Fase de Grupos — XGBoost Regresor
Predice la **posición ordinal** (1–48) de cada selección. El ranking dentro de cada grupo se determina directamente por el score del regresor — sin simulación de partidos, sin azar.

**Entrenamiento**: Qatar 2022 (`peso × 2.0`) + Rusia 2018 (`peso × 1.5`) = 64 muestras con `sample_weight`.

**Orden de importancia (gain)**:
```
wc_score > elim_norm > copa_norm > ranking_score > tm_cat > copa_score > eafc26 > primera_vez
```

### Eliminatorias — XGBoost Clasificador Binario
Para cada enfrentamiento, el modelo predice **P(A clasifica)**. Si P ≥ 50%, avanza A. Sin marcadores. Sin azar. Solo probabilidades.

- Input: `ΔX = X_A − X_B` (diferencia de features entre los dos equipos)
- Blend 60% clasificador + 40% score del regresor (evita extremos 0%/100%)
- Restricción de fase: equipos que llegan 2+ rondas más lejos de lo predicho tienen probabilidad forzada al 3%

**Equipos bloqueados** (no pueden clasificar a eliminatorias incluso como mejor tercero): Curazao · Haití · Jordania · Nueva Zelanda · Cabo Verde · Uzbekistán.

---

## 📐 Fixture

- **12 grupos** × 4 equipos = 48 selecciones
- **32 clasificados** a dieciseisavos: 12 primeros + 12 segundos + 8 mejores terceros
- Fixture oficial FIFA L01–L16 con fechas reales (28 jun – 3 jul 2026)
- Árbol: R32 → R16 → QF → SF → 3°/4° → Final

---

## 🗂️ Archivos del Proyecto

| Archivo | Descripción |
|---------|-------------|
| `dataset_mundial_2026_v6.csv` | Dataset principal — 48 equipos × 23 variables |
| `mundial_2026_xgboost.ipynb` | Notebook completo con modelo, dashboards y bracket |
| `Eliminatorias.csv` | Fuerza Final de eliminatorias por selección (fuente: cálculo propio) |
| `Copas_intercontinentales.csv` | Poder Final en copas continentales con ajuste por confederación |

---

## ⚡ Cómo ejecutarlo

```bash
pip install xgboost scikit-learn pandas numpy plotly
jupyter notebook mundial_2026_xgboost.ipynb
```

Ejecuta todas las celdas en orden. La celda 27 recalcula automáticamente los scores con los valores corregidos del dataset antes de generar el ranking de grupos.

---

## 🔢 Datos rápidos

| Métrica | Valor |
|---------|-------|
| Selecciones analizadas | **48** |
| Variables del modelo | **7** (+ 1 binaria) |
| Muestras de entrenamiento | **64** (WC2022×2.0 + WC2018×1.5) |
| Partidos totales modelados | **103** (72 grupos + 31 eliminatorias) |
| Valor total de mercado | **€14,135M** |
| Selección más valiosa | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 **Inglaterra — €1,400M** |
| Selección menos valiosa | 🇭🇹 **Haití — €8M** |
| Debutantes absolutos | **5** (Haití · Curazao · Cabo Verde · Jordania · Uzbekistán) |
| Confederaciones representadas | **6** |

---


