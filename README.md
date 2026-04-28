# Tech Challenge Fase 1 — Classificação de Desfecho Clínico em SRAG

**FIAP Pos-Tech — Machine Learning Engineering**

Modelo de classificação para prever o desfecho clínico (cura vs óbito) de pacientes hospitalizados com Síndrome Respiratória Aguda Grave (SRAG), usando dados do SIVEP-Gripe do Ministério da Saúde.

---

## Datasets

### 1. Dados Tabulares — SIVEP-Gripe

| Campo | Valor |
|-------|-------|
| Fonte | [OpenDataSUS — Ministério da Saúde](https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024) |
| Arquivo | `INFLUD24-26-06-2025.csv` |
| Registros | ~268 mil |
| Variáveis | 194 |
| Variável alvo | `EVOLUCAO` → codificada como `OBITO` (0=Cura, 1=Óbito) |

**Coloque o arquivo em `data/raw/` antes de executar o pipeline.**

### 2. Dados de Imagem — Radiografias Torácicas

| Campo | Valor |
|-------|-------|
| Fonte | [COVID-19 Radiography Database — Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) |
| Download | Automático via `kagglehub` (requer `~/.kaggle/kaggle.json`) |
| Classes | COVID, Normal, Lung\_Opacity, Viral Pneumonia |

---

## Estrutura do Projeto

```
tech-challenge-srag-v2/
├── data/
│   ├── raw/                    # CSV original (não versionado)
│   └── processed/              # Splits parquet + artefatos pkl
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation_interpretability.ipynb
│   └── 05_image_validation.ipynb
├── src/
│   ├── tabular/                # Módulos do pipeline sklearn/XGBoost
│   │   ├── load_data.py
│   │   ├── preprocessing.py
│   │   ├── modeling.py
│   │   └── evaluation.py
│   └── image/                  # Módulos do pipeline TensorFlow/Keras
│       ├── image_data.py
│       ├── image_preprocessing.py
│       ├── image_model.py
│       └── image_evaluation.py
├── results/
│   ├── figures/                # Gráficos gerados (confusion matrix, ROC, SHAP)
│   └── models/                 # Modelos serializados .pkl
├── tests/
│   ├── test_preprocessing.py
│   └── test_modeling.py
├── docs/
│   ├── relatorio_tecnico.md
│   └── dicionario-de-dados-2019-a-2025.pdf
├── run_pipeline.py
├── requirements.txt
├── Makefile
└── Dockerfile
```

---

## Requisitos

- **Python 3.11 ou 3.12** (evitar 3.13+ por incompatibilidade de wheels com numpy/pandas)
- ~4 GB de RAM para o pipeline tabular completo
- Credenciais do Kaggle (`~/.kaggle/kaggle.json`) para o notebook de imagens

---

## Instalação e Execução Local

```bash
# 1. Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Adicionar o dataset tabular
cp /caminho/para/INFLUD24-26-06-2025.csv data/raw/

# 4. Executar o pipeline completo
python run_pipeline.py

# 5. Rodar os testes
pytest tests/ -v
```

### Opções do pipeline

```bash
python run_pipeline.py --no-grid    # Sem GridSearchCV (mais rápido)
python run_pipeline.py --no-shap    # Sem SHAP (mais rápido)
python run_pipeline.py --nrows 10000  # Subconjunto dos dados
```

### Notebooks interativos

```bash
jupyter notebook notebooks/
```

---

## Execução com Docker

```bash
# Build
docker build -t tech-challenge-srag .

# Executar (abre Jupyter na porta 8888)
docker run -p 8888:8888 -v $(pwd)/data:/app/data tech-challenge-srag
```

Acesse: `http://localhost:8888`

---

## Modelos Treinados

### Dados Tabulares

| Modelo | Tipo | Papel |
|--------|------|-------|
| Regressão Logística | Linear | Baseline interpretável |
| Árvore de Decisão | Não-linear | Regras explícitas |
| Random Forest | Ensemble (bagging) | Generalização robusta |
| XGBoost | Ensemble (boosting) | Alta performance |

Todos otimizados com `GridSearchCV` + `StratifiedKFold(5)`, métrica: **F1-score (Óbito)**.

### Dados de Imagem

- CNN construída do zero com Keras Sequential
- `GlobalAveragePooling2D` para controle de parâmetros (~24k treináveis)
- Data augmentation leve (rotação ≤10°, zoom ≤10%)

---

## Métricas de Avaliação

| Métrica | Tabular | Imagem |
|---------|---------|--------|
| Accuracy | ✅ | ✅ |
| Precision | ✅ | ✅ |
| Recall | ✅ | ✅ |
| F1-score | ✅ (métrica principal) | ✅ |
| ROC-AUC | ✅ | — |
| Matriz de confusão | ✅ | ✅ |
| SHAP | ✅ | — |

---

## Makefile

```bash
make run    # Pipeline tabular completo
make test   # pytest
make lint   # ruff + mypy
```

---

## Relatório Técnico

Ver [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md) para detalhamento de:
- Estratégias de pré-processamento
- Modelos usados e justificativas
- Resultados e interpretação
