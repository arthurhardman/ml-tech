# Tech Challenge — Fase 1 e Fase 2 — SRAG

**FIAP Pos-Tech — Machine Learning Engineering**

Projeto de classificação para prever o desfecho clínico (cura vs óbito) de pacientes hospitalizados com Síndrome Respiratória Aguda Grave (SRAG), usando dados do SIVEP-Gripe do Ministério da Saúde.

Na **Fase 2**, o projeto foi estendido com otimização de hiperparâmetros via **Algoritmos Genéticos** e uma camada inicial de **LLM** para explicar as predições em linguagem natural.

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
│   ├── 05_image_validation.ipynb
│   ├── 06_genetic_optimization.ipynb
│   └── 07_llm_interpretation.ipynb
├── src/
│   ├── tabular/                # Módulos do pipeline sklearn/XGBoost
│   │   ├── load_data.py
│   │   ├── preprocessing.py
│   │   ├── modeling.py
│   │   └── evaluation.py
│   ├── image/                  # Módulos do pipeline TensorFlow/Keras
│   │   ├── image_data.py
│   │   ├── image_preprocessing.py
│   │   ├── image_model.py
│   │   └── image_evaluation.py
│   ├── genetic/                # Fase 2 — Algoritmo Genético
│   ├── llm/                    # Fase 2 — Explicações com LLM
│   └── monitoring/             # Fase 2 — Logs e histórico dos experimentos
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

### Execução da Fase 2

A Fase 2 parte dos artefatos gerados pelo pipeline tabular da Fase 1. Antes de rodar os notebooks novos, execute o pipeline pelo menos uma vez.

```bash
# Gera dados processados, modelos e figuras base
python run_pipeline.py --no-grid --no-shap

# Demonstração da otimização por Algoritmo Genético
jupyter notebook notebooks/06_genetic_optimization.ipynb

# Demonstração das explicações com LLM
jupyter notebook notebooks/07_llm_interpretation.ipynb
```

A integração com LLM usa Ollama quando disponível. Se o Ollama não estiver rodando, o projeto usa modo de demonstração/mock para manter os testes e o fluxo de apresentação funcionando.

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

## Fase 2 — Algoritmo Genético e LLM

A otimização da Fase 2 foca no modelo **XGBoost**, porque ele foi o principal candidato da etapa tabular e tem hiperparâmetros bem adequados para busca evolutiva.

O Algoritmo Genético trabalha com cromossomos que representam hiperparâmetros como `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma` e `scale_pos_weight`. A avaliação usa F1-score/recall para o desfecho de óbito, já que falsos negativos são especialmente sensíveis nesse tipo de problema.

A parte de LLM recebe a predição, a probabilidade estimada e os principais fatores do caso para gerar uma explicação em português. A saída deixa claro que o modelo serve como apoio à triagem e não substitui a avaliação médica.

Arquivos principais da Fase 2:

- `src/genetic/`: codificação dos genes, operadores e otimizador.
- `src/monitoring/`: histórico dos experimentos e curvas de convergência.
- `src/llm/`: cliente da LLM, prompts e explicador.
- `notebooks/06_genetic_optimization.ipynb`: execução dos experimentos com AG.
- `notebooks/07_llm_interpretation.ipynb`: exemplos de explicações geradas.

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
