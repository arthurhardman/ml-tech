# Tech Challenge — Fase 1 e Fase 2 — SRAG

**FIAP Pós-Tech — Machine Learning Engineering**  
Repositório: `arthurhardman/ml-tech`

Projeto de classificação para prever o desfecho clínico de pacientes hospitalizados com **SRAG** (Síndrome Respiratória Aguda Grave), usando dados do **SIVEP-Gripe / OpenDataSUS**.

A variável original do dataset tabular é `EVOLUCAO`. Para o modelo, ela é convertida para a variável binária `OBITO`:

- `0 = Cura`
- `1 = Óbito`

A **Fase 1** cobre o pipeline tabular de Machine Learning: EDA, pré-processamento, modelagem, avaliação e interpretabilidade. Ela também inclui uma validação complementar com imagens de radiografias torácicas.

A **Fase 2** o projeto foi estendido com otimização de hiperparâmetros por **Algoritmo Genético** e uma camada de **LLM** para explicar predições em linguagem natural.

---

## 1. Datasets

### Dados tabulares — SIVEP-Gripe / SRAG

| Campo | Valor |
|---|---|
| Fonte | [OpenDataSUS — SRAG 2019 a 2026](https://dadosabertos.saude.gov.br/dataset/srag-2019-a-2026) |
| Arquivo usado | `INFLUD24-26-06-2025.csv` |
| Tamanho original aproximado | 267.984 registros e 194 colunas |
| Registros após filtro de `EVOLUCAO` válida | aproximadamente 240.436 |
| Variável alvo | `EVOLUCAO` → `OBITO` (`0 = Cura`, `1 = Óbito`) |

O CSV bruto não fica versionado no repositório. Para rodar o pipeline, coloque o arquivo em:

```text
data/raw/INFLUD24-26-06-2025.csv
```

Na execução, use o parâmetro `--data` para deixar explícito qual arquivo deve ser lido:

```bash
python run_pipeline.py --data data/raw/INFLUD24-26-06-2025.csv
```

### Dados de imagem — Radiografias Torácicas

| Campo | Valor |
|---|---|
| Fonte | [COVID-19 Radiography Database — Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) |
| Download | Automático via `kagglehub` |
| Classes | COVID, Normal, Lung_Opacity, Viral Pneumonia |
| Notebook | `notebooks/05_image_validation.ipynb` |
| Código | `src/image/` |

A parte de imagem é um complemento da Fase 1. O carregamento fica em `src/image/image_data.py`, usando `kagglehub`, e o notebook permite controlar a quantidade de imagens por classe com `MAX_IMAGES_PER_CLASS`.

---

## 2. Estrutura do projeto

```text
ml-tech/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/                         # CSV bruto do SIVEP
│   └── processed/                   # pasta de apoio para dados processados
├── docs/
│   ├── plano_fase2.md
│   ├── relatorio_fase2.md
│   ├── relatorio_tecnico.md
│   ├── roteiro_video_fase2.md
│   └── dicionario-de-dados-2019-a-2025.pdf
├── experiments/
│   ├── experimentos.yaml
│   ├── exp_A/
│   ├── exp_B/
│   └── exp_C/
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation_interpretability.ipynb
│   ├── 05_image_validation.ipynb
│   ├── 06_genetic_optimization.ipynb
│   └── 07_llm_interpretation.ipynb
├── results/
│   ├── figures/                     # gráficos consolidados para relatório e vídeo
│   └── models/                      # modelos/artefatos consolidados
├── src/
│   ├── data/processed/              # splits e artefatos gerados pelo pipeline tabular
│   ├── tabular/                     # pipeline tabular sklearn/XGBoost
│   ├── image/                       # pipeline de imagem TensorFlow/Keras
│   ├── genetic/                     # Fase 2 — Algoritmo Genético
│   ├── llm/                         # Fase 2 — explicações com LLM
│   ├── monitoring/                  # tracking dos experimentos
│   └── results/                     # saídas de algumas execuções diretas dos módulos
├── tests/
├── run_pipeline.py
├── requirements.txt
├── Makefile
└── Dockerfile
```

### Observação sobre caminhos

Como o projeto foi evoluindo entre as fases, ficaram algumas pastas com nomes parecidos. Na prática:

- `data/raw/` é o local indicado para colocar o CSV original do SIVEP;
- `src/data/processed/` guarda splits e artefatos de pré-processamento usados por módulos e notebooks;
- `results/figures/` concentra os gráficos consolidados usados na entrega;
- `src/results/` ainda aparece como saída de algumas execuções diretas dos módulos tabulares;
- `experiments/` guarda configs, logs, históricos e resumos dos experimentos do Algoritmo Genético.

Antes de apagar, mover ou sobrescrever arquivos de resultado, confira qual notebook ou script usa aquele caminho.

---

## 3. Requisitos

| Recurso | Especificação |
|---|---|
| Python | Python 3.11 ou 3.12. O repositório traz `.python-version` com 3.12 e o `Dockerfile` usa Python 3.11. |
| Dependências | Pacotes listados em `requirements.txt`. |
| Dataset tabular | Arquivo `INFLUD24-26-06-2025.csv`, obtido no OpenDataSUS e salvo em `data/raw/`. |
| Dataset de imagem | Baixado automaticamente pelo código com `kagglehub`, usado no notebook `05_image_validation.ipynb`. |
| Memória | Recomenda-se pelo menos 4 GB de RAM para execução do pipeline tabular completo. |
| LLM local | Ollama com um modelo compatível, como `llama3.1:8b`, para executar as explicações localmente. Sem Ollama, o projeto mantém o fluxo com fallback/mock. |

Recomenda-se evitar Python 3.13+ por compatibilidade de algumas dependências científicas, como `numpy` e `pandas`.

---

## 4. Instalação e execução local

Os comandos abaixo devem ser executados na raiz do repositório.

### 4.1. Criar e ativar o ambiente virtual

Crie o ambiente virtual:

```bash
python -m venv venv
```

Ative o ambiente conforme o sistema operacional.

**Linux/Mac**

```bash
source venv/bin/activate
```

**Windows PowerShell**

```powershell
.\venv\Scripts\Activate.ps1
```

### 4.2. Instalar as dependências

Com o ambiente virtual ativo, instale os pacotes do projeto:

```bash
pip install -r requirements.txt
```

### 4.3. Adicionar o dataset tabular

O CSV bruto do SIVEP não é versionado no repositório. Baixe o arquivo `INFLUD24-26-06-2025.csv` no OpenDataSUS e copie para `data/raw/`.

**Linux/Mac**

```bash
mkdir -p data/raw
cp /caminho/para/INFLUD24-26-06-2025.csv data/raw/
```

**Windows PowerShell**

```powershell
New-Item -ItemType Directory -Force data\raw
Copy-Item "C:\caminho\para\INFLUD24-26-06-2025.csv" "data\raw\"
```

O caminho esperado pelo pipeline é:

```text
data/raw/INFLUD24-26-06-2025.csv
```

### 4.4. Executar o pipeline completo

Com o CSV no caminho padrão, execute:

```bash
python run_pipeline.py
```

Também é possível informar o caminho do arquivo explicitamente:

```bash
python run_pipeline.py --data data/raw/INFLUD24-26-06-2025.csv
```

O pipeline tabular executa as etapas de carregamento, pré-processamento, treino, avaliação e geração dos artefatos usados pelos notebooks e relatórios.

### 4.5. Opções do pipeline

```bash
python run_pipeline.py --no-grid                 # Executa sem GridSearchCV
python run_pipeline.py --no-shap                 # Executa sem geração de gráficos SHAP
python run_pipeline.py --nrows 10000             # Usa um subconjunto dos dados
python run_pipeline.py --no-grid --no-shap       # Execução reduzida para gerar artefatos base
```

Essas opções são úteis para validações rápidas, principalmente antes de abrir os notebooks da Fase 2.

### 4.6. Rodar os testes

```bash
pytest tests/ -v
```

Também é possível rodar com o módulo do Python:

```bash
python -m pytest tests -q
```

### 4.7. Execução da Fase 2

A Fase 2 parte dos artefatos gerados pelo pipeline tabular da Fase 1. Antes de rodar os notebooks novos, execute o pipeline pelo menos uma vez.

```bash
# Gera dados processados, modelos e figuras base
python run_pipeline.py --no-grid --no-shap
```

Depois disso, abra os notebooks da Fase 2:

```bash
# Demonstração da otimização por Algoritmo Genético
jupyter notebook notebooks/06_genetic_optimization.ipynb

# Demonstração das explicações com LLM
jupyter notebook notebooks/07_llm_interpretation.ipynb
```

A integração com LLM usa Ollama quando disponível. Sem uma instância local ativa, o projeto usa fallback/mock para manter os testes, os notebooks e o fluxo de demonstração funcionando.

### 4.8. Notebooks interativos

Para abrir a pasta de notebooks no Jupyter:

```bash
jupyter notebook notebooks/
```

Notebooks principais:

| Notebook | Conteúdo |
|---|---|
| `01_exploratory_data_analysis.ipynb` | Análise exploratória dos dados tabulares. |
| `02_preprocessing.ipynb` | Pré-processamento e preparação dos dados. |
| `03_modeling.ipynb` | Treinamento dos modelos tabulares. |
| `04_evaluation_interpretability.ipynb` | Avaliação, métricas e interpretabilidade. |
| `05_image_validation.ipynb` | Validação complementar com radiografias torácicas. |
| `06_genetic_optimization.ipynb` | Experimentos com Algoritmo Genético. |
| `07_llm_interpretation.ipynb` | Explicações em linguagem natural com LLM. |

### 4.9. Aplicação Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

## 5. Modelos da Fase 1

### Dados tabulares

| Modelo | Tipo | Papel no projeto |
|---|---|---|
| Logistic Regression | Linear | Baseline interpretável |
| Decision Tree | Não linear | Regras explícitas |
| Random Forest | Ensemble / Bagging | Modelo robusto para comparação |
| XGBoost | Ensemble / Boosting | Modelo principal usado na Fase 2 |

Os modelos tabulares usam métricas como accuracy, precision, recall, F1-score, ROC-AUC, matriz de confusão e interpretabilidade com SHAP/feature importance.

### Dados de imagem

A validação com radiografias usa uma CNN construída com Keras/TensorFlow. O modelo usa camadas convolucionais, `GlobalAveragePooling2D` e uma saída `softmax` para as quatro classes do dataset.

Também há data augmentation leve em `src/image/image_preprocessing.py`, com rotação, zoom e deslocamentos pequenos. Essa parte continua como validação complementar da Fase 1; a otimização genética da Fase 2 foca no XGBoost tabular.

---

## 6. Métricas de avaliação

| Métrica | Tabular | Imagem |
|---|---|---|
| Accuracy | Sim | Sim |
| Precision | Sim | Sim |
| Recall | Sim | Sim |
| F1-score | Sim | Sim |
| ROC-AUC | Sim | Não aplicado |
| Matriz de confusão | Sim | Sim |
| SHAP / importância de features | Sim | Não aplicado |

---

## 7. Fase 2 — Algoritmo Genético

A implementação do Algoritmo Genético fica em `src/genetic/`:

- `chromosome.py`: genes e conversão para hiperparâmetros do XGBoost;
- `operators.py`: seleção por torneio, crossover uniforme, mutação e elitismo;
- `fitness.py`: cálculo da função fitness;
- `ga_optimizer.py`: execução do processo evolutivo.

O cromossomo representa hiperparâmetros do XGBoost, incluindo:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`
- `scale_pos_weight`

A função fitness usa principalmente **F1-score** e também considera **recall**, porque o projeto trabalha com um problema clínico em que falsos negativos para óbito são especialmente sensíveis.

Os experimentos ficam em `experiments/`, com histórico por geração, logs e arquivos de resumo. Os gráficos consolidados ficam em `results/figures/`.

### Experimentos executados

| Experimento | População | Gerações | Crossover | Mutação | F1 da busca CV |
|---|---:|---:|---:|---:|---:|
| Exp A | 10 | 8 | 0.8 | 0.10 | 0.5057 |
| Exp B | 20 | 10 | 0.9 | 0.05 | 0.5091 |
| Exp C | 30 | 12 | 0.8 | 0.20 | 0.5126 |

O Exp C teve o maior F1 durante a busca por validação cruzada. No teste final, o melhor resultado por F1 ficou com o **XGBoost-AG Exp A**.

### Comparativo com o XGBoost original

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGBoost original da Fase 1 | 0.8523 | 0.3301 | 0.6928 | 0.4471 | 0.8797 |
| XGBoost-AG Exp A | 0.9113 | 0.4869 | 0.5429 | 0.5134 | 0.9055 |
| XGBoost-AG Exp B | 0.8996 | 0.4407 | 0.6124 | 0.5126 | 0.9055 |
| XGBoost-AG Exp C | 0.9140 | 0.5012 | 0.5243 | 0.5125 | 0.9059 |

O melhor resultado por F1 foi o **XGBoost-AG Exp A**, com **F1 = 0.5134**. Em relação ao XGBoost original, houve ganho em **F1** e **ROC-AUC**, com trade-off em recall.

---

## 8. Fase 2 — LLM

A camada de LLM gera explicações em português para apoiar a interpretação das predições do modelo.

Arquivos principais:

- `src/llm/client.py`: cliente Ollama e fallback/mock;
- `src/llm/prompts.py`: prompts em português;
- `src/llm/explainer.py`: montagem do contexto e geração da explicação;
- `notebooks/07_llm_interpretation.ipynb`: exemplos e avaliação qualitativa.

A LLM pode receber:

- predição do modelo;
- probabilidade estimada;
- principais fatores/features do caso;
- contexto das variáveis relevantes;
- SHAP/top features, quando disponíveis.

A explicação deixa claro que o modelo é um apoio à decisão e não substitui avaliação médica.

### Ollama

Quando o Ollama está disponível, o projeto pode usar `llama3.1:8b`:

```bash
ollama pull llama3.1:8b
```

**Linux/Mac**

```bash
export LLM_BACKEND=ollama
```

**Windows PowerShell**

```powershell
$env:LLM_BACKEND="ollama"
```

Se o Ollama não estiver rodando, o projeto usa fallback/mock para manter notebooks e testes funcionando.

A avaliação qualitativa das explicações considera fidelidade ao resultado do modelo, clareza, utilidade/acionabilidade e segurança.

---

## 9. Testes

```bash
python -m pytest tests -q
```

Na validação do projeto, os **37 testes** passaram.

Também há atalhos no `Makefile`:

```bash
make install
make test
make run
make run-fast
make run-quick
```

Os atalhos de execução usam o caminho padrão do script. Caso o CSV esteja em `data/raw/`, prefira executar com `--data`, como mostrado na seção de pipeline tabular.

---

## 10. Docker

```bash
docker build -t tech-challenge-srag .
docker run -p 8888:8888 -v $(pwd)/data:/app/data tech-challenge-srag
```

Depois de subir o container, acesse:

```text
http://localhost:8888
```

---

## 11. Documentação e entrega

Documentos principais:

- `docs/relatorio_tecnico.md`: relatório técnico da Fase 1;
- `docs/plano_fase2.md`: planejamento da Fase 2;
- `docs/relatorio_fase2.md`: relatório da Fase 2;
- `docs/roteiro_video_fase2.md`: roteiro do vídeo;
- `docs/dicionario-de-dados-2019-a-2025.pdf`: dicionário de dados usado como referência.

A Fase 2 cobre o Projeto 1 do Tech Challenge: otimização de hiperparâmetros com Algoritmo Genético, três experimentos, comparação com o modelo original, monitoramento/logging, integração com LLM, prompt engineering, avaliação das interpretações, documentação e testes.
