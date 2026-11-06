# Gerenciamento de Configuração em Sistemas de IA e MLOps

Em software tradicional, `código + dependências` determina o comportamento do sistema. Em
sistemas de IA isso não basta: o mesmo código produz modelos diferentes com outros dados ou
outros hiperparâmetros, e um agente LLM muda de comportamento quando muda o prompt ou a
versão do modelo. **Todos esses artefatos são itens de configuração** e precisam de
identificação, controle de versão, rastreabilidade e auditoria.

| Item de configuração | Problema | Ferramenta neste repositório | Onde fica versionado |
|---|---|---|---|
| Código | (o clássico) | Git | Git |
| Dataset (GB/TB) | Grande demais para o Git | **DVC** (`dvc add`) | Ponteiro `.dvc` no Git; conteúdo no cache/remote |
| Hiperparâmetros | Mudam a cada experimento | `params.yaml` + `dvc exp` | Git (+ refs de experimento) |
| Pesos do modelo | Binários grandes, derivados | Saída do pipeline DVC + `artifacts` | Hash no `dvc.lock`; binário no remote |
| Linhagem / métricas | "Qual dado e commit geraram este modelo?" | `model_meta.json`, `metrics.json` | Git |
| Prompts | Texto que se comporta como código | Registro com semver + lock SHA-256 | Git |
| Spec de agente | Modelo + prompt + ferramentas + guardrails | `AgentSpec` YAML + fingerprint | Git |

## Estrutura

```
├── params.yaml               # hiperparâmetros (Parte 2)
├── dvc.yaml                  # pipeline declarativo prepare → train → evaluate
├── dvc.lock                  # hashes exatos de cada entrada/saída (gerado)
├── data/raw/dataset.csv.dvc  # ponteiro para o dataset versionado (Parte 1)
├── src/
│   ├── get_data.py           # gera o dataset (v1, ou v2 com --extra-rows)
│   ├── prepare.py            # estágio prepare
│   ├── train.py              # estágio train  → models/model.pkl + model_meta.json
│   ├── evaluate.py           # estágio evaluate → metrics/metrics.json + quality gate
│   └── lineage.py            # commit Git, hash dos dados, hash dos parâmetros
├── prompts/                  # registro de prompts versionados (Parte 4)
│   ├── sentiment_classifier/1.0.0.yaml, 1.1.0.yaml
│   ├── support_agent_system/1.0.0.yaml
│   └── prompts.lock.json     # SHA-256 de cada versão publicada
├── agents/support_agent.yaml # especificação do agente (Parte 5)
├── evals/sentiment_golden.jsonl
├── llmops/                   # registry, validação de agentes, avaliação de prompts
├── tests/
└── .github/workflows/mlops-ci.yml   # CI (Parte 6)
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/get_data.py           # gera data/raw/dataset.csv (versão 1, determinística)
dvc repro                        # reproduz o pipeline completo
pytest
```

---

## Parte 1 — Versionamento de dados com DVC

O Git não foi feito para arquivos de gigabytes. O DVC guarda o arquivo em um cache
endereçado por conteúdo (`.dvc/cache`, sincronizado com um *remote* como S3, GCS, Azure,
SSH ou Google Drive) e coloca no Git apenas um pequeno ponteiro com o hash:

```yaml
# data/raw/dataset.csv.dvc  (vai para o Git)
outs:
- md5: 5daac8b43ff277b3f98cbcffffb004ed
  size: 162157
  hash: md5
  path: dataset.csv
```

Como este repositório foi criado (para reproduzir do zero):

```bash
git init && dvc init
dvc remote add -d storage s3://meu-bucket/dvc     # ou: /caminho/local, gdrive://..., ssh://...
python src/get_data.py
dvc add data/raw/dataset.csv                      # cria o .dvc e adiciona o CSV ao .gitignore
git add data/raw/dataset.csv.dvc data/raw/.gitignore
git commit -m "dados v1" && git tag data-v1
dvc push                                          # envia o conteúdo ao remote
```

Chega uma **nova versão dos dados**:

```bash
python src/get_data.py --extra-rows 400           # 569 → 969 linhas
dvc add data/raw/dataset.csv
git commit -am "dados v2" && git tag data-v2
dvc push
```

**Viagem no tempo**: o Git versiona o ponteiro e o DVC restaura o conteúdo correspondente:

```bash
git checkout data-v1 -- data/raw/dataset.csv.dvc
dvc checkout data/raw/dataset.csv                 # dataset volta a ter 569 linhas
```

Um colega (ou a CI) obtém exatamente os mesmos dados com `git clone` + `dvc pull`.

## Parte 2 — Pipeline reproduzível e hiperparâmetros

O [`dvc.yaml`](dvc.yaml) declara, para cada estágio, o comando, as dependências (código e
dados), os **parâmetros** de `params.yaml` que ele lê e as saídas. O `dvc repro` só
reexecuta os estágios afetados por uma mudança. Por exemplo, alterar `train.*` não
reexecuta o `prepare`. O `dvc.lock` registra o hash de tudo o que foi usado.

```bash
dvc dag                      # visualiza o grafo do pipeline
dvc repro                    # executa o que estiver desatualizado
dvc status                   # mostra o que mudou desde o último repro
```

**Experimentos com hiperparâmetros** sem criar commits ou branches à mão:

```bash
dvc exp run -S train.random_forest.n_estimators=300 -S train.random_forest.max_depth=10
dvc exp run -S train.model=logistic_regression
dvc exp show                 # tabela: parâmetros × métricas de cada experimento
dvc exp apply <nome-do-exp>  # traz o melhor para o workspace, depois é só commitar
```

Exemplo de saída (resumida):

| Experimento | accuracy | f1 | roc_auc | train.model | n_estimators | max_depth |
|---|---|---|---|---|---|---|
| main | 0.9474 | 0.9583 | 0.9947 | random_forest | 100 | 6 |
| ├── dowie-tale | 0.9474 | 0.9583 | 0.9937 | random_forest | 300 | 10 |
| └── jammy-dunt | **0.9825** | **0.9861** | **0.9954** | logistic_regression | – | – |

Comparando versões já commitadas:

```bash
dvc params diff data-v1
dvc metrics diff data-v1     # ex.: accuracy 0.9474 → 0.9794 após os dados v2
```

O estágio `evaluate` também funciona como **quality gate**: falha se a acurácia ficar
abaixo de `evaluate.min_accuracy`, o que impede a promoção de um modelo pior.

## Parte 3 — Pesos de modelos e linhagem

- `models/model.pkl` é uma saída do pipeline: o binário vai para o cache/remote do DVC e o
  hash fica no `dvc.lock`. Cada commit aponta para pesos específicos, e `dvc checkout`
  restaura os pesos de qualquer versão.
- `models/model_meta.json` (no Git) responde à pergunta de auditoria *"de onde veio este
  modelo?"*:

```json
{
  "model_type": "random_forest",
  "hyperparameters": {"n_estimators": 100, "max_depth": 6},
  "params_hash": "f0428116ffb64083",
  "git_commit": "67d0e5f9062a358f8f265a43897ecfa4191255f2",
  "training_data": {"path": "data/processed/train.csv", "md5": "56055803...", "rows": 455},
  "environment": {"python": "3.12.x", "scikit-learn": "1.x"}
}
```

- A seção `artifacts` do `dvc.yaml` registra o modelo no **DVC Model Registry** (baseado em
  Git/GTO): versões e estágios (`dev`, `staging`, `prod`) viram tags Git, por exemplo
  `gto register breast-cancer-classifier --version v1.0.0` e
  `gto assign breast-cancer-classifier --version v1.0.0 --stage prod`.
- Alternativa muito usada: **MLflow Model Registry**, com tracking de runs e aliases
  (`champion`/`challenger`) em um servidor central. O princípio é o mesmo: pesos imutáveis
  identificados por versão e ligados ao run (código, dados, parâmetros) que os produziu.

## Parte 4 — Prompts como itens de configuração

Em sistemas com LLM, o prompt *é* lógica de negócio. Uma vírgula pode mudar o
comportamento. Por isso aplicamos a ele as mesmas regras de qualquer item de configuração:

1. **Identificação**: `nome@MAJOR.MINOR.PATCH` (ex.: `sentiment_classifier@1.1.0`).
2. **Imutabilidade**: versão publicada não muda. Mudou o texto, cria-se uma nova versão.
3. **Rastreabilidade**: [`prompts.lock.json`](prompts/prompts.lock.json) guarda o SHA-256
   de cada versão. A CI falha se alguém editar uma versão já publicada.
4. **Validação**: avaliação contra um *golden set* antes de promover a versão.

```bash
python -m llmops.registry list
python -m llmops.registry diff sentiment_classifier 1.0.0 1.1.0
python -m llmops.registry verify          # usado na CI
python -m llmops.registry lock            # após criar uma NOVA versão
```

Avaliação de regressão comparando duas versões:

```text
$ python -m llmops.evaluate_prompt sentiment_classifier --compare 1.0.0 1.1.0
sentiment_classifier@1.0.0: 67%  ->  sentiment_classifier@1.1.0: 100%
  [CORRIGIDO] 'É uma caneca branca de 300 ml.': positivo -> neutro (esperado neutro)
  [CORRIGIDO] 'Bom acabamento, mas a bateria é ruim.': positivo -> neutro (esperado neutro)
  ...
```

O código da aplicação referencia sempre uma versão fixa:

```python
from llmops.registry import PromptRegistry

prompt = PromptRegistry().get("sentiment_classifier@1.1.0")
texto = prompt.render(review="Chegou rápido, recomendo!")
# logar prompt.ref e prompt.sha256 junto com cada resposta → auditoria
```

> **Sobre o modelo usado nos testes:** [`FakeSentimentLLM`](llmops/llm_client.py) é um
> modelo falso e determinístico, para que testes e CI rodem offline e sem custo. Ele só
> responde com rótulos que o prompt oferece, então a mudança de prompt de fato altera o
> resultado. Para usar um LLM real, implemente `complete()` chamando a API do provedor.

## Parte 5 — Especificação de agentes de IA

O comportamento de um agente depende de vários itens ao mesmo tempo. O
[`agents/support_agent.yaml`](agents/support_agent.yaml) declara todos eles em um único
arquivo versionado:

- **modelo** com versão exata (nunca `latest`, porque o provedor pode trocar o modelo por baixo);
- **system prompt** por referência imutável (`support_agent_system@1.0.0`);
- **ferramentas** com `input_schema`;
- **guardrails** (máximo de turnos, aprovação humana, tópicos bloqueados).

```text
$ python -m llmops.agent_spec agents/support_agent.yaml
OK   agents/support_agent.yaml (support-agent@2.1.0) fingerprint=0cbb12021646
```

O validador rejeita modelo sem versão fixa, prompt sem versão ou inexistente e guardrail
que cita ferramenta inexistente. O **fingerprint** é o hash do spec mais o conteúdo do
prompt referenciado: qualquer alteração em qualquer peça gera um novo fingerprint. Ele
funciona como o "build id" do agente para logs e auditoria.

## Parte 6 — Integração contínua

[`.github/workflows/mlops-ci.yml`](.github/workflows/mlops-ci.yml) tem dois jobs:

| Job | O que garante |
|---|---|
| `llm-config` | prompts publicados não foram alterados; specs de agentes válidos; testes; prompt `latest` com pass rate ≥ 90% |
| `ml-pipeline` | `dvc pull` dos dados, `dvc repro` (com quality gate) e, em PRs, `dvc params diff` / `dvc metrics diff` contra a `main` no resumo do job |

Para usar um remote real na CI, configure as credenciais (ex.: `AWS_ACCESS_KEY_ID`) em
*GitHub Secrets*. Sem remote, o exemplo regenera o dataset v1 de forma determinística.

## Referências

- DVC: <https://dvc.org/doc> (data versioning, pipelines, experiments, model registry)
- MLflow: <https://mlflow.org/docs/latest/> (tracking e model registry)
- Google Cloud: *MLOps: Continuous delivery and automation pipelines in machine learning*
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems* (NeurIPS 2015)
- Semantic Versioning 2.0.0: <https://semver.org>
