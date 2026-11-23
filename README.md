# MetalVision AI — inspeção exploratória de superfícies de aço

Projeto de **Ciência de Dados + Engenharia Mecânica** com upload de fotografias, análise de textura e, após treinamento, classificação experimental de defeitos superficiais.

## Publicação (sem instalar Python)
1. Acesse https://share.streamlit.io/ com a conta GitHub `Vinicius-Santoss-silva`.
2. Crie um app público do repositório `Vinicius-Santoss-silva/metalvision-ai`, branch `main`, arquivo principal `app.py`.
3. O Streamlit instalará as bibliotecas de `requirements.txt`.

## Treinar a IA
Abra `notebooks/treinar_colab.ipynb` no Google Colab. Esse notebook baixa a base NEU via KaggleHub, treina um SVM em características HOG e avalia em um conjunto separado. Depois, faça upload dos dois resultados `models/metalvision_svm.joblib` e `models/metricas.json` para o GitHub, na pasta `models/`.

**Sem o modelo treinado, o site funciona em modo de análise visual exploratória, sem simular previsão de IA.**

## Fonte de dados e limitações
A [NEU Surface Defect Database](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/) contém 1.800 imagens 200×200 em tons de cinza de **seis tipos de defeitos** em tiras de aço laminadas a quente. Classes: `Cr` (crazing), `In` (inclusion), `Pa` (patches), `PS` (pitted surface), `RS` (rolled-in scale) e `Sc` (scratches). **Não existe classe "peça saudável"** neste conjunto.

As imagens enviadas pelos usuários podem ter iluminação, material, escala e textura muito diferentes do conjunto de treinamento. As porcentagens, se exibidas, são estimativas internas do classificador e **não garantem acerto**, mesmo quando altas. Não é inspeção industrial validada, laudo ou ferramenta para decisões de segurança.

O mapa de contornos evidencia diferenças de textura; **não é detecção/localização de defeito pela IA**.

O aplicativo não cria uma base de dados com imagens enviadas; o serviço Streamlit processa a foto em memória. Evite enviar material confidencial.

## Estrutura
- `app.py`: interface Streamlit, upload e análise.
- `src/vision.py`: extrator de características, mapa de bordas, checagens de foto e predição.
- `training/train.py`: treino e avaliação reproduzíveis.
- `notebooks/treinar_colab.ipynb`: execução sem Python local.
- `requirements.txt`, `.streamlit/config.toml`: publicação.
