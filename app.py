"""MetalVision AI — Streamlit public demo."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
import streamlit as st
from src.vision import LABELS, open_image, image_quality, edge_map, load_model, predict

st.set_page_config(page_title="MetalVision AI | Inspeção Visual",page_icon="🔬",
                   layout="wide",initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{max-width:1250px;padding-top:2rem}
.hero{border-radius:20px;border:1px solid #235366;padding:26px 28px;background:linear-gradient(120deg,#14384b,#0d1d33)}
.hero h1{color:#efffff;font-size:2.6rem;margin:8px 0}
.hero p{color:#b5cedb;font-size:1.05rem}
.small-label{font-weight:700;letter-spacing:.15em;color:#37d9bf;font-size:.72rem}
[data-testid="stMetric"]{border:1px solid #284254;border-radius:14px;padding:12px;background:#102337}
</style>""",unsafe_allow_html=True)
st.markdown("""<div class="hero"><span class="small-label">METALVISION • COMPUTER VISION LAB</span>
<h1>O que uma superfície metálica pode revelar?</h1>
<p>Envie uma foto de uma peça, explore sua textura e, quando o modelo estiver disponível, obtenha uma classificação experimental de defeitos em aço.</p></div>""",unsafe_allow_html=True)
st.caption("Pesquisa e demonstração educacional · Não substitui inspeção técnica, ensaios não destrutivos ou controle de qualidade certificado.")
model_path=Path("models/metalvision_svm.joblib")
metrics_path=Path("models/metricas.json")

@st.cache_resource(show_spinner=False)
def get_model(path,mtime):
    return load_model(path)

with st.sidebar:
    st.title("🔬 MetalVision AI")
    st.write("Inspeção exploratória por visão computacional")
    if model_path.exists():
        st.success("Modelo de IA disponível")
    else:
        st.info("Modelo ainda não treinado — análise visual disponível")
    st.divider()
    st.markdown("**Para melhores resultados**")
    st.write("Fotografe de perto a superfície de aço, com iluminação uniforme, sem reflexos fortes e imagem nítida.")
    st.markdown("**Tipos de defeitos estudados**")
    for code,(title,_) in LABELS.items():
        st.caption(f"{code} · {title}")
    st.markdown("[📘 Código e metodologia](https://github.com/Vinicius-Santoss-silva/metalvision-ai)")
    st.markdown("[Dados de pesquisa NEU](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/)")

tab1,tab2,tab3=st.tabs(["📸 Analisar peça","🧠 Sobre a IA","📖 Metodologia"])
with tab1:
    uploaded=st.file_uploader("Envie uma foto de superfície metálica (JPG, PNG ou WEBP)",type=["jpg","jpeg","png","webp"])
    if uploaded is None:
        st.info("Selecione uma foto acima. O app funciona no celular e no computador.")
        col1,col2,col3=st.columns(3)
        col1.metric("Classes estudadas","6")
        col2.metric("Imagens do conjunto NEU","1.800")
        col3.metric("Upload","Até 10 MB")
        st.markdown("**Como funciona:** foto → checagens de qualidade → textura/bordas → classificador experimental (após treinamento).")
    else:
        try:
            if uploaded.size>10*1024*1024:
                st.error("Limite de 10 MB por arquivo.")
                st.stop()
            img=open_image(uploaded)
            q=image_quality(img)
            left,right=st.columns(2)
            with left:
                st.markdown("#### Sua imagem")
                st.image(img,use_container_width=True)
            with right:
                st.markdown("#### Mapa de contornos")
                st.image(Image.fromarray(edge_map(img)),use_container_width=True)
                st.caption("Contornos detectados matematicamente; **não** são regiões de defeito localizadas pela IA.")
            a,b,c=st.columns(3)
            a.metric("Luminosidade",f"{q['brightness']*100:.0f}%")
            b.metric("Contraste",f"{q['contrast']:.3f}")
            c.metric("Nitidez (indicador)",f"{q['sharpness']:.5f}")
            if q["warnings"]:
                st.warning("Qualidade de imagem possivelmente inadequada: "+" ".join(q["warnings"]))
            else:
                st.success("Nenhum problema evidente de qualidade nas verificações básicas.")
            st.divider()
            st.markdown("### Análise por inteligência artificial")
            if not model_path.exists():
                st.info("O modelo ainda não foi treinado/importado. Por enquanto, o app **não emite diagnóstico**, apenas analisa a imagem.")
                st.caption("Próxima etapa: executar o notebook Colab e adicionar o modelo treinado à pasta models.")
            elif q["warnings"]:
                st.warning("**Análise inconclusiva:** melhore a foto antes da classificação. Essas verificações são heurísticas.")
            else:
                package=get_model(str(model_path),model_path.stat().st_mtime)
                scores=predict(img,package)
                code,score=scores[0]
                st.metric("Classe com maior pontuação",LABELS[code][0])
                st.metric("Pontuação relativa do classificador",f"{score*100:.1f}%")
                if score<.60:
                    st.warning("**Resultado inconclusivo**: as pontuações não indicam classe claramente dominante.")
                else:
                    st.info("Possível característica: "+LABELS[code][1])
                table=pd.DataFrame([{"Classe":LABELS[cls][0],"Pontuação":float(p)} for cls,p in scores])
                fig=px.bar(table,x="Pontuação",y="Classe",orientation="h",text_auto=".1%",color_discrete_sequence=["#28ccb3"])
                fig.update_layout(template="plotly_dark",height=320,margin=dict(l=0,r=0,t=10,b=0),yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig,use_container_width=True)
                st.caption("Essas pontuações não são uma garantia de acerto. O modelo foi treinado para distinguir seis tipos de defeitos em imagens de laboratório; não distingue peças saudáveis.")
                if metrics_path.exists():
                    st.caption("Métricas de teste do conjunto NEU disponíveis na aba Sobre a IA.")
        except Exception as exc:
            st.error("Não foi possível analisar esta imagem: "+str(exc))
with tab2:
    st.subheader("Como o classificador aprende")
    st.markdown("O método inicial utiliza **HOG (gradientes)**, **LBP (textura)** e um **SVM com probabilidades estimadas**. O treino é feito com imagens NEU e a avaliação usa imagens separadas.")
    if metrics_path.exists():
        try:
            m=json.loads(metrics_path.read_text(encoding="utf-8"))
            x,y=st.columns(2)
            x.metric("Acurácia no teste NEU",f"{100*m['accuracy']:.1f}%")
            y.metric("Acurácia balanceada",f"{100*m['balanced_accuracy']:.1f}%")
            st.json(m)
        except (OSError,KeyError,ValueError):
            st.warning("Métricas de treinamento indisponíveis ou inválidas.")
    else:
        st.info("Ainda não há métricas reais: o modelo precisa ser treinado no Google Colab.")
    st.warning("Desempenho no conjunto NEU não representa precisão em fotografias arbitrárias de peças reais.")
with tab3:
    st.markdown("""
### Origem e limitações dos dados
A NEU Surface Defect Database tem **1.800 imagens de superfícies de aço laminado a quente com defeitos** (seis classes com 300 imagens cada).

**O conjunto de treinamento não possui exemplos rotulados como sem defeito.** Portanto, mesmo que uma peça esteja intacta, o classificador pode escolher uma das seis classes — resultado que não deve ser interpretado como prova de defeito.

**O que o aplicativo faz hoje**
- Lê uma imagem e calcula medidas básicas de iluminação, contraste e nitidez.
- Calcula contornos para ilustrar características da textura.
- Só apresenta uma classificação quando existe um modelo treinado na pasta `models/`.

**O que o aplicativo não faz:** emitir laudo, localizar automaticamente fissuras reais, substituir inspeção humana, validar segurança da peça ou reconhecer todos os materiais.

Fotografias enviadas não são salvas intencionalmente em um banco. Ainda assim, são processadas pelo serviço de hospedagem: evite conteúdo sigiloso.

[Mais informações sobre a NEU](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/)
""")
st.caption("MetalVision AI · Ciência de Dados × Engenharia Mecânica · Projeto experimental")
