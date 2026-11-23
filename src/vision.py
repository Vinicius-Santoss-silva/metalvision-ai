"""Pré-processamento, características visuais e inferência exploratória."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageOps
from skimage import color, filters, feature, transform, exposure
from sklearn.pipeline import Pipeline
import joblib

LABELS = {
    "Cr": ("Fissuração superficial (crazing)", "Rede de microfissuras na superfície."),
    "In": ("Inclusão (inclusion)", "Materiais ou partículas incorporadas à superfície."),
    "Pa": ("Manchas (patches)", "Áreas com textura ou aspecto superficial diferente."),
    "PS": ("Cavidades (pitted surface)", "Irregularidades ou pequenas depressões superficiais."),
    "RS": ("Carepa incrustada (rolled-in scale)", "Resíduos de óxido marcados na superfície laminada."),
    "Sc": ("Riscos (scratches)", "Marcas lineares na superfície."),
}

def open_image(upload):
    image=Image.open(upload)
    image=ImageOps.exif_transpose(image).convert("RGB")
    if image.width<80 or image.height<80:
        raise ValueError("Imagem muito pequena; envie uma foto com pelo menos 80×80 pixels.")
    if image.width*image.height>24_000_000:
        raise ValueError("Imagem grande demais; reduza a resolução.")
    return image

def gray_200(image):
    rgb=np.asarray(image.convert("RGB"),dtype=np.float32)/255.0
    resized=transform.resize(rgb,(200,200),preserve_range=True,anti_aliasing=True)
    return color.rgb2gray(resized).astype(np.float32)

def image_quality(image):
    img=gray_200(image)
    contrast=float(np.std(img))
    sharpness=float(np.var(filters.laplace(img)))
    brightness=float(np.mean(img))
    # Limiares apenas heurísticos: não são validação de qualidade industrial.
    warnings=[]
    if contrast<0.035: warnings.append("Pouco contraste ou superfície muito uniforme.")
    if sharpness<0.00018: warnings.append("Baixa definição ou pouco detalhe de textura.")
    if brightness<0.10: warnings.append("Imagem muito escura.")
    if brightness>0.92: warnings.append("Imagem excessivamente clara.")
    return {"contrast":contrast,"sharpness":sharpness,"brightness":brightness,"warnings":warnings}

def edge_map(image):
    arr=gray_200(image)
    edges=feature.canny(arr,sigma=1.5)
    return (edges.astype(np.uint8)*255)

def extract_features(image):
    x=gray_200(image)
    hog=feature.hog(x,orientations=9,pixels_per_cell=(16,16),
                    cells_per_block=(2,2),block_norm="L2-Hys")
    # Histograma de textura LBP, independente da posição na imagem.
    lbp=feature.local_binary_pattern((x*255).astype(np.uint8),8,1,method="uniform")
    hist=np.histogram(lbp,bins=np.arange(0,12),density=False)[0].astype(float)
    hist/=hist.sum()+1e-9
    return np.concatenate([hog,hist,np.array([x.mean(),x.std()],dtype=float)]).astype(np.float32)

def load_model(path):
    data=joblib.load(path)
    if not isinstance(data,dict) or not {"classifier","labels","feature_version"}.issubset(data):
        raise ValueError("Arquivo do modelo inválido.")
    if data["feature_version"]!="hog_lbp_v1":
        raise ValueError("Versão do extrator incompatível.")
    if set(data["labels"])!=set(LABELS):
        raise ValueError("Classes incompatíveis.")
    return data

def predict(image,package):
    classifier=package["classifier"]
    probs=classifier.predict_proba([extract_features(image)])[0]
    classes=list(classifier.classes_)
    out=sorted(zip(classes,map(float,probs)),key=lambda z:z[1],reverse=True)
    return out
