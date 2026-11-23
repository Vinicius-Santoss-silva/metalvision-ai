"""Treina classificador experimental NEU-CLS. Executar no Google Colab."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from src.vision import LABELS, extract_features

NAME_MAP={
    "cr":"Cr","crazing":"Cr","in":"In","inclusion":"In",
    "pa":"Pa","patches":"Pa","ps":"PS","pitted_surface":"PS","pitted surface":"PS",
    "rs":"RS","rolled-in_scale":"RS","rolled-in scale":"RS",
    "sc":"Sc","scratches":"Sc"
}

def get_label(path):
    # NEU naming, e.g. Cr_1.bmp / In_1.bmp, or class directories.
    parts=[path.parent.name.lower().strip(), path.stem.split("_")[0].lower().strip()]
    for name in parts:
        if name in NAME_MAP:
            return NAME_MAP[name]
    return None

def load_dataset(root):
    root=Path(root)
    images=[p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in (".bmp",".png",".jpg",".jpeg")]
    X,y=[],[]
    for p in sorted(images):
        label=get_label(p)
        if label is None:
            continue
        try:
            with Image.open(p) as im:
                X.append(extract_features(im.convert("RGB")))
            y.append(label)
        except (OSError, ValueError) as e:
            print("Ignorando arquivo não reconhecido:",p,e)
    if not y:
        raise ValueError("Nenhuma imagem NEU encontrada. Verifique a pasta ou os nomes Cr_1.bmp, In_1.bmp etc.")
    counts={k:y.count(k) for k in LABELS}
    print("Imagens por classe:",counts)
    if min(counts.values())<20:
        raise ValueError("Base incompleta. É necessário pelo menos 20 imagens por classe.")
    # Evitar usar inadvertidamente cópias de train e test da mesma imagem.
    return np.asarray(X),np.asarray(y)

def train(root,output="models"):
    X,y=load_dataset(root)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    clf=make_pipeline(StandardScaler(),SVC(kernel="rbf",C=2.0,gamma="scale",probability=True,random_state=42))
    clf.fit(X_train,y_train)
    pred=clf.predict(X_test)
    metrics={
        "accuracy":float(accuracy_score(y_test,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y_test,pred)),
        "train_images":int(len(y_train)),
        "test_images":int(len(y_test)),
        "class_counts":{k:int(np.sum(y==k)) for k in LABELS},
        "classes":list(clf.classes_),
        "confusion_matrix":confusion_matrix(y_test,pred,labels=list(LABELS)).tolist(),
        "classification_report":classification_report(y_test,pred,labels=list(LABELS),output_dict=True,zero_division=0),
        "dataset":"NEU-CLS",
        "split":"image-level 80/20 stratified, random_state=42",
        "caveat":"Metrics only on held-out NEU images; not generalizable to real phone photographs without testing."
    }
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    joblib.dump({"classifier":clf,"labels":list(LABELS),"feature_version":"hog_lbp_v1"},out/"metalvision_svm.joblib",compress=3)
    (out/"metricas.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Acurácia teste: {metrics['accuracy']:.3f} | Balanceada: {metrics['balanced_accuracy']:.3f}")
    print("Resultados salvos em:",out.resolve())
    return metrics

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",required=True,help="Diretório com arquivos NEU-CLS ou pastas por classe.")
    parser.add_argument("--output",default="models")
    args=parser.parse_args()
    train(args.data,args.output)
