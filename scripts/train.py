"""
train.py — ÉTAPE 1 : entraînement, évaluation et sauvegarde du classifieur texte.

Lancement (depuis la racine du projet) :
    python scripts/train.py

Produit :
    models/vectorizer.pkl   -> le TfidfVectorizer entraîné (vocabulaire + IDF)
    models/model.pkl        -> la LogisticRegression entraînée
    models/metrics.json     -> les scores (pour le README / la soutenance)
    docs/captures/matrice_confusion.png
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from app.preprocessing import nettoyer_texte  # noqa: E402

DATA = RACINE / "data"
MODELS = RACINE / "models"
CAPTURES = RACINE / "docs" / "captures"
CLASSES = ["negatif", "neutre", "positif"]
SEED = 42


def charger_donnees():
    """Jeu d'ENTRAÎNEMENT = textes.csv (24 avis).
    Jeu de TEST externe = les textes de référence des images (9) et des audios (9) :
    ces 18 phrases ne sont jamais vues à l'entraînement -> test honnête."""
    textes = pd.read_csv(DATA / "textes.csv")
    images = pd.read_csv(DATA / "images.csv")
    audios = pd.read_csv(DATA / "audios.csv")
    test = pd.concat([
        pd.DataFrame({"texte": images["texte_attendu_ocr"], "sentiment_ref": images["sentiment_ref"], "source": "image"}),
        pd.DataFrame({"texte": audios["transcription_reference"], "sentiment_ref": audios["sentiment_ref"], "source": "audio"}),
    ], ignore_index=True)
    return textes, test


def creer_vectorizer():
    # ngram_range=(1,2) : mots seuls ET paires de mots ("ne fonctionne", "fonctionne pas")
    # sublinear_tf=True  : 1 + log(tf) -> un mot répété 3 fois ne pèse pas 3x plus
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def creer_modele():
    # C=10 : régularisation faible (le jeu est minuscule, on laisse le modèle "apprendre")
    # LogisticRegression donne des PROBABILITÉS -> champ "confiance" de l'API
    return LogisticRegression(C=10, max_iter=1000, random_state=SEED)


def comparer_modeles(X, y):
    """Compare Régression Logistique et SVM linéaire par validation croisée
    stratifiée répétée (4 plis x 10 répétitions = 40 évaluations)."""
    cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=10, random_state=SEED)
    candidats = {
        "LogisticRegression": make_pipeline(creer_vectorizer(), creer_modele()),
        "LinearSVC": make_pipeline(creer_vectorizer(), LinearSVC(C=1, random_state=SEED)),
    }
    resultats = {}
    for nom, pipe in candidats.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
        resultats[nom] = {"f1_macro_moyen": round(float(scores.mean()), 3),
                          "ecart_type": round(float(scores.std()), 3)}
        print(f"  {nom:<20} F1-macro CV = {scores.mean():.2f} ± {scores.std():.2f}")
    return resultats


def sauver_matrice(cm, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title("Matrice de confusion (test externe, 18 avis)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout()
    fig.savefig(chemin, dpi=130)
    plt.close(fig)


def main():
    MODELS.mkdir(exist_ok=True)
    CAPTURES.mkdir(parents=True, exist_ok=True)

    # 1) Chargement ---------------------------------------------------------
    train, test = charger_donnees()
    print(f"Entraînement : {len(train)} avis | répartition : {train.sentiment_ref.value_counts().to_dict()}")
    print(f"Test externe : {len(test)} avis (textes de référence images + audios)")

    # 2) Nettoyage ----------------------------------------------------------
    X_train = train["texte"].map(nettoyer_texte)
    y_train = train["sentiment_ref"]
    X_test = test["texte"].map(nettoyer_texte)
    y_test = test["sentiment_ref"]
    print("\nExemple de nettoyage :")
    print("  AVANT :", train["texte"][1])
    print("  APRÈS :", X_train[1])

    # 3) Comparaison LogReg vs SVM (validation croisée sur le train) --------
    print("\nValidation croisée (4 plis x 10) sur les 24 avis :")
    cv_resultats = comparer_modeles(X_train, y_train)

    # 4) Entraînement final sur les 24 avis --------------------------------
    vectorizer = creer_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)      # fit = apprend vocabulaire + IDF
    modele = creer_modele().fit(X_train_vec, y_train)
    print(f"\nVocabulaire appris : {len(vectorizer.vocabulary_)} termes (mots + bigrammes)")

    # 5) Évaluation sur le test externe ------------------------------------
    X_test_vec = vectorizer.transform(X_test)            # transform SEULEMENT (pas de fit !)
    y_pred = modele.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
    rapport = classification_report(y_test, y_pred, labels=CLASSES, zero_division=0)
    print(f"\n=== Test externe : accuracy = {acc:.2f} | F1-macro = {f1:.2f} ===")
    print("Matrice de confusion (lignes = réel, colonnes = prédit)", CLASSES)
    print(cm)
    print(rapport)

    erreurs = test.assign(predit=y_pred)[y_pred != y_test]
    if len(erreurs):
        print("Erreurs :")
        for _, r in erreurs.iterrows():
            print(f"  [{r.source}] réel={r.sentiment_ref:<8} prédit={r.predit:<8} | {r.texte}")

    # 6) Mots les plus influents par classe (interprétabilité) -------------
    termes = np.array(vectorizer.get_feature_names_out())
    top = {}
    for idx, classe in enumerate(modele.classes_):
        top[classe] = termes[np.argsort(modele.coef_[idx])[::-1][:8]].tolist()
        print(f"Top termes '{classe}': {', '.join(top[classe])}")

    # 7) Sauvegarde des artefacts ------------------------------------------
    joblib.dump(vectorizer, MODELS / "vectorizer.pkl")
    joblib.dump(modele, MODELS / "model.pkl")
    sauver_matrice(cm, CAPTURES / "matrice_confusion.png")
    metrics = {
        "train_size": len(train), "test_size": len(test),
        "validation_croisee": cv_resultats,
        "test_externe": {"accuracy": round(acc, 3), "f1_macro": round(f1, 3),
                         "matrice_confusion": cm.tolist(), "labels": CLASSES},
        "top_termes": top,
    }
    (MODELS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nArtefacts sauvegardés dans models/ : vectorizer.pkl, model.pkl, metrics.json")


if __name__ == "__main__":
    main()
