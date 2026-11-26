"""
classifier.py — Le "cerveau" commun de l'architecture en cascade.

Texte, OCR et transcription finissent TOUS ici : c'est le seul endroit
où l'on prédit un sentiment. Les artefacts (.pkl) sont chargés UNE seule
fois (au premier appel) puis gardés en mémoire.
"""
from functools import lru_cache
from pathlib import Path

import joblib

from app.preprocessing import nettoyer_texte

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ModeleIndisponible(RuntimeError):
    """Levée si model.pkl / vectorizer.pkl n'existent pas (train.py pas lancé)."""


@lru_cache(maxsize=1)
def charger_artefacts():
    """Charge vectorizer + modèle une seule fois (lru_cache = mémorisation)."""
    vec_path, model_path = MODELS_DIR / "vectorizer.pkl", MODELS_DIR / "model.pkl"
    if not vec_path.exists() or not model_path.exists():
        raise ModeleIndisponible(
            "Modèle introuvable : lancez d'abord 'python scripts/train.py'.")
    return joblib.load(vec_path), joblib.load(model_path)


def predire_sentiment(texte: str) -> dict:
    """
    texte brut -> nettoyage -> TF-IDF -> LogisticRegression -> probabilités.
    Retourne : {"sentiment": "positif", "confiance": 0.71,
                "probabilites": {"negatif": 0.12, "neutre": 0.17, "positif": 0.71},
                "texte_nettoye": "..."}
    """
    vectorizer, modele = charger_artefacts()
    texte_nettoye = nettoyer_texte(texte)
    X = vectorizer.transform([texte_nettoye])          # transform, JAMAIS fit ici
    probas = modele.predict_proba(X)[0]                # ex: [0.12, 0.17, 0.71]
    idx = int(probas.argmax())                         # classe la plus probable
    return {
        "sentiment": str(modele.classes_[idx]),
        "confiance": round(float(probas[idx]), 4),
        "probabilites": {str(c): round(float(p), 4) for c, p in zip(modele.classes_, probas)},
        "texte_nettoye": texte_nettoye,
    }
