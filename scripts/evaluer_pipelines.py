"""
evaluer_pipelines.py — Évalue la cascade complète sur les 9 images et 9 audios.

Pour chaque fichier on mesure :
  - la qualité de l'extraction (OCR ou transcription) : WER et CER
      WER = Word Error Rate      = erreurs de MOTS / nombre de mots de référence
      CER = Character Error Rate = erreurs de CARACTÈRES / nombre de caractères
      (0 % = parfait ; on compare au texte de référence fourni dans le CSV)
  - si le sentiment prédit est correct.

Lancement :  python scripts/evaluer_pipelines.py
Résultat  :  docs/resultats_pipelines.csv + résumé dans le terminal
"""
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
from app.asr import traiter_audio  # noqa: E402
from app.ocr import traiter_image  # noqa: E402
from app.preprocessing import normaliser_unicode  # noqa: E402

DATA = RACINE / "data"


def distance_levenshtein(a, b):
    """Nombre minimal d'insertions / suppressions / substitutions pour passer
    de la séquence a à la séquence b (programmation dynamique, ligne par ligne)."""
    precedente = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        courante = [i]
        for j, y in enumerate(b, 1):
            courante.append(min(precedente[j] + 1,              # suppression
                                courante[j - 1] + 1,            # insertion
                                precedente[j - 1] + (x != y)))  # substitution
        precedente = courante
    return precedente[-1]


def normaliser(t: str) -> str:
    """Pour comparer équitablement : minuscules, sans ponctuation."""
    t = normaliser_unicode(t).lower()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in t).split())


def wer(ref, hyp):
    r, h = normaliser(ref).split(), normaliser(hyp).split()
    return distance_levenshtein(r, h) / max(len(r), 1)


def cer(ref, hyp):
    r, h = normaliser(ref), normaliser(hyp)
    return distance_levenshtein(r, h) / max(len(r), 1)


def main():
    lignes = []
    images = pd.read_csv(DATA / "images.csv")
    for _, r in images.iterrows():
        res = traiter_image(DATA / "images" / r.fichier)
        lignes.append(("image", r.fichier, r.texte_attendu_ocr, res, r.sentiment_ref))

    audios = pd.read_csv(DATA / "audios.csv")
    for _, r in audios.iterrows():
        chemin = DATA / "audios" / r.fichier
        if not chemin.exists():                     # le CSV dit .wav, les fichiers sont .mp3
            chemin = chemin.with_suffix(".mp3")
        res = traiter_audio(chemin)
        lignes.append(("audio", chemin.name, r.transcription_reference, res, r.sentiment_ref))

    df = pd.DataFrame([{
        "source": s, "fichier": f,
        "wer_%": round(100 * wer(ref, res["texte_extrait"]), 1),
        "cer_%": round(100 * cer(ref, res["texte_extrait"]), 1),
        "sentiment_ref": sref, "sentiment_predit": res["sentiment"],
        "correct": sref == res["sentiment"], "confiance": res["confiance"],
        "texte_extrait": res["texte_extrait"],
    } for s, f, ref, res, sref in lignes])

    sortie = RACINE / "docs" / "resultats_pipelines.csv"
    sortie.parent.mkdir(exist_ok=True)
    df.to_csv(sortie, index=False, encoding="utf-8-sig")   # utf-8-sig : accents OK dans Excel
    pd.set_option("display.width", 200)
    print(df.drop(columns="texte_extrait").to_string(index=False))
    print("\nRésumé par source :")
    print(df.groupby("source").agg(WER_moyen=("wer_%", "mean"), CER_moyen=("cer_%", "mean"),
                                   accuracy=("correct", "mean")).round(2))
    print(f"\nFichier écrit : {sortie}")


if __name__ == "__main__":
    main()
