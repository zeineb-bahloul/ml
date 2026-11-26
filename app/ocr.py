"""
ocr.py — ÉTAPE 2 (module Image) : image -> texte (pytesseract) -> sentiment.

Pré-requis système : le moteur Tesseract doit être installé
  - Windows : https://github.com/UB-Mannheim/tesseract/wiki (cocher "French")
  - Linux   : sudo apt install tesseract-ocr tesseract-ocr-fra
Si tesseract.exe n'est pas dans le PATH (fréquent sous Windows), définir
la variable d'environnement TESSERACT_CMD, ex :
  set TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""
import io
import os
import re
from pathlib import Path
from typing import Union

import pytesseract
from PIL import Image, ImageOps

from app.classifier import predire_sentiment

if os.getenv("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

# Le bandeau des captures fournies contient le nom du fichier
# ("CAPTURE AVIS CLIENT — I02.png") : ce n'est pas l'avis, on l'enlève.
BANDEAU = re.compile(r"^\s*CAPTURE AVIS CLIENT.*$", re.IGNORECASE | re.MULTILINE)


class OCRIndisponible(RuntimeError):
    """Levée si le binaire Tesseract n'est pas installé."""


def pretraiter_image(img: Image.Image) -> Image.Image:
    """Améliore la lisibilité pour Tesseract :
    1. conversion en niveaux de gris (L)  -> la couleur n'aide pas l'OCR
    2. agrandissement x2                  -> Tesseract préfère des lettres de ~30 px
    3. autocontraste                      -> texte bien noir sur fond bien blanc"""
    img = ImageOps.exif_transpose(img).convert("L")
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    return ImageOps.autocontrast(img)


def extraire_texte_image(source: Union[str, Path, bytes]) -> str:
    """Accepte un chemin de fichier OU des octets (upload de l'API)."""
    img = Image.open(io.BytesIO(source)) if isinstance(source, (bytes, bytearray)) else Image.open(source)
    try:
        # lang="fra" : dictionnaire français (accents !)
        # --psm 6    : "un bloc de texte uniforme" -> adapté aux captures d'avis
        brut = pytesseract.image_to_string(pretraiter_image(img), lang="fra", config="--psm 6")
    except pytesseract.TesseractNotFoundError as e:
        raise OCRIndisponible("Tesseract n'est pas installé ou introuvable (voir TESSERACT_CMD).") from e
    brut = BANDEAU.sub("", brut)
    lignes = [l.strip() for l in brut.splitlines() if l.strip()]
    return " ".join(lignes)


def traiter_image(source: Union[str, Path, bytes]) -> dict:
    """Pipeline complet de la cascade : image -> OCR -> classifieur texte."""
    texte = extraire_texte_image(source)
    return {"texte_extrait": texte, **predire_sentiment(texte)}


if __name__ == "__main__":  # démo : python -m app.ocr data/images/I02.png
    import sys
    for chemin in sys.argv[1:]:
        print(chemin, "->", traiter_image(chemin))
