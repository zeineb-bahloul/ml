"""
preprocessing.py — Nettoyage du texte (étape 1 du mini-projet).

Ce module est utilisé À LA FOIS :
  - pendant l'entraînement (scripts/train.py)
  - pendant la prédiction (API), pour que le texte saisi, le texte OCR
    et le texte transcrit soient nettoyés EXACTEMENT de la même façon.
Règle d'or du ML : même prétraitement à l'entraînement et en production.
"""
import re
import unicodedata

# ---------------------------------------------------------------------------
# Stop-words français (mots très fréquents qui ne portent pas de sentiment).
# On écrit la liste nous-mêmes pour éviter une dépendance (nltk/spacy).
# ATTENTION : on NE retire PAS les mots de négation ("ne", "pas", "sans",
# "aucun", "rien", "jamais", "plus"...). Pour l'analyse de sentiment, ils
# inversent le sens : "fonctionne" vs "ne fonctionne pas".
# ---------------------------------------------------------------------------
STOPWORDS_FR = {
    "a", "à", "au", "aux", "avec", "ce", "ces", "cet", "cette", "c", "d", "dans",
    "de", "des", "du", "elle", "elles", "en", "et", "eu", "il", "ils", "j", "je",
    "l", "la", "le", "les", "leur", "leurs", "lui", "m", "ma", "mais", "me",
    "mes", "moi", "mon", "n", "nos", "notre", "nous", "on", "ou", "par", "pour",
    "qu", "que", "qui", "s", "sa", "se", "ses", "son", "sur", "t", "ta", "te",
    "tes", "toi", "ton", "tu", "un", "une", "vos", "votre", "vous", "y",
    "est", "sont", "été", "être", "ai", "as", "avons", "avez", "ont", "avait",
    "était", "suis", "es", "sommes", "êtes", "comme", "lors", "ceci", "cela",
}
NEGATIONS = {"ne", "pas", "sans", "aucun", "aucune", "rien", "jamais", "plus", "non", "ni"}
STOPWORDS_FR -= NEGATIONS  # sécurité : jamais de négation dans les stop-words


def normaliser_unicode(texte: str) -> str:
    """Uniformise les apostrophes/guillemets typographiques (fréquents en OCR)."""
    texte = unicodedata.normalize("NFC", texte)
    return (texte.replace("’", "'").replace("‘", "'")
                 .replace("«", " ").replace("»", " ").replace("—", " "))


def nettoyer_texte(texte: str) -> str:
    """
    Nettoyage minimal demandé par l'énoncé :
      1. minuscules
      2. suppression des URLs, mentions @, numéros (#9921, 24h, 1,2 kg...)
      3. suppression de la ponctuation (on garde les lettres accentuées)
      4. découpage en mots + suppression des stop-words
    Exemple :
      "Article reçu cassé et le service client ne répond pas !"
      -> "article reçu cassé service client ne répond pas"
    """
    if not isinstance(texte, str):
        return ""
    texte = normaliser_unicode(texte).lower()
    texte = re.sub(r"https?://\S+|www\.\S+", " ", texte)   # URLs
    texte = re.sub(r"@\w+", " ", texte)                    # @mentions
    texte = re.sub(r"\d+", " ", texte)                     # chiffres
    texte = re.sub(r"[^a-zàâäçéèêëîïôöùûüÿœæ' ]", " ", texte)  # ponctuation, ★
    mots = re.split(r"[\s']+", texte)
    mots_gardes = []
    for m in mots:
        if not m:
            continue                      # chaîne vide (espaces multiples)
        if m in NEGATIONS:
            mots_gardes.append(m)         # on garde TOUJOURS les négations
        elif m not in STOPWORDS_FR and len(m) > 1:
            mots_gardes.append(m)         # mot "utile"
    return " ".join(mots_gardes)
