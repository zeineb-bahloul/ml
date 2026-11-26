"""
schemas.py — Schémas Pydantic : ce qui ENTRE dans l'API et ce qui en SORT.

Pydantic vérifie automatiquement les types et contraintes. Si une règle
n'est pas respectée, FastAPI renvoie 422 Unprocessable Entity avec le
détail de l'erreur, sans que l'on écrive un seul "if".
"""
from typing import Dict, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- Limites de sécurité ----------------------------------------------------
MAX_TEXTE = 5000                      # caractères
MAX_IMAGE = 5 * 1024 * 1024           # 5 Mo
MAX_AUDIO = 20 * 1024 * 1024          # 20 Mo

MIME_IMAGES = {"image/png", "image/jpeg", "image/bmp", "image/tiff", "image/webp"}
MIME_AUDIOS = {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave",
               "audio/mpeg", "audio/mp3", "audio/ogg", "audio/webm",
               "audio/flac", "audio/x-flac", "audio/mp4", "audio/x-m4a", "audio/m4a"}

Sentiment = Literal["positif", "negatif", "neutre"]
Source = Literal["texte", "image", "audio"]


# --- Entrée de /predict/text -------------------------------------------------
class TexteRequete(BaseModel):
    """Corps JSON attendu : {"texte": "Livraison rapide, je recommande !"}"""
    model_config = ConfigDict(json_schema_extra={
        "examples": [{"texte": "Produit conforme, livraison super rapide, je recommande !"}]})

    texte: str = Field(..., min_length=3, max_length=MAX_TEXTE,
                       description="Avis client à analyser (3 à 5000 caractères)")

    @field_validator("texte")
    @classmethod
    def pas_que_des_espaces(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("le texte doit contenir au moins 3 caractères non vides")
        return v


# --- Métadonnées d'un fichier uploadé (image ou audio) ---------------------
class FichierImage(BaseModel):
    nom: str
    type_mime: str
    taille: int = Field(..., gt=0, le=MAX_IMAGE, description="taille en octets (max 5 Mo)")

    @field_validator("type_mime")
    @classmethod
    def mime_image(cls, v: str) -> str:
        if v not in MIME_IMAGES:
            raise ValueError(f"type MIME '{v}' refusé ; acceptés : {sorted(MIME_IMAGES)}")
        return v


class FichierAudio(BaseModel):
    nom: str
    type_mime: str
    taille: int = Field(..., gt=0, le=MAX_AUDIO, description="taille en octets (max 20 Mo)")

    @field_validator("type_mime")
    @classmethod
    def mime_audio(cls, v: str) -> str:
        if v not in MIME_AUDIOS:
            raise ValueError(f"type MIME '{v}' refusé ; acceptés : {sorted(MIME_AUDIOS)}")
        return v


# --- Réponse unifiée (identique pour les 4 routes) --------------------------
class PredictionReponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{
        "source": "image",
        "texte_extrait": "RÉCLAMATION COMMANDE #9921 Colis ouvert et article manquant.",
        "sentiment_predit": "negatif",
        "confiance": 0.89,
        "probabilites": {"negatif": 0.89, "neutre": 0.06, "positif": 0.05},
        "duree_ms": 412.5}]})

    source: Source
    texte_extrait: str
    sentiment_predit: Sentiment
    confiance: float = Field(..., ge=0, le=1)
    probabilites: Dict[str, float]
    duree_ms: float
