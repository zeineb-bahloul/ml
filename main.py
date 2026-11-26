"""
main.py — ÉTAPE 3 : API REST FastAPI du classifieur de sentiment multi-source.

Lancer :   uvicorn main:app --reload
Swagger :  http://127.0.0.1:8000/docs     (Option A de l'étape 4)
ReDoc :    http://127.0.0.1:8000/redoc
Mini-app : http://127.0.0.1:8000/app/     (Option B de l'étape 4)

Routes :
  GET  /health          -> état du service (modèle chargé ? OCR ? ASR ?)
  POST /predict/text    -> JSON {"texte": "..."}
  POST /predict/image   -> multipart/form-data, champ "file" (png, jpg...)
  POST /predict/audio   -> multipart/form-data, champ "file" (wav, mp3...)
  POST /predict         -> détection automatique : champ "texte" OU champ "file"
"""
import mimetypes
import shutil
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app import asr, ocr
from app.classifier import ModeleIndisponible, charger_artefacts, predire_sentiment
from app.schemas import (FichierAudio, FichierImage, MIME_AUDIOS, MIME_IMAGES,
                         PredictionReponse, TexteRequete)

RACINE = Path(__file__).resolve().parent

app = FastAPI(
    title="Classifieur de Sentiment Multi-source",
    description=("E-commerce Customer Voice — architecture en cascade : "
                 "image --OCR--> texte, audio --ASR--> texte, puis un classifieur "
                 "texte unique (TF-IDF + Régression Logistique)."),
    version="1.0.0",
)

# CORS : autorise une page web servie ailleurs (ex. fichier ouvert en local) à appeler l'API
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Option B : la mini-application web statique est servie par l'API elle-même
app.mount("/app", StaticFiles(directory=RACINE / "webapp", html=True), name="webapp")


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------
def type_mime_reel(fichier: UploadFile) -> str:
    """Le navigateur envoie parfois 'application/octet-stream' : on devine
    alors le type à partir de l'extension (.mp3 -> audio/mpeg)."""
    mime = (fichier.content_type or "").split(";")[0].strip().lower()
    if mime in ("", "application/octet-stream"):
        mime = mimetypes.guess_type(fichier.filename or "")[0] or mime
    return mime


def lire_et_valider(fichier: UploadFile, schema):
    """Lit l'upload puis valide nom/type/taille avec le schéma Pydantic.
    Toute violation -> HTTP 422 avec un message clair."""
    contenu = fichier.file.read()
    try:
        schema(nom=fichier.filename or "fichier", type_mime=type_mime_reel(fichier), taille=len(contenu))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=[
            {"champ": ".".join(map(str, err["loc"])), "message": err["msg"]} for err in e.errors()])
    return contenu


def construire_reponse(source: str, resultat: dict, debut: float) -> PredictionReponse:
    return PredictionReponse(
        source=source,
        texte_extrait=resultat["texte_extrait"],
        sentiment_predit=resultat["sentiment"],
        confiance=resultat["confiance"],
        probabilites=resultat["probabilites"],
        duree_ms=round((time.perf_counter() - debut) * 1000, 1),
    )


def verifier_texte_extrait(texte: str, source: str):
    if len(texte.strip()) < 3:
        raise HTTPException(status_code=422,
                            detail=f"Aucun texte exploitable n'a été extrait du fichier {source}.")


# ---------------------------------------------------------------------------
# Traitements (partagés entre les routes dédiées et la route automatique)
# ---------------------------------------------------------------------------
def _traiter_texte(texte: str) -> PredictionReponse:
    debut = time.perf_counter()
    try:
        res = predire_sentiment(texte)
    except ModeleIndisponible as e:
        raise HTTPException(status_code=503, detail=str(e))
    return construire_reponse("texte", {"texte_extrait": texte, **res}, debut)


def _traiter_image(contenu: bytes) -> PredictionReponse:
    debut = time.perf_counter()
    try:
        texte = ocr.extraire_texte_image(contenu)
    except ocr.OCRIndisponible as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:  # Pillow ne sait pas ouvrir le fichier -> image corrompue
        raise HTTPException(status_code=422, detail="Fichier image illisible ou corrompu.")
    verifier_texte_extrait(texte, "image")
    return construire_reponse("image", {"texte_extrait": texte, **predire_sentiment(texte)}, debut)


def _traiter_audio(contenu: bytes, nom: str) -> PredictionReponse:
    debut = time.perf_counter()
    extension = Path(nom).suffix or ".wav"
    try:
        texte = asr.transcrire_octets(contenu, extension)
    except asr.ASRIndisponible as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError:  # ffmpeg n'a pas pu décoder -> fichier corrompu
        raise HTTPException(status_code=422, detail="Fichier audio illisible ou corrompu.")
    verifier_texte_extrait(texte, "audio")
    return construire_reponse("audio", {"texte_extrait": texte, **predire_sentiment(texte)}, debut)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def accueil():
    return RedirectResponse(url="/app/")


@app.get("/health", tags=["Service"], summary="État du service")
def health():
    try:
        charger_artefacts()
        modele_ok = True
    except ModeleIndisponible:
        modele_ok = False
    return {
        "status": "ok" if modele_ok else "degrade",
        "modele_texte": modele_ok,
        "ocr_tesseract": shutil.which(ocr.pytesseract.pytesseract.tesseract_cmd) is not None,
        "asr_moteur": moteur_asr_installe(),
        "asr_modele_whisper": asr.WHISPER_MODEL,
    }


def moteur_asr_installe() -> Optional[str]:
    """Indique quel moteur Whisper est installé (sans charger le modèle, qui est lourd)."""
    import importlib.util
    if importlib.util.find_spec("faster_whisper"):
        return "faster-whisper"
    if importlib.util.find_spec("whisper") and shutil.which("ffmpeg"):
        return "openai-whisper"
    return None


@app.post("/predict/text", response_model=PredictionReponse, tags=["Prédiction"],
          summary="Sentiment d'un avis texte (JSON)")
def predict_text(requete: TexteRequete):
    return _traiter_texte(requete.texte)


@app.post("/predict/image", response_model=PredictionReponse, tags=["Prédiction"],
          summary="Sentiment d'une capture d'écran (OCR -> modèle texte)")
def predict_image(file: UploadFile = File(..., description="Image PNG/JPEG (max 5 Mo)")):
    contenu = lire_et_valider(file, FichierImage)
    return _traiter_image(contenu)


@app.post("/predict/audio", response_model=PredictionReponse, tags=["Prédiction"],
          summary="Sentiment d'un message vocal (Whisper -> modèle texte)")
def predict_audio(file: UploadFile = File(..., description="Audio WAV/MP3/OGG (max 20 Mo)")):
    contenu = lire_et_valider(file, FichierAudio)
    return _traiter_audio(contenu, file.filename or "audio.wav")


@app.post("/predict", response_model=PredictionReponse, tags=["Prédiction"],
          summary="Détection automatique du type d'entrée")
def predict_auto(texte: Optional[str] = Form(None, description="Avis texte (si pas de fichier)"),
                 file: Optional[UploadFile] = File(None, description="Image ou audio")):
    """Envoyer SOIT un champ `texte`, SOIT un champ `file`.
    Le type est détecté via le type MIME (image/* ou audio/*)."""
    if file is not None and file.filename:
        mime = type_mime_reel(file)
        if mime in MIME_IMAGES:
            return _traiter_image(lire_et_valider(file, FichierImage))
        if mime in MIME_AUDIOS:
            return _traiter_audio(lire_et_valider(file, FichierAudio), file.filename)
        raise HTTPException(status_code=422,
                            detail=f"Type de fichier '{mime}' non reconnu (ni image, ni audio).")
    if texte is not None:
        try:
            requete = TexteRequete(texte=texte)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=[
                {"champ": "texte", "message": err["msg"]} for err in e.errors()])
        return _traiter_texte(requete.texte)
    raise HTTPException(status_code=422, detail="Envoyez un champ 'texte' ou un champ 'file'.")
