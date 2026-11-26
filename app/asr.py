"""
asr.py — ÉTAPE 2 (module Audio) : audio -> transcription (Whisper) -> sentiment.

ASR = Automatic Speech Recognition (reconnaissance automatique de la parole).
On utilise le modèle Whisper d'OpenAI, PRÉ-ENTRAÎNÉ (on ne l'entraîne pas).
Deux "moteurs" possibles pour faire tourner ce même modèle :

  1. faster-whisper (recommandé, défaut)
       pip install faster-whisper
       + pas besoin de PyTorch (≈ 2 Go en moins), pas besoin d'installer ffmpeg
       + 4x plus rapide sur CPU (moteur CTranslate2, poids quantifiés int8)
  2. openai-whisper (implémentation officielle, citée dans l'énoncé)
       pip install openai-whisper      + ffmpeg installé sur la machine
       (Windows : winget install ffmpeg)

Variables d'environnement (facultatives) :
  ASR_BACKEND   = auto (défaut) | faster-whisper | openai-whisper
  WHISPER_MODEL = tiny | base | small (défaut) | medium  ... ou un DOSSIER local
                  contenant un modèle déjà téléchargé (utile hors connexion)
Les poids sont téléchargés automatiquement au 1er lancement puis mis en cache.
"""
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Union

from app.classifier import predire_sentiment

ASR_BACKEND = os.getenv("ASR_BACKEND", "auto")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


class ASRIndisponible(RuntimeError):
    """Levée si aucun moteur Whisper n'est installé (ou ffmpeg manquant)."""


@lru_cache(maxsize=1)
def charger_whisper():
    """Charge le modèle UNE seule fois (plusieurs secondes) et le garde en mémoire.
    Retourne (nom_du_moteur, modele)."""
    if ASR_BACKEND in ("auto", "faster-whisper"):
        try:
            from faster_whisper import WhisperModel
            # int8 : poids quantifiés sur 8 bits -> 4x moins de mémoire, quasi même précision
            return "faster-whisper", WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        except ImportError:
            if ASR_BACKEND == "faster-whisper":
                raise ASRIndisponible("faster-whisper n'est pas installé (pip install faster-whisper).")
    try:
        import whisper
    except ImportError as e:
        raise ASRIndisponible(
            "Aucun moteur ASR installé : pip install faster-whisper (ou openai-whisper).") from e
    return "openai-whisper", whisper.load_model(WHISPER_MODEL)


def transcrire_audio(chemin: Union[str, Path]) -> str:
    moteur, modele = charger_whisper()
    try:
        if moteur == "faster-whisper":
            # language="fr" : on force le français (pas d'erreur de détection de langue)
            # beam_size=5   : garde les 5 meilleures hypothèses à chaque mot (plus précis)
            # vad_filter    : ignore les silences (détection d'activité vocale)
            segments, _info = modele.transcribe(str(chemin), language="fr", beam_size=5,
                                                vad_filter=True)
            return " ".join(s.text.strip() for s in segments).strip()
        # openai-whisper : fp16=False obligatoire sur CPU, temperature=0 -> déterministe
        resultat = modele.transcribe(str(chemin), language="fr", fp16=False, temperature=0)
        return resultat["text"].strip()
    except FileNotFoundError as e:  # openai-whisper sans ffmpeg -> FileNotFoundError: 'ffmpeg'
        raise ASRIndisponible("ffmpeg est introuvable : installez-le pour décoder l'audio.") from e


def transcrire_octets(contenu: bytes, extension: str = ".wav") -> str:
    """Pour l'API : Whisper lit un FICHIER, donc on écrit l'upload dans un
    fichier temporaire, on transcrit, puis on le supprime (bloc finally)."""
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
        tmp.write(contenu)
        chemin = tmp.name
    try:
        return transcrire_audio(chemin)
    except ASRIndisponible:
        raise
    except Exception as e:  # fichier non décodable (corrompu, faux .mp3...)
        raise RuntimeError(f"Décodage audio impossible : {e}") from e
    finally:
        os.remove(chemin)


def traiter_audio(source: Union[str, Path, bytes], extension: str = ".wav") -> dict:
    """Pipeline complet de la cascade : audio -> transcription -> classifieur texte."""
    if isinstance(source, (bytes, bytearray)):
        texte = transcrire_octets(source, extension)
    else:
        texte = transcrire_audio(source)
    return {"texte_extrait": texte, **predire_sentiment(texte)}


if __name__ == "__main__":  # démo : python -m app.asr data/audios/A01.mp3
    import sys
    for chemin in sys.argv[1:]:
        print(chemin, "->", traiter_audio(chemin))
