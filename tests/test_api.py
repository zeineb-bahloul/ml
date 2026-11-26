"""
test_api.py — ÉTAPE 5 : tests unitaires et d'intégration (pytest + TestClient/httpx).

Lancement :           pytest -v
Sans les tests lents : pytest -v -m "not lent"

- Tests UNITAIRES     : une fonction isolée (nettoyage, classifieur).
- Tests d'INTÉGRATION : une vraie requête HTTP envoyée à l'API (sans lancer
                        uvicorn : TestClient appelle l'application en mémoire).
"""
import io
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import main
from app import asr
from app.classifier import predire_sentiment
from app.preprocessing import nettoyer_texte

client = TestClient(main.app)
DATA = Path(__file__).resolve().parent.parent / "data"
CHAMPS_ATTENDUS = {"source", "texte_extrait", "sentiment_predit", "confiance", "probabilites", "duree_ms"}
SENTIMENTS = {"positif", "negatif", "neutre"}

tesseract_absent = pytest.mark.skipif(shutil.which(main.ocr.pytesseract.pytesseract.tesseract_cmd) is None,
                                      reason="Tesseract n'est pas installé")


def verifier_structure(json, source):
    """Vérifie la réponse JSON unifiée demandée par l'énoncé."""
    assert set(json) == CHAMPS_ATTENDUS
    assert json["source"] == source
    assert json["sentiment_predit"] in SENTIMENTS
    assert 0.0 <= json["confiance"] <= 1.0
    assert set(json["probabilites"]) == SENTIMENTS
    assert abs(sum(json["probabilites"].values()) - 1.0) < 1e-3   # des probabilités somment à 1
    assert json["confiance"] == max(json["probabilites"].values())


@pytest.fixture
def faux_asr(monkeypatch):
    """Remplace Whisper par une fausse transcription : test rapide (pas de modèle
    de 500 Mo à charger) et déterministe. On teste l'API, pas Whisper."""
    monkeypatch.setattr(asr, "transcrire_octets",
                        lambda contenu, extension=".wav": "Je demande le remboursement, le produit ne fonctionne pas.")


def png_en_octets(texte=None, taille=(400, 120)):
    img = Image.new("RGB", taille, "white")
    if texte:
        from PIL import ImageDraw
        ImageDraw.Draw(img).text((10, 50), texte, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =========================================================================
# 1. Tests UNITAIRES
# =========================================================================
class TestPretraitement:
    def test_minuscules_et_ponctuation(self):
        assert nettoyer_texte("SUPER Produit !!!") == "super produit"

    def test_stopwords_supprimes(self):
        assert "le" not in nettoyer_texte("le colis est arrivé").split()

    def test_negations_conservees(self):
        # "ne ... pas" inverse le sens : il NE FAUT PAS le supprimer
        mots = nettoyer_texte("Le produit ne fonctionne pas").split()
        assert "ne" in mots and "pas" in mots

    def test_chiffres_mentions_supprimes(self):
        assert nettoyer_texte("@ClientHeureux commande #9921") == "commande"

    def test_entree_non_texte(self):
        assert nettoyer_texte(None) == ""


class TestClassifieur:
    def test_structure_sortie(self):
        res = predire_sentiment("Excellent produit, je recommande")
        assert res["sentiment"] in SENTIMENTS
        assert 0 <= res["confiance"] <= 1

    @pytest.mark.parametrize("texte, attendu", [
        ("Très satisfait, excellente qualité, je recommande vivement", "positif"),
        ("Remboursement toujours en attente, le produit ne fonctionne pas", "negatif"),
        ("Colis livré en point relais comme prévu, conforme à la description", "neutre"),
    ])
    def test_cas_evidents(self, texte, attendu):
        assert predire_sentiment(texte)["sentiment"] == attendu


# =========================================================================
# 2. Tests d'INTÉGRATION — route /predict/text
# =========================================================================
class TestRouteTexte:
    def test_200_structure(self):
        r = client.post("/predict/text", json={"texte": "Livraison rapide, produit parfait, je recommande !"})
        assert r.status_code == 200
        verifier_structure(r.json(), "texte")
        assert r.json()["sentiment_predit"] == "positif"

    def test_422_champ_manquant(self):
        assert client.post("/predict/text", json={}).status_code == 422

    def test_422_mauvais_nom_de_champ(self):
        assert client.post("/predict/text", json={"text": "bonjour"}).status_code == 422

    def test_422_texte_vide_ou_espaces(self):
        assert client.post("/predict/text", json={"texte": ""}).status_code == 422
        assert client.post("/predict/text", json={"texte": "     "}).status_code == 422

    def test_422_texte_trop_long(self):
        assert client.post("/predict/text", json={"texte": "a" * 5001}).status_code == 422

    def test_422_mauvais_type(self):
        assert client.post("/predict/text", json={"texte": 123}).status_code == 422

    def test_422_json_invalide(self):
        r = client.post("/predict/text", content="pas du json", headers={"Content-Type": "application/json"})
        assert r.status_code == 422


# =========================================================================
# 3. Tests d'INTÉGRATION — route /predict/image
# =========================================================================
class TestRouteImage:
    @tesseract_absent
    @pytest.mark.parametrize("fichier, attendu", [("I02.png", "negatif"), ("I04.png", "positif"),
                                                  ("I06.png", "neutre")])
    def test_200_vraie_image(self, fichier, attendu):
        with open(DATA / "images" / fichier, "rb") as f:
            r = client.post("/predict/image", files={"file": (fichier, f, "image/png")})
        assert r.status_code == 200
        verifier_structure(r.json(), "image")
        assert r.json()["sentiment_predit"] == attendu
        assert len(r.json()["texte_extrait"]) > 20      # l'OCR a bien lu quelque chose

    def test_422_sans_fichier(self):
        assert client.post("/predict/image").status_code == 422

    def test_422_mauvais_type_mime(self):
        r = client.post("/predict/image", files={"file": ("avis.txt", b"bonjour", "text/plain")})
        assert r.status_code == 422

    def test_422_fichier_vide(self):
        r = client.post("/predict/image", files={"file": ("vide.png", b"", "image/png")})
        assert r.status_code == 422

    def test_422_fichier_trop_gros(self):
        gros = b"\x89PNG" + b"0" * (5 * 1024 * 1024 + 1)
        r = client.post("/predict/image", files={"file": ("gros.png", gros, "image/png")})
        assert r.status_code == 422

    def test_422_image_corrompue(self):
        r = client.post("/predict/image", files={"file": ("faux.png", b"ceci n'est pas un png", "image/png")})
        assert r.status_code == 422

    @tesseract_absent
    def test_422_image_sans_texte(self):
        r = client.post("/predict/image", files={"file": ("blanc.png", png_en_octets(), "image/png")})
        assert r.status_code == 422


# =========================================================================
# 4. Tests d'INTÉGRATION — route /predict/audio
# =========================================================================
class TestRouteAudio:
    def test_200_avec_faux_asr(self, faux_asr):
        r = client.post("/predict/audio", files={"file": ("A08.wav", b"RIFF....WAVE", "audio/wav")})
        assert r.status_code == 200
        verifier_structure(r.json(), "audio")
        assert r.json()["sentiment_predit"] == "negatif"

    def test_mime_devine_par_extension(self, faux_asr):
        # navigateur qui envoie "application/octet-stream" pour un .mp3
        r = client.post("/predict/audio", files={"file": ("A01.mp3", b"ID3...", "application/octet-stream")})
        assert r.status_code == 200

    def test_422_sans_fichier(self):
        assert client.post("/predict/audio").status_code == 422

    def test_422_image_envoyee_comme_audio(self):
        r = client.post("/predict/audio", files={"file": ("I01.png", png_en_octets(), "image/png")})
        assert r.status_code == 422

    def test_422_audio_trop_gros(self):
        r = client.post("/predict/audio", files={"file": ("long.wav", b"0" * (20 * 1024 * 1024 + 1), "audio/wav")})
        assert r.status_code == 422

    @pytest.mark.lent
    def test_200_vrai_audio_whisper(self):
        """Test de bout en bout avec le VRAI Whisper (lent : charge le modèle)."""
        try:
            asr.charger_whisper()
        except Exception as e:
            pytest.skip(f"Whisper indisponible : {e}")
        with open(DATA / "audios" / "A08.mp3", "rb") as f:
            r = client.post("/predict/audio", files={"file": ("A08.mp3", f, "audio/mpeg")})
        assert r.status_code == 200
        verifier_structure(r.json(), "audio")
        assert "remboursement" in r.json()["texte_extrait"].lower()
        assert r.json()["sentiment_predit"] == "negatif"


# =========================================================================
# 5. Route automatique /predict + /health
# =========================================================================
class TestRouteAuto:
    def test_texte(self):
        r = client.post("/predict", data={"texte": "Article reçu cassé, très déçu"})
        assert r.status_code == 200
        verifier_structure(r.json(), "texte")

    @tesseract_absent
    def test_image_detectee(self):
        with open(DATA / "images" / "I08.png", "rb") as f:
            r = client.post("/predict", files={"file": ("I08.png", f, "image/png")})
        assert r.status_code == 200
        assert r.json()["source"] == "image"

    def test_audio_detecte(self, faux_asr):
        r = client.post("/predict", files={"file": ("msg.mp3", b"ID3...", "audio/mpeg")})
        assert r.status_code == 200
        assert r.json()["source"] == "audio"

    def test_422_rien_envoye(self):
        assert client.post("/predict").status_code == 422

    def test_422_type_inconnu(self):
        r = client.post("/predict", files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")})
        assert r.status_code == 422

    def test_422_texte_trop_court(self):
        assert client.post("/predict", data={"texte": "ok"}).status_code == 422


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["modele_texte"] is True


def test_mini_app_servie():
    r = client.get("/app/")
    assert r.status_code == 200 and "Customer Voice" in r.text


def test_documentation_swagger():
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert {"/predict/text", "/predict/image", "/predict/audio", "/predict"} <= set(schema["paths"])
