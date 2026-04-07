"""Optional quality checks for TTS validation.

Each check gracefully returns None if its dependencies are not installed.
Install on gpu-server for full functionality:
    pip install speechmos resemblyzer

For language ID, SpeechBrain is needed:
    pip install speechbrain

All checks accept a file path and return a dict with results.
"""

import warnings
from pathlib import Path


def check_mos(audio_path):
    """Predict Mean Opinion Score using UTMOS via torch.hub.

    Returns: {"mos": float} or None if torch/torchaudio not installed.
    Score range: 1.0 (bad) to 5.0 (excellent). Flag below 3.5.
    """
    try:
        import torch
        import torchaudio
    except ImportError:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor = torch.hub.load(
                "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
            )
            wave, sr = torchaudio.load(str(audio_path))
            if sr != 16000:
                wave = torchaudio.functional.resample(wave, sr, 16000)
            score = predictor(wave, sr=16000)
        return {"mos": round(float(score.item()), 2)}
    except Exception as e:
        return {"mos": None, "mos_error": str(e)}


def check_speaker_similarity(audio_path, ref_path):
    """Compare speaker embeddings between generated audio and reference.

    Returns: {"speaker_similarity": float} or None if resemblyzer not installed.
    Score range: 0.0 (different speaker) to 1.0 (same speaker). Flag below 0.75.
    """
    if ref_path is None:
        return None

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        return None

    try:
        encoder = VoiceEncoder()
        wav_gen = preprocess_wav(str(audio_path))
        wav_ref = preprocess_wav(str(ref_path))
        embed_gen = encoder.embed_utterance(wav_gen)
        embed_ref = encoder.embed_utterance(wav_ref)
        # Cosine similarity
        import numpy as np
        norm_product = np.linalg.norm(embed_gen) * np.linalg.norm(embed_ref)
        if norm_product < 1e-10:
            similarity = 0.0
        else:
            similarity = float(np.dot(embed_gen, embed_ref) / norm_product)
        return {"speaker_similarity": round(similarity, 3)}
    except Exception as e:
        return {"speaker_similarity": None, "similarity_error": str(e)}


def check_language(audio_path, expected_language="en"):
    """Verify the spoken language matches expected using SpeechBrain lang-id.

    Returns: {"detected_language": str, "language_match": bool, "language_confidence": float}
    or None if speechbrain not installed.
    """
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        return None

    # Map our language codes to SpeechBrain's expected format
    LANG_MAP = {
        "en": "en",
        "de": "de",
        "nl": "nl",
    }
    expected = LANG_MAP.get(expected_language, expected_language)

    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            run_opts={"device": "cpu"},
        )
        prediction = classifier.classify_file(str(audio_path))
        # prediction returns (prob, score, index, label)
        detected = prediction[3][0].strip()
        confidence = float(prediction[1].exp().max())

        # SpeechBrain returns full language names or ISO codes depending on version
        # Normalize: take first 2 chars of detected language
        detected_code = detected[:2].lower() if len(detected) >= 2 else detected.lower()

        return {
            "detected_language": detected,
            "language_code": detected_code,
            "language_match": detected_code == expected,
            "language_confidence": round(confidence, 3),
        }
    except Exception as e:
        return {"detected_language": None, "language_error": str(e)}


def run_quality_checks(audio_path, ref_path=None, expected_language="en"):
    """Run all available quality checks on an audio file.

    Returns a dict with results from each available check.
    Checks that aren't installed are silently skipped.
    """
    results = {}

    mos = check_mos(audio_path)
    if mos is not None:
        results.update(mos)

    similarity = check_speaker_similarity(audio_path, ref_path)
    if similarity is not None:
        results.update(similarity)

    lang = check_language(audio_path, expected_language)
    if lang is not None:
        results.update(lang)

    return results


def get_available_checks():
    """Report which quality checks are available in this environment."""
    available = []

    try:
        import torch
        import torchaudio
        available.append("mos")
    except ImportError:
        pass

    try:
        from resemblyzer import VoiceEncoder
        available.append("speaker_similarity")
    except ImportError:
        pass

    try:
        from speechbrain.inference.classifiers import EncoderClassifier
        available.append("language_id")
    except ImportError:
        pass

    return available
