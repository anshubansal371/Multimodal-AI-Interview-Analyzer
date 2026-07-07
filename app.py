# app.py — AI Interview Analyzer v3
# Priority-1 rewrite: audio module, emotion-label removal,
# confidence-weighted fusion, richer speech metrics,
# template-based "AI-style" feedback synthesis.
import streamlit as st
import numpy as np
import json
import os
import re
import tempfile
import subprocess
import torch
import tensorflow as tf
from collections import Counter
import plotly.graph_objects as go
import os
import streamlit as st

from download_models import download_all

REQUIRED_FILES = [

    "models/face_model_best.keras",

    "models/audio_model_best.keras",

    "models/fusion_model_best.keras",

    "models/final_roberta_model/model.safetensors",

    "models/final_roberta_model/config.json",

    "models/final_roberta_model/tokenizer.json",

    "models/final_roberta_model/tokenizer_config.json",

    "models/final_roberta_model/emotion_map.json",
]

if not all(os.path.exists(f) for f in REQUIRED_FILES):

    with st.spinner("Downloading AI models... Please wait."):

        download_all()

st.set_page_config(
    page_title="AI Interview Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .score-card {
        background: linear-gradient(135deg,#667eea20,#764ba220);
        border-radius: 15px; padding: 1.5rem;
        border: 1px solid #667eea40;
        text-align: center; margin: 0.5rem 0;
    }
    .score-number {
        font-size: 2.5rem; font-weight: bold; color: #667eea;
    }
    .trait-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 0.8rem 1rem; margin: 0.4rem 0;
        border-left: 4px solid #667eea;
    }
    .feedback-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    .strength-item { color: #27ae60; }
    .improve-item  { color: #e67e22; }
    .tip-high      { color: #e74c3c; }
    .tip-med       { color: #f39c12; }
    .tip-low       { color: #27ae60; }
    .cert-box      { border-radius: 20px; padding: 2rem;
                       text-align: center; margin: 1rem 0; }
    .rec-badge {
        display: inline-block; padding: 0.5rem 1.5rem;
        border-radius: 25px; font-size: 1.2rem;
        font-weight: bold; color: white;
    }
</style>
""", unsafe_allow_html=True)

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models')

FACE_EMOTIONS  = {0:'angry',1:'disgust',2:'fear',3:'happy',
                   4:'neutral',5:'sad',6:'surprise'}
AUDIO_EMOTIONS = {'0':'angry','1':'disgust','2':'fearful',
                   '3':'happy','4':'neutral','5':'sad'}
FACE_TO_DIM  = {'angry':2,'disgust':2,'fear':2,'happy':1,
                 'neutral':4,'sad':2,'surprise':3}
AUDIO_TO_DIM = {'angry':2,'disgust':2,'fearful':2,'happy':1,
                 'neutral':4,'sad':2}
TEXT_TO_DIM  = {'angry':2,'anxious':2,'positive':1,'surprised':3}

FILLER_WORDS = ['um','uh','like','so','you know','basically',
                 'literally','actually','i mean','right','okay so']

SKILL_KEYWORDS = {
    'technical': [
        'python','java','c','c++','sql',
    'database','databases',
    'machine learning','deep learning',
    'tensorflow','pytorch',
    'docker','kubernetes',
    'web development','html','css','javascript',
    'api','software','project',
    'developed','built','implemented',
    'designed','created','cloud'],
    'leadership': [
        'led','managed','team','initiative','ownership',
        'mentored','coordinated','supervised','organized',
        'directed','handled','responsible','in charge',
        'head','guide','delegate','lead'],
    'problem_solving': [
        'solved','debugged','optimized','improved',
        'project','implemented','created','developed',
        'built','designed','application','system',
        'analyzed','designed','implemented','resolved',
        'fixed','approach','solution','challenge',
        'issue','problem','tackle','overcome','identify',
        'diagnose','worked on','figured out','addressed'],
    'communication': [
        'presented','explained','collaborated',
        'discussed','communicated','reported','told',
        'shared','informed','conveyed','worked with',
        'talked','meeting','spoke','described',
        'mentioned','expressed','interact'],
    'star_method': [
        'situation','task','action','result',
        'challenge','achieved','delivered','outcome',
        'background','context','responsible','goal',
        'objective','approach','decided','did','what i did',
        'as a result','led to','because of','therefore',
        'consequently','in the end','ultimately']}

STAR_COMPONENTS = {
    'situation': [
        'situation','context','background',
        'at the time','when i was','we were',
        'there was','i was working','in my previous',
        'in my role','during','at that time'],
    'task': [
        'task','responsible','goal','objective',
        'assigned','my job','i had to','i needed to',
        'my role was','i was asked','required to',
        'needed to','i was given'],
    'action': [
        'action','implemented','did','approach',
        'i decided','i analyzed','i built','i designed',
        'i created','i developed','i worked','i wrote',
        'i fixed','i solved','i set up','i used',
        'i started','i began','so i','what i did',
        'my approach','i chose','i applied'],
    'result': [
        'result','outcome','achieved','improved',
        'increased','decreased','delivered','led to',
        'as a result','consequently','therefore',
        'this helped','we managed','we succeeded',
        'successfully','in the end','ultimately',
        'the impact','percent','reduced','saved',
        'completed','finished']}


# ═══════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    models = {}
    try:
        face_path = os.path.join(MODELS_DIR, 'face_model_best.keras')
        if not os.path.exists(face_path):
            face_path = os.path.join(MODELS_DIR, 'face_model_best.h5')
        models['face'] = tf.keras.models.load_model(face_path)
        st.sidebar.success("✅ Face model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Face: {e}")
        models['face'] = None

    try:
        audio_path = os.path.join(MODELS_DIR, 'audio_model_best.keras')
        models['audio'] = tf.keras.models.load_model(audio_path)
        st.sidebar.success("✅ Audio model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Audio: {e}")
        models['audio'] = None

    try:
        fusion_path = os.path.join(MODELS_DIR, 'fusion_model_best.keras')
        models['fusion'] = tf.keras.models.load_model(fusion_path)
        st.sidebar.success("✅ Fusion model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Fusion: {e}")
        models['fusion'] = None

    try:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification)
        roberta_path = os.path.join(MODELS_DIR, 'final_roberta_model')
        models['roberta_tok'] = AutoTokenizer.from_pretrained(
            roberta_path, local_files_only=True)
        models['roberta'] = AutoModelForSequenceClassification\
            .from_pretrained(roberta_path, local_files_only=True)
        models['roberta'].eval()
        with open(os.path.join(roberta_path, 'emotion_map.json')) as f:
            emotion2id = json.load(f)
        models['id2emotion'] = {v: k for k, v in emotion2id.items()}
        st.sidebar.success("✅ Text model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Text: {e}")
        models['roberta'] = None

    return models


def video_to_wav(video_path, target_sr=16000):
    wav_path = video_path + "_converted.wav"
    subprocess.run([
        'ffmpeg', '-i', video_path, '-vn', '-ac', '1',
        '-ar', str(target_sr), wav_path, '-y', '-loglevel', 'quiet'],
        capture_output=True)
    return wav_path if os.path.exists(wav_path) else None


# ═══════════════════════════════════════════════════════
# PRIORITY 1 — AUDIO MODULE REWRITE
#
# Replaces RMS/ZCR proxies with:
#  - WebRTC VAD for real voiced/silence segmentation
#  - Pause ratio, average pause duration, silence %
#  - Praat/Parselmouth for pitch (more robust than
#    librosa.piptrack on conversational speech)
#  - Loudness variation from RMS envelope
#  - CNN emotion model used ONLY as a minor signal,
#    never displayed as a label
# ═══════════════════════════════════════════════════════

def run_vad(y, sr, frame_ms=30, energy_percentile=30):
    """
    Energy-threshold VAD (no external C-extension needed).
    Splits audio into fixed frames and marks each as
    voiced/unvoiced based on RMS energy relative to the
    signal's own energy distribution. Less precise than
    WebRTC's algorithm, but avoids the Visual C++ Build
    Tools dependency, and is adequate for pause-ratio /
    silence-percentage estimation in this context.
    """
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0 or len(y) < frame_len:
        return [], frame_ms / 1000

    n_frames = len(y) // frame_len
    frame_energies = []
    for i in range(n_frames):
        frame = y[i*frame_len:(i+1)*frame_len]
        frame_energies.append(float(np.sqrt(np.mean(frame**2))))

    frame_energies = np.array(frame_energies)
    if len(frame_energies) == 0:
        return [], frame_ms / 1000

    # Threshold set relative to this clip's own energy
    # distribution, since absolute RMS varies a lot by
    # mic/recording setup
    threshold = np.percentile(frame_energies, energy_percentile)
    threshold = max(threshold, 1e-4)  # floor to avoid pure-silence clips
    voiced_flags = [bool(e > threshold) for e in frame_energies]

    return voiced_flags, frame_ms / 1000


def compute_pause_metrics(voiced_flags, frame_dur):
    """
    From a voiced/unvoiced frame sequence, computes:
      - silence_pct: % of total time that is non-speech
      - pause_count: number of distinct silence gaps
        between speech segments (not leading/trailing silence)
      - avg_pause_duration: mean length of those gaps (sec)
      - pause_ratio: pause time / total speaking+pause time
    """
    if not voiced_flags:
        return {'silence_pct': 0, 'pause_count': 0,
                'avg_pause_duration': 0, 'pause_ratio': 0}

    total_frames = len(voiced_flags)
    silent_frames = sum(1 for v in voiced_flags if not v)
    silence_pct = round(100 * silent_frames / total_frames, 1)

    # Find pauses that occur BETWEEN speech (not at the very
    # start/end, which is just lead-in/lead-out silence)
    first_voice = next((i for i, v in enumerate(voiced_flags) if v), None)
    last_voice = next(
        (i for i in range(total_frames - 1, -1, -1)
         if voiced_flags[i]), None)

    pauses = []
    if first_voice is not None and last_voice is not None:
        current_pause = 0
        for i in range(first_voice, last_voice + 1):
            if not voiced_flags[i]:
                current_pause += 1
            else:
                if current_pause >= 3:  # ignore tiny gaps (<90ms)
                    pauses.append(current_pause * frame_dur)
                current_pause = 0

    pause_count = len(pauses)
    avg_pause_duration = round(
        float(np.mean(pauses)), 2) if pauses else 0.0
    speaking_frames = total_frames - silent_frames
    pause_ratio = round(
        silent_frames / max(total_frames, 1), 3)

    return {
        'silence_pct'        : silence_pct,
        'pause_count'        : pause_count,
        'avg_pause_duration' : avg_pause_duration,
        'pause_ratio'        : pause_ratio}


def estimate_pitch_praat(wav_path):
    """
    Uses Parselmouth (Praat) for pitch tracking — more
    robust on natural conversational speech than
    librosa.piptrack, which is tuned for musical pitch.
    """
    try:
        import parselmouth
        snd = parselmouth.Sound(wav_path)
        pitch = snd.to_pitch()
        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # remove unvoiced

        if len(pitch_values) < 10:
            return {'pitch_mean': 0, 'pitch_std': 0,
                     'pitch_stability': 50.0}

        pitch_mean = float(np.mean(pitch_values))
        pitch_std = float(np.std(pitch_values))
        pitch_stability = max(0, min(100,
            100 * (1 - pitch_std / (pitch_mean + 1e-6))))

        return {
            'pitch_mean': round(pitch_mean, 1),
            'pitch_std' : round(pitch_std, 1),
            'pitch_stability': round(pitch_stability, 1)}
    except Exception as e:
        return {'pitch_mean': 0, 'pitch_std': 0,
                 'pitch_stability': 50.0, 'error': str(e)}


def analyze_audio_from_video(video_path, audio_model,
                               transcript_word_count=None,
                               min_confidence=0.65,
                               decisive_margin=0.25):
    """
    Returns interview-relevant vocal metrics. No emotion
    label is ever surfaced — only: Voice Stability,
    Speaking Pace, Pause Control, Confidence (acoustic),
    Vocal Tone (calm/energetic/nervous — derived from
    stability+loudness, gated by CNN only as a minor signal).
    """
    import librosa

    wav_path = video_to_wav(video_path)
    if wav_path is None:
        return None

    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        duration_sec = len(y) / sr

        if duration_sec < 1.0:
            os.unlink(wav_path)
            return None

        # ── Voice Activity Detection ────────────────
        voiced_flags, frame_dur = run_vad(y, sr)
        pause_metrics = compute_pause_metrics(voiced_flags, frame_dur)

        # ── Pitch via Praat ──────────────────────────
        pitch_metrics = estimate_pitch_praat(wav_path)

        # ── Loudness variation (RMS envelope) ───────
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        loudness_variation = round(
            100 * rms_std / (rms_mean + 1e-6), 1)
        energy_stability = max(0, min(100,
            100 * (1 - rms_std / (rms_mean + 1e-6))))

        # ── Speaking rate from Whisper word count ───
        speaking_sec = duration_sec * (1 - pause_metrics['pause_ratio'])
        if transcript_word_count and speaking_sec > 0:
            wpm = round(transcript_word_count / (speaking_sec / 60), 0)
        else:
            wpm = None

        if wpm is None:
            pace_label = "Unknown"
        elif wpm < 100:
            pace_label = "Slow"
        elif wpm > 180:
            pace_label = "Fast"
        else:
            pace_label = "Good pace"

        # ── Gated CNN check (minor signal only) ─────
        cnn_label, cnn_decisive = None, False
        if audio_model is not None:
            from PIL import Image as PILImage
            input_shape = audio_model.input_shape
            img_h, img_w = input_shape[1], input_shape[2]
            n_channels = input_shape[-1]
            win_len = sr * 3
            all_predictions = []

            for start in range(0, len(y) - win_len, win_len):
                chunk = y[start:start + win_len]
                mel = librosa.feature.melspectrogram(
                    y=chunk, sr=sr, n_mels=img_h,
                    n_fft=2048, hop_length=512)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                mel_img = np.array(
                    PILImage.fromarray(mel_db).resize((img_w, img_h)))
                mn, mx = mel_img.min(), mel_img.max()
                if mx - mn < 1e-8:
                    continue
                mel_norm = (mel_img - mn) / (mx - mn)
                arr = (np.repeat(mel_norm[..., None], 3, axis=-1)
                       if n_channels == 3 else mel_norm[..., None])
                arr = np.expand_dims(arr.astype(np.float32), 0)
                probs = audio_model.predict(arr, verbose=0)[0]
                pred = int(np.argmax(probs))
                conf = float(probs[pred])
                all_predictions.append((pred, conf, probs))

            if all_predictions:
                confident = [p for p in all_predictions
                             if p[1] >= min_confidence]
                usable = confident if confident else all_predictions
                vote_counts = Counter(p[0] for p in usable)
                ranked = vote_counts.most_common()
                winning_class = ranked[0][0]
                winning_votes = ranked[0][1]
                runner_up = ranked[1][1] if len(ranked) > 1 else 0
                margin = (winning_votes - runner_up) / len(usable)
                cnn_decisive = (margin >= decisive_margin and
                                 len(confident) >= 3)
                if cnn_decisive:
                    cnn_label = AUDIO_EMOTIONS.get(
                        str(winning_class), 'neutral')

        os.unlink(wav_path)

        # ── Derive interview-facing labels ──────────
        pitch_stability = pitch_metrics['pitch_stability']

        if energy_stability > 65 and pitch_stability > 55:
            vocal_tone = "Calm and steady"
        elif energy_stability > 50 and pause_metrics['pause_ratio'] < 0.35:
            vocal_tone = "Confident"
        elif energy_stability < 35 or pause_metrics['pause_count'] > 8:
            vocal_tone = "Slightly nervous"
        else:
            vocal_tone = "Neutral"

        # CNN only nudges, never overrides acoustic evidence
        if cnn_decisive and cnn_label in ('fearful', 'sad') and \
                energy_stability < 50:
            vocal_tone = "Slightly nervous"
        elif cnn_decisive and cnn_label == 'happy' and \
                energy_stability > 55:
            vocal_tone = "Energetic and confident"

        if pause_metrics['pause_count'] == 0:
            pause_control = "Excellent — no awkward pauses"
        elif pause_metrics['avg_pause_duration'] < 1.0:
            pause_control = "Good — brief natural pauses"
        elif pause_metrics['avg_pause_duration'] < 2.5:
            pause_control = "Fair — some hesitation"
        else:
            pause_control = "Needs work — long pauses detected"

        voice_stability = round(
            (energy_stability + pitch_stability) / 2, 1)

        acoustic_confidence = max(0.4, min(0.95,
            1 - abs(energy_stability - pitch_stability) / 200))

        return {
            'vocal_tone'        : vocal_tone,
            'voice_stability'   : voice_stability,
            'energy_stability'  : round(energy_stability, 1),
            'pitch_stability'   : pitch_stability,
            'pitch_mean'        : pitch_metrics['pitch_mean'],
            'loudness_variation': loudness_variation,
            'pace_label'        : pace_label,
            'words_per_minute'  : wpm,
            'silence_pct'       : pause_metrics['silence_pct'],
            'pause_count'       : pause_metrics['pause_count'],
            'avg_pause_duration': pause_metrics['avg_pause_duration'],
            'pause_ratio'       : pause_metrics['pause_ratio'],
            'pause_control'     : pause_control,
            'confidence'        : acoustic_confidence,
            'probs'             : [0.14]*6,
            'dim'               : (
                1 if vocal_tone in ("Confident","Energetic and confident")
                else 2 if vocal_tone == "Slightly nervous" else 4),
            'cnn_label'         : cnn_label,
            'cnn_decisive'      : cnn_decisive}

    except Exception as e:
        if os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except Exception:
                pass
        st.warning(f"Audio analysis error: {e}")
        return None


# ═══════════════════════════════════════════════════════
# FACE ANALYSIS — "Camera Presence" naming, same
# majority-vote gating as before (this part was already
# validated against your debug logs)
# ═══════════════════════════════════════════════════════

def analyze_face_from_video(video_path, face_model,
                              sample_every=15,
                              min_confidence=0.65,
                              decisive_margin=0.20):
    import cv2

    if face_model is None:
        return None

    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            'haarcascade_frontalface_default.xml')

        input_shape = face_model.input_shape
        img_size, n_channels = input_shape[1], input_shape[-1]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        all_predictions = []
        face_found_count = 0
        smile_frames = 0
        frame_idx = 0
        total_sampled = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                total_sampled += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5,
                    minSize=(60, 60))
                if len(faces) > 0:
                    face_found_count += 1
                    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                    crop = frame[y:y+h, x:x+w]
                    resized = cv2.resize(crop, (img_size, img_size))
                    if n_channels == 3:
                        img_arr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    else:
                        img_arr = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                        img_arr = np.expand_dims(img_arr, -1)
                    arr = np.expand_dims(img_arr.astype(np.float32)/255.0, 0)
                    probs = face_model.predict(arr, verbose=0)[0]
                    pred = int(np.argmax(probs))
                    conf = float(probs[pred])
                    all_predictions.append((pred, conf, probs))
                    if pred == 3:
                        smile_frames += 1
            frame_idx += 1
        cap.release()

        if not all_predictions:
            return None

        camera_presence_pct = round(
            100 * face_found_count / max(total_sampled, 1), 1)
        smile_pct = round(
            100 * smile_frames / max(len(all_predictions), 1), 1)

        confident = [p for p in all_predictions if p[1] >= min_confidence]
        usable = confident if confident else all_predictions
        vote_counts = Counter(p[0] for p in usable)
        ranked = vote_counts.most_common()
        winning_class = ranked[0][0]
        winning_votes = ranked[0][1]
        runner_up_votes = ranked[1][1] if len(ranked) > 1 else 0
        margin = (winning_votes - runner_up_votes) / len(usable)
        is_decisive = margin >= decisive_margin

        matching = [p[2] for p in usable if p[0] == winning_class]
        avg_probs = np.mean(matching, axis=0)
        consistency = round(100 * winning_votes / len(usable), 1)

        raw_emotion = FACE_EMOTIONS.get(winning_class, 'neutral')
        if is_decisive and raw_emotion == 'happy':
            display_label = "Positive / engaged"
        elif is_decisive and raw_emotion == 'surprise':
            display_label = "Alert / responsive"
        else:
            display_label = "Neutral / composed"

        return {
            'display_label'      : display_label,
            'raw_emotion'        : raw_emotion,
            'is_decisive'        : is_decisive,
            'confidence'         : float(avg_probs[winning_class]),
            'camera_presence_pct': camera_presence_pct,
            'smile_pct'          : smile_pct,
            'consistency'        : consistency,
            'probs'              : avg_probs.tolist(),
            'dim'                : FACE_TO_DIM.get(
                raw_emotion if is_decisive else 'neutral', 4),
            'frames_analyzed'    : len(all_predictions),
            'frames_used'        : len(usable)}

    except Exception as e:
        st.warning(f"Face analysis error: {e}")
        return None


# ═══════════════════════════════════════════════════════
# SPEECH / TRANSCRIPT METRICS
# ═══════════════════════════════════════════════════════

def analyze_speech_quality(text):
    if not text:
        return {}
    words = text.lower().split()
    total = len(words)
    if total == 0:
        return {}

    fillers = sum(
        len(re.findall(r'\b' + re.escape(f) + r'\b', text.lower()))
        for f in FILLER_WORDS)
    ttr = len(set(words)) / total
    sents = [s for s in re.split(r'[.!?]+', text.strip())
             if len(s.strip()) > 0]
    avg_sentence_len = round(total / max(len(sents), 1), 1)

    word_counts = Counter(words)
    repeated_words = [w for w, c in word_counts.items()
                        if c > 3 and len(w) > 3]
    repetition_pct = round(
        sum(word_counts[w] for w in repeated_words) /
        max(total, 1) * 100, 1)

    # Rough grammar proxy: sentence-ending punctuation
    # present, no run-on sentences over 40 words
    long_sentences = sum(1 for s in sents if len(s.split()) > 40)
    grammar_score = max(0, min(100,
        100 - long_sentences * 10 - fillers/total*150))

    return {
        'total_words'        : total,
        'filler_count'       : fillers,
        'filler_ratio'       : round(fillers/total, 3),
        'vocabulary_richness': round(ttr, 3),
        'avg_sentence_length': avg_sentence_len,
        'repetition_pct'     : repetition_pct,
        'repeated_words'     : repeated_words[:5],
        'grammar_score'      : round(grammar_score, 1),
        'clarity_score'      : round(
            max(0, min(100, (1 - fillers/total*3)*100)), 1),
        'fluency_score'      : round(
            min(100, ttr*70 + min(avg_sentence_len/20, 1)*30), 1)}


def compute_star_completeness(text):

    text = text.lower()

    situation = any(x in text for x in [

        "when i",

        "during",

        "at that time",

        "in my project",

        "in college",

        "while working",

        "my internship",

        "our team"

    ])

    task = any(x in text for x in [

        "my task",

        "my responsibility",

        "i had to",

        "i needed to",

        "my goal",

        "objective",

        "assigned"

    ])

    action = any(x in text for x in [

        "i developed",

        "i created",

        "i built",

        "i designed",

        "i implemented",

        "i solved",

        "i worked",

        "i used",

        "i decided"

    ])

    result = any(x in text for x in [

        "as a result",

        "finally",

        "successfully",

        "improved",

        "increased",

        "reduced",

        "completed",

        "achieved",

        "therefore",

        "the result"

    ])

    components = {

        "situation": situation,

        "task": task,

        "action": action,

        "result": result

    }

    completeness = sum(components.values()) * 25

    return {

        "completeness": completeness,

        "components_found": components

    }


def predict_text_emotion(text, models):
    if not models.get('roberta'):
        return {'emotion':'positive','confidence':0.5,
                 'probs':[0.1,0.1,0.7,0.1],'dim':1}
    try:
        inp = models['roberta_tok'](
            text, return_tensors='pt', truncation=True,
            max_length=256, padding=True)
        with torch.no_grad():
            out = models['roberta'](**inp)
            probs = torch.softmax(out.logits, dim=1).numpy()[0]
        pred = np.argmax(probs)
        em = models['id2emotion'][pred]
        return {'emotion': em, 'confidence': float(probs[pred]),
                 'probs': probs.tolist(), 'dim': TEXT_TO_DIM.get(em, 4)}
    except Exception:
        return {'emotion':'positive','confidence':0.5,
                 'probs':[0.1,0.1,0.7,0.1],'dim':1}


def compute_keyword_score(text):
    import re

    text = text.lower()

    categories = {
        "technical": [
            r"\bpython\b", r"\bjava\b", r"\bc\+\+\b", r"\bc\b",
            r"\bmachine learning\b", r"\bdeep learning\b",
            r"\bai\b", r"\bapi\b", r"\bdatabase\b",
            r"\bsql\b", r"\bmodel\b", r"\bproject\b",
            r"\bsoftware\b", r"\bapplication\b",
            r"\bdevelop(ed|ing)?\b",
            r"\bcreat(ed|ing)?\b",
            r"\bbuild(s|ing|t)?\b",
            r"\bdesign(ed|ing)?\b"
        ],

        "problem_solving": [
            r"\bsolv(ed|e|ing)?\b",
            r"\bfix(ed|ing)?\b",
            r"\bdebug(ged|ging)?\b",
            r"\bimprov(ed|ing)?\b",
            r"\boptimiz(ed|ing)?\b",
            r"\bimplement(ed|ing)?\b",
            r"\bdevelop(ed|ing)?\b",
            r"\bdesign(ed|ing)?\b",
            r"\bcreat(ed|ing)?\b",
            r"\bbuild(s|ing|t)?\b",
            r"\bchallenge\b",
            r"\bissue\b",
            r"\bproblem\b",
            r"\bsolution\b",
            r"\bovercame\b",
            r"\bhandled\b"
        ],

        "communication": [
            r"\bexplain(ed|ing)?\b",
            r"\bpresent(ed|ing)?\b",
            r"\bcommunicat(ed|ing)?\b",
            r"\bcollaborat(ed|ing)?\b",
            r"\bdiscuss(ed|ing)?\b",
            r"\bteam\b",
            r"\bclient\b",
            r"\bmentor\b",
            r"\bmeeting\b",
            r"\bshared\b",
            r"\breported\b",
            r"\bworked with\b",
            r"\binteraction\b"
        ]
    }

    scores = {}

    for category, patterns in categories.items():

        found = []

        for pattern in patterns:

            if re.search(pattern, text):
                found.append(pattern)

        score = min(100, (len(found) / len(patterns)) * 100)

        scores[category] = {
            "score": score,
            "found": found
        }

    scores["overall"] = round(
        sum(v["score"] for v in scores.values()) /
        len(scores), 1
    )

    return scores


# ═══════════════════════════════════════════════════════
# PRIORITY — CONFIDENCE-WEIGHTED FUSION
#
# Instead of fixed text=50/face=35/audio=15, each
# modality's contribution is weighted by its OWN
# confidence reading for this specific sample, then
# normalized. A modality that's uncertain on this clip
# contributes less, regardless of its baseline reliability.
# Base reliability priors still anchor the weighting so a
# very confident audio reading can't outweigh text outright.
# ═══════════════════════════════════════════════════════

def confidence_weighted_fusion(text_r, face_r, audio_r, sq, star, kw):
    # Base reliability priors from validated model accuracy
    BASE_TEXT, BASE_FACE, BASE_AUDIO = 0.60, 0.30, 0.10

    text_conf = text_r['confidence']
    face_conf = face_r['confidence'] if face_r['is_decisive'] else 0.4
    audio_conf = audio_r['confidence']

    # Effective weight = base prior * this-sample confidence
    w_text = BASE_TEXT * text_conf
    w_face = BASE_FACE * face_conf
    w_audio = BASE_AUDIO * audio_conf
    total_w = w_text + w_face + w_audio + 1e-6
    w_text, w_face, w_audio = (
        w_text/total_w, w_face/total_w, w_audio/total_w)

    text_score = (
        text_r['confidence'] * 100 if text_r['emotion'] == 'positive'
        else text_r['confidence'] * 60 if text_r['emotion'] == 'surprised'
        else 100 - text_r['confidence'] * 70)

    face_score = (
        85 if face_r['display_label'] == "Positive / engaged" else
        75 if face_r['display_label'] == "Alert / responsive" else 60)
    face_score = min(100,
        face_score + (face_r['camera_presence_pct'] - 50) * 0.3)

    audio_score = (
        85 if audio_r['vocal_tone'] in
            ("Energetic and confident", "Confident") else
        75 if audio_r['vocal_tone'] == "Calm and steady" else
        45 if audio_r['vocal_tone'] == "Slightly nervous" else 65)
    if audio_r['pause_control'].startswith("Needs"):
        audio_score -= 10
    elif audio_r['pause_control'].startswith("Excellent"):
        audio_score += 5
    audio_score = max(0, min(100, audio_score))

    overall = round(
        text_score * w_text + face_score * w_face +
        audio_score * w_audio, 1)

    confidence_score = round(face_score, 1)
    clarity_score = round(
        sq.get('clarity_score', overall)*0.5 +
        sq.get('grammar_score', overall)*0.3 +
        sq.get('fluency_score', overall)*0.2, 1) if sq else overall
    answer_quality = round(min(100,
        kw.get('overall', 30)*0.4 + star['completeness']*0.4 +
        text_score*0.2), 1)
    professionalism = round(min(100,
        (100 - sq.get('repetition_pct', 0)*2) * 0.4 +
        sq.get('grammar_score', 70) * 0.3 +
        clarity_score * 0.3), 1)
    technical_relevance = round(kw.get('technical', {}).get('score', 0), 1)

    return {
        'overall'             : overall,
        'confidence'          : confidence_score,
        'clarity'             : clarity_score,
        'answer_quality'      : answer_quality,
        'professionalism'     : professionalism,
        'technical_relevance' : technical_relevance,
        'weights_used'        : {
            'text': round(w_text, 2), 'face': round(w_face, 2),
            'audio': round(w_audio, 2)}}


def get_recommendation(scores, star_completeness):
    overall = scores['overall']
    if overall >= 82 and star_completeness >= 50:
        return "Strong Hire", "#27ae60"
    elif overall >= 70:
        return "Hire", "#3498db"
    elif overall >= 50:
        return "Consider", "#f39c12"
    else:
        return "Needs Improvement", "#e74c3c"


# ═══════════════════════════════════════════════════════
# TEMPLATE-DRIVEN FEEDBACK SYNTHESIS
# (more granular than before — pulls specific numbers
# from transcript/speech/face data into each point rather
# than generic statements, to read closer to AI-generated
# feedback without requiring an external API)
# ═══════════════════════════════════════════════════════

def generate_feedback(scores, face_r, audio_r, text_r, sq, star, kw):
    overall = scores['overall']
    strengths, improvements = [], []

    if scores['confidence'] > 70:
        strengths.append(
            f"Composed on-camera presence "
            f"({face_r['camera_presence_pct']:.0f}% of frames "
            f"showed clear face visibility)")
    if scores['clarity'] > 70:
        strengths.append(
            f"Clear communication with "
            f"{sq.get('avg_sentence_length',0):.0f} words per "
            f"sentence on average — well-structured delivery")
    if star['completeness'] >= 75:
        strengths.append(
            "Strong STAR-method structure — covered "
            f"{sum(star['components_found'].values())}/4 components")
    if audio_r['vocal_tone'] in ("Calm and steady", "Confident",
                                   "Energetic and confident"):
        strengths.append(
            f"Vocal delivery came across as "
            f"{audio_r['vocal_tone'].lower()}, with "
            f"{audio_r['voice_stability']:.0f}% voice stability")
    if audio_r['pause_control'].startswith(("Excellent", "Good")):
        strengths.append(
            f"Good pause control — {audio_r['pause_count']} "
            f"natural pauses, no awkward silences")
    if kw.get('technical', {}).get('score', 0) > 50:
        found = kw['technical']['found']
        strengths.append(
            f"Demonstrated technical vocabulary: "
            f"{', '.join(found[:3])}")
    if not strengths:
        strengths = ["Shows genuine engagement with the question",
                       "Completed a full response without trailing off"]

    if scores['confidence'] < 60:
        improvements.append(
            f"On-camera presence was only "
            f"{face_r['camera_presence_pct']:.0f}% — try to keep "
            "your face clearly in frame and centered")
    if sq.get('filler_ratio', 0) > 0.06:
        improvements.append(
            f"Used filler words {sq.get('filler_count',0)} times — "
            "try pausing silently instead of saying 'um' or 'like'")
    if star['completeness'] < 50:
        missing = [k for k, v in star['components_found'].items() if not v]
        improvements.append(
            f"Answer structure was missing: {', '.join(missing)} — "
            "use the STAR method for clearer structure")
    if audio_r['vocal_tone'] == "Slightly nervous":
        improvements.append(
            "Vocal energy suggested some nervousness — practice "
            "deep breathing before answering to steady your voice")
    if audio_r['pause_control'].startswith("Needs"):
        improvements.append(
            f"Average pause length was "
            f"{audio_r['avg_pause_duration']:.1f}s — practice "
            "answers aloud to reduce long hesitations")
    if sq.get('repetition_pct', 0) > 15:
        improvements.append(
            f"Noticed repeated words ({', '.join(sq.get('repeated_words', [])[:3])}) "
            "— vary your vocabulary for stronger impact")
    if audio_r.get('words_per_minute') and (
            audio_r['words_per_minute'] < 100 or
            audio_r['words_per_minute'] > 180):
        improvements.append(
            f"Speaking pace was {audio_r['pace_label'].lower()} "
            f"({audio_r['words_per_minute']:.0f} words/min) — "
            "aim for 120-160 wpm for clarity")
    if not improvements:
        improvements = ["Continue refining answer specificity",
                          "Practice more mock interviews"]

    if overall >= 80:
        tips = ["HIGH: Research company culture deeply",
                "MED: Prepare 5 specific project examples",
                "LOW: Practice salary negotiation"]
    elif overall >= 60:
        tips = ["HIGH: Practice STAR method daily",
                "HIGH: Record yourself and review pacing",
                "MED: Expand technical vocabulary"]
    elif overall >= 40:
        tips = ["HIGH: Daily mock interview practice",
                "HIGH: Work on answer structure (STAR)",
                "MED: Improve vocabulary range"]
    else:
        tips = ["HIGH: Daily mirror practice",
                "HIGH: Study and apply STAR method",
                "HIGH: Work on building vocal confidence"]

    return {'strengths': strengths[:4],
             'improvements': improvements[:4], 'tips': tips}


def create_radar_chart(scores):
    categories = ['Communication', 'Confidence', 'Professionalism',
                   'Technical', 'Answer Quality']
    values = [scores['clarity'], scores['confidence'],
               scores['professionalism'], scores['technical_relevance'],
               scores['answer_quality']]
    vc = values + [values[0]]
    cc = categories + [categories[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vc, theta=cc, fill='toself',
        fillcolor='rgba(102,126,234,0.2)',
        line=dict(color='#667eea', width=2)))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=350,
        margin=dict(t=30, b=30, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)')
    return fig


# ═══════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════

def main():
    st.markdown('<div class="main-header">🎯 AI Interview Analyzer</div>',
                 unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#666;'>"
        "Interview evaluation platform — communication, "
        "delivery, and answer quality</p>",
        unsafe_allow_html=True)
    st.divider()

    st.sidebar.title("⚙️ Settings")
    st.sidebar.subheader("Model Status")
    with st.spinner("Loading AI models..."):
        models = load_models()

    st.sidebar.divider()
    job_title = st.sidebar.text_input(
        "🎯 Target Job Role",
        placeholder="e.g. Software Engineer, Data Scientist...",
        value="")

    st.sidebar.divider()
    st.sidebar.info(
        "📌 Upload a video of only the candidate "
        "answering questions for best results.")

    st.sidebar.divider()
    st.sidebar.markdown(
        "**📊 Evaluation dimensions:**\n"
        "- 💬 Communication & answer structure\n"
        "- 🎤 Voice stability, pace, pause control\n"
        "- 👁️ Camera presence\n"
        "- 🔑 Technical keyword relevance\n"
        "- ⭐ STAR method completeness\n\n"
        )

    tab1, tab2 = st.tabs(["📹 Upload Video/Audio", "✏️ Paste Transcript"])

    transcript = ""
    tmp_path = None
    face_r, audio_r = None, None
    is_video = False

    with tab1:
        st.markdown(
            "**Upload candidate video**")
        uploaded = st.file_uploader(
            "Choose video or audio file",
            type=['mp4','mov','avi','wav','mp3','m4a'],
            help="Max 200MB.")

        if uploaded:
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            is_video = uploaded.type.startswith('video')
            if is_video:
                st.video(uploaded)
            else:
                st.audio(uploaded)

            if st.button("🎯 Analyze Interview",
                          type="primary", key="btn_video"):
                progress = st.progress(0)
                status = st.empty()

                status.info("📝 Transcribing audio...")
                progress.progress(10)
                try:
                    import whisper
                    wm = whisper.load_model("tiny")
                    result = wm.transcribe(
                        tmp_path, language='en', fp16=False)
                    transcript = result['text'].strip()
                    transcript = transcript.lower()
                    transcript = transcript.replace("'ve", " have")
                    transcript = transcript.replace("'re", " are")
                    transcript = transcript.replace("'m", " am")
                    progress.progress(35)
                    st.success("✅ Transcription complete!")
                    st.text_area("📝 Transcript", transcript, height=120)
                except Exception as e:
                    st.error(f"Transcription error: {e}")
                    progress.progress(35)

                if is_video and models.get('face'):
                    status.info("👁️ Analyzing camera presence...")
                    progress.progress(50)
                    face_r = analyze_face_from_video(tmp_path, models['face'])
                    if face_r:
                        st.caption(
                            f"👁️ Camera presence: "
                            f"{face_r['camera_presence_pct']:.0f}% of "
                            f"frames | Smile frequency: "
                            f"{face_r['smile_pct']:.0f}% | "
                            f"Consistency: {face_r['consistency']:.0f}%")
                    else:
                        st.caption("👁️ No clear face detected in video")

                status.info("🎤 Analyzing voice (VAD, pitch, pace)...")
                progress.progress(70)
                word_count = len(transcript.split()) if transcript else None
                audio_r = analyze_audio_from_video(
                    tmp_path, models.get('audio'),
                    transcript_word_count=word_count)
                if audio_r:
                    st.caption(
                        f"🎤 Vocal tone: **{audio_r['vocal_tone']}** | "
                        f"Stability: {audio_r['voice_stability']:.0f}% | "
                        f"Pace: {audio_r['pace_label']} | "
                        f"Pauses: {audio_r['pause_count']}")

                progress.progress(100)
                status.empty()
                progress.empty()

    with tab2:
        st.markdown("Paste **your interview answers** for text-based analysis only.")
        transcript_input = st.text_area(
            "Paste your answers here",
            placeholder="In my previous role I led a team of 5 engineers to solve...",
            height=200)
        if st.button("🎯 Analyze Text", type="primary", key="btn_text"):
            transcript = transcript_input

    # ═══════════════════════════════════════
    if transcript and len(transcript) > 20:
        st.divider()
        st.subheader("📊 Interview Evaluation Report")

        with st.spinner("Running analysis..."):
            sq = analyze_speech_quality(transcript)
            star = compute_star_completeness(transcript)
            text_r = predict_text_emotion(transcript, models)
            kw = compute_keyword_score(transcript)

            if face_r is None:
                face_r = {'display_label':'Neutral / composed',
                           'raw_emotion':'neutral','is_decisive':False,
                           'confidence':0.5,'camera_presence_pct':0,
                           'smile_pct':0,'consistency':0,
                           'probs':[0.14]*7,'dim':4,
                           'frames_analyzed':0,'frames_used':0}
            if audio_r is None:
                audio_r = {
        'vocal_tone':'N/A',
        'voice_stability':None,
        'energy_stability':None,
        'pitch_stability':None,
        'pitch_mean':None,
        'loudness_variation':None,
        'pace_label':'N/A',
        'words_per_minute':None,
        'silence_pct':None,
        'pause_count':None,
        'avg_pause_duration':None,
        'pause_ratio':None,
        'pause_control':'N/A',
        'confidence':0.5,
        'probs':[0.14]*6,
        'dim':4,
        'cnn_label':None,
        'cnn_decisive':False
    }

            scores = confidence_weighted_fusion(
                text_r, face_r, audio_r, sq, star, kw)
            fb = generate_feedback(
                scores, face_r, audio_r, text_r, sq, star, kw)
            recommendation, rec_color = get_recommendation(
                scores, star['completeness'])

        # ── Headline metrics ──────────────
        s1, s2, s3, s4 = st.columns(4)
        with s1: st.metric("⏱️ Words", sq.get('total_words', 0))
        with s2: st.metric("🎯 Score", f"{scores['overall']}/100")
        with s3: st.metric("⭐ STAR", f"{star['completeness']:.0f}%")
        with s4: st.metric("🔑 Tech Match",
                              f"{scores['technical_relevance']:.0f}%")
        st.divider()

        # ── Recommendation badge ──────────
        st.markdown(
            f'<div style="text-align:center;margin:1rem 0">'
            f'<span class="rec-badge" style="background:{rec_color}">'
            f'Recommendation: {recommendation}</span></div>',
            unsafe_allow_html=True)
        st.caption(
            f"Fusion weights for this clip — "
            f"Text: {scores['weights_used']['text']*100:.0f}%, "
            f"Face: {scores['weights_used']['face']*100:.0f}%, "
            f"Audio: {scores['weights_used']['audio']*100:.0f}% "
            f"(scaled by each model's confidence on this sample)")
        st.divider()

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                f'<div class="score-card">'
                f'<div class="score-number" style="color:{rec_color}">'
                f'{scores["overall"]}</div>'
                f'<div style="color:#666;font-size:0.9rem">'
                f'Interview Score / 100</div></div>',
                unsafe_allow_html=True)
            for name, val in [
                ("🎯 Confidence", scores['confidence']),
                ("💬 Communication", scores['clarity']),
                ("👔 Professionalism", scores['professionalism']),
                ("🔧 Technical Relevance", scores['technical_relevance']),
                ("⭐ Answer Quality", scores['answer_quality'])]:
                ca, cb = st.columns([2,1])
                with ca:
                    st.markdown(f"**{name}**")
                    st.progress(int(val))
                with cb:
                    st.markdown(f"**{val:.0f}**")
        with col2:
            st.plotly_chart(create_radar_chart(scores), use_container_width=True)

        st.divider()

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("👁️ Camera Presence")
            st.markdown(
                f'<div class="trait-card"><b>Impression:</b> '
                f'{face_r["display_label"]}</div>',
                unsafe_allow_html=True)
            st.progress(int(face_r['camera_presence_pct']),
                         text=f"In frame: {face_r['camera_presence_pct']:.0f}%")
            st.progress(int(face_r['smile_pct']),
                         text=f"Smile frequency: {face_r['smile_pct']:.0f}%")
            st.progress(int(face_r['consistency']),
                         text=f"Consistency: {face_r['consistency']:.0f}%")

        with col4:
            st.subheader("🎤 Voice Analysis")

            st.markdown(f'<div class="trait-card"><b>Vocal tone:</b> {audio_r["vocal_tone"]}</div>',
        unsafe_allow_html=True)

            if audio_r["voice_stability"] is not None:
                st.progress(
        int(audio_r["voice_stability"]),
        text=f"Voice stability: {audio_r['voice_stability']:.0f}%"
    )

                if audio_r.get("words_per_minute"):
                    st.markdown(
            f"**Pace:** {audio_r['pace_label']} "
            f"({audio_r['words_per_minute']:.0f} wpm)"
        )

                st.markdown(f"**Pause control:** {audio_r['pause_control']}")

                st.caption(
        f"{audio_r['pause_count']} pauses | "
        f"{audio_r['silence_pct']:.0f}% silence"
    )

            else:
                st.write("**Voice Stability:** N/A")
                st.write("**Pause Control:** N/A")
                st.write("**Pace:** N/A")

        st.divider()

        st.subheader("🔑 Technical & Skill Assessment")

        insights = {
        "technical":
        "Demonstrates technical knowledge through programming languages, tools and project experience.",

        "problem_solving":
        "Shows practical problem-solving ability by describing implementation, development and technical challenges.",

        "communication":
        "Communicates ideas clearly and demonstrates teamwork and project discussion skills."
        }

        for cat, data in kw.items():
          
          if cat == "overall":
           continue

          score = data["score"]

          color = (
              "#27ae60" if score >= 70
              else "#f39c12" if score >= 40
              else "#e74c3c"
          )

          st.markdown(
        f"**{cat.replace('_',' ').title()}**: "
        f'<span style="color:{color}">{score:.0f}%</span>',
        unsafe_allow_html=True
          )

          st.progress(int(score))

          st.info(insights.get(cat, "Good overall performance."))

          st.markdown("")

        st.divider()

        st.subheader("⭐ STAR Method Coverage")
        star_cols = st.columns(4)
        for col, (comp, present) in zip(star_cols, star['components_found'].items()):
            with col:
                icon = "✅" if present else "❌"
                st.markdown(f"{icon} **{comp.title()}**")

        st.divider()

        st.subheader("💬 Feedback")
        c5, c6, c7 = st.columns(3)
        with c5:
            st.markdown("### ✅ Strengths")
            for s in fb['strengths']:
                st.markdown(
                    f'<div class="feedback-card"><span class="strength-item">'
                    f'✅ {s}</span></div>', unsafe_allow_html=True)
        with c6:
            st.markdown("### ⚠️ Improve")
            for im in fb['improvements']:
                st.markdown(
                    f'<div class="feedback-card"><span class="improve-item">'
                    f'⚠️ {im}</span></div>', unsafe_allow_html=True)
        with c7:
            st.markdown("### 💡 Action Tips")
            for t in fb['tips']:
                pr = t.split(':')[0]
                css = {'HIGH':'tip-high','MED':'tip-med','LOW':'tip-low'}.get(pr,'tip-med')
                st.markdown(
                    f'<div class="feedback-card"><span class="{css}">'
                    f'💡 {t}</span></div>', unsafe_allow_html=True)

        st.divider()

        st.subheader("📝 Speech Quality Detail")
        c8, c9 = st.columns(2)
        with c8:
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Word Count", sq.get('total_words', 0))
                st.metric("Filler Words", sq.get('filler_count', 0))
                st.metric("Avg Sentence Length", sq.get('avg_sentence_length', 0))
            with m2:
                st.metric("Vocab Richness", f"{sq.get('vocabulary_richness',0):.2f}")
                st.metric("Grammar Score", f"{sq.get('grammar_score',0):.0f}%")
                st.metric("Repetition", f"{sq.get('repetition_pct',0):.0f}%")
        with c9:
            st.text_area("Transcript analyzed:", transcript, height=200, disabled=True)

        st.divider()

        st.subheader("🏅 Interview Readiness Summary")
        st.markdown(
            f'<div class="cert-box" style="background:linear-gradient(135deg,'
            f'{rec_color}15,{rec_color}30);border:2px solid {rec_color};">'
            f'<h2 style="color:{rec_color}">🎯 Interview Evaluation Report</h2>'
            f'<h3 style="color:#333">Job Role: {job_title or "General"}</h3>'
            f'<div style="font-size:4rem;font-weight:bold;color:{rec_color}">'
            f'{scores["overall"]}/100</div>'
            f'<div style="font-size:1.5rem;color:{rec_color};font-weight:bold">'
            f'{recommendation}</div><hr style="border-color:{rec_color}40">'
            f'<div style="display:flex;justify-content:space-around;margin-top:1rem;flex-wrap:wrap">'
            + ''.join(
                f'<div style="margin:0.5rem"><div style="font-size:1.5rem;'
                f'font-weight:bold;color:{rec_color}">'
                f'{scores[k]:.0f}</div><div style="color:#666;font-size:0.85rem">'
                f'{lbl}</div></div>'
                for k, lbl in [('confidence','Confidence'),
                                ('clarity','Communication'),
                                ('professionalism','Professionalism'),
                                ('technical_relevance','Technical'),
                                ('answer_quality','Answer Quality')])
            + '</div></div>', unsafe_allow_html=True)

        st.divider()
        report = {
            'job_title': job_title, 'recommendation': recommendation,
            'scores': scores,
            'camera_presence': {
                'pct': face_r['camera_presence_pct'],
                'smile_pct': face_r['smile_pct'],
                'impression': face_r['display_label']},
            'voice': {
                'tone': audio_r['vocal_tone'],
                'stability': audio_r['voice_stability'],
                'pace': audio_r['pace_label'],
                'wpm': audio_r.get('words_per_minute'),
                'pause_control': audio_r['pause_control']},
            'star_completeness': star,
            'feedback': fb,
            'keywords': {k: v for k, v in kw.items() if isinstance(v, dict)},
            'speech_metrics': sq, 'transcript': transcript}
        st.download_button(
            label="📥 Download Full Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="interview_report.json", mime="application/json")

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;
             background:linear-gradient(135deg,#667eea10,#764ba210);
             border-radius:20px;margin:2rem 0">
        <h2>🚀 How it works</h2>
        <p style="color:#666;font-size:1.1rem">
        Upload a candidate-only interview video or paste your answer text
        </p></div>
        """, unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, emoji, title, desc in [
            (c1,"🎥","Upload","Candidate video"),
            (c2,"🤖","Analyze","Voice, presence & answer content"),
            (c3,"📊","Score","Confidence-weighted evaluation"),
            (c4,"💬","Decide","Hire recommendation + feedback")]:
            with col:
                st.markdown(
                    f'<div style="text-align:center;padding:1.5rem;'
                    f'background:#fff;border-radius:15px;'
                    f'box-shadow:0 2px 10px #0001;margin:0.5rem">'
                    f'<div style="font-size:2rem">{emoji}</div>'
                    f'<h4>{title}</h4>'
                    f'<p style="color:#666;font-size:0.85rem">{desc}</p>'
                    f'</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
