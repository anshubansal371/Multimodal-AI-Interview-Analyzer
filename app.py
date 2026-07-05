# app.py — AI Interview Analyzer v4 (Final)
import os

# ── Auto-download models on first run ─────────────────
_REQUIRED = [
    "models/face_model_best.keras",
    "models/audio_model_best.keras",
    "models/fusion_model_best.keras",
    "models/final_roberta_model/model.safetensors",
    "models/final_roberta_model/config.json",
    "models/final_roberta_model/tokenizer.json",
    "models/final_roberta_model/tokenizer_config.json",
    "models/final_roberta_model/emotion_map.json",
]

if not all(os.path.exists(f) for f in _REQUIRED):
    import streamlit as st
    with st.spinner(
            "⬇️ Downloading AI models... "
            "First run only — takes 2-3 min"):
        from download_models import download_all
        download_all()

import re
import json
import tempfile
import subprocess
import numpy as np
import torch
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go
from collections import Counter

st.set_page_config(
    page_title="AI Interview Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        font-size:2.5rem; font-weight:bold;
        text-align:center;
        background:linear-gradient(90deg,#667eea,#764ba2);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        padding:1rem 0;
    }
    .score-card {
        background:linear-gradient(135deg,#667eea20,#764ba220);
        border-radius:15px; padding:1.5rem;
        border:1px solid #667eea40;
        text-align:center; margin:0.5rem 0;
    }
    .score-number { font-size:2.5rem; font-weight:bold; color:#667eea; }
    .trait-card {
        background:#f8f9fa; border-radius:10px;
        padding:0.8rem 1rem; margin:0.4rem 0;
        border-left:4px solid #667eea;
    }
    .feedback-card {
        background:#f8f9fa; border-radius:10px;
        padding:1rem; margin:0.5rem 0;
        border-left:4px solid #667eea;
    }
    .strength-item { color:#27ae60; }
    .improve-item  { color:#e67e22; }
    .tip-high      { color:#e74c3c; }
    .tip-med       { color:#f39c12; }
    .tip-low       { color:#27ae60; }
    .cert-box      { border-radius:20px; padding:2rem;
                       text-align:center; margin:1rem 0; }
    .rec-badge {
        display:inline-block; padding:0.5rem 1.5rem;
        border-radius:25px; font-size:1.2rem;
        font-weight:bold; color:white;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models')

FACE_EMOTIONS = {
    0:'angry',1:'disgust',2:'fear',3:'happy',
    4:'neutral',5:'sad',6:'surprise'}
AUDIO_EMOTIONS = {
    '0':'angry','1':'disgust','2':'fearful',
    '3':'happy','4':'neutral','5':'sad'}
FACE_TO_DIM = {
    'angry':2,'disgust':2,'fear':2,'happy':1,
    'neutral':4,'sad':2,'surprise':3}
AUDIO_TO_DIM = {
    'angry':2,'disgust':2,'fearful':2,
    'happy':1,'neutral':4,'sad':2}
TEXT_TO_DIM = {
    'angry':2,'anxious':2,'positive':1,'surprised':3}

FILLER_WORDS = [
    'um','uh','like','so','you know','basically',
    'literally','actually','i mean','right','okay so']

# Weighted technical skills dictionary
TECHNICAL_SKILLS = {
    'python'          : 10, 'java'           : 10,
    'c++'             : 10, 'c'              :  5,
    'sql'             :  8, 'database'       :  8,
    'docker'          : 10, 'kubernetes'     : 10,
    'machine learning': 12, 'deep learning'  : 12,
    'tensorflow'      : 10, 'pytorch'        : 10,
    'opencv'          :  8, 'git'            :  8,
    'github'          :  8, 'api'            :  8,
    'web development' : 10, 'project'        : 12,
    'internship'      : 12, 'streamlit'      :  8,
    'algorithm'       :  8, 'data structure' :  8,
    'cloud'           :  8, 'aws'            : 10,
    'flask'           :  8, 'django'         :  8,
    'nlp'             : 10, 'neural network' : 10,
    'research'        :  8, 'developed'      :  6,
    'implemented'     :  6, 'designed'       :  6}

# Intro patterns — STAR not applicable for introductions
INTRO_PATTERNS = [
    'my name is', 'i am from', 'i am pursuing',
    'i am doing', 'my hobbies', 'my strengths',
    'thank you for allowing me', 'tell me about yourself',
    'introduce myself', 'i completed my',
    'i am currently', 'i have done my',
    'i belong to', 'i was born']

STAR_COMPONENTS = {
    'situation': [
        'situation','context','background',
        'at the time','when i was','we were',
        'there was','i was working','in my previous',
        'in my role','during','at that time'],
    'task': [
        'task','responsible','goal','objective',
        'assigned','my job','i had to','i needed to',
        'my role was','i was asked','required to'],
    'action': [
        'action','implemented','did','approach',
        'i decided','i analyzed','i built','i designed',
        'i created','i developed','i worked','i wrote',
        'i fixed','i solved','i set up','i used',
        'so i','what i did','my approach'],
    'result': [
        'result','outcome','achieved','improved',
        'increased','decreased','delivered','led to',
        'as a result','consequently','therefore',
        'this helped','we succeeded','successfully',
        'in the end','ultimately','percent',
        'reduced','saved','completed']}


# ═══════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    models = {}
    try:
        face_path = os.path.join(
            MODELS_DIR, 'face_model_best.keras')
        if not os.path.exists(face_path):
            face_path = os.path.join(
                MODELS_DIR, 'face_model_best.h5')
        models['face'] = tf.keras.models.load_model(face_path)
        st.sidebar.success("✅ Face model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Face: {e}")
        models['face'] = None

    try:
        models['audio'] = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, 'audio_model_best.keras'))
        st.sidebar.success("✅ Audio model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Audio: {e}")
        models['audio'] = None

    try:
        models['fusion'] = tf.keras.models.load_model(
            os.path.join(MODELS_DIR, 'fusion_model_best.keras'))
        st.sidebar.success("✅ Fusion model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Fusion: {e}")
        models['fusion'] = None

    try:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification)
        rp = os.path.join(MODELS_DIR, 'final_roberta_model')
        models['roberta_tok'] = AutoTokenizer.from_pretrained(
            rp, local_files_only=True)
        models['roberta'] = \
            AutoModelForSequenceClassification.from_pretrained(
                rp, local_files_only=True)
        models['roberta'].eval()
        with open(os.path.join(rp, 'emotion_map.json')) as f:
            emotion2id = json.load(f)
        models['id2emotion'] = {
            v: k for k, v in emotion2id.items()}
        st.sidebar.success("✅ Text model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Text: {e}")
        models['roberta'] = None

    return models


def video_to_wav(video_path, target_sr=16000):
    wav_path = video_path + "_conv.wav"
    subprocess.run([
        'ffmpeg', '-i', video_path, '-vn',
        '-ac', '1', '-ar', str(target_sr),
        wav_path, '-y', '-loglevel', 'quiet'],
        capture_output=True)
    return wav_path if os.path.exists(wav_path) else None


# ═══════════════════════════════════════════════════════
# FACE ANALYSIS
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
        img_size, n_ch = input_shape[1], input_shape[-1]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        all_preds, face_found, smile_frames = [], 0, 0
        frame_idx, total_sampled = 0, 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                total_sampled += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, 1.2, 5, minSize=(60, 60))
                if len(faces) > 0:
                    face_found += 1
                    x, y, w, h = max(
                        faces, key=lambda f: f[2]*f[3])
                    crop = frame[y:y+h, x:x+w]
                    resized = cv2.resize(
                        crop, (img_size, img_size))
                    if n_ch == 3:
                        img_arr = cv2.cvtColor(
                            resized, cv2.COLOR_BGR2RGB)
                    else:
                        img_arr = cv2.cvtColor(
                            resized, cv2.COLOR_BGR2GRAY)
                        img_arr = np.expand_dims(img_arr, -1)
                    arr = np.expand_dims(
                        img_arr.astype(np.float32)/255.0, 0)
                    probs = face_model.predict(
                        arr, verbose=0)[0]
                    pred = int(np.argmax(probs))
                    conf = float(probs[pred])
                    all_preds.append((pred, conf, probs))
                    if pred == 3:
                        smile_frames += 1
            frame_idx += 1
        cap.release()

        if not all_preds:
            return None

        camera_pct = round(
            100 * face_found / max(total_sampled, 1), 1)
        smile_pct = round(
            100 * smile_frames / max(len(all_preds), 1), 1)

        confident = [p for p in all_preds if p[1] >= min_confidence]
        usable = confident if confident else all_preds
        vote_counts = Counter(p[0] for p in usable)
        ranked = vote_counts.most_common()
        wc = ranked[0][0]
        wv = ranked[0][1]
        ru = ranked[1][1] if len(ranked) > 1 else 0
        margin = (wv - ru) / len(usable)
        is_decisive = margin >= decisive_margin

        matching = [p[2] for p in usable if p[0] == wc]
        avg_probs = np.mean(matching, axis=0)
        consistency = round(100 * wv / len(usable), 1)
        raw_em = FACE_EMOTIONS.get(wc, 'neutral')

        if is_decisive and raw_em == 'happy':
            display_label = "Positive / engaged"
        elif is_decisive and raw_em == 'surprise':
            display_label = "Alert / responsive"
        else:
            display_label = "Neutral / composed"

        return {
            'display_label'      : display_label,
            'raw_emotion'        : raw_em,
            'is_decisive'        : is_decisive,
            'confidence'         : float(avg_probs[wc]),
            'camera_presence_pct': camera_pct,
            'smile_pct'          : smile_pct,
            'consistency'        : consistency,
            'probs'              : avg_probs.tolist(),
            'dim'                : FACE_TO_DIM.get(
                raw_em if is_decisive else 'neutral', 4),
            'frames_analyzed'    : len(all_preds),
            'frames_used'        : len(usable)}

    except Exception as e:
        st.warning(f"Face analysis error: {e}")
        return None


# ═══════════════════════════════════════════════════════
# AUDIO ANALYSIS — no emotion labels, vocal traits only
# ═══════════════════════════════════════════════════════

def run_vad(y, sr, frame_ms=30, energy_percentile=30):
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0 or len(y) < frame_len:
        return [], frame_ms / 1000
    n_frames = len(y) // frame_len
    energies = [
        float(np.sqrt(np.mean(
            y[i*frame_len:(i+1)*frame_len]**2)))
        for i in range(n_frames)]
    energies = np.array(energies)
    if len(energies) == 0:
        return [], frame_ms / 1000
    threshold = max(
        np.percentile(energies, energy_percentile), 1e-4)
    return [bool(e > threshold) for e in energies], frame_ms/1000


def compute_pause_metrics(voiced_flags, frame_dur):
    if not voiced_flags:
        return {'silence_pct':0,'pause_count':0,
                'avg_pause_duration':0,'pause_ratio':0}
    total = len(voiced_flags)
    silent = sum(1 for v in voiced_flags if not v)
    silence_pct = round(100 * silent / total, 1)

    first_v = next((i for i,v in enumerate(voiced_flags) if v), None)
    last_v  = next((i for i in range(total-1,-1,-1)
                    if voiced_flags[i]), None)
    pauses = []
    if first_v is not None and last_v is not None:
        cur = 0
        for i in range(first_v, last_v+1):
            if not voiced_flags[i]:
                cur += 1
            else:
                if cur >= 3:
                    pauses.append(cur * frame_dur)
                cur = 0

    return {
        'silence_pct'       : silence_pct,
        'pause_count'       : len(pauses),
        'avg_pause_duration': round(
            float(np.mean(pauses)), 2) if pauses else 0.0,
        'pause_ratio'       : round(silent/max(total,1), 3)}


def estimate_pitch_praat(wav_path):
    try:
        import parselmouth
        snd = parselmouth.Sound(wav_path)
        pitch = snd.to_pitch()
        pv = pitch.selected_array['frequency']
        pv = pv[pv > 0]
        if len(pv) < 10:
            return {'pitch_mean':0,'pitch_std':0,
                     'pitch_stability':50.0}
        pm = float(np.mean(pv))
        ps = float(np.std(pv))
        return {
            'pitch_mean'     : round(pm, 1),
            'pitch_std'      : round(ps, 1),
            'pitch_stability': round(
                max(0, min(100,
                    100*(1-ps/(pm+1e-6)))), 1)}
    except Exception:
        return {'pitch_mean':0,'pitch_std':0,
                 'pitch_stability':50.0}


def analyze_audio_from_video(video_path, audio_model,
                               transcript_word_count=None,
                               min_confidence=0.65,
                               decisive_margin=0.25):
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

        voiced_flags, frame_dur = run_vad(y, sr)
        pause_metrics = compute_pause_metrics(
            voiced_flags, frame_dur)
        pitch_metrics = estimate_pitch_praat(wav_path)

        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        rms_std  = float(np.std(rms))
        energy_stability = max(0, min(100,
            100*(1-rms_std/(rms_mean+1e-6))))
        loudness_variation = round(
            100*rms_std/(rms_mean+1e-6), 1)

        speaking_sec = duration_sec * (
            1 - pause_metrics['pause_ratio'])
        if transcript_word_count and speaking_sec > 0:
            wpm = round(
                transcript_word_count/(speaking_sec/60), 0)
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

        pitch_stability = pitch_metrics['pitch_stability']
        cnn_label, cnn_decisive = None, False

        if audio_model is not None:
            from PIL import Image as PILImage
            ish = audio_model.input_shape
            ih, iw, ich = ish[1], ish[2], ish[-1]
            win_len = sr * 3
            all_preds = []
            for start in range(0, len(y)-win_len, win_len):
                chunk = y[start:start+win_len]
                mel = librosa.feature.melspectrogram(
                    y=chunk, sr=sr, n_mels=ih,
                    n_fft=2048, hop_length=512)
                mel_db = librosa.power_to_db(
                    mel, ref=np.max)
                mel_img = np.array(
                    PILImage.fromarray(mel_db).resize(
                        (iw, ih)))
                mn, mx = mel_img.min(), mel_img.max()
                if mx - mn < 1e-8:
                    continue
                mel_norm = (mel_img-mn)/(mx-mn)
                arr = (np.repeat(mel_norm[...,None],3,axis=-1)
                       if ich==3 else mel_norm[...,None])
                arr = np.expand_dims(
                    arr.astype(np.float32), 0)
                probs = audio_model.predict(
                    arr, verbose=0)[0]
                pred = int(np.argmax(probs))
                conf = float(probs[pred])
                all_preds.append((pred, conf, probs))

            if all_preds:
                confident = [p for p in all_preds
                             if p[1] >= min_confidence]
                usable = confident if confident else all_preds
                vc = Counter(p[0] for p in usable)
                ranked = vc.most_common()
                wc = ranked[0][0]
                wv = ranked[0][1]
                ru = ranked[1][1] if len(ranked)>1 else 0
                margin = (wv-ru)/len(usable)
                cnn_decisive = (margin >= decisive_margin
                                 and len(confident) >= 3)
                if cnn_decisive:
                    cnn_label = AUDIO_EMOTIONS.get(
                        str(wc), 'neutral')

        os.unlink(wav_path)

        if energy_stability > 65 and pitch_stability > 55:
            vocal_tone = "Calm and steady"
        elif energy_stability > 50 and \
                pause_metrics['pause_ratio'] < 0.35:
            vocal_tone = "Confident"
        elif energy_stability < 35 or \
                pause_metrics['pause_count'] > 8:
            vocal_tone = "Slightly nervous"
        else:
            vocal_tone = "Neutral"

        if cnn_decisive and cnn_label in (
                'fearful','sad') and energy_stability < 50:
            vocal_tone = "Slightly nervous"
        elif cnn_decisive and cnn_label == 'happy' and \
                energy_stability > 55:
            vocal_tone = "Energetic and confident"

        if pause_metrics['pause_count'] == 0:
            pause_control = "Excellent"
        elif pause_metrics['avg_pause_duration'] < 1.0:
            pause_control = "Good"
        elif pause_metrics['avg_pause_duration'] < 2.5:
            pause_control = "Fair"
        else:
            pause_control = "Needs work"

        voice_stability = round(
            (energy_stability+pitch_stability)/2, 1)
        audio_conf = max(0.4, min(0.95,
            1-abs(energy_stability-pitch_stability)/200))

        return {
            'vocal_tone'        : vocal_tone,
            'voice_stability'   : voice_stability,
            'energy_stability'  : round(energy_stability,1),
            'pitch_stability'   : pitch_stability,
            'pitch_mean'        : pitch_metrics['pitch_mean'],
            'loudness_variation': loudness_variation,
            'pace_label'        : pace_label,
            'words_per_minute'  : wpm,
            'silence_pct'       : pause_metrics['silence_pct'],
            'pause_count'       : pause_metrics['pause_count'],
            'avg_pause_duration': pause_metrics[
                'avg_pause_duration'],
            'pause_ratio'       : pause_metrics['pause_ratio'],
            'pause_control'     : pause_control,
            'confidence'        : audio_conf,
            'probs'             : [0.14]*6,
            'dim'               : (
                1 if vocal_tone in (
                    "Confident","Energetic and confident")
                else 2 if vocal_tone=="Slightly nervous"
                else 4),
            'cnn_label'         : cnn_label,
            'cnn_decisive'      : cnn_decisive}

    except Exception as e:
        if os.path.exists(wav_path):
            try: os.unlink(wav_path)
            except Exception: pass
        st.warning(f"Audio analysis error: {e}")
        return None


# ═══════════════════════════════════════════════════════
# SPEECH QUALITY
# ═══════════════════════════════════════════════════════

def analyze_speech_quality(text):
    if not text:
        return {}
    words = text.lower().split()
    total = len(words)
    if total == 0:
        return {}
    fillers = sum(
        len(re.findall(
            r'\b'+re.escape(f)+r'\b', text.lower()))
        for f in FILLER_WORDS)
    ttr = len(set(words)) / total
    sents = [s for s in re.split(r'[.!?]+', text.strip())
             if len(s.strip()) > 0]
    avg_sent = round(total/max(len(sents),1), 1)
    wc = Counter(words)
    repeated = [w for w,c in wc.items()
                if c > 3 and len(w) > 3]
    rep_pct = round(
        sum(wc[w] for w in repeated)/max(total,1)*100, 1)
    long_sents = sum(1 for s in sents
                     if len(s.split()) > 40)
    grammar_score = max(0, min(100,
        100 - long_sents*10 - fillers/total*150))
    return {
        'total_words'        : total,
        'filler_count'       : fillers,
        'filler_ratio'       : round(fillers/total, 3),
        'vocabulary_richness': round(ttr, 3),
        'avg_sentence_length': avg_sent,
        'repetition_pct'     : rep_pct,
        'repeated_words'     : repeated[:5],
        'grammar_score'      : round(grammar_score, 1),
        'clarity_score'      : round(
            max(0, min(100,
                (1-fillers/total*3)*100)), 1),
        'fluency_score'      : round(
            min(100, ttr*70+min(avg_sent/20,1)*30), 1)}


def is_introduction(text):
    text_lower = text.lower()
    return any(p in text_lower for p in INTRO_PATTERNS)


def compute_star_completeness(text):
    if is_introduction(text):
        return {'completeness': None,
                'components_found': None,
                'is_intro': True}
    text_lower = text.lower()
    found = {}
    for comp, kws in STAR_COMPONENTS.items():
        found[comp] = any(
            re.search(r'\b'+re.escape(kw)+r'\b',
                      text_lower) for kw in kws)
    completeness = round(
        100*sum(found.values())/len(STAR_COMPONENTS), 1)
    return {'completeness': completeness,
             'components_found': found,
             'is_intro': False}


def compute_keyword_score(text):
    text_lower = text.lower()

    # Weighted technical score
    tech_score = 0
    tech_found = []
    for skill, weight in TECHNICAL_SKILLS.items():
        if re.search(r'\b'+re.escape(skill)+r'\b',
                     text_lower):
            tech_score += weight
            tech_found.append(skill)
    tech_score = min(100, tech_score)

    # Other categories
    OTHER_KEYWORDS = {
        'leadership': [
            'led','managed','team','initiative',
            'ownership','mentored','coordinated',
            'supervised','organized','directed',
            'handled','responsible','in charge',
            'head','guide','delegate','lead'],
        'problem_solving': [
            'solved','debugged','optimized',
            'improved','analyzed','designed',
            'implemented','resolved','fixed',
            'approach','solution','challenge',
            'issue','problem','tackle','overcome',
            'identify','diagnose','worked on',
            'figured out','addressed'],
        'communication': [
            'presented','explained','collaborated',
            'discussed','communicated','reported',
            'told','shared','informed','conveyed',
            'worked with','talked','meeting',
            'spoke','described','mentioned',
            'expressed','interact'],
        'star_method': [
            'situation','task','action','result',
            'challenge','achieved','delivered',
            'outcome','background','context',
            'responsible','goal','objective',
            'approach','decided','did',
            'what i did','as a result','led to',
            'because of','therefore',
            'consequently','in the end',
            'ultimately']}

    scores = {'technical': {'score': tech_score,
                              'found': tech_found[:6]}}
    for cat, kws in OTHER_KEYWORDS.items():
        found = [k for k in kws
                 if re.search(
                     r'\b'+re.escape(k)+r'\b',
                     text_lower)]
        scores[cat] = {
            'score': min(100,
                len(found)/max(len(kws),1)*300),
            'found': found[:4]}

    scores['overall'] = round(
        np.mean([v['score']
                 for v in scores.values()]), 1)
    return scores


def predict_text_emotion(text, models):
    if not models.get('roberta'):
        return {'emotion':'positive','confidence':0.5,
                 'probs':[0.1,0.1,0.7,0.1],'dim':1}
    try:
        inp = models['roberta_tok'](
            text, return_tensors='pt',
            truncation=True, max_length=256,
            padding=True)
        with torch.no_grad():
            out = models['roberta'](**inp)
            probs = torch.softmax(
                out.logits, dim=1).numpy()[0]
        pred = np.argmax(probs)
        em = models['id2emotion'][pred]
        return {
            'emotion'   : em,
            'confidence': float(probs[pred]),
            'probs'     : probs.tolist(),
            'dim'       : TEXT_TO_DIM.get(em, 4)}
    except Exception:
        return {'emotion':'positive','confidence':0.5,
                 'probs':[0.1,0.1,0.7,0.1],'dim':1}


# ═══════════════════════════════════════════════════════
# SCORING — confidence-weighted fusion
# ═══════════════════════════════════════════════════════

def compute_scores(text_r, face_r, audio_r,
                    sq, star, kw):
    """
    Final scores use confidence-weighted fusion so
    a low-confidence modality contributes less.
    Overall is a weighted blend of sub-dimensions —
    not directly from the fusion model — matching
    how a real recruiter would weight these signals.
    """
    # Sub-dimension scores
    communication = round(
        sq.get('clarity_score',50)*0.4 +
        sq.get('fluency_score',50)*0.3 +
        sq.get('grammar_score',50)*0.3, 1)

    professionalism = round(min(100,
        (100-sq.get('repetition_pct',0)*2)*0.4 +
        sq.get('grammar_score',70)*0.3 +
        communication*0.3), 1)

    technical = round(kw.get('technical',{})
                       .get('score',0), 1)

    star_comp = star['completeness'] if (
        star['completeness'] is not None) else 50
    answer_quality = round(min(100,
        kw.get('overall',30)*0.4 +
        star_comp*0.4 + (
        text_r['confidence']*100 if
        text_r['emotion']=='positive' else 50)*0.2), 1)

    # Face-derived confidence score
    face_score = (
        85 if face_r['display_label']=="Positive / engaged"
        else 75 if face_r['display_label']=="Alert / responsive"
        else 60)
    face_score = min(100, face_score +
        (face_r['camera_presence_pct']-50)*0.3)
    confidence_score = round(face_score, 1)

    # Confidence-weighted overall
    BASE_TEXT, BASE_FACE, BASE_AUDIO = 0.55, 0.30, 0.15
    w_t = BASE_TEXT * text_r['confidence']
    w_f = BASE_FACE * face_r['confidence']
    w_a = BASE_AUDIO * audio_r['confidence']
    total_w = w_t + w_f + w_a + 1e-6
    w_t, w_f, w_a = w_t/total_w, w_f/total_w, w_a/total_w

    text_score = (
        text_r['confidence']*100 if
        text_r['emotion']=='positive'
        else text_r['confidence']*60 if
        text_r['emotion']=='surprised'
        else 100-text_r['confidence']*70)

    audio_score = (
        85 if audio_r['vocal_tone'] in
            ("Energetic and confident","Confident")
        else 75 if audio_r['vocal_tone']=="Calm and steady"
        else 45 if audio_r['vocal_tone']=="Slightly nervous"
        else 65)
    if audio_r['pause_control'] == "Needs work":
        audio_score -= 10
    elif audio_r['pause_control'] == "Excellent":
        audio_score += 5
    audio_score = max(0, min(100, audio_score))

    fusion_overall = round(min(100, max(0,
        text_score*w_t + face_score*w_f +
        audio_score*w_a)), 1)

    # Final overall = weighted blend of sub-dimensions
    overall = round(
        communication    * 0.30 +
        professionalism  * 0.20 +
        technical        * 0.20 +
        answer_quality   * 0.20 +
        confidence_score * 0.10, 1)

    # Blend with fusion estimate
    overall = round(overall*0.7 + fusion_overall*0.3, 1)

    return {
        'overall'            : overall,
        'communication'      : communication,
        'professionalism'    : professionalism,
        'technical'          : technical,
        'answer_quality'     : answer_quality,
        'confidence'         : confidence_score,
        'weights_used'       : {
            'text' : round(w_t,2),
            'face' : round(w_f,2),
            'audio': round(w_a,2)}}


def get_recommendation(scores, star):
    overall = scores['overall']
    technical = scores['technical']
    star_comp = star['completeness']

    if overall >= 85 and technical >= 65:
        return "Strong Hire", "#27ae60"
    elif overall >= 70 and (
            star_comp is None or star_comp >= 50):
        return "Hire", "#3498db"
    elif overall >= 55:
        return "Consider", "#f39c12"
    else:
        return "Needs Improvement", "#e74c3c"


# ═══════════════════════════════════════════════════════
# FEEDBACK — personalized based on actual scores
# ═══════════════════════════════════════════════════════

def generate_feedback(scores, face_r, audio_r,
                       text_r, sq, star, kw):
    strengths, improvements, tips = [], [], []

    # Strengths
    if scores['communication'] > 70:
        strengths.append(
            f"Clear communication with strong vocabulary "
            f"richness ({sq.get('vocabulary_richness',0):.2f})")
    if scores['confidence'] > 70:
        strengths.append(
            f"Good on-camera presence — face visible in "
            f"{face_r['camera_presence_pct']:.0f}% of frames")
    if (star['completeness'] is not None and
            star['completeness'] >= 75):
        found = sum(star['components_found'].values())
        strengths.append(
            f"Strong STAR structure — covered {found}/4 "
            f"components in your answer")
    if audio_r['vocal_tone'] in (
            "Calm and steady","Confident",
            "Energetic and confident"):
        strengths.append(
            f"Vocal delivery was "
            f"{audio_r['vocal_tone'].lower()} with "
            f"{audio_r['voice_stability']:.0f}% stability")
    if scores['technical'] > 60:
        found_tech = kw.get('technical',{}).get('found',[])
        if found_tech:
            strengths.append(
                f"Good technical vocabulary: "
                f"{', '.join(found_tech[:3])}")
    if not strengths:
        strengths = [
            "Shows genuine engagement with the question",
            "Completed a full response without trailing off"]

    # Improvements with personalized numbers
    if scores['technical'] < 60:
        improvements.append(
            "Discuss more technical projects and mention "
            "specific technologies you used (Python, SQL, "
            "cloud tools, etc.)")
    if scores['communication'] < 70:
        improvements.append(
            "Maintain better eye contact and speak more "
            "confidently — reduce filler words "
            f"({sq.get('filler_count',0)} detected)")
    if scores['professionalism'] < 70:
        improvements.append(
            "Use more formal language and avoid filler "
            "words — repetition detected: "
            f"{', '.join(sq.get('repeated_words',[])[:3])}")
    if (star['completeness'] is not None and
            star['completeness'] < 50):
        missing = [k for k,v in
                   star['components_found'].items()
                   if not v]
        improvements.append(
            f"Answer missing STAR components: "
            f"{', '.join(missing)} — structure your "
            "answer as Situation → Task → Action → Result")
    if audio_r['vocal_tone'] == "Slightly nervous":
        improvements.append(
            "Practice deep breathing before answering — "
            "your vocal energy suggested some nervousness")
    if audio_r['pause_control'] == "Needs work":
        improvements.append(
            f"Reduce long pauses (avg "
            f"{audio_r['avg_pause_duration']:.1f}s) — "
            "practice answers aloud to reduce hesitation")
    if face_r['camera_presence_pct'] < 50:
        improvements.append(
            "Keep your face clearly in frame — "
            f"only visible in {face_r['camera_presence_pct']:.0f}% "
            "of frames")
    if not improvements:
        improvements = [
            "Continue refining answer specificity",
            "Practice more mock interviews"]

    # Personalised tips
    overall = scores['overall']
    if overall >= 80:
        tips = [
            "HIGH: Research company culture and recent news",
            "MED: Prepare 5 specific project examples",
            "LOW: Practice salary negotiation responses"]
    elif overall >= 60:
        tips = [
            "HIGH: Practice STAR method daily with timer",
            "HIGH: Record yourself and review pacing",
            "MED: Expand technical keyword vocabulary"]
    elif overall >= 40:
        tips = [
            "HIGH: Daily mock interview practice (30 min)",
            "HIGH: Work on STAR answer structure",
            "HIGH: Reduce filler words — pause instead"]
    else:
        tips = [
            "HIGH: Daily mirror practice for confidence",
            "HIGH: Study and apply STAR method",
            "HIGH: Work on vocal confidence and pace"]

    return {
        'strengths'  : strengths[:4],
        'improvements': improvements[:4],
        'tips'        : tips}


# ═══════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════

def create_radar_chart(scores):
    cats = ['Communication','Confidence',
             'Professionalism','Technical',
             'Answer Quality']
    vals = [scores['communication'],
             scores['confidence'],
             scores['professionalism'],
             scores['technical'],
             scores['answer_quality']]
    vc = vals + [vals[0]]
    cc = cats + [cats[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vc, theta=cc, fill='toself',
        fillcolor='rgba(102,126,234,0.2)',
        line=dict(color='#667eea',width=2)))
    fig.update_layout(
        polar=dict(radialaxis=dict(
            visible=True, range=[0,100])),
        showlegend=False, height=350,
        margin=dict(t=30,b=30,l=30,r=30),
        paper_bgcolor='rgba(0,0,0,0)')
    return fig


def create_benchmark_chart(scores, color):
    bm = {
        'Your Score'       : scores['overall'],
        'Average Candidate': 58.0,
        'Good Candidate'   : 72.0,
        'Top Candidate'    : 88.0}
    fig = go.Figure(go.Bar(
        x=list(bm.values()), y=list(bm.keys()),
        orientation='h',
        marker_color=[color,'#95a5a6',
                       '#3498db','#27ae60'],
        text=[f"{v:.1f}" for v in bm.values()],
        textposition='auto'))
    fig.update_layout(
        xaxis=dict(range=[0,100]), height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10,b=10,l=10,r=10))
    return fig


# ═══════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════

def main():
    st.markdown(
        '<div class="main-header">'
        '🎯 AI Interview Analyzer</div>',
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#666;'>"
        "Interview evaluation platform — communication, "
        "delivery and answer quality</p>",
        unsafe_allow_html=True)
    st.divider()

    st.sidebar.title("⚙️ Settings")
    st.sidebar.subheader("Model Status")
    with st.spinner("Loading AI models..."):
        models = load_models()

    st.sidebar.divider()
    job_title = st.sidebar.text_input(
        "🎯 Target Job Role",
        placeholder="e.g. Software Engineer, "
                    "Data Scientist...",
        value="")

    st.sidebar.divider()
    st.sidebar.info(
        "📌 Upload a video of only the candidate "
        "for best results.")

    st.sidebar.divider()
    st.sidebar.markdown(
        "**📊 Evaluation dimensions:**\n"
        "- 💬 Communication & answer structure\n"
        "- 🎤 Voice stability, pace, pauses\n"
        "- 👁️ Camera presence\n"
        "- 🔑 Technical keyword relevance\n"
        "- ⭐ STAR method (skipped for intros)\n\n"
        "ℹ️ Fusion weights scale dynamically "
        "with each model's confidence on your clip.")

    tab1, tab2 = st.tabs([
        "📹 Upload Video/Audio",
        "✏️ Paste Transcript"])

    transcript = ""
    tmp_path = None
    face_r, audio_r = None, None
    is_video = False

    with tab1:
        st.markdown(
            "**Upload candidate video** "
            "(single speaker) for full analysis.")
        uploaded = st.file_uploader(
            "Choose video or audio file",
            type=['mp4','mov','avi',
                  'wav','mp3','m4a'],
            help="Max 200MB.")

        if uploaded:
            suffix = os.path.splitext(
                uploaded.name)[1]
            with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            is_video = uploaded.type.startswith('video')
            if is_video:
                st.video(uploaded)
            else:
                st.audio(uploaded)

            if st.button("🎯 Analyze Interview",
                          type="primary",
                          key="btn_video"):
                progress = st.progress(0)
                status = st.empty()

                status.info("📝 Transcribing audio...")
                progress.progress(10)
                try:
                    import whisper
                    wm = whisper.load_model("tiny")
                    result = wm.transcribe(
                        tmp_path, language='en',
                        fp16=False)
                    transcript = result['text'].strip()
                    progress.progress(35)
                    st.success("✅ Transcription complete!")
                    st.text_area("📝 Transcript",
                                  transcript, height=120)
                except Exception as e:
                    st.error(f"Transcription error: {e}")
                    progress.progress(35)

                if is_video and models.get('face'):
                    status.info(
                        "👁️ Analyzing camera presence...")
                    progress.progress(50)
                    face_r = analyze_face_from_video(
                        tmp_path, models['face'])
                    if face_r:
                        st.caption(
                            f"👁️ Camera presence: "
                            f"{face_r['camera_presence_pct']:.0f}% | "
                            f"Smile: {face_r['smile_pct']:.0f}% | "
                            f"Consistency: "
                            f"{face_r['consistency']:.0f}%")

                status.info("🎤 Analyzing voice...")
                progress.progress(70)
                wc = len(transcript.split()) if transcript else None
                audio_r = analyze_audio_from_video(
                    tmp_path, models.get('audio'),
                    transcript_word_count=wc)
                if audio_r:
                    st.caption(
                        f"🎤 Vocal tone: "
                        f"**{audio_r['vocal_tone']}** | "
                        f"Stability: "
                        f"{audio_r['voice_stability']:.0f}% | "
                        f"Pace: {audio_r['pace_label']}")

                progress.progress(100)
                status.empty()
                progress.empty()

    with tab2:
        st.markdown(
            "Paste **your interview answers** "
            "for text-based analysis.")
        transcript_input = st.text_area(
            "Paste your answers here",
            placeholder=
                "In my previous role I led a team...",
            height=200)
        if st.button("🎯 Analyze Text",
                      type="primary",
                      key="btn_text"):
            transcript = transcript_input

    # ═══════════════════════════════════════
    if transcript and len(transcript) > 20:
        st.divider()
        st.subheader("📊 Interview Evaluation Report")

        with st.spinner("Running analysis..."):
            sq    = analyze_speech_quality(transcript)
            star  = compute_star_completeness(transcript)
            text_r = predict_text_emotion(
                transcript, models)
            kw    = compute_keyword_score(transcript)

            if face_r is None:
                face_r = {
                    'display_label'      : 'Neutral / composed',
                    'raw_emotion'        : 'neutral',
                    'is_decisive'        : False,
                    'confidence'         : 0.5,
                    'camera_presence_pct': 0,
                    'smile_pct'          : 0,
                    'consistency'        : 0,
                    'probs'              : [0.14]*7,
                    'dim'                : 4,
                    'frames_analyzed'    : 0,
                    'frames_used'        : 0}
            if audio_r is None:
                audio_r = {
                    'vocal_tone'        : 'Neutral',
                    'voice_stability'   : 50,
                    'energy_stability'  : 50,
                    'pitch_stability'   : 50,
                    'pitch_mean'        : 0,
                    'loudness_variation': 0,
                    'pace_label'        : 'Unknown',
                    'words_per_minute'  : None,
                    'silence_pct'       : 0,
                    'pause_count'       : 0,
                    'avg_pause_duration': 0,
                    'pause_ratio'       : 0,
                    'pause_control'     : 'Unknown',
                    'confidence'        : 0.5,
                    'probs'             : [0.14]*6,
                    'dim'               : 4,
                    'cnn_label'         : None,
                    'cnn_decisive'      : False}

            scores = compute_scores(
                text_r, face_r, audio_r, sq, star, kw)
            fb = generate_feedback(
                scores, face_r, audio_r,
                text_r, sq, star, kw)
            recommendation, rec_color = \
                get_recommendation(scores, star)

        # ── Headline stats ─────────────────
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("⏱️ Words",
                       sq.get('total_words',0))
        with s2:
            st.metric("🎯 Score",
                       f"{scores['overall']}/100")
        with s3:
            st.metric("🔧 Technical",
                       f"{scores['technical']:.0f}%")
        with s4:
            star_display = (
                "N/A (intro)" if star['is_intro']
                else f"{star['completeness']:.0f}%")
            st.metric("⭐ STAR", star_display)
        st.divider()

        # ── Recommendation ─────────────────
        st.markdown(
            f'<div style="text-align:center;'
            f'margin:1rem 0">'
            f'<span class="rec-badge" '
            f'style="background:{rec_color}">'
            f'Recommendation: {recommendation}'
            f'</span></div>',
            unsafe_allow_html=True)
        st.caption(
            f"Confidence-weighted fusion — "
            f"Text: {scores['weights_used']['text']*100:.0f}%, "
            f"Face: {scores['weights_used']['face']*100:.0f}%, "
            f"Audio: {scores['weights_used']['audio']*100:.0f}%")
        st.divider()

        # ── Score + Radar ───────────────────
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown(
                f'<div class="score-card">'
                f'<div class="score-number" '
                f'style="color:{rec_color}">'
                f'{scores["overall"]}</div>'
                f'<div style="color:#666;font-size:0.9rem">'
                f'Interview Score / 100</div></div>',
                unsafe_allow_html=True)
            for name, key in [
                ("💬 Communication", 'communication'),
                ("👔 Professionalism",'professionalism'),
                ("🔧 Technical",     'technical'),
                ("⭐ Answer Quality",'answer_quality'),
                ("🎯 Confidence",    'confidence')]:
                val = scores[key]
                ca, cb = st.columns([2,1])
                with ca:
                    st.markdown(f"**{name}**")
                    st.progress(int(val))
                with cb:
                    st.markdown(f"**{val:.0f}**")
        with col2:
            st.plotly_chart(
                create_radar_chart(scores),
                use_container_width=True)

        st.divider()

        # ── Benchmark ──────────────────────
        st.subheader("📊 Benchmark Comparison")
        st.plotly_chart(
            create_benchmark_chart(scores, rec_color),
            use_container_width=True)
        st.divider()

        # ── Presence + Voice ───────────────
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("👁️ Camera Presence")
            st.markdown(
                f'<div class="trait-card">'
                f'<b>Impression:</b> '
                f'{face_r["display_label"]}</div>',
                unsafe_allow_html=True)
            st.progress(
                int(face_r['camera_presence_pct']),
                text=f"In frame: "
                     f"{face_r['camera_presence_pct']:.0f}%")
            st.progress(
                int(face_r['smile_pct']),
                text=f"Smile frequency: "
                     f"{face_r['smile_pct']:.0f}%")
            st.progress(
                int(face_r['consistency']),
                text=f"Consistency: "
                     f"{face_r['consistency']:.0f}%")

        with col4:
            st.subheader("🎤 Voice Analysis")
            st.markdown(
                f'<div class="trait-card">'
                f'<b>Vocal tone:</b> '
                f'{audio_r["vocal_tone"]}</div>',
                unsafe_allow_html=True)
            st.progress(
                int(audio_r['voice_stability']),
                text=f"Voice stability: "
                     f"{audio_r['voice_stability']:.0f}%")
            if audio_r.get('words_per_minute'):
                st.markdown(
                    f"**Pace:** {audio_r['pace_label']} "
                    f"({audio_r['words_per_minute']:.0f} wpm)")
            st.markdown(
                f"**Pause control:** "
                f"{audio_r['pause_control']}")
            st.caption(
                f"{audio_r['pause_count']} pauses | "
                f"{audio_r['silence_pct']:.0f}% silence")

        st.divider()

        # ── Keywords ───────────────────────
        st.subheader("🔑 Keyword Coverage")
        for cat, data in kw.items():
            if cat == 'overall':
                continue
            sc   = data['score']
            found = data['found']
            ck = ('#27ae60' if sc > 60
                  else '#e67e22' if sc > 30
                  else '#e74c3c')
            st.markdown(
                f"**{cat.replace('_',' ').title()}**: "
                f'<span style="color:{ck}">'
                f'{sc:.0f}%</span>',
                unsafe_allow_html=True)
            if found:
                st.markdown(
                    f"  *Found: "
                    f"{', '.join(found[:4])}*")
            st.progress(int(sc))

        st.divider()

        # ── STAR ───────────────────────────
        st.subheader("⭐ STAR Method")
        if star['is_intro']:
            st.info(
                "ℹ️ Introduction detected — "
                "STAR analysis not applicable")
        else:
            star_cols = st.columns(4)
            for col, (comp, present) in zip(
                    star_cols,
                    star['components_found'].items()):
                with col:
                    icon = "✅" if present else "❌"
                    st.markdown(
                        f"{icon} **{comp.title()}**")

        st.divider()

        # ── Feedback ───────────────────────
        st.subheader("💬 Feedback")
        c5, c6, c7 = st.columns(3)
        with c5:
            st.markdown("### ✅ Strengths")
            for s in fb['strengths']:
                st.markdown(
                    f'<div class="feedback-card">'
                    f'<span class="strength-item">'
                    f'✅ {s}</span></div>',
                    unsafe_allow_html=True)
        with c6:
            st.markdown("### ⚠️ Improve")
            for im in fb['improvements']:
                st.markdown(
                    f'<div class="feedback-card">'
                    f'<span class="improve-item">'
                    f'⚠️ {im}</span></div>',
                    unsafe_allow_html=True)
        with c7:
            st.markdown("### 💡 Action Tips")
            for t in fb['tips']:
                pr  = t.split(':')[0]
                css = {'HIGH':'tip-high',
                        'MED' :'tip-med',
                        'LOW' :'tip-low'}.get(
                    pr,'tip-med')
                st.markdown(
                    f'<div class="feedback-card">'
                    f'<span class="{css}">'
                    f'💡 {t}</span></div>',
                    unsafe_allow_html=True)

        st.divider()

        # ── Speech detail ──────────────────
        st.subheader("📝 Speech Quality")
        c8, c9 = st.columns(2)
        with c8:
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Word Count",
                           sq.get('total_words',0))
                st.metric("Filler Words",
                           sq.get('filler_count',0))
                st.metric("Avg Sentence Len",
                           sq.get('avg_sentence_length',0))
            with m2:
                st.metric("Vocab Richness",
                           f"{sq.get('vocabulary_richness',0):.2f}")
                st.metric("Grammar Score",
                           f"{sq.get('grammar_score',0):.0f}%")
                st.metric("Repetition",
                           f"{sq.get('repetition_pct',0):.0f}%")
        with c9:
            st.text_area("Transcript analyzed:",
                          transcript, height=200,
                          disabled=True)

        st.divider()

        # ── Certificate ────────────────────
        st.subheader("🏅 Interview Readiness Summary")
        st.markdown(
            f'<div class="cert-box" style="'
            f'background:linear-gradient(135deg,'
            f'{rec_color}15,{rec_color}30);'
            f'border:2px solid {rec_color};">'
            f'<h2 style="color:{rec_color}">'
            f'🎯 Interview Evaluation Report</h2>'
            f'<h3 style="color:#333">'
            f'Job Role: {job_title or "General"}'
            f'</h3>'
            f'<div style="font-size:4rem;'
            f'font-weight:bold;color:{rec_color}">'
            f'{scores["overall"]}/100</div>'
            f'<div style="font-size:1.5rem;'
            f'color:{rec_color};font-weight:bold">'
            f'{recommendation}</div>'
            f'<hr style="border-color:{rec_color}40">'
            f'<div style="display:flex;'
            f'justify-content:space-around;'
            f'flex-wrap:wrap;margin-top:1rem">'
            + ''.join(
                f'<div style="margin:0.5rem">'
                f'<div style="font-size:1.5rem;'
                f'font-weight:bold;color:{rec_color}">'
                f'{scores[k]:.0f}</div>'
                f'<div style="color:#666;'
                f'font-size:0.85rem">{lbl}</div>'
                f'</div>'
                for k, lbl in [
                    ('communication',  'Communication'),
                    ('professionalism','Professionalism'),
                    ('technical',      'Technical'),
                    ('answer_quality', 'Answer Quality'),
                    ('confidence',     'Confidence')])
            + '</div></div>',
            unsafe_allow_html=True)

        st.divider()

        # ── Download ───────────────────────
        report = {
            'job_title'      : job_title,
            'recommendation' : recommendation,
            'scores'         : scores,
            'camera_presence': {
                'pct'        : face_r['camera_presence_pct'],
                'smile_pct'  : face_r['smile_pct'],
                'impression' : face_r['display_label']},
            'voice'          : {
                'tone'          : audio_r['vocal_tone'],
                'stability'     : audio_r['voice_stability'],
                'pace'          : audio_r['pace_label'],
                'wpm'           : audio_r.get(
                    'words_per_minute'),
                'pause_control' : audio_r['pause_control']},
            'star'           : star,
            'feedback'       : fb,
            'keywords'       : {
                k: v for k,v in kw.items()
                if isinstance(v,dict)},
            'speech_metrics' : sq,
            'transcript'     : transcript}

        st.download_button(
            label="📥 Download Full Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="interview_report.json",
            mime="application/json")

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;
             background:linear-gradient(
                135deg,#667eea10,#764ba210);
             border-radius:20px;margin:2rem 0">
        <h2>🚀 How it works</h2>
        <p style="color:#666;font-size:1.1rem">
        Upload a candidate-only interview video
        or paste your answer text to get started
        </p></div>
        """, unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        for col,emoji,title,desc in [
            (c1,"🎥","Upload","Candidate video"),
            (c2,"🤖","Analyze",
             "Voice + presence + content"),
            (c3,"📊","Score",
             "Weighted evaluation"),
            (c4,"💬","Decide",
             "Hire recommendation")]:
            with col:
                st.markdown(
                    f'<div style="text-align:center;'
                    f'padding:1.5rem;background:#fff;'
                    f'border-radius:15px;'
                    f'box-shadow:0 2px 10px #0001;'
                    f'margin:0.5rem">'
                    f'<div style="font-size:2rem">'
                    f'{emoji}</div>'
                    f'<h4>{title}</h4>'
                    f'<p style="color:#666;'
                    f'font-size:0.85rem">'
                    f'{desc}</p></div>',
                    unsafe_allow_html=True)


if __name__ == "__main__":
    main()
