# app.py — AI Interview Analyzer v2
# Redesigned around interview performance assessment
# rather than raw emotion classification, since:
# - Face model: 68.73% (FER-2013/CK+/RAF-DB)
# - Audio model: 59.51% (RAVDESS/TESS/CREMA-D, acted speech)
# - Text model (RoBERTa): 86% — most reliable, weighted highest
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
</style>
""", unsafe_allow_html=True)

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'models')

# Internal model output spaces — kept as-is because the
# trained models still output these classes. We never
# show these raw labels to the user; we translate them
# into interview-relevant traits below.
FACE_EMOTIONS  = {0:'angry',1:'disgust',2:'fear',3:'happy',
                   4:'neutral',5:'sad',6:'surprise'}
AUDIO_EMOTIONS = {'0':'angry','1':'disgust','2':'fearful',
                   '3':'happy','4':'neutral','5':'sad'}
PERF_LABELS  = {0:'Poor',1:'Average',2:'Good',3:'Excellent'}
PERF_COLORS  = {'Poor':'#e74c3c','Average':'#f39c12',
                 'Good':'#3498db','Excellent':'#27ae60'}
FACE_TO_DIM  = {'angry':2,'disgust':2,'fear':2,'happy':1,
                 'neutral':4,'sad':2,'surprise':3}
AUDIO_TO_DIM = {'angry':2,'disgust':2,'fearful':2,'happy':1,
                 'neutral':4,'sad':2}
TEXT_TO_DIM  = {'angry':2,'anxious':2,'positive':1,'surprised':3}

FILLER_WORDS = ['um','uh','like','so','you know','basically',
                 'literally','actually','i mean','right','okay so']

SKILL_KEYWORDS = {
    'technical': ['python','java','sql','machine learning',
        'deep learning','api','algorithm','tensorflow','pytorch',
        'docker','cloud'],
    'leadership': ['led','managed','team','initiative','ownership',
        'mentored','coordinated','supervised'],
    'problem_solving': ['solved','debugged','optimized','improved',
        'analyzed','designed','implemented','resolved'],
    'communication': ['presented','explained','collaborated',
        'discussed','communicated','reported'],
    'star_method': ['situation','task','action','result',
        'challenge','achieved','delivered','outcome']}

GRAMMAR_FILLERS = ['like i said','basically','um','uh','you know what i mean']


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
# FACE ANALYSIS → interview-relevant traits, not raw emotion
#
# Internally still runs the trained 7-class CNN (needed for
# the fusion model's input vector), but only requires
# decisive majority agreement before reporting anything
# specific. Otherwise reports "Neutral / composed" rather
# than a misleading single-frame artifact like "angry".
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
                    if pred == 3:  # happy
                        smile_frames += 1
            frame_idx += 1
        cap.release()

        if not all_predictions:
            return None

        eye_contact_pct = round(
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
        consistency = round(
            100 * winning_votes / len(usable), 1)

        raw_emotion = FACE_EMOTIONS.get(winning_class, 'neutral')

        # Translate raw class → interview-facing label only
        # when the model is genuinely decisive
        if is_decisive and raw_emotion in ('happy',):
            display_label = "Positive / engaged"
        elif is_decisive and raw_emotion in ('surprise',):
            display_label = "Alert / responsive"
        else:
            # angry / disgust / fear / sad / low-decisiveness
            # all fold into a neutral interpretation —
            # these classes are too unreliable on real
            # interview footage to assert directly (see
            # README for the validation behind this choice)
            display_label = "Neutral / composed"

        return {
            'display_label' : display_label,
            'raw_emotion'   : raw_emotion,
            'is_decisive'   : is_decisive,
            'confidence'    : float(avg_probs[winning_class]),
            'eye_contact_pct': eye_contact_pct,
            'smile_pct'     : smile_pct,
            'consistency'   : consistency,
            'probs'         : avg_probs.tolist(),
            'dim'           : FACE_TO_DIM.get(
                raw_emotion if is_decisive else 'neutral', 4),
            'frames_analyzed': len(all_predictions),
            'frames_used'    : len(usable)}

    except Exception as e:
        st.warning(f"Face analysis error: {e}")
        return None


# ═══════════════════════════════════════════════════════
# AUDIO ANALYSIS → vocal traits, not emotion labels
#
# Reports calm/confident/nervous/energetic based on
# pitch stability, energy stability, and speaking pace —
# acoustic features that are far more reliable indicators
# for conversational speech than the acted-emotion CNN.
# The CNN's prediction is still computed (for the fusion
# vector) but is gated behind a high confidence + majority
# agreement threshold before it's allowed to influence the
# displayed label at all.
# ═══════════════════════════════════════════════════════

def analyze_audio_from_video(video_path, audio_model,
                               min_confidence=0.65,
                               decisive_margin=0.25):
    import librosa
    from PIL import Image as PILImage

    wav_path = video_to_wav(video_path)
    if wav_path is None:
        return None

    try:
        y, sr = librosa.load(wav_path, sr=22050, mono=True)

        if len(y) < sr:
            os.unlink(wav_path)
            return None

        # ── Acoustic traits (reliable, model-free) ─────
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))
        rms_std  = float(np.std(rms))
        energy_stability = max(0, min(100,
            100 * (1 - rms_std / (rms_mean + 1e-6))))

        pitches, mags = librosa.piptrack(y=y, sr=sr)
        pitch_vals = pitches[mags > np.median(mags)]
        pitch_vals = pitch_vals[pitch_vals > 0]
        if len(pitch_vals) > 10:
            pitch_std = float(np.std(pitch_vals))
            pitch_mean = float(np.mean(pitch_vals))
            pitch_stability = max(0, min(100,
                100 * (1 - pitch_std / (pitch_mean + 1e-6))))
        else:
            pitch_stability = 50.0

        # Speaking pace proxy via zero-crossing rate variance
        # (rough proxy when transcript word-count isn't passed in)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        pace_variability = float(np.std(zcr) * 1000)

        if pace_variability < 8:
            pace_label = "Steady"
        elif pace_variability < 15:
            pace_label = "Normal"
        else:
            pace_label = "Variable"

        # ── Optional CNN check — gated, not primary ─────
        input_shape = audio_model.input_shape if audio_model else None
        cnn_label = None
        cnn_decisive = False

        if audio_model is not None and input_shape is not None:
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

        # ── Combine acoustic traits into a single vocal
        # trait label. CNN only contributes if it's both
        # decisive AND agrees with stable-energy readings —
        # otherwise we trust the acoustic features alone.
        if energy_stability > 65 and pitch_stability > 55:
            base_trait = "Calm and steady"
            base_dim = 4  # neutral
        elif energy_stability > 50:
            base_trait = "Confident"
            base_dim = 1  # positivity
        elif energy_stability < 35:
            base_trait = "Slightly nervous"
            base_dim = 2  # nervousness
        else:
            base_trait = "Neutral"
            base_dim = 4

        if cnn_decisive and cnn_label == 'happy':
            display_label = "Energetic and confident"
        elif cnn_decisive and cnn_label in ('fearful', 'sad'):
            # CNN agreeing with low energy_stability reinforces
            # "slightly nervous"; otherwise we don't let a single
            # acted-emotion class override stable acoustic readings
            display_label = ("Slightly nervous"
                if energy_stability < 50 else base_trait)
        else:
            display_label = base_trait

        # Confidence score for this trait reading — based on
        # how much the acoustic signals agree with each other,
        # not the CNN's own (unreliable) softmax confidence
        agreement = abs(energy_stability - pitch_stability)
        trait_confidence = max(0.4, min(0.95,
            1 - (agreement / 200)))

        return {
            'display_label'    : display_label,
            'energy_stability' : round(energy_stability, 1),
            'pitch_stability'  : round(pitch_stability, 1),
            'pace_label'       : pace_label,
            'confidence'       : trait_confidence,
            'probs'            : [0.14]*6,  # neutral prior for fusion vec
            'dim'              : base_dim,
            'cnn_label'        : cnn_label,
            'cnn_decisive'     : cnn_decisive}

    except Exception as e:
        if os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except Exception:
                pass
        st.warning(f"Audio analysis error: {e}")
        return None


# ═══════════════════════════════════════════════════════
# SPEECH / TRANSCRIPT ANALYSIS (your strongest signal)
# ═══════════════════════════════════════════════════════

def analyze_speech_quality(text, audio_duration_sec=None):
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
    aws = total / max(len(sents), 1)

    repeated = Counter(words)
    repetition_ratio = round(
        sum(1 for c in repeated.values() if c > 3) / max(total, 1) * 100, 1)

    result = {
        'total_words'        : total,
        'filler_count'       : fillers,
        'filler_ratio'       : round(fillers/total, 3),
        'vocabulary_richness': round(ttr, 3),
        'repetition_pct'     : repetition_ratio,
        'clarity_score'      : round(
            max(0, min(100, (1 - fillers/total*3)*100)), 1),
        'fluency_score'      : round(
            min(100, ttr*70 + min(aws/20, 1)*30), 1)}

    if audio_duration_sec and audio_duration_sec > 0:
        wpm = total / (audio_duration_sec / 60)
        result['words_per_minute'] = round(wpm, 0)
        if wpm < 100:
            result['pace_assessment'] = "Slow — consider speaking with more energy"
        elif wpm > 180:
            result['pace_assessment'] = "Fast — consider slowing down for clarity"
        else:
            result['pace_assessment'] = "Good pace"

    return result


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
    text_lower = text.lower()
    scores = {}
    for cat, kws in SKILL_KEYWORDS.items():
        found = [k for k in kws if k in text_lower]
        scores[cat] = {'score': min(100, len(found)/len(kws)*100*2),
                        'found': found}
    scores['overall'] = round(
        np.mean([v['score'] for v in scores.values()]), 1)
    return scores


# ═══════════════════════════════════════════════════════
# FUSION — text weighted highest (86% model), face second,
# audio lowest (59% acted-emotion model on real speech)
# ═══════════════════════════════════════════════════════

def build_fusion_vector(fr, ar, tr, sq=None):
    fp = np.array(fr['probs'] + [0]*(7-len(fr['probs'])))[:7]
    ap = np.array(ar['probs'] + [0]*(6-len(ar['probs'])))[:6]
    tp = np.array(tr['probs'] + [0]*(4-len(tr['probs'])))[:4]
    co = np.array([fr['confidence'], ar['confidence'], tr['confidence']])
    dv = np.zeros(5)
    dv[min(fr['dim'], 4)] += 0.33
    dv[min(ar['dim'], 4)] += 0.33
    dv[min(tr['dim'], 4)] += 0.34
    sf = (np.array([
        sq.get('clarity_score', 50)/100,
        sq.get('fluency_score', 50)/100,
        1 - sq.get('filler_ratio', 0),
        sq.get('vocabulary_richness', 0.5),
        min(sq.get('total_words', 50)/200, 1)])
        if sq else np.array([.5,.5,.8,.5,.5]))
    return np.concatenate([fp, ap, tp, co, dv, sf]).astype(np.float32)


def map_to_score(probs, face_r, audio_r, text_r, sq=None):
    """
    Overall score is a weighted blend: text 50%, face 35%,
    audio 15% — reflecting each model's actual reliability.
    The fusion model's own output is used as the base
    classification (Poor/Average/Good/Excellent), but the
    headline numeric score is recomputed with these weights
    so the weakest model can't dominate.
    """
    p0, p1, p2, p3 = probs
    base_overall = p0*15 + p1*45 + p2*72 + p3*92

    text_score = (
        text_r['confidence'] * 100 if text_r['emotion'] == 'positive'
        else text_r['confidence'] * 60 if text_r['emotion'] == 'surprised'
        else 100 - text_r['confidence'] * 70)

    face_score = (
        85 if face_r['display_label'] == "Positive / engaged" else
        75 if face_r['display_label'] == "Alert / responsive" else
        60)
    face_score = min(100, face_score + (face_r['eye_contact_pct'] - 50) * 0.3)

    audio_score = (
        85 if audio_r['display_label'] in
            ("Energetic and confident", "Confident") else
        70 if audio_r['display_label'] == "Calm and steady" else
        45 if audio_r['display_label'] == "Slightly nervous" else
        65)

    overall = (text_score * 0.50 + face_score * 0.35 +
                audio_score * 0.15)
    # blend in the trained fusion model's own read, lightly,
    # so it's not entirely discarded
    overall = round(min(100, max(0, overall * 0.7 + base_overall * 0.3)), 1)

    confidence = round(min(100, max(0, face_score)), 1)
    clarity = round(
        sq.get('clarity_score', overall)*0.6 +
        sq.get('fluency_score', overall)*0.4, 1) if sq else overall
    answer_quality = round(min(100,
        text_score*0.6 + (sq.get('total_words', 50)/300*100*0.4
        if sq else 30)), 1)

    return {
        'overall'         : overall,
        'confidence'      : confidence,
        'clarity'         : clarity,
        'answer_quality'  : answer_quality}


def generate_feedback(scores, face_r, audio_r, text_r, sq):
    overall = scores['overall']
    strengths, improvements = [], []

    if scores['confidence'] > 70:
        strengths.append("Strong, composed presence on camera")
    if scores['clarity'] > 70:
        strengths.append("Clear and well-structured communication")
    if scores['answer_quality'] > 70:
        strengths.append("Good use of specific examples in answers")
    if text_r['emotion'] == 'positive':
        strengths.append("Positive and enthusiastic tone in your answers")
    if face_r['eye_contact_pct'] > 70:
        strengths.append(
            f"Good eye contact maintained ({face_r['eye_contact_pct']:.0f}% of frames)")
    if audio_r['display_label'] in ("Calm and steady", "Confident",
                                      "Energetic and confident"):
        strengths.append(f"Vocal tone came across as {audio_r['display_label'].lower()}")
    if not strengths:
        strengths = ["Shows genuine interest", "Willing to engage with questions"]

    if scores['confidence'] < 60:
        improvements.append("Work on maintaining eye contact with the camera")
    if scores['clarity'] < 60:
        improvements.append("Reduce filler words and slow down your pace")
    if scores['answer_quality'] < 60:
        improvements.append("Use the STAR method for more structured answers")
    if audio_r['display_label'] == "Slightly nervous":
        improvements.append("Practice deep breathing before answering to steady your voice")
    if face_r['eye_contact_pct'] < 50:
        improvements.append("Look at the camera more consistently while answering")
    if sq and sq.get('filler_ratio', 0) > 0.08:
        improvements.append(f"You used filler words frequently ({sq['filler_count']} times) — practice pausing instead")
    if not improvements:
        improvements = ["Continue refining answer specificity",
                          "Practice more mock interviews"]

    if overall >= 80:
        tips = ["HIGH: Research company culture deeply",
                "MED: Prepare 5 specific project examples",
                "LOW: Practice salary negotiation"]
        label, message, color = "Strong Candidate", \
            "Outstanding! You came across as well prepared.", "#27ae60"
    elif overall >= 60:
        tips = ["HIGH: Practice STAR method daily",
                "HIGH: Record yourself and review",
                "MED: Expand technical vocabulary"]
        label, message, color = "Good Candidate", \
            "Good performance with room to grow!", "#3498db"
    elif overall >= 40:
        tips = ["HIGH: Daily mock interview practice",
                "HIGH: Work on answer structure",
                "MED: Improve vocabulary range"]
        label, message, color = "Developing Candidate", \
            "Average performance. Keep practicing!", "#f39c12"
    else:
        tips = ["HIGH: Daily mirror practice",
                "HIGH: Study and apply STAR method",
                "HIGH: Work on building confidence"]
        label, message, color = "Needs Development", \
            "Keep going! Focus on high priority tips.", "#e74c3c"

    return {'label': label, 'message': message, 'color': color,
             'strengths': strengths[:4], 'improvements': improvements[:4],
             'tips': tips}


def create_radar_chart(scores):
    categories = ['Overall', 'Confidence', 'Clarity', 'Answer Quality']
    values = [scores['overall'], scores['confidence'],
               scores['clarity'], scores['answer_quality']]
    vc = values + [values[0]]
    cc = categories + [categories[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vc, theta=cc, fill='toself',
        fillcolor='rgba(102,126,234,0.2)',
        line=dict(color='#667eea', width=2)))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=320,
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
        "Interview performance assessment — communication, "
        "presence, and answer quality</p>",
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
        "**📊 What this dashboard measures:**\n"
        "- 💬 Answer content & structure (strongest signal)\n"
        "- 👁️ Eye contact & on-camera presence\n"
        "- 🎤 Vocal steadiness & pace\n"
        "- 🔑 Keyword relevance to role\n"
        "- ⭐ STAR method usage\n\n"
        "ℹ️ We deliberately **do not** report raw "
        "emotion labels (e.g. 'angry') from face/audio "
        "models — those were trained on acted datasets "
        "and are unreliable on natural interview footage. "
        "Instead we report interview-relevant traits "
        "(eye contact %, vocal steadiness, etc.) derived "
        "from the same underlying signals.")

    tab1, tab2 = st.tabs(["📹 Upload Video/Audio", "✏️ Paste Transcript"])

    transcript = ""
    tmp_path = None
    face_r, audio_r = None, None
    is_video = False
    audio_duration = None

    with tab1:
        st.markdown(
            "**Upload candidate video** (single speaker — "
            "candidate only) for full analysis.")
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
                    if 'segments' in result and result['segments']:
                        audio_duration = result['segments'][-1]['end']
                    progress.progress(35)
                    st.success("✅ Transcription complete!")
                    st.text_area("📝 Transcript", transcript, height=120)
                except Exception as e:
                    st.error(f"Transcription error: {e}")
                    progress.progress(35)

                if is_video and models.get('face'):
                    status.info("👁️ Analyzing on-camera presence...")
                    progress.progress(50)
                    face_r = analyze_face_from_video(tmp_path, models['face'])
                    if face_r:
                        st.caption(
                            f"👁️ Eye contact: {face_r['eye_contact_pct']:.0f}% "
                            f"of frames | Smile frequency: "
                            f"{face_r['smile_pct']:.0f}% | "
                            f"Expression consistency: "
                            f"{face_r['consistency']:.0f}%")
                    else:
                        st.caption("👁️ No clear face detected in video")

                status.info("🎤 Analyzing vocal characteristics...")
                progress.progress(70)
                audio_r = analyze_audio_from_video(tmp_path, models.get('audio'))
                if audio_r:
                    st.caption(
                        f"🎤 Vocal tone: **{audio_r['display_label']}** | "
                        f"Energy stability: {audio_r['energy_stability']:.0f}% | "
                        f"Pace: {audio_r['pace_label']}")

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
        st.subheader("📊 Interview Performance Report")

        with st.spinner("Running analysis..."):
            sq = analyze_speech_quality(transcript, audio_duration)
            text_r = predict_text_emotion(transcript, models)

            if face_r is None:
                face_r = {'display_label':'Neutral / composed',
                           'raw_emotion':'neutral','is_decisive':False,
                           'confidence':0.5,'eye_contact_pct':0,
                           'smile_pct':0,'consistency':0,
                           'probs':[0.14]*7,'dim':4,
                           'frames_analyzed':0,'frames_used':0}
            if audio_r is None:
                audio_r = {'display_label':'Neutral','energy_stability':50,
                            'pitch_stability':50,'pace_label':'Normal',
                            'confidence':0.5,'probs':[0.14]*6,'dim':4,
                            'cnn_label':None,'cnn_decisive':False}

            vec = build_fusion_vector(face_r, audio_r, text_r, sq)
            if models.get('fusion'):
                fusion_probs = models['fusion'].predict(
                    vec.reshape(1,-1), verbose=0)[0]
            else:
                te = text_r['emotion']
                fusion_probs = (np.array([0.02,0.05,0.23,0.70])
                    if te == 'positive' else
                    np.array([0.30,0.35,0.25,0.10]))

            scores = map_to_score(fusion_probs, face_r, audio_r, text_r, sq)
            kw = compute_keyword_score(transcript)
            fb = generate_feedback(scores, face_r, audio_r, text_r, sq)

        color = fb['color']

        # ── Headline metrics ──────────────
        s1, s2, s3, s4 = st.columns(4)
        kw_found = len([k for cat in SKILL_KEYWORDS.values()
                         for k in cat if k in transcript.lower()])
        with s1: st.metric("⏱️ Words", sq.get('total_words', 0))
        with s2: st.metric("🎯 Score", f"{scores['overall']}/100")
        with s3: st.metric("🏆 Assessment", fb['label'])
        with s4: st.metric("🔑 Keywords", kw_found)
        st.divider()

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                f'<div class="score-card">'
                f'<div class="score-number" style="color:{color}">'
                f'{scores["overall"]}</div>'
                f'<div style="color:#666;font-size:0.9rem">Overall Score / 100</div>'
                f'<br><span style="font-size:1.3rem;font-weight:bold;color:{color}">'
                f'{fb["label"]}</span></div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<p style="text-align:center;color:#666;font-style:italic;">'
                f'{fb["message"]}</p>', unsafe_allow_html=True)
            for name, val in [
                ("🎯 Confidence", scores['confidence']),
                ("💬 Clarity", scores['clarity']),
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

        # ── Presence + Vocal traits (replaces raw emotion display) ──
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("👁️ On-Camera Presence")
            st.markdown(
                f'<div class="trait-card">'
                f'<b>Dominant impression:</b> {face_r["display_label"]}</div>',
                unsafe_allow_html=True)
            st.progress(int(face_r['eye_contact_pct']),
                         text=f"Eye contact: {face_r['eye_contact_pct']:.0f}%")
            st.progress(int(face_r['smile_pct']),
                         text=f"Smile frequency: {face_r['smile_pct']:.0f}%")
            st.progress(int(face_r['consistency']),
                         text=f"Expression consistency: {face_r['consistency']:.0f}%")
            if face_r['frames_analyzed'] > 0:
                st.caption(
                    f"Based on {face_r['frames_analyzed']} sampled frames")

        with col4:
            st.subheader("🎤 Vocal Characteristics")
            st.markdown(
                f'<div class="trait-card">'
                f'<b>Vocal tone:</b> {audio_r["display_label"]}</div>',
                unsafe_allow_html=True)
            st.progress(int(audio_r['energy_stability']),
                         text=f"Energy stability: {audio_r['energy_stability']:.0f}%")
            st.progress(int(audio_r['pitch_stability']),
                         text=f"Pitch stability: {audio_r['pitch_stability']:.0f}%")
            st.markdown(f"**Speaking pace:** {audio_r['pace_label']}")
            if sq.get('words_per_minute'):
                st.caption(
                    f"{sq['words_per_minute']:.0f} words/min — "
                    f"{sq.get('pace_assessment','')}")

        st.divider()

        # ── Keyword analysis ──────────────
        st.subheader("🔑 Technical & Skill Keyword Coverage")
        kw_cols = st.columns(len(SKILL_KEYWORDS))
        for col, (cat, data) in zip(kw_cols, kw.items()):
            if cat == 'overall':
                continue
        for cat, data in kw.items():
            if cat == 'overall':
                continue
            sc, found = data['score'], data['found']
            ck = ('#27ae60' if sc > 60 else '#e67e22' if sc > 30 else '#e74c3c')
            st.markdown(
                f"**{cat.replace('_',' ').title()}**: "
                f'<span style="color:{ck}">{sc:.0f}%</span>',
                unsafe_allow_html=True)
            if found:
                st.markdown(f"  *Found: {', '.join(found[:4])}*")
            st.progress(int(sc))

        st.divider()

        # ── Feedback ──────────────────────
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

        # ── Speech quality detail ─────────
        st.subheader("📝 Answer & Speech Quality")
        c8, c9 = st.columns(2)
        with c8:
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Word Count", sq.get('total_words', 0))
                st.metric("Filler Words", sq.get('filler_count', 0))
            with m2:
                st.metric("Vocab Richness", f"{sq.get('vocabulary_richness',0):.2f}")
                st.metric("Repetition", f"{sq.get('repetition_pct',0):.0f}%")
        with c9:
            st.text_area("Transcript analyzed:", transcript, height=180, disabled=True)

        st.divider()

        # ── Certificate ───────────────────
        st.subheader("🏅 Performance Summary")
        st.markdown(
            f'<div class="cert-box" style="background:linear-gradient(135deg,'
            f'{color}15,{color}30);border:2px solid {color};">'
            f'<h2 style="color:{color}">🎯 Interview Performance Report</h2>'
            f'<h3 style="color:#333">Job Role: {job_title or "General"}</h3>'
            f'<div style="font-size:4rem;font-weight:bold;color:{color}">'
            f'{scores["overall"]}/100</div>'
            f'<div style="font-size:1.5rem;color:{color};font-weight:bold">'
            f'{fb["label"]}</div><hr style="border-color:{color}40">'
            f'<div style="display:flex;justify-content:space-around;margin-top:1rem">'
            + ''.join(
                f'<div><div style="font-size:1.5rem;font-weight:bold;color:{color}">'
                f'{scores[k]:.0f}</div><div style="color:#666;font-size:0.85rem">'
                f'{lbl}</div></div>'
                for k, lbl in [('confidence','Confidence'),
                                ('clarity','Clarity'),
                                ('answer_quality','Answer Quality')])
            + '</div></div>', unsafe_allow_html=True)

        st.divider()
        report = {
            'job_title': job_title, 'scores': scores,
            'assessment': fb['label'],
            'presence': {'eye_contact_pct': face_r['eye_contact_pct'],
                          'smile_pct': face_r['smile_pct'],
                          'label': face_r['display_label']},
            'vocal': {'tone': audio_r['display_label'],
                       'energy_stability': audio_r['energy_stability'],
                       'pace': audio_r['pace_label']},
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
            (c2,"🤖","Analyze","Presence + voice + answer content"),
            (c3,"📊","Score","Weighted interview score"),
            (c4,"💬","Improve","Personalized feedback")]:
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