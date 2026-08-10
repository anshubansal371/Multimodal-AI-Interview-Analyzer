# app.py — AI Interview Analyzer (Production Final)
#
# Pipeline:
#   AssemblyAI  → cloud transcription + speaker diarization
#                 + filler words + speech rate (15-30 sec)
#   Face CNN    → facial emotion from real video frames
#   Audio CNN   → vocal traits from mel spectrogram
#   RoBERTa     → text emotion classification (86% acc)
#   Fusion MLP  → confidence-weighted multimodal blend
#   Groq LLaMA  → contextual technical scoring + HR feedback
#
# Speaker separation: AssemblyAI cloud diarization
#   - No local DLLs, no WhisperX conflicts
#   - Works in 15-30 seconds on any video length
#   - User selects which speaker is the candidate
#   - Only candidate speech is analyzed
#
# Score accuracy fixes:
#   - Technical uses LLM context (70%) + keywords (30%)
#   - STAR weight = 0 for introductions
#   - Overall = weighted sum of 5 dimensions (validated)
#   - Confidence = face (60%) + audio (40%)
#   - AssemblyAI filler count used instead of regex estimate

import os, re, json, shutil, tempfile, subprocess
import sqlite3
from datetime import datetime
from collections import Counter

import numpy as np
import torch
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go

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
    with st.spinner(
            "⬇️ Downloading AI models (first run only)..."):
        from download_models import download_all
        download_all()

st.set_page_config(
    page_title="AI Interview Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{font-size:2.5rem;font-weight:bold;
  text-align:center;
  background:linear-gradient(90deg,#667eea,#764ba2);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;padding:1rem 0}
.score-card{background:linear-gradient(135deg,
  #667eea20,#764ba220);border-radius:15px;
  padding:1.5rem;border:1px solid #667eea40;
  text-align:center;margin:.5rem 0}
.score-number{font-size:2.5rem;font-weight:bold;
  color:#667eea}
.trait-card{background:#f8f9fa;border-radius:10px;
  padding:.8rem 1rem;margin:.4rem 0;
  border-left:4px solid #667eea}
.feedback-card{background:#f8f9fa;border-radius:10px;
  padding:1rem;margin:.5rem 0;
  border-left:4px solid #667eea}
.llm-card{background:#f0f7ff;border-radius:10px;
  padding:1rem;margin:.5rem 0;
  border-left:4px solid #3498db}
.strength-item{color:#27ae60}
.improve-item{color:#e67e22}
.tip-high{color:#e74c3c}
.tip-med{color:#f39c12}
.tip-low{color:#27ae60}
.cert-box{border-radius:20px;padding:2rem;
  text-align:center;margin:1rem 0}
.rec-badge{display:inline-block;padding:.5rem 1.5rem;
  border-radius:25px;font-size:1.2rem;
  font-weight:bold;color:white}
.aai-badge{display:inline-block;padding:.15rem .5rem;
  border-radius:8px;font-size:.7rem;font-weight:bold;
  color:white;background:#e74c3c;margin-left:.3rem}
.groq-badge{display:inline-block;padding:.15rem .5rem;
  border-radius:8px;font-size:.7rem;font-weight:bold;
  color:white;background:#3498db;margin-left:.3rem}
.star-box{background:#f0fff4;
  border-left:4px solid #27ae60;
  padding:1.5rem;border-radius:10px;
  font-size:1rem;line-height:1.7}
.encourage-box{background:linear-gradient(135deg,
  #667eea15,#764ba215);border-radius:15px;
  padding:1.5rem;text-align:center;
  border:1px solid #667eea30}
.candidate-turn{background:#f0fff4;border-radius:8px;
  padding:.5rem .8rem;margin:.2rem 0;
  border-left:3px solid #27ae60}
.interviewer-turn{background:#f8f9fa;border-radius:8px;
  padding:.5rem .8rem;margin:.2rem 0;
  border-left:3px solid #95a5a6}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'models')

FACE_EMOTIONS  = {
    0:'angry',1:'disgust',2:'fear',
    3:'happy',4:'neutral',5:'sad',6:'surprise'}
AUDIO_EMOTIONS = {
    '0':'angry','1':'disgust','2':'fearful',
    '3':'happy','4':'neutral','5':'sad'}
FACE_TO_DIM    = {
    'angry':2,'disgust':2,'fear':2,'happy':1,
    'neutral':4,'sad':2,'surprise':3}
AUDIO_TO_DIM   = {
    'angry':2,'disgust':2,'fearful':2,
    'happy':1,'neutral':4,'sad':2}
TEXT_TO_DIM    = {
    'angry':2,'anxious':2,
    'positive':1,'surprised':3}

FILLER_WORDS = [
    'um','uh','like','so','you know',
    'basically','literally','actually',
    'i mean','right','okay so']

TECHNICAL_SKILLS = {
    'python':10,'java':10,'c++':10,'c':5,
    'sql':8,'database':8,'docker':10,
    'kubernetes':10,'machine learning':12,
    'deep learning':12,'tensorflow':10,
    'pytorch':10,'opencv':8,'git':8,
    'github':8,'api':8,'web development':10,
    'project':12,'internship':12,'streamlit':8,
    'algorithm':8,'data structure':8,'cloud':8,
    'aws':10,'flask':8,'django':8,'nlp':10,
    'neural network':10,'research':8,
    'developed':6,'implemented':6,'designed':6}
TECH_MAX = sum(TECHNICAL_SKILLS.values())

INTRO_PATTERNS = [
    'my name is','i am from','i am pursuing',
    'i am doing','my hobbies','my strengths',
    'tell me about yourself','introduce myself',
    'i completed my','i am currently',
    'i have done my','i belong to','i was born']

STAR_COMPONENTS = {
    'situation': [
        'situation','context','background',
        'when i was','in my previous','during',
        'at the time','there was'],
    'task': [
        'task','responsible','goal','objective',
        'assigned','i had to','i needed to',
        'my role was','i was asked'],
    'action': [
        'i implemented','i built','i designed',
        'i developed','i created','i analyzed',
        'i solved','so i','my approach',
        'i decided','i worked','i wrote','i fixed'],
    'result': [
        'result','outcome','achieved','improved',
        'increased','decreased','as a result',
        'successfully','percent','reduced',
        'saved','delivered','completed',
        'in the end']}


# ══════════════════════════════════════════════════════
# DATABASE
# ══════════════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect('interview_progress.db')
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT, job_title TEXT,
            date TEXT, overall REAL,
            communication REAL, technical REAL,
            answer_quality REAL, confidence REAL,
            professionalism REAL,
            recommendation TEXT, transcript TEXT)""")
    conn.commit()
    conn.close()


def save_session(name, job, scores, rec, transcript):
    try:
        conn = sqlite3.connect('interview_progress.db')
        conn.execute("""
            INSERT INTO sessions(
                student_name,job_title,date,overall,
                communication,technical,answer_quality,
                confidence,professionalism,
                recommendation,transcript)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (name, job,
             datetime.now().strftime("%Y-%m-%d %H:%M"),
             scores.get('overall',0),
             scores.get('communication',0),
             scores.get('technical',0),
             scores.get('answer_quality',0),
             scores.get('confidence',0),
             scores.get('professionalism',0),
             rec, transcript[:500]))
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_progress(name):
    try:
        conn = sqlite3.connect('interview_progress.db')
        rows = conn.execute("""
            SELECT date,overall,communication,
                   technical,answer_quality,
                   confidence,recommendation
            FROM sessions
            WHERE student_name=?
            ORDER BY date""", (name,)).fetchall()
        conn.close()
        return rows
    except Exception:
        return []


# ══════════════════════════════════════════════════════
# API CLIENTS
# ══════════════════════════════════════════════════════

@st.cache_resource
def get_groq_client(key):
    if not key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=key)
    except Exception:
        return None


# ══════════════════════════════════════════════════════
# ASSEMBLYAI — transcription + speaker diarization
# Replaces: Whisper + WhisperX + pyannote + VAD
# Speed: 30-sec video processed in ~10 seconds
# Free tier: 100 hours/month
# ══════════════════════════════════════════════════════

def transcribe_and_diarize_assemblyai(
        audio_path, aai_key, num_speakers=2):
    """
    Sends audio to AssemblyAI and receives:
      - Transcript with word-level timestamps
      - Speaker labels per utterance (diarization)
      - Filler word count
      - Words per minute
      - Confidence scores
    Returns (result_dict, error_string)
    """
    try:
        import assemblyai as aai
        aai.settings.api_key = aai_key

        config = aai.TranscriptionConfig(
            speaker_labels    = True,
            speakers_expected = num_speakers,
            punctuate         = True,
            format_text       = True,
            language_code     = "en")

        transcriber = aai.Transcriber()
        transcript  = transcriber.transcribe(
            audio_path, config=config)

        if transcript.status == \
                aai.TranscriptStatus.error:
            return None, transcript.error

        utterances    = transcript.utterances or []
        speaker_times = {}
        speaker_segs  = {}
        exchange      = []

        for utt in utterances:
            spk = utt.speaker
            dur = (utt.end - utt.start) / 1000.0
            speaker_times.setdefault(spk, 0)
            speaker_segs.setdefault(spk, [])
            speaker_times[spk] += dur
            seg = dict(
                text      = utt.text,
                start_ms  = utt.start,
                end_ms    = utt.end,
                confidence= utt.confidence or 0.8)
            speaker_segs[spk].append(seg)
            exchange.append(dict(
                speaker  = spk,
                text     = utt.text,
                start_ms = utt.start,
                end_ms   = utt.end))

        sorted_speakers = sorted(
            speaker_times.items(),
            key=lambda x: x[1], reverse=True)

        txt_lower    = (transcript.text or "").lower()
        filler_count = sum(
            len(re.findall(
                r'\b'+re.escape(f)+r'\b', txt_lower))
            for f in FILLER_WORDS)

        words     = (transcript.text or "").split()
        total_dur = transcript.audio_duration or 1
        wpm       = round(
            len(words)/(total_dur/60), 1)
        avg_conf  = transcript.confidence or 0.8

        return dict(
            full_transcript  = transcript.text or "",
            utterances       = utterances,
            exchange         = exchange,
            speaker_times    = speaker_times,
            speaker_segments = speaker_segs,
            sorted_speakers  = sorted_speakers,
            filler_count     = filler_count,
            words_per_minute = wpm,
            audio_confidence = avg_conf,
            total_words      = len(words),
            audio_duration   = total_dur), None

    except Exception as e:
        return None, str(e)


def get_candidate_transcript(aai_result,
                              candidate_speaker):
    segs = aai_result['speaker_segments'].get(
        candidate_speaker, [])
    return " ".join(
        s['text'] for s in segs
        if s.get('text','').strip())


def build_exchange_html(exchange, candidate_speaker):
    lines = []
    for turn in exchange:
        spk = turn['speaker']
        txt = turn['text']
        if spk == candidate_speaker:
            lines.append(
                f'<div class="candidate-turn">'
                f'🙋 <b>You:</b> {txt}</div>')
        else:
            lines.append(
                f'<div class="interviewer-turn">'
                f'👔 <b>Interviewer:</b> {txt}</div>')
    return "\n".join(lines)


# ══════════════════════════════════════════════════════
# GROQ LLM
# ══════════════════════════════════════════════════════

def analyze_technical_llm(transcript, job_title,
                            question, client):
    if not client or not transcript:
        return None
    q = f"Question: {question}\n" if question else ""
    prompt = f"""Senior technical interviewer for: {job_title}
{q}Answer: {transcript[:1500]}

Return ONLY JSON:
{{
  "technical_score": <0-100>,
  "skills_demonstrated": ["skill1","skill2"],
  "technical_depth": "shallow|moderate|deep",
  "concepts_understood": ["concept1"],
  "missing_skills": ["skill1"],
  "technical_feedback": "one specific sentence"
}}
Give credit for concepts even without exact keywords.
'Optimized slow queries' = SQL credit.
Return ONLY valid JSON."""
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role":"user","content":prompt}],
            temperature=0.1, max_tokens=600)
        raw = r.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(
                r'```json|```','',raw).strip()
        return json.loads(raw)
    except Exception:
        return None


def generate_llm_feedback(transcript, scores,
                            face_r, audio_r,
                            aai_data, sq, star,
                            job_title, question,
                            client):
    if not client or not transcript:
        return None

    filler = (aai_data.get('filler_count',0)
               if aai_data else
               sq.get('filler_count',0))
    wpm    = (aai_data.get('words_per_minute','N/A')
               if aai_data else 'N/A')
    star_info = (
        "Not applicable — introduction"
        if star['is_intro'] else
        f"{star.get('completeness',0):.0f}% complete")
    missing = ([] if star['is_intro'] else [
        k for k,v in
        (star.get('components_found') or {}).items()
        if not v])
    q_line = (f"Question: {question}\n"
               if question else "")

    prompt = f"""Expert interview coach for {job_title}.
{q_line}
CANDIDATE ANSWER: {transcript[:1500]}

SCORES: Overall={scores.get('overall',0)}/100
Communication={scores.get('communication',0)}/100
Technical={scores.get('technical',0)}/100
Answer Quality={scores.get('answer_quality',0)}/100

DELIVERY:
Filler words={filler}, Pace={wpm} WPM
Vocal tone={audio_r.get('vocal_tone','N/A')}
Stability={audio_r.get('voice_stability',0)}%
Camera={face_r.get('camera_presence_pct',0)}%
Expression={face_r.get('display_label','N/A')}

STAR: {star_info}, Missing={missing}

Return ONLY JSON:
{{
  "strengths": [
    "specific strength quoting their exact words",
    "specific strength 2 with evidence",
    "specific strength 3"
  ],
  "improvements": [
    "specific fix with better phrasing example",
    "specific fix 2 referencing their words",
    "specific fix 3"
  ],
  "star_rewrite": "Rewrite in 4-5 STAR sentences. null if introduction.",
  "one_thing": "Single most important improvement",
  "encouragement": "One genuine encouraging sentence"
}}
Quote their actual words. Be specific. ONLY JSON."""
    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role":"user","content":prompt}],
            temperature=0.3, max_tokens=1200)
        raw = r.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(
                r'```json|```','',raw).strip()
        return json.loads(raw)
    except Exception:
        return None


# ══════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    m = {}
    for name, fname in [
        ('face',  'face_model_best.keras'),
        ('audio', 'audio_model_best.keras'),
        ('fusion','fusion_model_best.keras')]:
        fp  = os.path.join(MODELS_DIR, fname)
        alt = fp.replace('.keras','.h5')
        try:
            m[name] = tf.keras.models.load_model(
                fp if os.path.exists(fp) else alt)
            st.sidebar.success(
                f"✅ {name.title()} model")
        except Exception as e:
            st.sidebar.error(f"❌ {name}: {e}")
            m[name] = None

    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification)
        rp = os.path.join(
            MODELS_DIR,'final_roberta_model')
        m['roberta_tok'] = \
            AutoTokenizer.from_pretrained(
                rp, local_files_only=True)
        m['roberta'] = \
            AutoModelForSequenceClassification\
            .from_pretrained(
                rp, local_files_only=True)
        m['roberta'].eval()
        with open(os.path.join(
                rp,'emotion_map.json')) as f:
            e2id = json.load(f)
        m['id2emotion'] = {
            v:k for k,v in e2id.items()}
        st.sidebar.success("✅ Text (RoBERTa)")
    except Exception as e:
        st.sidebar.error(f"❌ Text: {e}")
        m['roberta'] = None
    return m


# ══════════════════════════════════════════════════════
# AUDIO / VIDEO HELPERS
# ══════════════════════════════════════════════════════

def extract_audio(video_path, sr=16000):
    if not shutil.which("ffmpeg"):
        return None
    wav = video_path + "_audio.wav"
    subprocess.run([
        'ffmpeg','-i',video_path,
        '-vn','-ac','1','-ar',str(sr),
        wav,'-y','-loglevel','quiet'],
        capture_output=True)
    return wav if os.path.exists(wav) else None


def run_vad(y, sr, frame_ms=30, pct=30):
    fl = int(sr*frame_ms/1000)
    if fl<=0 or len(y)<fl:
        return [], frame_ms/1000
    n  = len(y)//fl
    en = np.array([
        float(np.sqrt(np.mean(
            y[i*fl:(i+1)*fl]**2)))
        for i in range(n)])
    th = max(np.percentile(en,pct), 1e-4)
    return [bool(e>th) for e in en], frame_ms/1000


def compute_pause_metrics(flags, dur):
    if not flags:
        return dict(
            silence_pct=0, pause_count=0,
            avg_pause_duration=0, pause_ratio=0)
    total  = len(flags)
    silent = sum(1 for v in flags if not v)
    fv = next(
        (i for i,v in enumerate(flags) if v), None)
    lv = next(
        (i for i in range(total-1,-1,-1)
         if flags[i]), None)
    pauses = []
    if fv is not None and lv is not None:
        cur = 0
        for i in range(fv, lv+1):
            if not flags[i]:
                cur += 1
            else:
                if cur >= 3:
                    pauses.append(cur*dur)
                cur = 0
    return dict(
        silence_pct=round(100*silent/total, 1),
        pause_count=len(pauses),
        avg_pause_duration=round(
            float(np.mean(pauses)), 2)
            if pauses else 0.0,
        pause_ratio=round(
            silent/max(total,1), 3))


def estimate_pitch(wav_path):
    try:
        import parselmouth
        s  = parselmouth.Sound(wav_path)
        p  = s.to_pitch()
        pv = p.selected_array['frequency']
        pv = pv[pv>0]
        if len(pv)<10:
            return dict(
                pitch_mean=0,
                pitch_stability=50.0)
        pm,ps = float(np.mean(pv)),float(np.std(pv))
        return dict(
            pitch_mean=round(pm,1),
            pitch_stability=round(
                max(0,min(100,
                    100*(1-ps/(pm+1e-6)))),1))
    except Exception:
        return dict(pitch_mean=0,pitch_stability=50.0)


# ══════════════════════════════════════════════════════
# FACE ANALYSIS
# ══════════════════════════════════════════════════════

def analyze_face(video_path, face_model,
                  every=15, min_conf=0.65,
                  margin=0.20):
    import cv2
    if not face_model:
        return None
    try:
        cas = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            'haarcascade_frontalface_default.xml')
        ish   = face_model.input_shape
        sz,ch = ish[1],ish[-1]
        cap   = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        preds,found,smiles = [],[],0
        idx,sampled = 0,0

        while True:
            ret,frm = cap.read()
            if not ret:
                break
            if idx % every == 0:
                sampled += 1
                gr  = cv2.cvtColor(
                    frm, cv2.COLOR_BGR2GRAY)
                fcs = cas.detectMultiScale(
                    gr,1.2,5,minSize=(60,60))
                if len(fcs)>0:
                    found.append(1)
                    x,y,w,h = max(
                        fcs,
                        key=lambda f:f[2]*f[3])
                    crop = cv2.resize(
                        frm[y:y+h,x:x+w],
                        (sz,sz))
                    img = (
                        cv2.cvtColor(
                            crop,
                            cv2.COLOR_BGR2RGB)
                        if ch==3 else
                        np.expand_dims(
                            cv2.cvtColor(
                                crop,
                                cv2.COLOR_BGR2GRAY),
                            -1))
                    arr   = np.expand_dims(
                        img.astype(np.float32)/255.0,
                        0)
                    probs = face_model.predict(
                        arr, verbose=0)[0]
                    pred  = int(np.argmax(probs))
                    conf  = float(probs[pred])
                    preds.append((pred,conf,probs))
                    if pred==3:
                        smiles += 1
                else:
                    found.append(0)
            idx += 1
        cap.release()

        if not preds:
            return None

        cam   = round(
            100*sum(found)/max(sampled,1), 1)
        smpct = round(
            100*smiles/max(len(preds),1), 1)
        conf_p = [p for p in preds
                   if p[1]>=min_conf]
        use   = conf_p if conf_p else preds
        vc    = Counter(p[0] for p in use)
        ranked= vc.most_common()
        wc,wv = ranked[0]
        ru    = ranked[1][1] if len(ranked)>1 else 0
        is_d  = (wv-ru)/len(use) >= margin
        avgp  = np.mean(
            [p[2] for p in use if p[0]==wc],
            axis=0)
        raw   = FACE_EMOTIONS.get(wc,'neutral')
        lbl   = (
            "Positive / engaged"
            if is_d and raw=='happy' else
            "Alert / responsive"
            if is_d and raw=='surprise' else
            "Neutral / composed")

        return dict(
            display_label=lbl,
            raw_emotion=raw,
            is_decisive=is_d,
            confidence=float(avgp[wc]),
            camera_presence_pct=cam,
            smile_pct=smpct,
            consistency=round(
                100*wv/len(use),1),
            probs=avgp.tolist(),
            dim=FACE_TO_DIM.get(
                raw if is_d else 'neutral',4),
            frames_analyzed=len(preds),
            frames_used=len(use))

    except Exception as e:
        st.warning(f"Face error: {e}")
        return None


# ══════════════════════════════════════════════════════
# AUDIO EMOTION — vocal traits not raw emotion labels
# ══════════════════════════════════════════════════════

def analyze_audio_emotion(wav_path, audio_model,
                            wpm=None,
                            min_conf=0.65,
                            dec_margin=0.25):
    import librosa
    if not wav_path:
        return None
    try:
        y,sr  = librosa.load(
            wav_path, sr=16000, mono=True)
        dur   = len(y)/sr
        if dur < 1:
            return None

        flags,fd = run_vad(y, sr)
        pm       = compute_pause_metrics(flags, fd)
        pitch    = estimate_pitch(wav_path)
        rms      = librosa.feature.rms(y=y)[0]
        rmm,rms2 = float(np.mean(rms)),float(np.std(rms))
        es = max(0, min(100,
            100*(1-rms2/(rmm+1e-6))))
        ps = pitch['pitch_stability']

        cnn_lbl, cnn_dec = None, False
        if audio_model:
            from PIL import Image as PILImage
            ish = audio_model.input_shape
            ih,iw,ich = ish[1],ish[2],ish[-1]
            wl = sr*3; ap = []
            for s2 in range(0, len(y)-wl, wl):
                ch2 = y[s2:s2+wl]
                mel = librosa.feature.melspectrogram(
                    y=ch2, sr=sr, n_mels=ih,
                    n_fft=2048, hop_length=512)
                md  = librosa.power_to_db(
                    mel, ref=np.max)
                mi  = np.array(
                    PILImage.fromarray(
                        md).resize((iw,ih)))
                mn,mx = mi.min(),mi.max()
                if mx-mn < 1e-8:
                    continue
                mn2 = (mi-mn)/(mx-mn)
                a   = (
                    np.repeat(
                        mn2[...,None],3,axis=-1)
                    if ich==3 else
                    mn2[...,None])
                a   = np.expand_dims(
                    a.astype(np.float32), 0)
                pp  = audio_model.predict(
                    a, verbose=0)[0]
                ap.append((
                    int(np.argmax(pp)),
                    float(pp[np.argmax(pp)]),
                    pp))

            if ap:
                cp = [x for x in ap
                       if x[1]>=min_conf]
                u  = cp if cp else ap
                vc = Counter(x[0] for x in u)
                top= vc.most_common()
                wc2,wv2 = top[0]
                ru2 = top[1][1] if len(top)>1 else 0
                m2  = (wv2-ru2)/len(u)
                cnn_dec = (
                    m2>=dec_margin and len(cp)>=3)
                if cnn_dec:
                    cnn_lbl = AUDIO_EMOTIONS.get(
                        str(wc2),'neutral')

        if es>65 and ps>55:
            tone = "Calm and steady"
        elif es>50 and pm['pause_ratio']<0.35:
            tone = "Confident"
        elif es<35 or pm['pause_count']>8:
            tone = "Slightly nervous"
        else:
            tone = "Neutral"

        if (cnn_dec and
                cnn_lbl in ('fearful','sad') and
                es<50):
            tone = "Slightly nervous"
        elif (cnn_dec and
                cnn_lbl=='happy' and es>55):
            tone = "Energetic and confident"

        pc = (
            "Excellent"
            if pm['pause_count']==0 else
            "Good"
            if pm['avg_pause_duration']<1.0 else
            "Fair"
            if pm['avg_pause_duration']<2.5 else
            "Needs work")

        vs = round((es+ps)/2, 1)
        ac = max(0.4, min(0.95,
            1-abs(es-ps)/200))

        return dict(
            vocal_tone=tone,
            voice_stability=vs,
            energy_stability=round(es,1),
            pitch_stability=ps,
            pitch_mean=pitch['pitch_mean'],
            pace_label=(
                "Unknown" if not wpm else
                "Slow"    if wpm<100  else
                "Fast"    if wpm>180  else
                "Good pace"),
            words_per_minute=wpm,
            silence_pct=pm['silence_pct'],
            pause_count=pm['pause_count'],
            avg_pause_duration=pm[
                'avg_pause_duration'],
            pause_ratio=pm['pause_ratio'],
            pause_control=pc,
            confidence=ac,
            probs=[0.14]*6,
            dim=(
                1 if tone in (
                    "Confident",
                    "Energetic and confident")
                else 2 if tone=="Slightly nervous"
                else 4))

    except Exception as e:
        st.warning(f"Audio emotion error: {e}")
        return None


# ══════════════════════════════════════════════════════
# TEXT ANALYSIS
# ══════════════════════════════════════════════════════

def analyze_speech_quality(text, aai_data=None):
    if not text:
        return {}
    words = text.lower().split()
    total = len(words)
    if total == 0:
        return {}

    # Use AssemblyAI filler count if available
    if (aai_data and
            aai_data.get('filler_count')
            is not None):
        fillers = int(aai_data['filler_count'])
    else:
        fillers = sum(
            len(re.findall(
                r'\b'+re.escape(f)+r'\b',
                text.lower()))
            for f in FILLER_WORDS)

    ttr   = len(set(words)) / total
    sents = [s for s in
             re.split(r'[.!?]+', text.strip())
             if len(s.strip()) > 0]
    aws   = round(
        total / max(len(sents), 1), 1)
    wc    = Counter(words)
    rep   = [w for w, c in wc.items()
              if c > 3 and len(w) > 3]
    rep_p = round(
        sum(wc[w] for w in rep) /
        max(total, 1) * 100, 1)

    ls    = sum(
        1 for s in sents
        if len(s.split()) > 40)

    # Filler ratio — avoid division issues
    filler_ratio = fillers / max(total, 1)

    # Grammar score
    gram = max(0, min(100,
        100 -
        ls * 8 -
        filler_ratio * 100))

    # Penalties — proportional, capped
    fp = min(30, filler_ratio * 200)
    rp = min(15, rep_p * 0.4)

    # Vocabulary score — scaled properly
    # ttr of 0.6 = good diversity = 72 score
    vocab_score = min(100, ttr * 120)

    # Sentence length score
    # 15-20 words per sentence = ideal
    if aws >= 8 and aws <= 25:
        sent_score = 100
    elif aws < 8:
        sent_score = max(0, aws * 12)
    else:
        sent_score = max(0,
            100 - (aws - 25) * 3)

    # COMMUNICATION — weighted blend
    communication = max(0, min(100,
        gram        * 0.30 +
        (100 - fp)  * 0.25 +
        (100 - rp)  * 0.15 +
        vocab_score * 0.20 +
        sent_score  * 0.10))

    # Debug print — remove after testing
    print(f"DEBUG sq: total={total}, "
          f"fillers={fillers}, "
          f"filler_ratio={filler_ratio:.3f}, "
          f"gram={gram:.1f}, "
          f"fp={fp:.1f}, rp={rp:.1f}, "
          f"vocab={vocab_score:.1f}, "
          f"sent={sent_score:.1f}, "
          f"COMM={communication:.1f}")

    return dict(
        total_words          = total,
        filler_count         = fillers,
        filler_ratio         = round(
            filler_ratio, 3),
        vocabulary_richness  = round(ttr, 3),
        avg_sentence_length  = aws,
        repetition_pct       = rep_p,
        repeated_words       = rep[:5],
        grammar_score        = round(gram, 1),
        clarity_score        = round(max(0, min(
            100, (1 - filler_ratio*3)*100)), 1),
        fluency_score        = round(min(
            100, ttr*70+min(aws/20,1)*30), 1),
        communication_score  = round(
            communication, 1))


def is_intro(text):
    """
    Only returns True if the MAJORITY of the text
    is introduction. Not if intro phrases appear
    anywhere in a longer answer.
    """
    tl    = text.lower()
    words = tl.split()
    total = len(words)
    if total == 0:
        return False

    # Count intro-pattern words vs total
    intro_word_count = 0
    for p in INTRO_PATTERNS:
        if p in tl:
            # Count words in this pattern
            intro_word_count += len(p.split())

    # Only flag as intro if intro content is
    # more than 60% of the text AND text is short
    intro_ratio = intro_word_count / max(total, 1)

    # Short text (< 80 words) with intro patterns
    if total < 80 and intro_ratio > 0.1:
        return True

    # Long text — only flag if STARTS with intro
    # and intro is very dominant
    if total >= 80:
        # Check first 50 words only
        first_50 = ' '.join(words[:50])
        has_intro_start = any(
            p in first_50
            for p in INTRO_PATTERNS)
        # Long text with Q&A after intro = not intro
        if has_intro_start and total > 150:
            return False
        if has_intro_start and total <= 150:
            return True

    return False


def compute_star(text):
    """
    For long transcripts with intro + Q&A,
    skip the intro portion and analyze the rest.
    """
    tl    = text.lower()
    words = tl.split()
    total = len(words)

    # If short and is intro — skip STAR
    if total < 80 and is_intro(text):
        return dict(
            completeness=None,
            components_found=None,
            is_intro=True)

    # For long transcripts — remove intro portion
    # (first 60 words typically) and analyze rest
    if total > 100:
        # Find where Q&A likely starts
        # (after introductory phrases)
        intro_end_idx = 0
        for i, p in enumerate(INTRO_PATTERNS):
            if p in ' '.join(words[:80]):
                # Intro found in first 80 words
                # Analyze from word 60 onwards
                intro_end_idx = min(60, total//4)
                break

        # Analyze the non-intro portion
        analysis_text = ' '.join(
            words[intro_end_idx:])
    else:
        analysis_text = tl

    found = {}
    for comp, kws in STAR_COMPONENTS.items():
        found[comp] = any(
            re.search(
                r'\b'+re.escape(kw)+r'\b',
                analysis_text)
            for kw in kws)

    completeness = round(
        100*sum(found.values())/4, 1)

    return dict(
        completeness=completeness,
        components_found=found,
        is_intro=False)


def compute_keywords(text):
    tl = text.lower()
    raw, fnd = 0, []
    for sk, w in TECHNICAL_SKILLS.items():
        if re.search(
                r'\b'+re.escape(sk)+r'\b', tl):
            raw += w
            fnd.append(sk)
    tech = min(100, (raw/TECH_MAX)*100*0.8*5)

    OTHER = {
        'leadership': [
            'led','managed','team','initiative',
            'ownership','mentored','coordinated',
            'supervised','organized','directed',
            'handled','responsible','in charge',
            'head','guide','delegate','lead',
            'our team','my team','i led',
            'i managed','i headed','i guided'],

        'problem_solving': [
            'solved','debugged','optimized',
            'improved','analyzed','designed',
            'implemented','resolved','fixed',
            'approach','challenge','issue',
            'problem','solution','overcome',
            'identify','diagnose','addressed',
            'worked on','figured out','built',
            'created','developed','tackled'],

        # FIXED — communication keywords now
        # match natural interview speech
        'communication': [
            # Direct communication words
            'explained','presented','discussed',
            'communicated','reported','described',
            'mentioned','expressed','shared',
            'told','asked','answered','spoke',
            'talked','informed','conveyed',
            # Collaboration words
            'collaborated','worked with',
            'coordinated','cooperated',
            'contributed','supported','assisted',
            # Team/people words
            'team','client','customer','manager',
            'colleague','stakeholder','member',
            'meeting','interview','presentation',
            # Natural phrases
            'i explained','i presented',
            'i discussed','i worked with',
            'i told','i shared','i asked',
            'we discussed','we worked',
            'with my team','with the team',
            'with my manager','with clients'],

        'star_method': [
            'situation','task','action','result',
            'challenge','achieved','delivered',
            'outcome','background','context',
            'responsible','goal','objective',
            'approach','decided','as a result',
            'led to','therefore','in the end',
            'successfully','improved','increased',
            'during','when i','i was given',
            'i needed to','i had to']}

    sc = {'technical': dict(
        score=round(tech,1), found=fnd[:6])}

    for cat, kws in OTHER.items():
        found = []
        for k in kws:
            # Use word boundary for single words
            # Use simple 'in' for phrases
            if ' ' in k:
                if k in tl:
                    found.append(k)
            else:
                if re.search(
                        r'\b'+re.escape(k)+r'\b',
                        tl):
                    found.append(k)
        # Scale: finding 5+ keywords = 100%
        score = min(100,
            len(found)/max(len(kws),1)*400)
        sc[cat] = dict(
            score=round(score,1),
            found=found[:5])

    sc['overall'] = round(np.mean(
        [v['score'] for v in sc.values()]), 1)
    return sc


def predict_text_emotion(text, models):
    if not models.get('roberta'):
        return dict(
            emotion='positive', confidence=0.5,
            probs=[0.1,0.1,0.7,0.1], dim=1)
    try:
        inp = models['roberta_tok'](
            text, return_tensors='pt',
            truncation=True,
            max_length=256, padding=True)
        with torch.no_grad():
            out   = models['roberta'](**inp)
            probs = torch.softmax(
                out.logits, dim=1).numpy()[0]
        pred = np.argmax(probs)
        em   = models['id2emotion'][pred]
        return dict(
            emotion=em,
            confidence=float(probs[pred]),
            probs=probs.tolist(),
            dim=TEXT_TO_DIM.get(em, 4))
    except Exception:
        return dict(
            emotion='positive', confidence=0.5,
            probs=[0.1,0.1,0.7,0.1], dim=1)


# ══════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════

def compute_scores(text_r, face_r, audio_r,
                    sq, star, kw,
                    llm_tech=None,
                    aai_data=None):

    # FIXED — safe .get with fallback calculation
    comm_score = sq.get('communication_score', None)

    if comm_score is None or comm_score == 0:
        # Recalculate directly from sq values
        # in case key was missing
        gram   = sq.get('grammar_score', 70)
        fr     = sq.get('filler_ratio', 0)
        rep_p  = sq.get('repetition_pct', 0)
        ttr    = sq.get('vocabulary_richness', 0.5)
        aws    = sq.get('avg_sentence_length', 15)

        fp = min(30, fr * 200)
        rp = min(15, rep_p * 0.4)
        vs = min(100, ttr * 120)
        ss = (100 if 8 <= aws <= 25 else
               max(0, aws*12) if aws < 8 else
               max(0, 100-(aws-25)*3))

        communication = max(0, min(100,
            gram*0.30 + (100-fp)*0.25 +
            (100-rp)*0.15 + vs*0.20 +
            ss*0.10))
        print(f"DEBUG: Recalculated comm = "
              f"{communication:.1f}")
    else:
        communication = float(comm_score)

    print(f"DEBUG compute_scores: "
          f"communication = {communication:.1f}")
    
    # ── TECHNICAL ─────────────────────────────────
    # LLM gives contextual score (70%)
    # Keywords give surface score (30%)
    kw_tech = kw.get(
        'technical',{}).get('score', 0)
    if llm_tech and \
            llm_tech.get('technical_score') \
            is not None:
        llm_t = float(
            llm_tech['technical_score'])
        technical = round(
            llm_t * 0.70 + kw_tech * 0.30, 1)
    else:
        technical = round(kw_tech, 1)
    technical = max(0, min(100, technical))

    # ── ANSWER QUALITY ────────────────────────────
    # STAR completeness (40%)
    # Keyword relevance (40%)
    # Text emotion (20%)
    star_comp = star.get('completeness')
    star_w    = (0.0 if star['is_intro']
                  or star_comp is None
                  else float(star_comp))

    # Text score — positive emotion = better answer
    if text_r['emotion'] == 'positive':
        text_s = min(100,
            text_r['confidence'] * 100)
    elif text_r['emotion'] == 'surprised':
        text_s = min(100,
            text_r['confidence'] * 70)
    else:
        text_s = max(20,
            100 - text_r['confidence'] * 60)

    kw_overall = kw.get('overall', 30)
    answer_quality = round(min(100,
        kw_overall * 0.40 +
        star_w     * 0.40 +
        text_s     * 0.20), 1)

    # ── CONFIDENCE ────────────────────────────────
    # Face presence (40%) + Vocal stability (40%)
    # + AAI audio confidence (20%)
    face_s = (
        90 if face_r['display_label'] ==
            "Positive / engaged" else
        75 if face_r['display_label'] ==
            "Alert / responsive" else 60)
    # Camera presence adjusts face score
    cam_pct = face_r['camera_presence_pct']
    if cam_pct > 80:
        face_s = min(100, face_s + 5)
    elif cam_pct < 40:
        face_s = max(0, face_s - 15)

    # Audio/vocal contribution
    vt = audio_r.get('vocal_tone', 'Neutral')
    audio_s = (
        90 if vt == "Energetic and confident"
        else 80 if vt == "Confident"
        else 70 if vt == "Calm and steady"
        else 40 if vt == "Slightly nervous"
        else 60)
    pc = audio_r.get('pause_control', 'Fair')
    if pc == "Needs work": audio_s -= 10
    elif pc == "Excellent": audio_s += 5
    audio_s = max(0, min(100, audio_s))

    # AAI transcription confidence as proxy
    aai_conf_bonus = 0
    if aai_data:
        aai_c = aai_data.get(
            'audio_confidence', 0.8)
        aai_conf_bonus = (aai_c - 0.7) * 20

    confidence = round(
        face_s  * 0.40 +
        audio_s * 0.40 +
        min(10, max(-10,
            aai_conf_bonus)), 1)
    confidence = max(0, min(100, confidence))

    # ── PROFESSIONALISM ───────────────────────────
    # Grammar (35%) + low fillers (30%)
    # + low repetition (15%) + vocabulary (20%)
    grammar  = sq.get('grammar_score', 70)
    filler_r = sq.get('filler_ratio', 0)
    rep_pct  = sq.get('repetition_pct', 0)
    vocab    = sq.get('vocabulary_richness', 0.5)

    filler_prof = max(0, 100 - filler_r * 300)
    rep_prof    = max(0, 100 - rep_pct * 2)
    vocab_prof  = min(100, vocab * 120)

    professionalism = round(min(100,
        grammar      * 0.35 +
        filler_prof  * 0.30 +
        rep_prof     * 0.15 +
        vocab_prof   * 0.20), 1)

    # ── FUSION — confidence weighted ──────────────
    B_T, B_F, B_A = 0.55, 0.30, 0.15
    w_t = B_T * text_r['confidence']
    w_f = B_F * face_r['confidence']
    w_a = B_A * audio_r['confidence']
    tot = w_t + w_f + w_a + 1e-6
    w_t,w_f,w_a = w_t/tot,w_f/tot,w_a/tot

    fusion_est = round(min(100, max(0,
        text_s  * w_t +
        face_s  * w_f +
        audio_s * w_a)), 1)

    # ── OVERALL — validated weighted blend ────────
    # Weights reflect importance in interview
    dim_overall = (
        communication   * 0.30 +
        professionalism * 0.20 +
        technical       * 0.20 +
        answer_quality  * 0.20 +
        confidence      * 0.10)

    # Blend: 70% dimensional, 30% fusion estimate
    overall = round(min(100, max(0,
        dim_overall * 0.70 +
        fusion_est  * 0.30)), 1)

    return dict(
        overall          = overall,
        communication    = round(communication,1),
        professionalism  = professionalism,
        technical        = technical,
        technical_relevance = technical,
        answer_quality   = answer_quality,
        confidence       = confidence,
        weights_used     = dict(
            text  = round(w_t,2),
            face  = round(w_f,2),
            audio = round(w_a,2)))


def get_recommendation(scores, star_comp):
    o = scores['overall']
    t = scores['technical']
    c = scores['communication']
    if o>=85 and t>=65 and c>=70:
        return "Strong Hire","#27ae60"
    elif o>=70 and (
            star_comp is None or star_comp>=50):
        return "Hire","#3498db"
    elif o>=55:
        return "Consider","#f39c12"
    else:
        return "Needs Improvement","#e74c3c"


def fallback_feedback(scores, face_r, audio_r,
                       sq, star, kw):
    s,im,tips = [],[],[]
    if scores['communication']>70:
        s.append(
            f"Clear communication — vocab "
            f"{sq.get('vocabulary_richness',0):.2f}, "
            f"grammar "
            f"{sq.get('grammar_score',0):.0f}%")
    if scores['confidence']>70:
        s.append(
            f"Good presence — "
            f"{face_r['camera_presence_pct']:.0f}%"
            f" in frame, tone: "
            f"{audio_r['vocal_tone'].lower()}")
    if (not star['is_intro'] and
            star.get('completeness') and
            star['completeness']>=75):
        s.append(
            f"Strong STAR — "
            f"{sum(star['components_found'].values())}"
            f"/4 components covered")
    if not s:
        s = ["Shows genuine engagement",
              "Completed a full response"]

    if scores['technical']<60:
        im.append(
            "Mention specific technologies used")
    if sq.get('filler_count',0)>10:
        im.append(
            f"Reduce filler words "
            f"({sq.get('filler_count',0)} detected)")
    if (not star['is_intro'] and
            star.get('completeness') is not None and
            star['completeness']<50):
        m = [k for k,v in
             (star.get('components_found') or {})
             .items() if not v]
        im.append(f"STAR missing: {', '.join(m)}")
    if not im:
        im = ["Refine answer specificity",
               "Practice more mock interviews"]

    o = scores['overall']
    if o>=80:
        tips = [
            "HIGH: Research company culture",
            "MED: Prepare 5 project examples",
            "LOW: Practice salary negotiation"]
    elif o>=60:
        tips = [
            "HIGH: Practice STAR daily",
            "HIGH: Record and review yourself",
            "MED: Expand technical vocabulary"]
    else:
        tips = [
            "HIGH: Daily mock interviews",
            "HIGH: Work on STAR structure",
            "HIGH: Reduce filler words"]
    return dict(
        strengths=s[:3],
        improvements=im[:3],
        tips=tips)


# ══════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════

def radar_chart(scores):
    cats = [
        'Communication','Confidence',
        'Professionalism','Technical',
        'Answer Quality']
    vals = [
        scores['communication'],
        scores['confidence'],
        scores['professionalism'],
        scores['technical'],
        scores['answer_quality']]
    vc,cc = vals+[vals[0]],cats+[cats[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vc, theta=cc, fill='toself',
        fillcolor='rgba(102,126,234,0.2)',
        line=dict(color='#667eea',width=2)))
    fig.update_layout(
        polar=dict(radialaxis=dict(
            visible=True,range=[0,100])),
        showlegend=False, height=340,
        margin=dict(t=30,b=30,l=30,r=30),
        paper_bgcolor='rgba(0,0,0,0)')
    return fig


def benchmark_chart(scores, color):
    bm = {
        'Your Score'       : scores['overall'],
        'Average Candidate': 58.0,
        'Good Candidate'   : 72.0,
        'Top Candidate'    : 88.0}
    fig = go.Figure(go.Bar(
        x=list(bm.values()),
        y=list(bm.keys()),
        orientation='h',
        marker_color=[
            color,'#95a5a6','#3498db','#27ae60'],
        text=[f"{v:.1f}" for v in bm.values()],
        textposition='auto'))
    fig.update_layout(
        xaxis=dict(range=[0,100]), height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10,b=10,l=10,r=10))
    return fig


def progress_chart(history):
    if not history or len(history)<2:
        return None
    dates = [h[0] for h in history]
    fig   = go.Figure()
    for y,name,col in [
        ([h[1] for h in history],
         'Overall','#667eea'),
        ([h[2] for h in history],
         'Communication','#27ae60'),
        ([h[3] for h in history],
         'Technical','#e74c3c')]:
        fig.add_trace(go.Scatter(
            x=dates, y=y,
            mode='lines+markers', name=name,
            line=dict(color=col,width=2),
            marker=dict(size=7)))
    fig.update_layout(
        title='Your Progress Over Time',
        yaxis=dict(range=[0,100]), height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=40,b=20,l=20,r=20))
    return fig


# ══════════════════════════════════════════════════════
# DEFAULTS
# ══════════════════════════════════════════════════════

def default_face():
    return dict(
        display_label='N/A',
        raw_emotion='neutral',
        is_decisive=False, confidence=0.5,
        camera_presence_pct=0, smile_pct=0,
        consistency=0, probs=[0.14]*7, dim=4,
        frames_analyzed=0, frames_used=0)


def default_audio():
    return dict(
        vocal_tone='N/A', voice_stability=0,
        energy_stability=50, pitch_stability=50,
        pitch_mean=0, loudness_variation=0,
        pace_label='N/A', words_per_minute=None,
        silence_pct=0, pause_count=0,
        avg_pause_duration=0, pause_ratio=0,
        pause_control='N/A', confidence=0.5,
        probs=[0.14]*6, dim=4,
        cnn_label=None, cnn_decisive=False)


# ══════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════

def main():
    init_db()

    st.markdown(
        '<div class="main-header">'
        '🎯 AI Interview Analyzer</div>',
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:#666'>"
        "AssemblyAI • Groq LLaMA • "
        "Multimodal Deep Learning</p>",
        unsafe_allow_html=True)
    st.divider()

    # ── Sidebar ───────────────────────────────────
    st.sidebar.title("⚙️ Settings")
    st.sidebar.subheader("🤖 Models")
    with st.spinner("Loading AI models..."):
        models = load_models()

    st.sidebar.divider()
    student_name = st.sidebar.text_input(
        "👤 Your Name",
        placeholder="For progress tracking")
    job_title = st.sidebar.text_input(
        "🎯 Target Job Role",
        placeholder="e.g. Software Engineer")

    st.sidebar.divider()
    st.sidebar.subheader("🔑 API Keys")

    aai_key = st.sidebar.text_input(
        "AssemblyAI Key",
        type="password",
        placeholder="your_key_here",
        help="Free at assemblyai.com\n"
             "Handles: transcription + speaker\n"
             "diarization + filler + pace\n"
             "Speed: 30-sec video in ~10 sec")
    aai_ok = False
    if aai_key:
        try:
            import assemblyai
            aai_ok = True
            st.sidebar.success(
                "✅ AssemblyAI\n"
                "Speaker separation enabled")
        except ImportError:
            st.sidebar.error(
                "❌ pip install assemblyai")
    else:
        st.sidebar.info(
            "ℹ️ Add AssemblyAI key for\n"
            "automatic speaker separation")

    groq_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com\n"
             "Handles: technical scoring + feedback")
    groq_client = get_groq_client(groq_key)
    if groq_key and groq_client:
        st.sidebar.success(
            "✅ Groq — AI feedback")
    elif groq_key:
        st.sidebar.error("❌ Invalid Groq key")
    else:
        st.sidebar.info(
            "ℹ️ Add Groq key for AI feedback")

    st.sidebar.divider()
    st.sidebar.markdown(
        "**📊 Pipeline:**\n"
        "1. 🎙️ AssemblyAI → transcribe + diarize\n"
        "2. 😊 Face CNN → expression analysis\n"
        "3. 🎤 Audio CNN → vocal traits\n"
        "4. 💬 RoBERTa → answer emotion\n"
        "5. 🔀 Fusion → combine signals\n"
        "6. 🤖 Groq → personalized feedback")

    # ── Tabs ──────────────────────────────────────
    tab1,tab2,tab3 = st.tabs([
        "📹 Upload Video/Audio",
        "✏️ Paste Transcript",
        "📈 My Progress"])

    transcript      = ""
    tmp_path        = None
    face_r          = None
    audio_r         = None
    aai_data        = None
    is_video        = False
    transcript_only = False
    interview_q     = ""

    # ── Tab 1 ─────────────────────────────────────
    with tab1:
        interview_q = st.text_input(
            "❓ Interview question (optional)",
            placeholder="e.g. Tell me about a "
                        "challenging project...",
            help="Helps AI evaluate your answer "
                 "more accurately")

        num_spk = st.number_input(
            "👥 Number of speakers in video",
            min_value=1, max_value=4, value=2,
            help="2 = candidate + interviewer")

        uploaded = st.file_uploader(
            "Upload interview video or audio",
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
            is_video = uploaded.type.startswith(
                'video')
            if is_video: st.video(uploaded)
            else: st.audio(uploaded)

            if st.button(
                    "🎯 Analyze Interview",
                    type="primary",
                    key="btn_v"):

                progress = st.progress(0)
                status   = st.empty()

                # ── AssemblyAI ────────────────────
                if aai_key and aai_ok:
                    status.info(
                        "🎙️ AssemblyAI: transcribing "
                        "and separating speakers... "
                        "(15-30 seconds)")
                    progress.progress(10)

                    wav_tmp = extract_audio(tmp_path)
                    audio_src = wav_tmp or tmp_path

                    aai_result, err = \
                        transcribe_and_diarize_assemblyai(
                            audio_src,
                            aai_key,
                            num_speakers=int(num_spk))

                    if wav_tmp and \
                            os.path.exists(wav_tmp):
                        try: os.unlink(wav_tmp)
                        except: pass

                    if aai_result:
                        sorted_spk = \
                            aai_result['sorted_speakers']
                        n_spk = len(sorted_spk)

                        if n_spk >= 2:
                            st.success(
                                f"✅ Found {n_spk} "
                                f"speakers — select "
                                f"which is YOU")

                            c1,c2 = st.columns(2)
                            with c1:
                                st.markdown(
                                    "**🎤 Speaking time:**")
                                for i,(spk,dur) in \
                                        enumerate(
                                            sorted_spk):
                                    icon = ("🙋"
                                             if i==0
                                             else "👔")
                                    lbl  = ("Likely YOU"
                                             if i==0 else
                                             "Likely interviewer")
                                    st.markdown(
                                        f"{icon} {lbl}: "
                                        f"**{dur:.0f}s**")

                            with c2:
                                opts = [
                                    f"Speaker {i+1} "
                                    f"({dur:.0f}s speaking)"
                                    for i,(spk,dur)
                                    in enumerate(sorted_spk)]
                                sel = st.selectbox(
                                    "✅ Select YOUR voice:",
                                    opts, index=0)
                                cand_spk = sorted_spk[
                                    opts.index(sel)][0]

                            transcript = \
                                get_candidate_transcript(
                                    aai_result,
                                    cand_spk)
                            aai_data = aai_result

                            with st.expander(
                                    "📝 Full conversation "
                                    "(click to expand)",
                                    expanded=False):
                                html = \
                                    build_exchange_html(
                                        aai_result[
                                            'exchange'],
                                        cand_spk)
                                st.markdown(
                                    html,
                                    unsafe_allow_html=True)

                            st.text_area(
                                "📝 YOUR transcript "
                                "(interviewer removed)",
                                transcript,
                                height=120)

                            m1,m2,m3,m4 = st.columns(4)
                            with m1: st.metric(
                                "🔤 Words",
                                aai_data['total_words'])
                            with m2: st.metric(
                                "⚡ Pace",
                                f"{aai_data['words_per_minute']:.0f} WPM")
                            with m3: st.metric(
                                "😶 Fillers",
                                aai_data['filler_count'])
                            with m4: st.metric(
                                "🎯 Confidence",
                                f"{aai_data['audio_confidence']*100:.0f}%")

                        else:
                            st.info(
                                "ℹ️ Single speaker "
                                "detected — treating "
                                "as candidate only")
                            transcript = \
                                aai_result[
                                    'full_transcript']
                            aai_data = aai_result
                            st.text_area(
                                "📝 Transcript",
                                transcript,
                                height=120)

                    else:
                        st.warning(
                            f"⚠️ AssemblyAI error: "
                            f"{err}\nFalling back to "
                            f"local Whisper.")
                        try:
                            import whisper
                            wm = whisper.load_model(
                                "tiny")
                            r = wm.transcribe(
                                tmp_path,
                                language='en',
                                fp16=False)
                            transcript = \
                                r['text'].strip()
                            st.text_area(
                                "📝 Transcript",
                                transcript,
                                height=120)
                        except Exception as e:
                            st.error(
                                f"Transcription: {e}")

                    progress.progress(40)

                else:
                    status.info(
                        "📝 Transcribing... "
                        "(add AssemblyAI key for "
                        "speaker separation)")
                    progress.progress(10)
                    try:
                        import whisper
                        wm = whisper.load_model("tiny")
                        r  = wm.transcribe(
                            tmp_path,
                            language='en',
                            fp16=False)
                        transcript = r['text'].strip()
                        st.success(
                            "✅ Transcription done!")
                        st.info(
                            "💡 Add AssemblyAI key "
                            "to separate interviewer "
                            "and candidate speech")
                        st.text_area(
                            "📝 Transcript",
                            transcript,
                            height=120)
                    except Exception as e:
                        st.error(f"Error: {e}")
                    progress.progress(40)

                # ── Face analysis ─────────────────
                if is_video and models.get('face'):
                    status.info(
                        "😊 Analyzing facial "
                        "expressions...")
                    progress.progress(55)
                    face_r = analyze_face(
                        tmp_path, models['face'])
                    if face_r:
                        st.caption(
                            f"😊 Camera: "
                            f"{face_r['camera_presence_pct']:.0f}%"
                            f" | Smile: "
                            f"{face_r['smile_pct']:.0f}%"
                            f" | "
                            f"{face_r['display_label']}")

                # ── Audio emotion ─────────────────
                status.info(
                    "🎤 Analyzing vocal tone...")
                progress.progress(75)
                wav_emo = extract_audio(tmp_path)
                wpm_val = (
                    aai_data.get('words_per_minute')
                    if aai_data else None)
                audio_r = analyze_audio_emotion(
                    wav_emo,
                    models.get('audio'),
                    wpm=wpm_val)
                if wav_emo and \
                        os.path.exists(wav_emo):
                    try: os.unlink(wav_emo)
                    except: pass
                if audio_r:
                    st.caption(
                        f"🎤 Tone: "
                        f"**{audio_r['vocal_tone']}** "
                        f"| Stability: "
                        f"{audio_r['voice_stability']:.0f}%")

                progress.progress(100)
                status.empty()
                progress.empty()

    # ── Tab 2 ─────────────────────────────────────
    with tab2:
        iq2 = st.text_input(
            "❓ Interview question (optional)",
            key="q_t2")
        st.markdown(
            "Paste your answer for "
            "text-only analysis.")
        t_input = st.text_area(
            "Your answer:", height=200,
            placeholder=
                "In my previous role I led...")
        if st.button(
                "🎯 Analyze Text",
                type="primary",
                key="btn_t"):
            transcript = t_input
            transcript_only = True
            if iq2:
                interview_q = iq2

    # ── Tab 3 ─────────────────────────────────────
    with tab3:
        st.subheader("📈 Your Interview Progress")
        if not student_name:
            st.info(
                "Enter your name in sidebar "
                "to track progress")
        else:
            history = get_progress(student_name)
            if not history:
                st.info(
                    f"No sessions yet for "
                    f"{student_name}")
            else:
                fig_p = progress_chart(history)
                if fig_p:
                    st.plotly_chart(
                        fig_p,
                        use_container_width=True)
                st.subheader("Session History")
                for h in reversed(history[-10:]):
                    c1,c2,c3 = st.columns([2,1,1])
                    with c1:
                        st.markdown(f"**{h[0]}**")
                    with c2:
                        st.markdown(
                            f"Score: "
                            f"**{h[1]:.0f}/100**")
                    with c3:
                        st.markdown(f"{h[6]}")
                if len(history) >= 2:
                    delta = (history[-1][1] -
                              history[0][1])
                    col = ("#27ae60"
                            if delta>=0 else "#e74c3c")
                    arr = "↑" if delta>=0 else "↓"
                    st.markdown(
                        f'<div style="text-align:'
                        f'center;padding:1rem;'
                        f'background:{col}15;'
                        f'border-radius:10px;'
                        f'border:1px solid {col}40">'
                        f'<h3 style="color:{col}">'
                        f'{arr} {abs(delta):.1f} pts'
                        f'</h3><p style="color:#666">'
                        f'improvement from first to '
                        f'latest</p></div>',
                        unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    # ANALYSIS + RESULTS
    # ══════════════════════════════════════════════
    if transcript and len(transcript.split()) >= 20:

        st.divider()
        st.subheader(
            "📊 Interview Evaluation Report")

        with st.spinner(
                "🔀 Running multimodal analysis..."):
            sq     = analyze_speech_quality(
                transcript, aai_data)
            star   = compute_star(transcript)
            text_r = predict_text_emotion(
                transcript, models)
            kw     = compute_keywords(transcript)
            if face_r  is None: face_r  = default_face()
            if audio_r is None: audio_r = default_audio()

        # LLM
        llm_tech, llm_fb = None, None
        if groq_client:
            c1,c2 = st.columns(2)
            with c1:
                with st.spinner(
                        "🤖 Groq: analyzing "
                        "technical depth..."):
                    llm_tech = analyze_technical_llm(
                        transcript,
                        job_title or "Software Engineer",
                        interview_q,
                        groq_client)
            # Need scores for LLM feedback prompt
            temp_scores = compute_scores(
                text_r, face_r, audio_r,
                sq, star, kw, llm_tech, aai_data)
            with c2:
                with st.spinner(
                        "🤖 Groq: generating "
                        "feedback..."):
                    llm_fb = generate_llm_feedback(
                        transcript,
                        temp_scores,
                        face_r, audio_r,
                        aai_data, sq, star,
                        job_title or "Software Engineer",
                        interview_q,
                        groq_client)

        scores = compute_scores(
            text_r, face_r, audio_r,
            sq, star, kw, llm_tech, aai_data)

        if transcript_only:
            scores['confidence'] = round(
                text_r['confidence']*100, 1)

        fb  = fallback_feedback(
            scores, face_r, audio_r, sq, star, kw)
        rec, rc = get_recommendation(
            scores, star.get('completeness'))

        if student_name:
            save_session(
                student_name,
                job_title or "General",
                scores, rec, transcript)

        use_llm = llm_fb is not None
        use_aai = aai_data is not None

        # Headline
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric(
            "⏱️ Words",sq.get('total_words',0))
        with c2: st.metric(
            "🎯 Score",f"{scores['overall']}/100")
        with c3: st.metric(
            "🔧 Technical",
            f"{scores['technical']:.0f}%")
        with c4:
            sd = ("N/A (intro)" if star['is_intro']
                  else
                  f"{star.get('completeness',0):.0f}%")
            st.metric("⭐ STAR", sd)
        st.divider()

        ab = ('<span class="aai-badge">AAI ✓</span>'
               if use_aai else '')
        gb = ('<span class="groq-badge">Groq ✓</span>'
               if use_llm else '')
        st.markdown(
            f'<div style="text-align:center;'
            f'margin:1rem 0">'
            f'<span class="rec-badge" '
            f'style="background:{rc}">'
            f'Recommendation: {rec}</span>'
            f'{ab}{gb}</div>',
            unsafe_allow_html=True)
        w = scores.get('weights_used',{})
        st.caption(
            f"Confidence-weighted fusion — "
            f"Text: {w.get('text',0)*100:.0f}% | "
            f"Face: {w.get('face',0)*100:.0f}% | "
            f"Audio: {w.get('audio',0)*100:.0f}%")
        st.divider()

        # AAI metrics
        if use_aai:
            st.subheader(
                "🎙️ AssemblyAI Speech Analytics")
            m1,m2,m3,m4 = st.columns(4)
            with m1: st.metric(
                "Words",aai_data['total_words'])
            with m2: st.metric(
                "Pace",
                f"{aai_data['words_per_minute']:.0f} WPM")
            with m3: st.metric(
                "Filler Words",
                aai_data['filler_count'])
            with m4: st.metric(
                "Transcription Confidence",
                f"{aai_data['audio_confidence']*100:.0f}%")
            st.divider()

        # Score + Radar
        col1,col2 = st.columns([1,2])
        with col1:
            st.markdown(
                f'<div class="score-card">'
                f'<div class="score-number" '
                f'style="color:{rc}">'
                f'{scores["overall"]}</div>'
                f'<div style="color:#666;'
                f'font-size:.9rem">'
                f'Interview Score / 100</div></div>',
                unsafe_allow_html=True)
            for name,key in [
                ("💬 Communication",
                 'communication'),
                ("👔 Professionalism",
                 'professionalism'),
                ("🔧 Technical",'technical'),
                ("⭐ Answer Quality",
                 'answer_quality'),
                ("🎯 Confidence",'confidence')]:
                v = scores[key]
                ca,cb = st.columns([2,1])
                with ca:
                    st.markdown(f"**{name}**")
                    st.progress(int(v))
                with cb:
                    st.markdown(f"**{v:.0f}**")
        with col2:
            st.plotly_chart(
                radar_chart(scores),
                use_container_width=True)

        st.divider()
        st.subheader("📊 Benchmark")
        st.plotly_chart(
            benchmark_chart(scores,rc),
            use_container_width=True)
        st.divider()

        # Camera + Voice
        if not transcript_only:
            c3,c4 = st.columns(2)
            with c3:
                st.subheader("👁️ Camera Presence")
                st.markdown(
                    f'<div class="trait-card">'
                    f'<b>Impression:</b> '
                    f'{face_r["display_label"]}'
                    f'</div>',
                    unsafe_allow_html=True)
                if face_r['frames_analyzed']>0:
                    st.progress(int(
                        face_r['camera_presence_pct']),
                        text=f"In frame: "
                             f"{face_r['camera_presence_pct']:.0f}%")
                    st.progress(
                        int(face_r['smile_pct']),
                        text=f"Smile: "
                             f"{face_r['smile_pct']:.0f}%")
                    st.progress(
                        int(face_r['consistency']),
                        text=f"Consistency: "
                             f"{face_r['consistency']:.0f}%")
                else:
                    st.info("Upload video for face analysis")

            with c4:
                st.subheader("🎤 Vocal Analysis")
                st.markdown(
                    f'<div class="trait-card">'
                    f'<b>Vocal tone:</b> '
                    f'{audio_r["vocal_tone"]}'
                    f'</div>',
                    unsafe_allow_html=True)
                if audio_r['voice_stability']>0:
                    st.progress(int(
                        audio_r['voice_stability']),
                        text=f"Stability: "
                             f"{audio_r['voice_stability']:.0f}%")
                    if audio_r.get('words_per_minute'):
                        st.markdown(
                            f"**Pace:** "
                            f"{audio_r['pace_label']} "
                            f"({audio_r['words_per_minute']:.0f} wpm)")
                    st.markdown(
                        f"**Pause control:** "
                        f"{audio_r['pause_control']}")
                else:
                    st.info("Upload video for voice analysis")
            st.divider()

        # Keywords
        st.subheader("🔑 Skill Coverage")
        for cat,data in kw.items():
            if cat=='overall': continue
            sc2   = data['score']
            found = data['found']
            ck = ('#27ae60' if sc2>60 else
                   '#e67e22' if sc2>30 else '#e74c3c')
            st.markdown(
                f"**{cat.replace('_',' ').title()}**: "
                f'<span style="color:{ck}">'
                f'{sc2:.0f}%</span>',
                unsafe_allow_html=True)
            if found:
                st.markdown(
                    f"  *Found: "
                    f"{', '.join(found[:4])}*")
            st.progress(int(sc2))

        # LLM technical
        if llm_tech:
            st.divider()
            st.subheader(
                "🤖 AI Technical Analysis")
            tc1,tc2 = st.columns(2)
            with tc1:
                if llm_tech.get(
                        'skills_demonstrated'):
                    st.markdown(
                        "**✅ Skills shown:**")
                    for sk in llm_tech[
                            'skills_demonstrated']:
                        st.markdown(f"  ✅ {sk}")
                if llm_tech.get(
                        'concepts_understood'):
                    st.markdown("**💡 Concepts:**")
                    for c in llm_tech[
                            'concepts_understood']:
                        st.markdown(f"  💡 {c}")
            with tc2:
                if llm_tech.get('missing_skills'):
                    st.markdown(
                        "**⚠️ Add next time:**")
                    for sk in llm_tech[
                            'missing_skills']:
                        st.markdown(f"  ⚠️ {sk}")
                dep = llm_tech.get(
                    'technical_depth','moderate')
                dc  = ('#27ae60' if dep=='deep' else
                        '#f39c12' if dep=='moderate'
                        else '#e74c3c')
                st.markdown(
                    f"**Depth:** "
                    f'<span style="color:{dc};'
                    f'font-weight:bold">'
                    f'{dep.title()}</span>',
                    unsafe_allow_html=True)
                if llm_tech.get(
                        'technical_feedback'):
                    st.info(
                        llm_tech['technical_feedback'])

        st.divider()

        # STAR
        st.subheader("⭐ STAR Method")
        if star['is_intro']:
            st.info(
                "ℹ️ Introduction detected — "
                "STAR not applicable")
        else:
            sc3 = st.columns(4)
            for col,(comp,pres) in zip(
                    sc3,
                    star['components_found'].items()):
                with col:
                    st.markdown(
                        f"{'✅' if pres else '❌'} "
                        f"**{comp.title()}**")
        st.divider()

        # Feedback
        st.subheader(
            "💬 " +
            ("AI Feedback (Groq)"
             if use_llm else "Feedback"))
        strg = (llm_fb['strengths'] if use_llm
                 else fb['strengths'])
        impr = (llm_fb['improvements'] if use_llm
                 else fb['improvements'])
        cf1,cf2,cf3 = st.columns(3)
        with cf1:
            st.markdown("### ✅ Strengths")
            for s in strg:
                css = ("llm-card" if use_llm
                        else "feedback-card")
                st.markdown(
                    f'<div class="{css}">'
                    f'<span class="strength-item">'
                    f'✅ {s}</span></div>',
                    unsafe_allow_html=True)
        with cf2:
            st.markdown("### ⚠️ Improve")
            for im in impr:
                css = ("llm-card" if use_llm
                        else "feedback-card")
                st.markdown(
                    f'<div class="{css}">'
                    f'<span class="improve-item">'
                    f'⚠️ {im}</span></div>',
                    unsafe_allow_html=True)
        with cf3:
            st.markdown("### 💡 Tips")
            if use_llm and llm_fb.get('one_thing'):
                st.markdown(
                    f'<div class="llm-card">'
                    f'<span class="tip-high">'
                    f'🎯 <b>TOP PRIORITY:</b><br>'
                    f'{llm_fb["one_thing"]}'
                    f'</span></div>',
                    unsafe_allow_html=True)
            for t in fb['tips']:
                pr  = t.split(':')[0]
                css = {
                    'HIGH':'tip-high',
                    'MED' :'tip-med',
                    'LOW' :'tip-low'
                }.get(pr,'tip-med')
                st.markdown(
                    f'<div class="feedback-card">'
                    f'<span class="{css}">'
                    f'💡 {t}</span></div>',
                    unsafe_allow_html=True)

        # LLM extras
        if use_llm:
            st.divider()
            if (not star['is_intro'] and
                    llm_fb.get('star_rewrite') and
                    str(llm_fb['star_rewrite']
                        ).strip() not in
                    ('null','None','')):
                st.subheader(
                    "✨ Your Answer (STAR Format)")
                st.markdown(
                    f'<div class="star-box">'
                    f'{llm_fb["star_rewrite"]}'
                    f'</div>',
                    unsafe_allow_html=True)
            if llm_fb.get('encouragement'):
                st.divider()
                st.markdown(
                    f'<div class="encourage-box">'
                    f'<p style="font-size:1.1rem;'
                    f'color:#667eea;'
                    f'font-style:italic;margin:0">'
                    f'💬 {llm_fb["encouragement"]}'
                    f'</p></div>',
                    unsafe_allow_html=True)

        st.divider()

        # Speech detail
        st.subheader("📝 Speech Quality")
        c8,c9 = st.columns(2)
        with c8:
            m1,m2 = st.columns(2)
            with m1:
                st.metric("Word Count",
                    sq.get('total_words',0))
                st.metric("Filler Words",
                    sq.get('filler_count',0))
                st.metric("Avg Sentence",
                    sq.get(
                        'avg_sentence_length',0))
            with m2:
                st.metric("Vocab Richness",
                    f"{sq.get('vocabulary_richness',0):.2f}")
                st.metric("Grammar",
                    f"{sq.get('grammar_score',0):.0f}%")
                st.metric("Repetition",
                    f"{sq.get('repetition_pct',0):.0f}%")
        with c9:
            st.text_area(
                "Transcript analyzed:",
                transcript, height=200,
                disabled=True)

        st.divider()

        # Certificate
        st.subheader("🏅 Performance Summary")
        st.markdown(
            f'<div class="cert-box" style="'
            f'background:linear-gradient(135deg,'
            f'{rc}15,{rc}30);border:2px solid {rc};">'
            f'<h2 style="color:{rc}">'
            f'🎯 Interview Evaluation Report</h2>'
            f'<h3 style="color:#333">'
            f'Role: {job_title or "General"}</h3>'
            f'<div style="font-size:4rem;'
            f'font-weight:bold;color:{rc}">'
            f'{scores["overall"]}/100</div>'
            f'<div style="font-size:1.5rem;'
            f'color:{rc};font-weight:bold">'
            f'{rec}</div>'
            f'<hr style="border-color:{rc}40">'
            f'<div style="display:flex;'
            f'justify-content:space-around;'
            f'flex-wrap:wrap;margin-top:1rem">'
            + ''.join(
                f'<div style="margin:.5rem">'
                f'<div style="font-size:1.5rem;'
                f'font-weight:bold;color:{rc}">'
                f'{scores[k]:.0f}</div>'
                f'<div style="color:#666;'
                f'font-size:.85rem">{lb}</div></div>'
                for k,lb in [
                    ('communication',
                     'Communication'),
                    ('professionalism',
                     'Professionalism'),
                    ('technical','Technical'),
                    ('answer_quality',
                     'Answer Quality'),
                    ('confidence','Confidence')])
            + '</div></div>',
            unsafe_allow_html=True)

        if student_name:
            st.success(
                f"✅ Session saved for "
                f"{student_name}")

        st.divider()

        report = dict(
            student_name=student_name,
            job_title=job_title,
            date=datetime.now().strftime(
                "%Y-%m-%d %H:%M"),
            recommendation=rec,
            scores=scores,
            aai_metrics=(dict(
                words=aai_data['total_words'],
                wpm=aai_data['words_per_minute'],
                fillers=aai_data['filler_count'],
                confidence=aai_data[
                    'audio_confidence'])
                if aai_data else {}),
            camera=dict(
                pct=face_r['camera_presence_pct'],
                smile=face_r['smile_pct'],
                label=face_r['display_label']),
            voice=dict(
                tone=audio_r['vocal_tone'],
                stability=audio_r['voice_stability'],
                pace=audio_r['pace_label'],
                pause_control=audio_r[
                    'pause_control']),
            star=star,
            llm_feedback=llm_fb,
            llm_technical=llm_tech,
            fallback_feedback=fb,
            keywords={
                k:v for k,v in kw.items()
                if isinstance(v,dict)},
            speech_metrics=sq,
            transcript=transcript)

        st.download_button(
            "📥 Download Full Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=(
                f"interview_report_"
                f"{datetime.now().strftime('%Y%m%d_%H%M')}"
                f".json"),
            mime="application/json")

    elif (transcript and
            len(transcript.split()) < 20):
        st.warning(
            "⚠️ Response too short — please "
            "provide a more complete answer.")

    else:
        st.markdown("""
        <div style="text-align:center;
          padding:3rem;
          background:linear-gradient(135deg,
            #667eea10,#764ba210);
          border-radius:20px;margin:2rem 0">
        <h2>🚀 How it works</h2>
        <p style="color:#666;font-size:1.1rem">
        Add your API keys in the sidebar,<br>
        then upload your interview video
        </p></div>""",
        unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)
        for col,em,ti,de in [
            (c1,"🎙️","AssemblyAI",
             "Transcribe + separate speakers"),
            (c2,"😊","Emotion AI",
             "Face + voice + text models"),
            (c3,"📊","Smart Score",
             "6-dimension evaluation"),
            (c4,"🤖","Groq Feedback",
             "Personalized HR-style report")]:
            with col:
                st.markdown(
                    f'<div style="text-align:'
                    f'center;padding:1.5rem;'
                    f'background:#fff;'
                    f'border-radius:15px;'
                    f'box-shadow:0 2px 10px #0001;'
                    f'margin:.5rem">'
                    f'<div style="font-size:2rem">'
                    f'{em}</div>'
                    f'<h4>{ti}</h4>'
                    f'<p style="color:#666;'
                    f'font-size:.85rem">'
                    f'{de}</p></div>',
                    unsafe_allow_html=True)


if __name__ == "__main__":
    main()
