# 🎯 Multimodal AI Interview Analyzer

A complete AI-powered interview evaluation system that analyzes a candidate's interview using **Facial Expressions, Voice Analysis, and Natural Language Processing (NLP)** to generate an objective interview score, hiring recommendation, and personalized feedback.

The system combines **Computer Vision, Speech Processing, Deep Learning, and Transformer-based NLP** into a single interactive Streamlit dashboard.

---
# LIVE DEMO:- https://multimodal-ai-interview-analyzer.streamlit.app/

---

# 🚀 Features

✅ Facial Emotion Recognition

- Face detection using Haar Cascade
- CNN with Channel Attention
- Frame-wise emotion prediction
- Majority voting
- Camera presence analysis
- Smile frequency estimation
- Facial consistency analysis

---

✅ Audio Analysis

- Automatic speech extraction from video
- Voice Activity Detection (VAD)
- Pitch estimation using Praat
- Speaking pace calculation
- Pause detection
- Voice stability analysis
- Mel Spectrogram generation
- CNN-based speech emotion recognition

---

✅ Text Analysis

- Whisper Automatic Speech Recognition (ASR)
- RoBERTa-based emotion classification
- Emotion prediction
- Technical keyword matching
- Communication analysis
- Problem-solving analysis
- STAR Method evaluation
- Speech quality analysis

---

✅ Multimodal Fusion

The outputs from Face, Audio and Text models are combined using a confidence-weighted late fusion strategy.

Fusion includes:

- Face probabilities
- Audio probabilities
- Text probabilities
- Model confidence
- Speech quality features
- Voting features

A trained Dual-Branch MLP predicts the final interview performance.

---

✅ Dashboard Features

- Upload Interview Video
- Paste Interview Transcript
- Automatic Speech Transcription
- Face Analysis
- Audio Analysis
- Text Analysis
- Interview Score
- Hiring Recommendation
- Radar Chart
- Benchmark Comparison
- STAR Method Analysis
- Technical Skill Assessment
- Voice Metrics
- Personalized Feedback

---

# 🧠 Project Architecture

```
                   Video / Transcript
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
     Face Module      Audio Module      Text Module
          │                │                │
          └────────────────┼────────────────┘
                           ▼
             Confidence Weighted Fusion
                           ▼
                  Scoring Engine
                           ▼
                 Feedback Generator
                           ▼
                 Streamlit Dashboard
```

---

# 📂 Project Structure

```
Multimodal-AI-Interview-Analyzer
│
├── app.py
├── download_models.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── face_model_best.keras
│   ├── audio_model_best.keras
│   ├── fusion_model_best.keras
│   └── final_roberta_model/
│
├── notebooks/
│   ├── face_training.ipynb
│   ├── Audio.ipynb
│   ├── text_training.ipynb
│   ├── text_pipeline.ipynb
│   ├── fusion_training.ipynb
│   └── score_feedback.ipynb
│
└── assets/
```

---

# ⚙️ Technologies Used

## Programming

- Python

## Deep Learning

- TensorFlow
- Keras
- PyTorch

## Computer Vision

- OpenCV
- Haar Cascade

## NLP

- Transformers
- RoBERTa
- Whisper

## Audio Processing

- Librosa
- Praat-Parselmouth
- FFmpeg

## Dashboard

- Streamlit
- Plotly

---

# 📊 Models Used

| Module | Model |
|---------|-------|
| Face | CNN + Channel Attention |
| Audio | CNN on Mel Spectrogram |
| Text | RoBERTa-base |
| Fusion | Dual-Branch MLP |

---

# 📈 Evaluation Metrics

The dashboard evaluates:

- Communication
- Professionalism
- Technical Relevance
- Confidence
- Answer Quality
- Overall Interview Score

It also provides:

- Hiring Recommendation
- STAR Analysis
- Keyword Coverage
- Speech Quality
- Personalized Feedback

---

# 🧩 Working Pipeline

1. Upload Interview Video or Transcript

2. Extract Frames

3. Detect Face

4. Predict Facial Emotion

5. Extract Audio

6. Convert Speech to Text using Whisper

7. Analyze Voice Features

8. Analyze Text using RoBERTa

9. Perform STAR Analysis

10. Perform Keyword Matching

11. Generate Fusion Features

12. Predict Overall Performance

13. Display Dashboard Results

---

# 📊 Dataset

## Face

- FER-2013
- CK+
- RAF-DB (combined)

## Audio

- RAVDESS
- CREMA-D
- SAVEE

## Text

- DailyDialog
- GoEmotions

Final interview emotion classes:

- Positive
- Angry
- Anxious
- Surprised

---

# 🎯 Dashboard Outputs

The dashboard generates:

- Overall Interview Score
- Hiring Recommendation
- Face Emotion
- Voice Analysis
- Text Emotion
- Communication Score
- Technical Score
- Confidence Score
- Professionalism Score
- STAR Coverage
- Radar Chart
- Benchmark Comparison
- Action Tips
- Strengths
- Areas of Improvement

---

# ☁️ Deployment

The application is deployed using:

- Streamlit Cloud

Source Code:

- GitHub

Model Storage:

- Google Drive

Models are downloaded automatically during the first execution.

---

# 💡 Advantages

- Fully automated interview evaluation
- Multimodal AI approach
- End-to-end deployment
- Real-time dashboard
- Easy-to-use interface
- Explainable scoring system
- Personalized feedback

---

# ⚠️ Limitations

- Face accuracy depends on lighting and camera quality
- Audio analysis is affected by background noise
- Text analysis depends on transcription quality
- STAR evaluation uses rule-based matching
- Supports English interviews only

---

# 🔮 Future Work

- Real-time webcam interviews
- Live emotion tracking
- Speaker diarization
- PDF report generation
- Resume analysis
- Job-specific interview evaluation
- Cloud database integration
- Larger multimodal datasets
- Improved face and audio models

---

# 👨‍💻 Author

**Anshu Bansal**

B.Tech Computer Science Engineering

Amity University Noida

---

# 📜 License

This project is developed for academic and research purposes.
