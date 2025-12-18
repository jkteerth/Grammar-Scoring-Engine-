🎙️ Grammar Scoring Engine from Voice 

(Python | AI | Offline ASR | ML Grammar Scoring)

📌 Project Overview

This project is an AI-based Grammar Scoring Engine that evaluates spoken English from audio  inputs.
It converts speech to text, corrects grammatical errors using pretrained transformer models, and produces a grammar score out of 100 with visual feedback.

The system supports:

🎧 Audio upload (WAV, MP3, M4A, FLAC, OGG)

🎤 Live voice recording (Start / Stop)

📊 Grammar score visualization

🌊 Audio waveform visualization
 
📊  provide the the corrected sentence and input sentence 

🖥️ Modern CustomTkinter UI

🌐 Flask web version (optional)

This project was developed as part of an SHL assessment and follows industry-grade design practices.

🚀 Key Features

● Offline Speech Recognition (Vosk – no internet required)

● Grammar Correction using Transformers (T5-base)

● ML-based Grammar Scoring (0–100)

● Audio Support

● Waveform Visualization

● Animated Score Visualization

● Threaded Processing (No UI Freeze)

● Cross-Platform (Windows tested)

🧠 Architecture Pipeline
Audio / Video Input
        ↓
FFmpeg (Normalize & Extract Audio)
        ↓
Vosk ASR (Speech → Text)
        ↓
T5 Grammar Correction Model
        ↓
Grammar Scoring Logic
        ↓
Visualization (Waveform + Score)

📁 Project Structure
SHL/
│
├── app.py                     # Main UI launcher
├── README.md                  # Project documentation
├── requirements.txt
│
├── models/
│   ├── speech_to_text.py      # Vosk ASR
│   ├── grammar_corrector_ml.py
│   └── grammar_scorer_ml.py
│
├── audio/
│   ├── recorder.py
│   └── audio_utils.py
│
├── ui/
│   └── main_ui.py             # CustomTkinter UI
│
├── utils/
│   └── text_compare.py
│
└── vosk-model-en-us-0.22-lgraph/

⚙️ Technologies Used
Component	Technology
Language	Python 3.11
UI	CustomTkinter
ASR	Vosk (Offline)
Grammar Correction	T5-base Transformer
Audio Processing	FFmpeg, PyDub
Visualization	Matplotlib
Threading	Python threading
Optional Web	Flask
🧪 Supported Input Formats
🎧 Audio

WAV

MP3

M4A

FLAC

OGG

🛠️ Installation & Setup (Windows)
1️⃣ Clone or Download Project
cd C:\Projects

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

🔊 FFmpeg Setup (Required)
Download FFmpeg

https://www.gyan.dev/ffmpeg/builds/

Extract to:

C:\ffmpeg-8.0.1-essentials_build\


Add to System PATH:

C:\ffmpeg-8.0.1-essentials_build\bin


Verify:

ffmpeg -version

🧠 Download Vosk Model (Offline ASR)

Download:

vosk-model-en-us-0.22-lgraph


Place it in the project root:

SHL/vosk-model-en-us-0.22-lgraph/

▶️ Run the Application
python app.py

🖥️ How to Use

Upload Audio OR Upload Video

OR click Start Recording → Stop Recording

Click Score & Process

View:

● Original text

● Corrected sentence

● Grammar score

● Waveform

Animated score chart

📊 Grammar Scoring Logic

 ‣ Grammar is corrected using a pretrained T5 transformer

 ‣ Score is calculated based on:

‣ Degree of correction

‣ Structural differences

‣ Score range: 0–100

Designed to produce realistic human-like scores

“The system uses offline speech recognition, transformer-based grammar correction, and ML-driven scoring to evaluate spoken English from audio and video inputs. It is fully offline, scalable, and reproducible.”

🔮 Future Enhancements

● Browser-based microphone & camera

● CEFR level prediction (A1–C2)

● PDF report export

● Web deployment (Flask / HuggingFace Spaces)

● Confidence scoring per sentence

👨‍💻 Author
Focused on ML, NLP, and Speech Processing 
