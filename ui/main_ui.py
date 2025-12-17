import customtkinter as ctk
from tkinter import filedialog
import threading
import pyaudio
import wave
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
from models.speech_to_text import transcribe
from models.grammar_corrector_ml import correct_grammar_ml
from models.grammar_scorer_ml import grammar_score_ml

# Suppress matplotlib warnings
warnings.filterwarnings("ignore")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Global variable to track if app is running
app_running = True

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Try to import audio processing libraries
try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.frames = []
        
    def start_recording(self):
        self.recording = True
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      input=True,
                                      frames_per_buffer=CHUNK)
        threading.Thread(target=self._record, daemon=True).start()
        
    def _record(self):
        while self.recording and app_running:
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
            except:
                break
            
    def stop_recording(self):
        self.recording = False
        if hasattr(self, 'stream') and self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
        
        # Save the recording
        filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        
        return filename

def load_audio_file(audio_path):
    """Load audio file using appropriate library based on file format"""
    try:
        if audio_path.lower().endswith('.wav'):
            # Use scipy for WAV files
            sample_rate, audio_data = wavfile.read(audio_path)
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            return sample_rate, audio_data.astype(np.float32)
        else:
            # Use librosa for other formats (MP3, M4A, FLAC, etc.)
            if LIBROSA_AVAILABLE:
                audio_data, sample_rate = librosa.load(audio_path, sr=None, mono=True)
                return sample_rate, audio_data
            else:
                raise ImportError("librosa not installed for MP3 support")
    except Exception as e:
        print(f"Error loading audio: {e}")
        raise

def show_waveform(audio_path, waveform_frame):
    """Display waveform of audio file"""
    if not app_running:
        return
    
    # Clear previous waveform
    for widget in waveform_frame.winfo_children():
        widget.destroy()
    
    try:
        # Load audio data
        sample_rate, audio_data = load_audio_file(audio_path)
        
        # Calculate duration
        duration = len(audio_data) / sample_rate
        time = np.linspace(0, duration, len(audio_data))
        
        # Create plot on main thread
        app.after(0, _create_waveform_plot, waveform_frame, time, audio_data, sample_rate)
        
    except Exception as e:
        app.after(0, _show_error, waveform_frame, f"Cannot display waveform: {str(e)}")

def _create_waveform_plot(waveform_frame, time, audio_data, sample_rate):
    """Create waveform plot"""
    if not app_running:
        return
    
    try:
        # Clear the frame
        for widget in waveform_frame.winfo_children():
            widget.destroy()
        
        # Create figure for waveform
        fig, ax = plt.subplots(figsize=(8, 3), facecolor='#2b2b2b')
        
        # Normalize audio data for better visualization
        if len(audio_data) > 0:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
        
        # Plot waveform
        ax.plot(time, audio_data, color='#1f6aa5', linewidth=0.5, alpha=0.8)
        
        # Fill under the waveform
        ax.fill_between(time, audio_data, color='#1f6aa5', alpha=0.3)
        
        # Customize plot
        ax.set_xlabel('Time (s)', color='white', fontsize=10)
        ax.set_ylabel('Amplitude', color='white', fontsize=10)
        ax.set_title('Audio Waveform', color='white', fontsize=12, pad=10)
        
        # Set background and grid
        ax.set_facecolor('#1e1e1e')
        ax.grid(True, alpha=0.2, color='white')
        ax.tick_params(colors='white')
        
        # Set spines color
        for spine in ax.spines.values():
            spine.set_color('white')
            
        fig.patch.set_facecolor('#2b2b2b')
        fig.tight_layout()
        
        # Create canvas and embed in tkinter frame
        canvas = FigureCanvasTkAgg(fig, master=waveform_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)
        
    except Exception as e:
        _show_error(waveform_frame, f"Error creating plot: {str(e)}")

def _show_error(frame, message):
    """Show error message in frame"""
    if not app_running:
        return
    
    for widget in frame.winfo_children():
        widget.destroy()
    
    error_label = ctk.CTkLabel(
        frame,
        text=message,
        text_color="red",
        font=("Arial", 12),
        wraplength=400
    )
    error_label.pack(pady=50)

def process_grammar_and_show_score(audio_path, output, score_frame, status_label):
    """Process audio for grammar and show score"""
    try:
        if not app_running:
            return
        
        app.after(0, lambda: status_label.configure(
            text="Processing grammar...", 
            text_color="yellow"
        ))
        
        # Transcribe and process
        text = transcribe(audio_path)
        corrected = correct_grammar_ml(text)
        score = grammar_score_ml(text, corrected)
        
        # Update text output on main thread
        app.after(0, _update_output, output, text, corrected, score)
        
        # Update score visualization on main thread
        app.after(0, _create_score_plot, score_frame, score)
        
        app.after(0, lambda: status_label.configure(
            text="Grammar analysis completed!", 
            text_color="green"
        ))
        
    except Exception as e:
        if app_running:
            app.after(0, lambda: status_label.configure(
                text=f"Error in grammar processing: {str(e)}", 
                text_color="red"
            ))

def _update_output(output, text, corrected, score):
    """Update output text widget"""
    if not app_running:
        return
    
    output.configure(state="normal")
    output.delete("1.0", "end")
    output.insert("end", f"Original Text:\n{text}\n\n")
    output.insert("end", f"Corrected Text:\n{corrected}\n\n")
    output.insert("end", f"Grammar Score: {score}/100")
    output.configure(state="disabled")

class PieChartAnimation:
    """Class to manage pie chart animation with proper cleanup"""
    def __init__(self, canvas, wedges):
        self.canvas = canvas
        self.wedges = wedges
        self.animating = True
        self.animation_id = None
        
    def start(self):
        """Start the animation"""
        if self.animating:
            self._rotate()
    
    def _rotate(self):
        """Rotate the pie chart"""
        if not self.animating or not app_running:
            return
        
        try:
            for wedge in self.wedges:
                wedge.set_theta1(wedge.theta1 + 1)
                wedge.set_theta2(wedge.theta2 + 1)
            self.canvas.draw()
            
            # Schedule next rotation
            if app_running:
                self.animation_id = self.canvas.get_tk_widget().after(30, self._rotate)
        except:
            # Window probably closed
            self.stop()
    
    def stop(self):
        """Stop the animation"""
        self.animating = False
        if self.animation_id:
            try:
                self.canvas.get_tk_widget().after_cancel(self.animation_id)
            except:
                pass

def _create_score_plot(score_frame, score):
    """Create score plot"""
    if not app_running:
        return
    
    # Clear previous visualization and stop any existing animation
    for widget in score_frame.winfo_children():
        widget.destroy()
    
    # Clear any existing animation
    if hasattr(score_frame, '_animation'):
        score_frame._animation.stop()
    
    try:
        # Create figure for pie chart
        fig, ax = plt.subplots(figsize=(5, 4), facecolor='#2b2b2b')
        
        # Data for pie chart
        categories = ['Correct', 'Incorrect']
        values = [score, 100 - score] if score >= 0 else [0, 100]
        colors = ['#4CAF50', '#F44336']
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(values, labels=categories, colors=colors, 
                                           autopct='%1.1f%%', startangle=90,
                                           explode=(0.1, 0))
        
        # Customize text colors
        for text in texts + autotexts:
            text.set_color('white')
            text.set_fontsize(12)
        
        # Set title
        ax.set_title(f'Grammar Score: {score}/100', color='white', fontsize=14, pad=20)
        
        # Set background color
        ax.set_facecolor('#2b2b2b')
        fig.patch.set_facecolor('#2b2b2b')
        
        # Create canvas
        canvas = FigureCanvasTkAgg(fig, master=score_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create and start animation
        score_frame._animation = PieChartAnimation(canvas, wedges)
        score_frame._animation.start()
        
    except Exception as e:
        _show_error(score_frame, f"Error creating score plot: {str(e)}")

def upload_audio_file(waveform_frame, status_label, output):
    """Handle audio file upload"""
    if not app_running:
        return
    
    path = filedialog.askopenfilename(
        filetypes=[
            ("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg"),
            ("WAV Files", "*.wav"),
            ("MP3 Files", "*.mp3"),
            ("All Files", "*.*")
        ]
    )
    if path:
        # Store audio path globally
        app.audio_path = path
        
        # Update status
        status_label.configure(text="Loading audio file...", text_color="yellow")
        
        # Show waveform in background thread
        threading.Thread(
            target=show_waveform,
            args=(path, waveform_frame),
            daemon=True
        ).start()
        
        # Update output
        output.configure(state="normal")
        output.delete("1.0", "end")
        output.insert("end", f"📁 Audio File: {os.path.basename(path)}\n\n")
        output.insert("end", "Waveform loading... Click 'Score' to check grammar.")
        output.configure(state="disabled")
        
        # Enable score button after a short delay
        def enable_score_button():
            if app_running and hasattr(app, 'audio_path') and app.audio_path:
                score_btn.configure(state="normal")
                status_label.configure(
                    text="Audio loaded. Click 'Score' to check grammar.", 
                    text_color="cyan"
                )
        
        app.after(1000, enable_score_button)

def start_recording(status_label, record_btn):
    """Start voice recording"""
    if not app_running:
        return
    
    if not hasattr(record_btn, 'recorder'):
        record_btn.recorder = AudioRecorder()
    
    record_btn.configure(text="⏹ Stop Recording", fg_color="#F44336")
    status_label.configure(text="Recording... Speak now!", text_color="yellow")
    record_btn.recorder.start_recording()
    record_btn.recording = True

def stop_recording(waveform_frame, status_label, record_btn, output):
    """Stop voice recording and save"""
    if not app_running:
        return
    
    if hasattr(record_btn, 'recorder') and record_btn.recorder.recording:
        record_btn.configure(text="🎤 Record Voice", fg_color="#1f6aa5")
        status_label.configure(text="Saving recording...", text_color="yellow")
        
        filename = record_btn.recorder.stop_recording()
        record_btn.recording = False
        
        if filename and os.path.exists(filename):
            # Store audio path globally
            app.audio_path = filename
            
            # Show waveform in background thread
            threading.Thread(
                target=show_waveform,
                args=(filename, waveform_frame),
                daemon=True
            ).start()
            
            # Update output
            output.configure(state="normal")
            output.delete("1.0", "end")
            output.insert("end", f"🎤 Recording: {os.path.basename(filename)}\n\n")
            output.insert("end", "Waveform loading... Click 'Score' to check grammar.")
            output.configure(state="disabled")
            
            # Enable score button after a short delay
            def enable_score_button():
                if app_running and hasattr(app, 'audio_path') and app.audio_path:
                    score_btn.configure(state="normal")
                    status_label.configure(
                        text="Recording saved. Click 'Score' to check grammar.", 
                        text_color="cyan"
                    )
            
            app.after(1000, enable_score_button)
        else:
            status_label.configure(text="Error saving recording", text_color="red")

def toggle_recording(waveform_frame, status_label, record_btn, output):
    """Toggle recording state"""
    if not app_running:
        return
    
    if hasattr(record_btn, 'recording') and record_btn.recording:
        stop_recording(waveform_frame, status_label, record_btn, output)
    else:
        start_recording(status_label, record_btn)

def score_current_audio(output, score_frame, status_label):
    """Score the currently loaded audio"""
    if not app_running:
        return
    
    if hasattr(app, 'audio_path') and os.path.exists(app.audio_path):
        # Disable buttons during processing
        upload_btn.configure(state="disabled")
        record_btn.configure(state="disabled")
        score_btn.configure(state="disabled")
        
        # Process in separate thread
        thread = threading.Thread(
            target=process_grammar_and_show_score,
            args=(app.audio_path, output, score_frame, status_label),
            daemon=True
        )
        thread.start()
        
        # Re-enable buttons after a delay
        def reenable_buttons():
            if app_running:
                upload_btn.configure(state="normal")
                record_btn.configure(state="normal")
                score_btn.configure(state="normal")
        
        app.after(5000, reenable_buttons)  # Re-enable after 5 seconds
    else:
        status_label.configure(text="No audio loaded. Please upload or record first.", 
                              text_color="red")

def on_closing():
    """Handle window closing"""
    global app_running
    app_running = False
    
    # Stop any recording
    if hasattr(record_btn, 'recorder') and record_btn.recorder.recording:
        record_btn.recorder.stop_recording()
    
    # Stop any animations
    if hasattr(score_frame, '_animation'):
        score_frame._animation.stop()
    
    # Close the app
    app.destroy()

def launch_ui():
    global app, upload_btn, record_btn, score_btn, score_frame
    
    app = ctk.CTk()
    app.title("Grammar Scoring Engine")
    app.geometry("1400x900")
    
    # Handle window closing
    app.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Store audio path as app attribute
    app.audio_path = None
    
    # Configure grid
    app.grid_rowconfigure(2, weight=1)
    app.grid_columnconfigure(0, weight=1)
    
    # Title
    title = ctk.CTkLabel(
        app, 
        text="🎙 Grammar Scoring Engine", 
        font=("Arial", 28, "bold")
    )
    title.grid(row=0, column=0, pady=(20, 10))
    
    # Button Frame
    btn_frame = ctk.CTkFrame(app)
    btn_frame.grid(row=1, column=0, pady=(0, 20))
    
    # Option 1: Upload Files
    upload_btn = ctk.CTkButton(
        btn_frame,
        text="📁 Upload Audio File",
        width=200,
        height=50,
        font=("Arial", 14, "bold"),
        corner_radius=10
    )
    upload_btn.pack(side="left", padx=10)
    
    # Option 2: Record Voice
    record_btn = ctk.CTkButton(
        btn_frame,
        text="🎤 Record Voice",
        width=200,
        height=50,
        font=("Arial", 14, "bold"),
        corner_radius=10
    )
    record_btn.pack(side="left", padx=10)
    
    # Common Score Button
    score_btn = ctk.CTkButton(
        btn_frame,
        text="📊 Score",
        width=200,
        height=50,
        font=("Arial", 14, "bold"),
        corner_radius=10,
        fg_color="#4CAF50",
        hover_color="#45a049",
        state="disabled"
    )
    score_btn.pack(side="left", padx=10)
    
    # Main content area
    content_frame = ctk.CTkFrame(app)
    content_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 20))
    content_frame.grid_rowconfigure(0, weight=1)
    content_frame.grid_columnconfigure(0, weight=1)
    content_frame.grid_columnconfigure(1, weight=1)
    
    # Left Panel: Text Output
    left_panel = ctk.CTkFrame(content_frame)
    left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=10)
    left_panel.grid_rowconfigure(0, weight=1)
    left_panel.grid_columnconfigure(0, weight=1)
    
    text_label = ctk.CTkLabel(
        left_panel, 
        text="Text Output", 
        font=("Arial", 16, "bold")
    )
    text_label.pack(pady=(10, 5))
    
    output = ctk.CTkTextbox(
        left_panel, 
        font=("Arial", 12),
        wrap="word"
    )
    output.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    # Right Panel: Visualizations
    right_panel = ctk.CTkFrame(content_frame)
    right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=10)
    right_panel.grid_rowconfigure(0, weight=1)
    right_panel.grid_rowconfigure(1, weight=1)
    right_panel.grid_columnconfigure(0, weight=1)
    
    # Top Right: Waveform
    waveform_frame_container = ctk.CTkFrame(right_panel)
    waveform_frame_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
    waveform_frame_container.grid_columnconfigure(0, weight=1)
    waveform_frame_container.grid_rowconfigure(0, weight=1)
    
    waveform_label = ctk.CTkLabel(
        waveform_frame_container, 
        text="Audio Waveform", 
        font=("Arial", 16, "bold")
    )
    waveform_label.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
    
    waveform_frame = ctk.CTkFrame(waveform_frame_container)
    waveform_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
    
    # Bottom Right: Score Visualization
    score_frame_container = ctk.CTkFrame(right_panel)
    score_frame_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 10))
    score_frame_container.grid_columnconfigure(0, weight=1)
    score_frame_container.grid_rowconfigure(0, weight=1)
    
    score_label = ctk.CTkLabel(
        score_frame_container, 
        text="Grammar Score", 
        font=("Arial", 16, "bold")
    )
    score_label.grid(row=0, column=0, pady=(10, 5), sticky="w", padx=10)
    
    score_frame = ctk.CTkFrame(score_frame_container)
    score_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
    
    # Status Bar
    status_frame = ctk.CTkFrame(app, height=40)
    status_frame.grid(row=3, column=0, sticky="ew", padx=20, pady=(0, 10))
    status_frame.grid_columnconfigure(0, weight=1)
    status_frame.grid_propagate(False)
    
    status_label = ctk.CTkLabel(
        status_frame,
        text="Welcome! Upload an audio file or record your voice, then click 'Score'",
        font=("Arial", 12),
        text_color="gray"
    )
    status_label.grid(row=0, column=0, padx=20, pady=10, sticky="w")
    
    # Configure button commands
    upload_btn.configure(
        command=lambda: upload_audio_file(waveform_frame, status_label, output)
    )
    record_btn.configure(
        command=lambda: toggle_recording(waveform_frame, status_label, record_btn, output)
    )
    score_btn.configure(
        command=lambda: score_current_audio(output, score_frame, status_label)
    )
    
    # Add recording flag to record button
    record_btn.recording = False
    
    # Initial instructions
    output.configure(state="normal")
    output.insert("end", "Instructions:\n\n")
    output.insert("end", "1. 📁 Upload Audio File - Select an audio file (WAV, MP3, M4A, FLAC, OGG)\n")
    output.insert("end", "2. 🎤 Record Voice - Record your voice (toggle button)\n")
    output.insert("end", "3. 📊 Score - Check grammar and get score (available after step 1 or 2)\n\n")
    output.insert("end", "Note: Close the application properly using the window close button.")
    output.configure(state="disabled")
    
    app.mainloop()

if __name__ == "__main__":
    launch_ui()