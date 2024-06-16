import os
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import soundfile as sf
import sounddevice as sd
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import customtkinter as ctk

# Constants
CONVERTED_AUDIOS_FOLDER = "converted_audios"

class AudioFunctions:
    def __init__(self, app_instance):
        self.app = app_instance
        self.selected_folder_path = None
        self.check_vars = []

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.app.select_folder_button.configure(text="Choose again")
            self.clear_audio_select_frame()

            self.selected_folder_path = folder_path
            audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp3', '.wav', '.flac'))]

            for file in audio_files:
                var = tk.BooleanVar()
                checkbox = ctk.CTkCheckBox(self.app.audio_select_frame, text=file, variable=var)
                checkbox.pack(anchor="w", padx=10, pady=5)
                self.check_vars.append((file, var))

            self.app.st_button.configure(state="disabled")
            for _, var in self.check_vars:
                var.trace_add("write", lambda name, index, mode, var=var: self.update_start_button())

            self.app.audio_select_frame.update_idletasks()
            self.app.canvas.config(scrollregion=self.app.canvas.bbox("all"))
            self.app.play_button.configure(state="normal")

    def clear_audio_select_frame(self):
        for widget in self.app.audio_select_frame.winfo_children():
            widget.destroy()
        self.check_vars = []

    def update_start_button(self):
        for _, var in self.check_vars:
            if var.get():
                self.app.st_button.configure(state="normal")
                return
        self.app.st_button.configure(state="disabled")

    def play_audio(self):
        if self.selected_folder_path:
            checked_paths = []
            for file, var in self.check_vars:
                if var.get():
                    audio_path = os.path.join(self.selected_folder_path, file)
                    checked_paths.append(audio_path)
            if len(checked_paths) > 1:
                messagebox.showwarning("Warning", "Please select only one audio file to play at a time.")
            elif checked_paths:
                self.app.play_button.configure(state="disabled")
                self.open_audio_window(checked_paths[0])

    def open_audio_window(self, file_path):
        self.app.playing = True
        self.app.play_window = ctk.CTkToplevel(self.app)
        self.app.play_window.title("Playing Audio")
        self.app.play_window.geometry("300x100")

        play_label = ctk.CTkLabel(self.app.play_window, text=f"Playing: {os.path.basename(file_path)}")
        play_label.pack(pady=10)

        stop_button = ctk.CTkButton(self.app.play_window, text="Stop", command=self.stop_audio)
        stop_button.pack(pady=10)

        self.app.playback_thread = threading.Thread(target=self.play_audio_file, args=(file_path,))
        self.app.playback_thread.start()

    def play_audio_file(self, file_path):
        try:
            data, samplerate = sf.read(file_path)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Error playing {file_path}: {e}")
            try:
                print(f"Attempting to convert {file_path} to WAV...")
                wav_file = self.convert_to_wav(file_path)
                if wav_file:
                    data, samplerate = sf.read(wav_file)
                    sd.play(data, samplerate)
                    sd.wait()
                else:
                    print(f"Failed to convert {file_path} to WAV")
            except Exception as convert_error:
                print(f"Error converting {file_path} to WAV: {convert_error}")
        finally:
            self.audio_playback_finished()

    def stop_audio(self):
        sd.stop()
        if self.app.play_window:
            self.app.play_window.destroy()
        self.audio_playback_finished()

    def audio_playback_finished(self):
        self.app.playing = False
        self.app.play_button.configure(state="normal")

    def convert_to_wav(self, input_file):
        try:
            if not os.path.exists(CONVERTED_AUDIOS_FOLDER):
                os.makedirs(CONVERTED_AUDIOS_FOLDER)
            output_file = os.path.join(CONVERTED_AUDIOS_FOLDER, os.path.splitext(os.path.basename(input_file))[0] + '.wav')

            if os.path.exists(output_file):
                print(f"Converted file already exists: {output_file}")
                return output_file

            audio = AudioSegment.from_file(input_file)
            audio.export(output_file, format='wav')
            print(f"Converted {input_file} to {output_file}")
            return output_file

        except CouldntDecodeError as decode_error:
            print(f"Error decoding {input_file}: {decode_error}")
            return None
        except Exception as e:
            print(f"Error converting {input_file} to WAV: {e}")
            return None

    def check_all(self):
        for child in self.app.audio_select_frame.winfo_children():
            if isinstance(child, tk.Checkbutton):
                child.select()
