import tkinter
import tkinter.messagebox
import tkinter.filedialog as filedialog
import customtkinter
import os
import soundfile as sf
import sounddevice as sd
import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.preprocessing import StandardScaler
from applymodel import applyindivmodel
from massapplymodel import massapplymodelfunc
import librosa
import audiofunc
from PIL import Image, ImageTk
import math
import sys
import massapplymodel
import csv

customtkinter.set_appearance_mode("Light")
customtkinter.set_default_color_theme("green")

CONVERTED_AUDIOS_FOLDER = "converted_audios"
script_directory = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_directory, r'Image')
output_file_path = os.path.join(script_directory, 'prediction_results.txt')
history_file_path = os.path.join(script_directory, 'history.txt')

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
        #i learned with my mistakes and everything should be dynamic
        total_files = self.total_files = 0
        legit_files = self.legit_files = 0
        modified_files = self.modified_files = 0
        
        self.legit_file_list = []
        self.modified_file_list = []
        
        
        self.audio_functions = audiofunc.AudioFunctions(self)
        self.selected_files = False
        self.playback_thread = None
        self.playing = False

        self.title("CTRL ALT DELETE")
        self.geometry(f"{900}x{480}")

        self.grid_columnconfigure(1, weight=2)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.sidebar_frame = customtkinter.CTkFrame(self, width=150, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame,
                                                 text="CTRL ALT DELETE",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=10, pady=(10, 10))
        
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, 
                                                        text="Analyze Audio",
                                                        command=self.sidebar_button_event)
        self.sidebar_button_1.grid(row=1, column=0, padx=10, pady=10)
        self.sidebar_button_1.configure(state="disabled")
        
        self.audio_tabview = customtkinter.CTkTabview(self, width=250)
        self.audio_tabview.grid(row=0, column=1, padx=(20,20), pady=(20,20), sticky="nsew")
        self.audio_tabview.add("Files")
        self.audio_tabview.add("Spectrogram")
        self.audio_tabview.add("Summary")

        self.slider_progressbar_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.slider_progressbar_frame.grid(row=1, column=1, columnspan=2, padx=(20, 20), pady=(0, 0), sticky="ew")
        self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
        self.progressbar_1 = customtkinter.CTkProgressBar(self.slider_progressbar_frame)
        self.progressbar_1.grid(row=0, column=0, padx=(20, 20), pady=(10, 10), sticky="ew")

        self.audio_tabview.tab("Files").grid_columnconfigure(0, weight=1)
        self.audio_tabview.tab("Files").grid_columnconfigure(1, weight=1)

        self.select_folder_button = customtkinter.CTkButton(self.audio_tabview.tab("Files"),
                                                            text="Choose Folder",
                                                            command=self.audio_functions.open_folder)
        self.select_folder_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        self.st_button = customtkinter.CTkButton(self.audio_tabview.tab("Files"),
                                                 text="Start",
                                                 command=self.start_button)
        self.st_button.grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.st_button.configure(state="disabled")
        self.test_all_button = customtkinter.CTkButton(self.audio_tabview.tab("Files"),
                                                       text="Test All",
                                                       command=self.test_all_files)
        self.test_all_button.grid(row=4, column=0, padx=10, pady=10, sticky="w")

        self.play_button = customtkinter.CTkButton(self.audio_tabview.tab("Files"),
                                                   text="Play Audio",
                                                   command=self.audio_functions.play_audio)
        self.play_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")
        self.play_button.configure(state="disabled")

        self.canvas = tkinter.Canvas(self.audio_tabview.tab("Files"), height=0)
        self.canvas.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.scrollbar = customtkinter.CTkScrollbar(self.audio_tabview.tab("Files"), orientation="vertical", command=self.canvas.yview)
        self.scrollbar.grid(row=0, column=2, sticky="ns")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.audio_select_frame = tkinter.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.audio_select_frame, anchor="nw")

        #results/spectrogram
        self.audio_tabview.tab("Spectrogram").grid_columnconfigure(0, weight=1)
        self.hover_label = tkinter.Label(self.audio_tabview.tab("Spectrogram"))
        
        self.spectrogram_labels = []  # To store references to spectrogram image labels

        self.about_frame = customtkinter.CTkButton(self.sidebar_frame,
                                                   text="About",
                                                   command=self.about_button_event)
        self.about_frame.grid(row=3, column=0, padx=10, pady=10)
        
        
        #summary
        self.save_csv_button = customtkinter.CTkButton(self.audio_tabview.tab("Summary"),
                                               text="Save Results to CSV",
                                               command=self.save_results_to_csv)
        self.save_csv_button.pack(padx=10, pady=10, anchor='w')
        
          # Add buttons to show filenames
        self.show_legit_files_button = customtkinter.CTkButton(self.audio_tabview.tab("Summary"),
                                                               text="Show Legitimate Files",
                                                               command=self.show_legit_files)
        self.show_legit_files_button.pack(padx=10, pady=10, anchor='w')

        self.show_modified_files_button = customtkinter.CTkButton(self.audio_tabview.tab("Summary"),
                                                                  text="Show Modified Files",
                                                                  command=self.show_modified_files)
        self.show_modified_files_button.pack(padx=10, pady=10, anchor='w')

        
        
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, 
                                                        text="Documentation",
                                                        command=self.show_documentation)
        self.sidebar_button_2.grid(row=2, column=0, padx=10, pady=10)
        
        self.documentation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.documentation_frame.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=10, pady=10)
        self.documentation_frame.grid_remove()
        
        self.documentation_text = tkinter.Text(self.documentation_frame, wrap="word")
        self.documentation_text.pack(expand=True, fill="both", padx=10, pady=10)
        self.documentation_text.config(state="disabled") 
        self.documentation_text.tag_configure("font", font=("arial", 12))  

        self.result_text = tkinter.Text(self.audio_tabview.tab("Summary"), wrap="word")
        self.result_text.pack(expand=True, fill="both", padx=10, pady=10)
        self.result_text.config(state="disabled") 

        # Navigation buttons
        self.previous_button = customtkinter.CTkButton(self.audio_tabview.tab("Spectrogram"),
                                                       text="Previous",
                                                       command=self.show_previous_image)
        self.previous_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")
        
        self.next_button = customtkinter.CTkButton(self.audio_tabview.tab("Spectrogram"),
                                                   text="Next",
                                                   command=self.show_next_image)
        self.next_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")
        
        self.current_image_index = -1
        self.spectrogram_paths = []

        
    def open_folder(self):
        self.audio_functions.open_folder()

    def update_start_button(self):
        self.audio_functions.update_start_button()

    def play_audio(self):
        self.audio_functions.play_audio()

    def open_audio_window(self, file_path):
        self.audio_functions.open_audio_window(file_path)

    def play_audio_file(self, file_path):
        self.audio_functions.play_audio_file(file_path)

    def stop_audio(self):
        self.audio_functions.stop_audio()

    def on_closing_audio_window(self):
        self.audio_functions.on_closing_audio_window()

    def sidebar_button_event(self):
        print("Sidebar button clicked")

    def show_documentation(self):
        self.documentation_frame.grid()
        self.load_documentation_content()
    
    def about_button_event(self):
        tkinter.messagebox.showinfo("About", "This is an audio analysis application.")

    def load_documentation_content(self):
        documentation_text = """
        Audio Analysis Application Documentation

        Features:
        - Choose a folder containing audio files
        - Analyze audio files and generate spectrograms
        - Play selected audio files
        - View results of the analysis

        How to Use:
        1. Click on "Choose Folder" to select a folder containing audio files.
        2. Select the audio files you want to analyze.
        3. Click on "Start" to begin the analysis.
        4. View the spectrogram and results in their respective tabs.
        5. Click on "Play Audio" to listen to the selected audio file.
        """
        self.documentation_text.config(state="normal")
        self.documentation_text.delete(1.0, tkinter.END)
        self.documentation_text.insert(tkinter.END, documentation_text, "font")
        self.documentation_text.config(state="disabled")
        
    def generate_spectrogram(self, file_path, output_folder):
        if file_path:
            # Convert MP3 to WAV
            wav_file_path = self.audio_functions.convert_to_wav(file_path)
            if not wav_file_path:
                return

            # Generate spectrogram
            spectrogram_save_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_spectrogram.png")
            self.spectrogram_paths.append(spectrogram_save_path)  # Save the spectrogram path

            self.mp3_to_spectrogram(wav_file_path, spectrogram_save_path)
            
            # Classify the spectrogram and update the lists
            result_text = self.apply_model_and_display_prediction(spectrogram_save_path)
            if "LEGITIMATE" in result_text:
                self.legit_files += 1
                self.legit_file_list.append(file_path)
            elif "MODIFIED" in result_text:
                self.modified_files += 1
                self.modified_file_list.append(file_path)
        
        # Update total files count
        self.total_files += 1

        # Display updated summary
        self.display_summary()
        
        if len(self.spectrogram_paths) == 1:
            self.show_image(0)  # Show the first image after generating the first spectrogram

                
            self.apply_model_and_display_prediction(spectrogram_save_path)

    def mp3_to_spectrogram(self, file_path, save_path=None):
        try:
            y, sr = librosa.load(file_path, sr=16000, duration=5.0)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        f, t, Zxx = spectrogram(y, fs=sr, window='hamming', nperseg=int(sr * 0.108), noverlap=int(sr * 0.01))
        log_Zxx = np.log1p(np.abs(Zxx))
        scaler = StandardScaler()
        z_normalized_log_Zxx = scaler.fit_transform(log_Zxx.T).T
        max_db_value = 11.0
        z_normalized_log_Zxx = np.clip(z_normalized_log_Zxx, -np.inf, max_db_value)

        plt.figure(figsize=(8, 6))
        plt.imshow(z_normalized_log_Zxx, aspect='auto', origin='lower', cmap='viridis', vmax=max_db_value)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()

    def show_image(self, index):
        self.current_image_index = index
        image_path = self.spectrogram_paths[index]
        original_image = Image.open(image_path)
        resized_image = original_image.resize((250, 250))
        tk_image = ImageTk.PhotoImage(resized_image)
        
        # Create a label for the spectrogram image
        if hasattr(self, 'spectrogram_label'):
            self.spectrogram_label.destroy()
        
        self.spectrogram_label = tkinter.Label(self.audio_tabview.tab("Spectrogram"), image=tk_image)
        self.spectrogram_label.image = tk_image  # Keep a reference to avoid garbage collection
        self.spectrogram_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nesw")
        
        # Display predictions
        result_text = self.apply_model_and_display_prediction(image_path)
        
        # Create or update the text box for displaying prediction results
        if hasattr(self, 'spectrogram_result'):
            self.spectrogram_result.destroy()
            
        self.spectrogram_result = customtkinter.CTkTextbox(self.audio_tabview.tab("Spectrogram"))
        self.spectrogram_result.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        self.spectrogram_result.insert("0.0", result_text)



    def show_next_image(self):
        if self.current_image_index < len(self.spectrogram_paths) - 1:
            self.show_image(self.current_image_index + 1)

    def show_previous_image(self):
        if self.current_image_index > 0:
            self.show_image(self.current_image_index - 1)

    def apply_model_and_display_prediction(self, spectrogram_save_path):
        try:
            file_name = os.path.basename(spectrogram_save_path).replace('_spectrogram.png', '')

            with open(output_file_path, 'r') as output_file:
                lines = output_file.readlines()
            result = applyindivmodel(spectrogram_save_path)

            if result is None:
                return "Error: Model could not process the spectrogram."

            predicted_class_line = lines[0].strip()
            class_probabilities_line = lines[1].strip()

            predicted_class = int(predicted_class_line.split(":")[1].strip())
            class_probabilities_str = class_probabilities_line.split(":")[1].strip()
            class_probabilities = [round(float(value) * 100, 2) for value in class_probabilities_str.strip("[]").split(",")]

            predicted_class, class_probabilities = result

            result_text = f"File: {file_name}\n"
            if predicted_class == 0:
                result_text += 'The audio file is LEGITIMATE\n'
            else:
                result_text += 'The audio file is MODIFIED\n'
            
            modification_types = {
                0: 'Unmodified',
                1: 'Voice Synthesis',
                2: 'Voice Changer',
                3: 'Voice Splicing'
            }

            result_text += f'Modification Type: {modification_types.get(predicted_class, "Unknown")}\n'
            
            e = math.e
            k = 10
            x = class_probabilities[predicted_class]  # probability of the predicted class
            funnum = (1 / (1 + e**(-k * (x - 0.3)))) * 100
            funnum = round(funnum, 2)
            result_text += f'Confidence Level: {funnum}%\n'
            return result_text

        except Exception as e:
            return f"An error occurred: {e}"

    def start_button(self):
        # Clear previous spectrogram labels
        self.spectrogram_paths.clear()
        if hasattr(self, 'spectrogram_label'):
            self.spectrogram_label.destroy()
        
        checked_paths = []
        for file, var in self.audio_functions.check_vars:
            if var.get():
                audio_path = os.path.join(self.audio_functions.selected_folder_path, file)
                checked_paths.append(audio_path)
                
        output_folder = "output_spectrograms"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Start a new thread for spectrogram generation
        threading.Thread(target=self.generate_spectrograms_thread, args=(checked_paths, output_folder)).start()

    def generate_spectrograms_thread(self, file_paths, output_folder):
        for file_path in file_paths:
            self.generate_spectrogram(file_path, output_folder)
            
    def test_all_files(self):
        selected_folder_path = self.audio_functions.selected_folder_path

        if not selected_folder_path:
            tkinter.messagebox.showerror("Error", "Please select a folder first.")
            return
        
        output_folder = "output_spectrograms"
        output_file_path = "test_all_results.csv"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        try:
            # Step 1: Gather all audio file paths (both MP3 and WAV) from the selected folder
            file_paths = []
            for file in os.listdir(selected_folder_path):
                if file.endswith('.mp3') or file.endswith('.wav'):
                    file_paths.append(os.path.join(selected_folder_path, file))
            
            # Step 2: Generate spectrograms for all audio files in the selected folder
            for file_path in file_paths:
                self.generate_spectrogram(file_path, output_folder)

            # Step 3: Apply the model to classify the generated spectrograms
            massapplymodel.massapplymodelfunc(output_folder, output_file_path)
            tkinter.messagebox.showinfo("Success", f"Results saved to {output_file_path}")
        except Exception as e:
            tkinter.messagebox.showerror("Error", f"An error occurred: {e}")
            
    def display_summary(self):
        summary_text = f"""
        Total Files: {self.total_files}
        Legitimate Files: {self.legit_files}
        Modified Files: {self.modified_files}
        """
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tkinter.END)
        self.result_text.insert(tkinter.END, summary_text)
        self.result_text.config(state="disabled")

    def show_legit_files(self):
        self.show_file_list(self.legit_file_list, "Legitimate Files")

    def show_modified_files(self):
        self.show_file_list(self.modified_file_list, "Modified Files")

    def show_file_list(self, file_list, title):
        file_list_text = f"{title}:\n\n" + "\n".join(file_list)
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tkinter.END)
        self.result_text.insert(tkinter.END, file_list_text)
        self.result_text.config(state="disabled")

    def save_results_to_csv(self):
        self.save_list_to_csv(self.legit_file_list, "legit_files.csv")
        self.save_list_to_csv(self.modified_file_list, "modified_files.csv")
        tkinter.messagebox.showinfo("CSV Saved", "Results saved to CSV successfully.")

    def save_list_to_csv(self, file_list, filename):
        try:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Filename"])
                for file_path in file_list:
                    writer.writerow([file_path])
        except Exception as e:
            tkinter.messagebox.showerror("Error", f"Failed to save {filename}: {e}")    

if __name__ == "__main__":
    app = App()
    app.mainloop()
