import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector
from pydub import AudioSegment
from pydub.playback import play
import os
from datetime import datetime
import random
import sounddevice as sd
import threading
import time

# Desactiva el modo interactivo de matplotlib para mejorar el rendimiento
plt.ioff()

# Obtén la ruta de la carpeta del proyecto
PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def normalizar_audio(audio):
    """Normaliza el audio para que su volumen sea más consistente."""
    return librosa.util.normalize(audio)

def detectar_picos(y, sr, umbral=0.1, espacio_minimo=8000):
    """Detecta los picos de energía en el audio, con un espacio mínimo entre ellos."""
    energia = librosa.feature.rms(y=y)[0]
    indices_picos = np.where(energia > umbral * np.max(energia))[0]
    picos_filtrados = []
    for i in range(len(indices_picos)):
        if len(picos_filtrados) == 0 or indices_picos[i] - picos_filtrados[-1] > espacio_minimo:
            picos_filtrados.append(indices_picos[i])
    return picos_filtrados, energia

def extraer_sampler_automatico(y, sr, duracion=4.0, num_samples=3, espacio_minimo=8000):
    """Extrae automáticamente samplers de una canción basado en picos de energía."""
    picos, energia = detectar_picos(y, sr)
    samplers = []
    usadas = []
    num_frames = len(y)
    frames_por_sample = int(duracion * sr)
    max_inicio = num_frames - frames_por_sample
    
    while len(samplers) < num_samples:
        inicio = random.randint(0, max_inicio)
        fin = inicio + frames_por_sample
        if all(abs(inicio - prev_inicio) > frames_por_sample for prev_inicio in usadas):
            samplers.append(y[inicio:fin])
            usadas.append(inicio)
    
    return samplers, energia, picos, sr

def filtro_bajo(audio, sr, corte=1000):
    """Aplica un filtro pasa-bajo para eliminar ruidos de alta frecuencia."""
    from scipy.signal import butter, filtfilt
    b, a = butter(4, corte / (0.5 * sr), btype='low')
    return filtfilt(b, a, audio)

def exportar_sampler(sampler, nombre_base, sr, formato="wav"):
    """Exporta un sampler a un archivo de audio con un nombre único."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{nombre_base}_{timestamp}.{formato}"
    ruta_completa = os.path.join(PROJECT_FOLDER, nombre_archivo)
    sf.write(ruta_completa, sampler, sr)
    print(f"Sampler guardado como: {ruta_completa}")

def extraer_sampler_manual(y, sr, inicio, fin):
    """Extrae un sampler manualmente basado en los tiempos de inicio y fin."""
    inicio_frame = int(inicio * sr)
    fin_frame = int(fin * sr)
    sampler = y[inicio_frame:fin_frame]
    return sampler

class SamplerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SamplerApp v1.0")

        # Crear menú desplegable clásico
        menu_bar = tk.Menu(self.root)
        self.root.config(menu=menu_bar)  # Esto enlaza el menú a la ventana principal

        # Menú de archivo
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Importar Audio", command=self.cargar_audio)
        file_menu.add_command(label="Salir", command=self.root.quit)
        menu_bar.add_cascade(label="Archivo", menu=file_menu)
        
        # self.btn_cargar = tk.Button(self.frame, text="Cargar Audio", command=self.cargar_audio)
        # self.btn_cargar.pack(pady=5)

        # Menú de ayuda
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Ver Ayuda", command=self.show_help)
        menu_bar.add_cascade(label="Ayuda", menu=help_menu)
        
        self.archivo_audio = None
        self.y = None
        self.sr = 22050  # Tasa de muestreo reducida
        self.inicio = None
        self.fin = None
        self.play_obj = None  # Objeto para controlar la reproducción
        
        
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        # Slider para cambiar el pitch
        self.label_pitch = tk.Label(self.frame, text="Cambiar Pitch:")
        self.label_pitch.pack(pady=5)
        self.slider_pitch = tk.Scale(self.frame, from_=-12, to=12, orient="horizontal", label="Pitch Shift (Semitonos)")
        self.slider_pitch.set(0)  # Valor inicial en 0 (sin cambio)
        self.slider_pitch.pack(pady=5)
        
        # Botón de Play con símbolo Unicode
        self.btn_play = tk.Button(self.frame, text="▶️ Play", command=self.reproducir_audio)
        self.btn_play.pack(pady=5)
        
        # Botón de Stop con símbolo Unicode
        self.btn_stop = tk.Button(self.frame, text="⏹️ Stop", command=self.detener_audio)
        self.btn_stop.pack(pady=5)
        
        # Extracción automática
        self.btn_auto = tk.Button(self.frame, text="Extraer Automáticamente", command=self.extraer_auto)
        self.btn_auto.pack(pady=5)
        
        # Extracción manual
        self.btn_manual = tk.Button(self.frame, text="Extraer Manualmente", command=self.extraer_manual)
        self.btn_manual.pack(pady=5)
        
       
        
      
        
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(pady=10)
        
        # Inicializar SpanSelector para selección con el ratón
        self.span = SpanSelector(
            self.ax, self.on_select, 'horizontal', useblit=True,
            props=dict(facecolor='red', alpha=0.5)
        )
        self.marcador_inicio = None
        self.marcador_fin = None
        
        # Conectar eventos de la rueda del ratón
        self.canvas.get_tk_widget().bind("<MouseWheel>", self.on_mousewheel)
        
        # Conectar la barra espaciadora para pausar/reanudar
        self.root.bind("<space>", self.pausar_reanudar_audio)

    def cargar_audio(self):
        archivo = filedialog.askopenfilename(filetypes=[("Archivos de audio", "*.wav;*.mp3;*.flac")])
        if not archivo:  # Si el usuario cancela, no hace nada
            return
        try:
            y, sr = librosa.load(archivo, sr=None)
            if y is None or sr is None:
                raise ValueError("El archivo de audio no se pudo cargar correctamente.")
            self.y, self.sr = y, sr

            # Calcular la energía del audio
            self.energia = librosa.feature.rms(y=self.y)[0]  # Calcula la energía (RMS)
            self.archivo_audio = archivo  # Guardar la ruta del archivo
            self.actualizar_grafico()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo de audio: {str(e)}")
    
    def actualizar_grafico(self):
        """Muestra la energía del audio en un gráfico."""

        if hasattr(self, 'energia') and self.energia is not None:  # Verificar si self.energia existe

            self.ax.clear()  # Limpiar el gráfico anterior
            tiempo = librosa.frames_to_time(np.arange(len(self.energia)), sr=self.sr)
            self.ax.plot(tiempo, self.energia, label='Energía del audio', color='blue')
            self.ax.set_title("Energía del audio")
            self.ax.set_xlabel("Tiempo (s)")
            self.ax.set_ylabel("Energía")
            self.ax.legend(loc='upper right')
            self.canvas.draw()
        else:
            messagebox.showwarning("Advertencia", "No se ha calculado la energía del audio.")
    
    def on_select(self, inicio, fin):
        """Maneja la selección de la zona a samplear."""
        self.inicio = inicio
        self.fin = fin
        self.dibujar_marcadores()
    
    def dibujar_marcadores(self):
        """Dibuja líneas verticales para marcar el inicio y fin seleccionados."""
        if self.marcador_inicio:
            self.marcador_inicio.remove()
        if self.marcador_fin:
            self.marcador_fin.remove()
        
        self.marcador_inicio = self.ax.axvline(self.inicio, color='green', linestyle='--', label='Inicio')
        self.marcador_fin = self.ax.axvline(self.fin, color='red', linestyle='--', label='Fin')
        self.canvas.draw()
    
    def on_mousewheel(self, event):
        """Ajusta el zoom del gráfico con la rueda del ratón."""
        scale_factor = 1.1 if event.delta > 0 else 0.9
        x_min, x_max = self.ax.get_xlim()
        x_range = (x_max - x_min) * scale_factor
        self.ax.set_xlim([x_min, x_min + x_range])
        self.canvas.draw()

    def reproducir_audio(self):
        """Reproduce solo el fragmento seleccionado de la canción."""
        if self.y is not None and self.sr is not None:
            if self.inicio is not None and self.fin is not None:
                # Extraer el fragmento seleccionado
                inicio_muestra = int(self.inicio * self.sr)
                fin_muestra = int(self.fin * self.sr)
                fragmento_audio = self.y[inicio_muestra:fin_muestra]

                # Aplicar el cambio de pitch si es necesario
                pitch_shift = self.slider_pitch.get()
                if pitch_shift != 0:
                    fragmento_audio = librosa.effects.pitch_shift(y=fragmento_audio, sr=self.sr, n_steps=pitch_shift)

                # Reproducir el fragmento en un hilo separado
                threading.Thread(target=self._reproducir_audio_thread, args=(fragmento_audio,)).start()
            else:
                messagebox.showwarning("Advertencia", "Selecciona un fragmento de la canción primero.")
        else:
            messagebox.showwarning("Advertencia", "No se ha cargado ningún archivo de audio.")
    
    def _reproducir_audio_thread(self, audio):
        """Reproduce audio en un hilo separado."""
        try:
            sd.play(audio, self.sr)
            sd.wait()  # Espera a que termine la reproducción
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo reproducir el audio: {str(e)}")

    def detener_audio(self):
        """Detiene la reproducción del audio."""
        sd.stop()

    def pausar_reanudar_audio(self, event=None):
        """Pausa o reanuda la reproducción del audio."""
        if self.play_obj is not None:
            self.play_obj.stop()
            self.play_obj = None
        else:
            self.reproducir_audio()
        
    def extraer_auto(self):
        """Extrae samplers automáticamente y los guarda."""
        if self.archivo_audio:
            samplers, energia, picos, sr = extraer_sampler_automatico(self.y, self.sr)
            for i, sampler in enumerate(samplers):
                exportar_sampler(sampler, f"sample_auto_{i+1}", sr)
            self.texto_estado.insert(tk.END, f"Se han extraído {len(samplers)} samplers automáticamente.\n")
            self.texto_estado.yview(tk.END)
        else:
            messagebox.showwarning("Advertencia", "Por favor, carga un archivo de audio primero.")

    def extraer_manual(self):
        """Extrae un sampler manualmente basándose en los tiempos de inicio y fin proporcionados."""
        try:
            inicio = float(self.entry_inicio.get())
            fin = float(self.entry_fin.get())
            if fin > inicio:
                sampler = extraer_sampler_manual(self.y, self.sr, inicio, fin)
                exportar_sampler(sampler, "sample_manual", self.sr)
                self.texto_estado.insert(tk.END, f"Sampler extraído manualmente y guardado.\n")
                self.texto_estado.yview(tk.END)
            else:
                messagebox.showwarning("Advertencia", "El tiempo de fin debe ser mayor que el de inicio.")
        except ValueError:
            messagebox.showwarning("Advertencia", "Por favor ingresa valores numéricos válidos para inicio y fin.")
    
    def limpiar_seleccion(self):
        """Limpia cualquier selección anterior en la interfaz."""
        self.inicio = None
        self.fin = None
        self.dibujar_marcadores()

    def get_text(self, key):
        if key == "Help":
            return "Ayuda"
        return key

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title(self.get_text("Help"))
        help_text = """
        Bienvenido a la ayuda del programa SamplerApp V_1.0.
        - Selecciona un archivo de audio.
        - Selecciona una región del audio para extraer el sampler ó extrae automaticamente.
        - Usa el sampler en tus producciones.
        """
        tk.Label(help_window, text=help_text, justify=tk.LEFT).pack(padx=10, pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = SamplerApp(root)
    root.mainloop()
