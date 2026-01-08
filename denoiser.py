import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')

class AudioDenoiserFFT:
    """
    Herramienta de denoising de audio con FFT y validación Parseval
    """
    
    def __init__(self):
        self.fs = None
        self.audio = None
        self.audio_filtrado = None
        self.fft_filtrado = None
        
    def cargar_audio(self):
        """Carga archivo WAV desde entrada del usuario"""
        archivo = input("\n📁 Nombre del archivo .wav (ej: audio.wav): ").strip()
        
        try:
            self.fs, audio = wavfile.read(archivo)
            
            # Convertir a mono si es estéreo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Normalizar a [-1, 1]
            audio = audio.astype(np.float32)
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            
            self.audio = audio
            print(f"✓ Audio cargado: {archivo}")
            print(f"  - Frecuencia: {self.fs} Hz")
            print(f"  - Duración: {len(audio)/self.fs:.2f} s")
            print(f"  - Muestras: {len(audio)}")
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def calcular_fft(self, señal=None):
        """Calcula FFT con ventana de Hann"""
        if señal is None:
            señal = self.audio
        
        n = len(señal)
        ventana = np.hanning(n)
        señal_ventaneada = señal * ventana
        fft_resultado = np.fft.fft(señal_ventaneada)
        frecuencias = np.fft.fftfreq(n, 1/self.fs)
        
        return fft_resultado, frecuencias, ventana
    
    def mostrar_espectro(self):
        """Muestra el espectro para que el usuario decida el filtro"""
        fft_audio, frecuencias, _ = self.calcular_fft()
        
        # Solo frecuencias positivas
        idx_pos = frecuencias >= 0
        frec_pos = frecuencias[idx_pos]
        magnitud = np.abs(fft_audio[idx_pos])
        
        plt.figure(figsize=(10, 4))
        plt.plot(frec_pos, magnitud, 'b', linewidth=1)
        plt.title('ESPECTRO DEL AUDIO (para decidir filtro)')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, self.fs/2])
        plt.yscale('log')
        
        # Marcar rangos de frecuencia típicos
        plt.axvline(1000, color='r', linestyle='--', alpha=0.3, label='1 kHz')
        plt.axvline(2000, color='r', linestyle='--', alpha=0.3, label='2 kHz')
        plt.axvline(5000, color='r', linestyle='--', alpha=0.3, label='5 kHz')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*60)
        print("ANÁLISIS DEL ESPECTRO")
        print("="*60)
        print("\nRecomendaciones:")
        print("1. Si hay mucho ruido en altas frecuencias → PASA-BAJAS")
        print("2. Si hay ruido en bajas frecuencias → PASA-ALTAS")
        print("3. Si hay pico específico de ruido → NOTCH")
        print("4. Si quieres solo un rango específico → PASA-BANDA")
    
    def diseñar_filtro(self, tipo, parametros):
        """Diseña diferentes tipos de filtros"""
        n = len(self.audio)
        frecuencias = np.fft.fftfreq(n, 1/self.fs)
        mascara = np.ones(n, dtype=np.complex128)
        
        if tipo == 'lowpass':
            fc = parametros['frecuencia_corte']
            mascara[np.abs(frecuencias) > fc] = 0
            print(f"✓ Filtro pasa-bajas diseñado: corte en {fc} Hz")
            
        elif tipo == 'highpass':
            fc = parametros['frecuencia_corte']
            mascara[np.abs(frecuencias) < fc] = 0
            print(f"✓ Filtro pasa-altas diseñado: corte en {fc} Hz")
            
        elif tipo == 'bandpass':
            f_low = parametros['frecuencia_baja']
            f_high = parametros['frecuencia_alta']
            mascara[(np.abs(frecuencias) < f_low) | (np.abs(frecuencias) > f_high)] = 0
            print(f"✓ Filtro pasa-banda diseñado: {f_low}-{f_high} Hz")
            
        elif tipo == 'notch':
            f_center = parametros['frecuencia_central']
            ancho = parametros['ancho_banda']
            f_low = f_center - ancho/2
            f_high = f_center + ancho/2
            mascara[(np.abs(frecuencias) >= f_low) & (np.abs(frecuencias) <= f_high)] = 0
            print(f"✓ Filtro notch diseñado: rechaza {f_low}-{f_high} Hz")
        
        # Suavizar bordes del filtro
        from scipy.ndimage import gaussian_filter1d
        mascara = gaussian_filter1d(mascara.real, sigma=5)
        
        return mascara
    
    def aplicar_filtro(self):
        """Aplica el filtro seleccionado por el usuario"""
        print("\n" + "="*60)
        print("SELECCIÓN DE FILTRO")
        print("="*60)
        
        print("\nTipos de filtro disponibles:")
        print("1. Pasa-bajas (elimina frecuencias ALTAS)")
        print("2. Pasa-altas (elimina frecuencias BAJAS)")
        print("3. Pasa-banda (solo deja un rango)")
        print("4. Notch (elimina frecuencia específica)")
        
        opcion = input("\nSeleccione filtro (1-4): ").strip()
        
        if opcion == '1':
            fc = float(input("Frecuencia de corte (Hz): "))
            mascara = self.diseñar_filtro('lowpass', {'frecuencia_corte': fc})
            
        elif opcion == '2':
            fc = float(input("Frecuencia de corte (Hz): "))
            mascara = self.diseñar_filtro('highpass', {'frecuencia_corte': fc})
            
        elif opcion == '3':
            f_low = float(input("Frecuencia baja (Hz): "))
            f_high = float(input("Frecuencia alta (Hz): "))
            mascara = self.diseñar_filtro('bandpass', {
                'frecuencia_baja': f_low,
                'frecuencia_alta': f_high
            })
            
        elif opcion == '4':
            f_center = float(input("Frecuencia central a eliminar (Hz): "))
            ancho = float(input("Ancho de banda a eliminar (Hz): "))
            mascara = self.diseñar_filtro('notch', {
                'frecuencia_central': f_center,
                'ancho_banda': ancho
            })
        
        else:
            print("Opción no válida. Usando filtro pasa-bajas por defecto.")
            mascara = self.diseñar_filtro('lowpass', {'frecuencia_corte': 1000})
        
        # Aplicar filtro
        fft_audio, _, ventana = self.calcular_fft()
        self.fft_filtrado = fft_audio * mascara
        audio_filtrado = np.fft.ifft(self.fft_filtrado).real
        
        # Compensar ventana y normalizar
        audio_filtrado = audio_filtrado / np.mean(ventana)
        max_val = np.max(np.abs(audio_filtrado))
        if max_val > 0:
            audio_filtrado = audio_filtrado / max_val
        
        self.audio_filtrado = audio_filtrado
        print("✓ Filtro aplicado correctamente")
    
    def calcular_metricas(self):
        """Calcula MSE y SNR"""
        mse = np.mean((self.audio - self.audio_filtrado) ** 2)
        
        potencia_señal = np.mean(self.audio ** 2)
        potencia_ruido = np.mean((self.audio - self.audio_filtrado) ** 2)
        
        if potencia_ruido > 0:
            snr = 10 * np.log10(potencia_señal / potencia_ruido)
        else:
            snr = float('inf')
        
        return {'MSE': mse, 'SNR (dB)': snr}
    
    def verificar_parseval(self, señal=None, fft_señal=None):
        """Verifica el teorema de Parseval"""
        if señal is None:
            señal = self.audio
            fft_señal, _, _ = self.calcular_fft(señal)
        
        n = len(señal)
        energia_tiempo = np.sum(señal ** 2)
        energia_frecuencia = np.sum(np.abs(fft_señal) ** 2) / n
        
        diferencia = abs(energia_tiempo - energia_frecuencia)
        diferencia_porcentual = (diferencia / energia_tiempo) * 100 if energia_tiempo > 0 else 0
        
        return {
            'Energía tiempo': energia_tiempo,
            'Energía frecuencia': energia_frecuencia,
            'Diferencia %': diferencia_porcentual
        }
    
    def mostrar_graficas(self):
        """Muestra las gráficas requeridas: tiempo y espectro"""
        fft_original, frecuencias, _ = self.calcular_fft(self.audio)
        fft_filtrado, _, _ = self.calcular_fft(self.audio_filtrado)
        
        t = np.arange(len(self.audio)) / self.fs
        
        # Figura 1: Señales en tiempo
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        ax1.plot(t[:1000], self.audio[:1000], 'b', linewidth=1)
        ax1.set_title('Señal Original (con ruido)')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Amplitud')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(t[:1000], self.audio_filtrado[:1000], 'r', linewidth=1)
        ax2.set_title('Señal Filtrada')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Amplitud')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Figura 2: Espectros
        fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 6))
        
        idx_pos = frecuencias >= 0
        ax3.plot(frecuencias[idx_pos], np.abs(fft_original[idx_pos]), 'b', alpha=0.7)
        ax3.set_title('Espectro Original')
        ax3.set_xlabel('Frecuencia (Hz)')
        ax3.set_ylabel('Magnitud')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([0, self.fs/4])
        ax3.set_yscale('log')
        
        ax4.plot(frecuencias[idx_pos], np.abs(fft_filtrado[idx_pos]), 'r', alpha=0.7)
        ax4.set_title('Espectro Filtrado')
        ax4.set_xlabel('Frecuencia (Hz)')
        ax4.set_ylabel('Magnitud')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, self.fs/4])
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def guardar_resultado(self):
        """Guarda el audio filtrado"""
        nombre = input("\n💾 Nombre para guardar (ej: audio_filtrado.wav): ").strip()
        if not nombre.endswith('.wav'):
            nombre += '.wav'
        
        audio_int16 = (self.audio_filtrado * 32767).astype(np.int16)
        wavfile.write(nombre, self.fs, audio_int16)
        print(f"✓ Audio guardado: {nombre}")


def main():
    """Programa principal"""
    print("="*60)
    print("DENOISING DE AUDIO CON FFT + VALIDACIÓN PARSEVAL")
    print("="*60)
    
    while True:
        # Crear denoiser
        denoiser = AudioDenoiserFFT()
        
        # 1. Cargar audio
        print("\n" + "="*60)
        print("CARGA DE AUDIO")
        print("="*60)
        
        if not denoiser.cargar_audio():
            print("Intenta de nuevo...")
            continue
        
        # 2. Mostrar espectro para decisión
        print("\n" + "="*60)
        print("ANÁLISIS DEL ESPECTRO")
        print("="*60)
        denoiser.mostrar_espectro()
        
        # 3. Aplicar filtro
        denoiser.aplicar_filtro()
        
        # 4. Calcular métricas
        print("\n" + "="*60)
        print("MÉTRICAS DE CALIDAD")
        print("="*60)
        metricas = denoiser.calcular_metricas()
        print(f"MSE: {metricas['MSE']:.6f}")
        print(f"SNR: {metricas['SNR (dB)']:.2f} dB")
        
        # 5. Verificar Parseval
        print("\n" + "="*60)
        print("VERIFICACIÓN PARSEVAL")
        print("="*60)
        
        # Antes
        parseval_antes = denoiser.verificar_parseval()
        print("\nANTES de filtrar:")
        print(f"  Energía en tiempo:    {parseval_antes['Energía tiempo']:.6f}")
        print(f"  Energía en frecuencia: {parseval_antes['Energía frecuencia']:.6f}")
        print(f"  Diferencia:           {parseval_antes['Diferencia %']:.4f}%")
        
        # Después
        parseval_despues = denoiser.verificar_parseval(
            denoiser.audio_filtrado, 
            denoiser.fft_filtrado
        )
        print("\nDESPUÉS de filtrar:")
        print(f"  Energía en tiempo:    {parseval_despues['Energía tiempo']:.6f}")
        print(f"  Energía en frecuencia: {parseval_despues['Energía frecuencia']:.6f}")
        print(f"  Diferencia:           {parseval_despues['Diferencia %']:.4f}%")
        
        # 6. Mostrar gráficas
        print("\n" + "="*60)
        print("GRÁFICAS COMPARATIVAS")
        print("="*60)
        input("Presiona Enter para mostrar gráficas...")
        denoiser.mostrar_graficas()
        
        # 7. Guardar resultado
        guardar = input("\n¿Guardar audio filtrado? (s/n): ").strip().lower()
        if guardar == 's':
            denoiser.guardar_resultado()
        
        # 8. Preguntar por otro audio
        otro = input("\n¿Procesar otro audio? (s/n): ").strip().lower()
        if otro != 's':
            print("\n¡Programa terminado!")
            break


if __name__ == "__main__":
    main()