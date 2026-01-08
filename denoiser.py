import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import warnings
warnings.filterwarnings('ignore')

class AudioDenoiserFFT:
    """
    Sistema completo para eliminación de ruido en audio usando FFT
    """
    
    def __init__(self):
        self.fs = None
        self.audio = None
        self.audio_filtrado = None
        self.fft_filtrado = None
        self.fft_filtrado_sin_ventana = None
        
    def cargar_audio(self):
        """Solicita y carga un archivo WAV del usuario"""
        archivo = input("\n📁 Nombre del archivo .wav (ej: audio.wav): ").strip()
        
        try:
            self.fs, audio = wavfile.read(archivo)
            
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
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
    
    def calcular_fft_con_ventana(self, señal=None):
        """Calcula FFT aplicando ventana de Hann (para filtrado)"""
        if señal is None:
            señal = self.audio
        
        n = len(señal)
        ventana = np.hanning(n)
        señal_ventaneada = señal * ventana
        fft_resultado = np.fft.fft(señal_ventaneada)
        frecuencias = np.fft.fftfreq(n, 1/self.fs)
        
        return fft_resultado, frecuencias, ventana
    
    def calcular_fft_sin_ventana(self, señal=None):
        """Calcula FFT sin ventana (para validación Parseval)"""
        if señal is None:
            señal = self.audio
        
        n = len(señal)
        fft_resultado = np.fft.fft(señal)
        frecuencias = np.fft.fftfreq(n, 1/self.fs)
        
        return fft_resultado, frecuencias
    
    def mostrar_espectro(self):
        """Visualiza el espectro para ayudar a seleccionar filtro"""
        fft_audio, frecuencias, _ = self.calcular_fft_con_ventana()
        
        idx_pos = frecuencias >= 0
        frec_pos = frecuencias[idx_pos]
        magnitud = np.abs(fft_audio[idx_pos])
        
        plt.figure(figsize=(10, 4))
        plt.plot(frec_pos, magnitud, 'b', linewidth=1)
        plt.title('ESPECTRO DEL AUDIO - Analice para elegir filtro')
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, self.fs/2])
        plt.yscale('log')
        
        plt.axvline(1000, color='r', linestyle='--', alpha=0.3, label='1 kHz')
        plt.axvline(2000, color='r', linestyle='--', alpha=0.3, label='2 kHz')
        plt.axvline(5000, color='r', linestyle='--', alpha=0.3, label='5 kHz')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*60)
        print("GUÍA PARA SELECCIÓN DE FILTRO")
        print("="*60)
        print("\nBasado en el espectro:")
        print("• Ruido en altas frecuencias → PASA-BAJAS")
        print("• Ruido en bajas frecuencias → PASA-ALTAS")
        print("• Pico específico de ruido → NOTCH")
        print("• Solo rango específico → PASA-BANDA")
    
    def diseñar_filtro(self, tipo, parametros):
        """Genera máscara de filtro en dominio de frecuencia"""
        n = len(self.audio)
        frecuencias = np.fft.fftfreq(n, 1/self.fs)
        mascara = np.ones(n, dtype=np.complex128)
        
        if tipo == 'lowpass':
            fc = parametros['frecuencia_corte']
            mascara[np.abs(frecuencias) > fc] = 0
            print(f"✓ Filtro pasa-bajas: corte en {fc} Hz")
            
        elif tipo == 'highpass':
            fc = parametros['frecuencia_corte']
            mascara[np.abs(frecuencias) < fc] = 0
            print(f"✓ Filtro pasa-altas: corte en {fc} Hz")
            
        elif tipo == 'bandpass':
            f_low = parametros['frecuencia_baja']
            f_high = parametros['frecuencia_alta']
            mascara[(np.abs(frecuencias) < f_low) | (np.abs(frecuencias) > f_high)] = 0
            print(f"✓ Filtro pasa-banda: {f_low}-{f_high} Hz")
            
        elif tipo == 'notch':
            f_center = parametros['frecuencia_central']
            ancho = parametros['ancho_banda']
            f_low = f_center - ancho/2
            f_high = f_center + ancho/2
            mascara[(np.abs(frecuencias) >= f_low) & (np.abs(frecuencias) <= f_high)] = 0
            print(f"✓ Filtro notch: elimina {f_low}-{f_high} Hz")
        
        from scipy.ndimage import gaussian_filter1d
        mascara = gaussian_filter1d(mascara.real, sigma=5)
        
        return mascara
    
    def aplicar_filtro(self):
        """Aplica el filtro seleccionado por el usuario"""
        print("\n" + "="*60)
        print("CONFIGURACIÓN DE FILTRO")
        print("="*60)
        
        print("\nOpciones disponibles:")
        print("1. Pasa-bajas (elimina altas frecuencias)")
        print("2. Pasa-altas (elimina bajas frecuencias)")
        print("3. Pasa-banda (conserva rango específico)")
        print("4. Notch (elimina banda específica)")
        
        opcion = input("\nSeleccione tipo de filtro (1-4): ").strip()
        
        if opcion == '1':
            fc = float(input("Frecuencia de corte (Hz): "))
            mascara = self.diseñar_filtro('lowpass', {'frecuencia_corte': fc})
            
        elif opcion == '2':
            fc = float(input("Frecuencia de corte (Hz): "))
            mascara = self.diseñar_filtro('highpass', {'frecuencia_corte': fc})
            
        elif opcion == '3':
            f_low = float(input("Frecuencia inferior (Hz): "))
            f_high = float(input("Frecuencia superior (Hz): "))
            mascara = self.diseñar_filtro('bandpass', {
                'frecuencia_baja': f_low,
                'frecuencia_alta': f_high
            })
            
        elif opcion == '4':
            f_center = float(input("Frecuencia central a eliminar (Hz): "))
            ancho = float(input("Ancho de banda (Hz): "))
            mascara = self.diseñar_filtro('notch', {
                'frecuencia_central': f_center,
                'ancho_banda': ancho
            })
        
        else:
            print("Opción no válida. Usando filtro pasa-bajas por defecto.")
            mascara = self.diseñar_filtro('lowpass', {'frecuencia_corte': 1000})
        
        fft_audio, _, ventana = self.calcular_fft_con_ventana()
        self.fft_filtrado = fft_audio * mascara
        audio_filtrado = np.fft.ifft(self.fft_filtrado).real
        
        audio_filtrado = audio_filtrado / np.mean(ventana)
        max_val = np.max(np.abs(audio_filtrado))
        if max_val > 0:
            audio_filtrado = audio_filtrado / max_val
        
        self.audio_filtrado = audio_filtrado
        self.fft_filtrado_sin_ventana, _ = self.calcular_fft_sin_ventana(audio_filtrado)
        
        print("✓ Procesamiento completado")
    
    def calcular_metricas(self):
        """Evalúa calidad del procesamiento con MSE y SNR"""
        mse = np.mean((self.audio - self.audio_filtrado) ** 2)
        
        potencia_señal = np.mean(self.audio ** 2)
        potencia_ruido = np.mean((self.audio - self.audio_filtrado) ** 2)
        
        if potencia_ruido > 0:
            snr = 10 * np.log10(potencia_señal / potencia_ruido)
        else:
            snr = float('inf')
        
        return {'MSE': mse, 'SNR (dB)': snr}
    
    def verificar_parseval(self, señal=None, fft_señal=None):
        """Comprueba conservación de energía entre tiempo y frecuencia"""
        if señal is None:
            señal = self.audio
        
        n = len(señal)
        
        if fft_señal is None:
            fft_señal, _ = self.calcular_fft_sin_ventana(señal)
        
        energia_tiempo = np.sum(señal ** 2)
        energia_frecuencia = np.sum(np.abs(fft_señal) ** 2) / n
        
        diferencia = abs(energia_tiempo - energia_frecuencia)
        diferencia_porcentual = (diferencia / energia_tiempo) * 100
        
        return {
            'Energía tiempo': energia_tiempo,
            'Energía frecuencia': energia_frecuencia,
            'Diferencia %': diferencia_porcentual
        }
    
    def mostrar_graficas(self):
        """Genera visualizaciones comparativas"""
        fft_original, frecuencias, _ = self.calcular_fft_con_ventana(self.audio)
        fft_filtrado, _, _ = self.calcular_fft_con_ventana(self.audio_filtrado)
        
        t = np.arange(len(self.audio)) / self.fs
        
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        ax1.plot(t[:1000], self.audio[:1000], 'b', linewidth=1)
        ax1.set_title('Señal Original')
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Amplitud')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(t[:1000], self.audio_filtrado[:1000], 'r', linewidth=1)
        ax2.set_title('Señal Procesada')
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Amplitud')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
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
        ax4.set_title('Espectro Procesado')
        ax4.set_xlabel('Frecuencia (Hz)')
        ax4.set_ylabel('Magnitud')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([0, self.fs/4])
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def guardar_resultado(self):
        """Almacena el audio procesado en disco"""
        nombre = input("\n💾 Nombre para archivo resultante: ").strip()
        if not nombre.endswith('.wav'):
            nombre += '.wav'
        
        audio_int16 = (self.audio_filtrado * 32767).astype(np.int16)
        wavfile.write(nombre, self.fs, audio_int16)
        print(f"✓ Archivo guardado: {nombre}")


def main():
    """Función principal del programa"""
    print("="*60)
    print("SISTEMA DE PROCESAMIENTO DE AUDIO - FFT + PARSEVAL")
    print("="*60)
    
    while True:
        procesador = AudioDenoiserFFT()
        
        print("\n" + "="*60)
        print("CARGA DE ARCHIVO DE AUDIO")
        print("="*60)
        
        if not procesador.cargar_audio():
            print("Intente nuevamente...")
            continue
        
        print("\n" + "="*60)
        print("ANÁLISIS ESPECTRAL")
        print("="*60)
        procesador.mostrar_espectro()
        
        procesador.aplicar_filtro()
        
        print("\n" + "="*60)
        print("EVALUACIÓN DE RESULTADOS")
        print("="*60)
        metricas = procesador.calcular_metricas()
        print(f"Error Cuadrático Medio (MSE): {metricas['MSE']:.6f}")
        print(f"Relación Señal-Ruido (SNR): {metricas['SNR (dB)']:.2f} dB")
        
        print("\n" + "="*60)
        print("VALIDACIÓN TEÓRICA - TEOREMA DE PARSEVAL")
        print("="*60)
        
        resultado_inicial = procesador.verificar_parseval()
        print("\nEstado inicial:")
        print(f"  Energía en dominio temporal:    {resultado_inicial['Energía tiempo']:.6f}")
        print(f"  Energía en dominio frecuencial: {resultado_inicial['Energía frecuencia']:.6f}")
        print(f"  Discrepancia:                   {resultado_inicial['Diferencia %']:.6f}%")
        
        resultado_final = procesador.verificar_parseval(
            procesador.audio_filtrado, 
            procesador.fft_filtrado_sin_ventana
        )
        print("\nEstado procesado:")
        print(f"  Energía en dominio temporal:    {resultado_final['Energía tiempo']:.6f}")
        print(f"  Energía en dominio frecuencial: {resultado_final['Energía frecuencia']:.6f}")
        print(f"  Discrepancia:                   {resultado_final['Diferencia %']:.6f}%")
        
        print("\n" + "="*60)
        print("VISUALIZACIÓN DE RESULTADOS")
        print("="*60)
        input("Presione Enter para continuar...")
        procesador.mostrar_graficas()
        
        guardar = input("\n¿Desea guardar el resultado? (s/n): ").strip().lower()
        if guardar == 's':
            procesador.guardar_resultado()
        
        continuar = input("\n¿Procesar otro archivo? (s/n): ").strip().lower()
        if continuar != 's':
            print("\nFinalizando ejecución...")
            break


if __name__ == "__main__":
    main()
