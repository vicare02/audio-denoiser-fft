# Denoising de Audio con FFT + Validación Parseval

## Descripción
Herramienta para eliminar ruido de archivos de audio usando transformada de Fourier (FFT), con validación del teorema de Parseval.

## Características
- Carga archivos WAV
- Calcula espectro con FFT
- 4 tipos de filtros: pasa-bajas, pasa-altas, pasa-banda, notch
- Reconstruye señal con IFFT
- Gráficas comparativas (tiempo y frecuencia)
- Métricas: MSE y SNR
- Verificación del teorema de Parseval
- Interfaz interactiva por terminal

## Requisitos
- Python 3.6+
- numpy
- matplotlib
- scipy
- 
## Autor
Víctor Eduardo Aparicio Arenas

## Uso
```bash
python denoiser.py

