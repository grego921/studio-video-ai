import subprocess
import numpy as np
import scipy.io.wavfile as wavfile
import os
import sys

def main():
    print("=" * 70)
    print("TESTE DE MIXAGEM DE ÁUDIO COM FFMPEG (-map 0:v:0 -map 1:a:0)")
    print("=" * 70)

    video_440 = "temp_test_video_440hz.mp4"
    audio_880 = "temp_test_audio_880hz.wav"
    mixed_with_map = "temp_test_mixed_map.mp4"
    mixed_without_map = "temp_test_mixed_nomap.mp4"
    extracted_wav = "temp_test_extracted.wav"
    extracted_nomap_wav = "temp_test_extracted_nomap.wav"

    try:
        # 1. Gerar vídeo MP4 com tom de 440 Hz (simula o vídeo retornado pelo ComfyUI)
        print("1. Gerando vídeo MP4 de teste com áudio embutido de 440 Hz...")
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=24",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-c:v", "libx264", "-c:a", "aac",
            video_440
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Gerar arquivo de áudio com tom de 880 Hz (simula o áudio enviado pelo usuário)
        print("2. Gerando arquivo de áudio de teste com tom de 880 Hz...")
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "sine=frequency=880:duration=2",
            "-c:a", "pcm_s16le",
            audio_880
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. Executar o comando ANTIGO (sem -map) para demonstrar o problema
        print("\n3. Executando comando ANTIGO (sem -map)...")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_440, "-i", audio_880,
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            mixed_without_map
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 4. Executar o comando NOVO (com -map 0:v:0 -map 1:a:0)
        print("4. Executando comando NOVO (com -map 0:v:0 -map 1:a:0)...")
        subprocess.run([
            "ffmpeg", "-y", "-i", video_440, "-i", audio_880,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            mixed_with_map
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 5. Extrair áudio dos dois resultados para análise espectral via FFT
        subprocess.run([
            "ffmpeg", "-y", "-i", mixed_without_map, "-vn", "-c:a", "pcm_s16le", extracted_nomap_wav
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        subprocess.run([
            "ffmpeg", "-y", "-i", mixed_with_map, "-vn", "-c:a", "pcm_s16le", extracted_wav
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 6. Analisar frequências dominantes
        def get_dominant_freq(wav_path):
            sr, data = wavfile.read(wav_path)
            if data.ndim > 1:
                data = data[:, 0]
            # FFT
            fft_data = np.abs(np.fft.rfft(data))
            freqs = np.fft.rfftfreq(len(data), d=1.0/sr)
            # Ignorar DC (0 Hz)
            fft_data[0] = 0
            peak_idx = np.argmax(fft_data)
            peak_freq = freqs[peak_idx]
            
            # Amplitude em 440 Hz e 880 Hz
            idx_440 = np.argmin(np.abs(freqs - 440))
            idx_880 = np.argmin(np.abs(freqs - 880))
            amp_440 = fft_data[idx_440]
            amp_880 = fft_data[idx_880]
            return peak_freq, amp_440, amp_880

        freq_nomap, a440_nomap, a880_nomap = get_dominant_freq(extracted_nomap_wav)
        freq_map, a440_map, a880_map = get_dominant_freq(extracted_wav)

        print("\n" + "=" * 70)
        print("RESULTADOS DA ANÁLISE ESPECTRAL (FFT):")
        print("=" * 70)
        print(f"🔴 SEM -MAP (comportamento antigo com bug):")
        print(f"   Frequência dominante detectada: {freq_nomap:.1f} Hz")
        print(f"   Potência em 440 Hz (vídeo ref) : {a440_nomap:.2e}")
        print(f"   Potência em 880 Hz (áudio user): {a880_nomap:.2e}")
        print(f"   -> O áudio do usuário foi ignorado e o áudio de 440 Hz foi mantido!")

        print(f"\n🟢 COM -MAP 0:v:0 -MAP 1:a:0 (comportamento corrigido):")
        print(f"   Frequência dominante detectada: {freq_map:.1f} Hz")
        print(f"   Potência em 440 Hz (vídeo ref) : {a440_map:.2e}")
        print(f"   Potência em 880 Hz (áudio user): {a880_map:.2e}")
        print(f"   -> O áudio do usuário (880 Hz) substituiu perfeitamente o áudio de 440 Hz!")

        assert abs(freq_map - 880.0) < 5.0, f"Erro: frequência esperada 880 Hz, mas obteve {freq_map} Hz"
        assert a880_map > a440_map * 1000, "Erro: potência de 880 Hz deveria superar 440 Hz por ordens de grandeza"

        print("\n" + "=" * 70)
        print("✅ PROVA MATEMÁTICA CONCLUÍDA COM SUCESSO! O VÍDEO FINAL POSSUI 880 HZ.")
        print("=" * 70 + "\n")

    finally:
        # Limpeza de arquivos temporários do teste
        for f in [video_440, audio_880, mixed_with_map, mixed_without_map, extracted_wav, extracted_nomap_wav]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

if __name__ == "__main__":
    main()
