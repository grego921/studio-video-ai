"""
Motor de Animação Facial e Sincronia Labial (Lip-Sync) em Alta Qualidade
Gera o vídeo do personagem maçônico falando com voz neural realista
"""

import os
import sys
import math
import cv2
import numpy as np
from scipy.io import wavfile

def main():
    print("=== INICIANDO GERAÇÃO DO VÍDEO COM SINCRONIA LABIAL REAL ===")
    
    # 1. Carrega a imagem do personagem
    img_path = "personagem_maconaria.jpg"
    if not os.path.exists(img_path):
        print(f"Erro: {img_path} não encontrado!")
        return
    
    base_img = cv2.imread(img_path)
    img_h, img_w, _ = base_img.shape
    print(f"Imagem carregada: {img_w}x{img_h}")
    
    # 2. Carrega e analisa o áudio
    wav_path = "fala.wav"
    if not os.path.exists(wav_path):
        print("Convertendo fala_maconica.mp3 para WAV...")
        os.system('ffmpeg -y -i fala_maconica.mp3 -ar 16000 -ac 1 fala.wav')
    
    sample_rate, audio_data = wavfile.read(wav_path)
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32768.0
    else:
        audio_float = audio_data.astype(np.float32)
        
    duration = len(audio_float) / sample_rate
    fps = 30
    total_frames = int(duration * fps)
    print(f"Áudio: {duration:.2f}s | Taxa: {sample_rate}Hz | Total de frames: {total_frames} @ {fps}fps")
    
    # 3. Calcula o envelope de energia acústica por frame (sincronia labial)
    samples_per_frame = sample_rate / fps
    raw_envelope = []
    
    for i in range(total_frames):
        start = int(i * samples_per_frame)
        end = int(min(len(audio_float), (i + 1) * samples_per_frame))
        chunk = audio_float[start:end]
        if len(chunk) > 0:
            rms = np.sqrt(np.mean(chunk**2))
        else:
            rms = 0.0
        raw_envelope.append(rms)
        
    raw_envelope = np.array(raw_envelope)
    
    # Normalização e suavização do envelope
    max_rms = np.percentile(raw_envelope, 95) if len(raw_envelope) > 0 else 0.1
    if max_rms <= 0: max_rms = 0.1
    norm_env = np.clip(raw_envelope / max_rms, 0.0, 1.0)
    
    # Suavização com média móvel para movimento biológico natural (sem trepidações)
    kernel_size = 3
    smooth_env = np.convolve(norm_env, np.ones(kernel_size)/kernel_size, mode='same')
    
    # Cria pasta temporária para os frames
    frames_dir = "temp_render_frames"
    os.makedirs(frames_dir, exist_ok=True)
    
    # Coordenadas anatômicas faciais medidas no personagem
    cx, cy = 510, 312      # Centro exato da fenda labial
    mw = 38                # Meia largura da boca
    mh_down = 38           # Influência vertical no queixo
    
    # Olhos (para piscar natural)
    eye_left_x, eye_y = 465, 243
    eye_right_x = 555
    eye_radius = 20
    
    # Malha de coordenadas da imagem base
    grid_y, grid_x = np.mgrid[0:img_h, 0:img_w].astype(np.float32)
    
    print("Renderizando quadros de animação facial...")
    
    # Canvas vertical 1080x1920 (padrão profissional vertical 9:16)
    out_w, out_h = 1080, 1920
    
    for f in range(total_frames):
        t = f / fps
        speech_val = float(smooth_env[f])
        
        # 1. Deformação labial e do queixo
        open_val = math.pow(speech_val, 0.8) # Curva de resposta natural
        
        # Deslocamento vertical do lábio inferior e queixo
        dy_lower = open_val * 9.5
        dist_lower = ((grid_x - cx) / (mw * 1.25))**2 + ((grid_y - cy) / (mh_down * 1.5))**2
        weight_lower = np.exp(-dist_lower) * np.clip((grid_y - (cy - 2)) / 8.0, 0, 1.0)
        
        # Deslocamento do lábio superior (sobe levemente)
        dy_upper = -open_val * 1.8
        dist_upper = ((grid_x - cx) / (mw * 1.2))**2 + ((grid_y - cy) / 16.0)**2
        weight_upper = np.exp(-dist_upper) * np.clip(((cy + 2) - grid_y) / 8.0, 0, 1.0)
        
        disp_y = dy_lower * weight_lower + dy_upper * weight_upper
        
        # 2. Piscar de olhos (a cada ~3.5 segundos: frames 50-56 e frames 170-176)
        blink_factor = 0.0
        for blink_center in [52, 172, 290]:
            diff = abs(f - blink_center)
            if diff <= 3:
                blink_factor = math.cos((diff / 3.0) * (math.pi / 2.0))
                break
                
        if blink_factor > 0:
            # Puxa a pálpebra superior para baixo sobre os olhos
            for ex in [eye_left_x, eye_right_x]:
                dist_eye = ((grid_x - ex) / 18.0)**2 + ((grid_y - eye_y) / 10.0)**2
                weight_eye = np.exp(-dist_eye) * np.clip(((eye_y + 4) - grid_y) / 10.0, 0, 1.0)
                disp_y += blink_factor * 8.0 * weight_eye
                
        # Remapeia a face
        map_y = grid_y - disp_y
        map_x = grid_x
        frame_img = cv2.remap(base_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 3. Cavidade bucal e dentes quando a boca se abre
        gap = open_val * 6.5
        if gap > 0.8:
            norm_x = (grid_x - cx) / (mw * 0.75)
            norm_y = (grid_y - cy) / (gap * 0.7)
            inside = np.exp(-(norm_x**4 + norm_y**2)) * min(1.0, open_val * 1.3)
            
            # Sombra interna da cavidade bucal (vermelho escuro natural)
            cavity_bgr = np.array([24, 20, 36], dtype=np.float32)
            # Dentes superiores
            teeth_mask = np.exp(-((grid_x - cx) / (mw * 0.6))**2 - ((grid_y - (cy - gap * 0.2)) / (gap * 0.4))**2)
            teeth_bgr = np.array([195, 192, 202], dtype=np.float32)
            
            mask_3d = inside[:, :, np.newaxis]
            teeth_3d = (teeth_mask * inside * 0.75)[:, :, np.newaxis]
            
            frame_float = frame_img.astype(np.float32)
            frame_float = frame_float * (1.0 - mask_3d) + cavity_bgr * mask_3d
            frame_float = frame_float * (1.0 - teeth_3d) + teeth_bgr * teeth_3d
            frame_img = np.clip(frame_float, 0, 255).astype(np.uint8)
            
        # 4. Movimento orgânico de respiração e micro-balanço da cabeça
        sway_x = int(round(1.2 * math.sin(t * 1.2)))
        sway_y = int(round(1.5 * math.sin(t * 2.0)))
        
        # 5. Composição no formato Vertical 9:16 (1080x1920)
        # Cria fundo elegante escurecido com o próprio templo
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        # Fundo desfocado com gradiente
        bg_blur = cv2.resize(frame_img, (out_w, out_h))
        bg_blur = cv2.GaussianBlur(bg_blur, (51, 51), 0)
        bg_blur = (bg_blur.astype(np.float32) * 0.25).astype(np.uint8)
        canvas = bg_blur
        
        # Micro-zoom cinematográfico suave (de 1.0 a 1.03)
        zoom = 1.0 + 0.03 * (f / total_frames)
        zoom_w = int(img_w * zoom)
        zoom_h = int(img_h * zoom)
        zoomed_img = cv2.resize(frame_img, (zoom_w, zoom_h), interpolation=cv2.INTER_LINEAR)
        
        # Posicionamento centralizado na parte superior do formato vertical (altura de retrato)
        pos_x = (out_w - zoom_w) // 2 + sway_x
        pos_y = 120 + (out_h - 120 - zoom_h) // 4 + sway_y
        
        # Recorta limites se necessário
        x1_dst = max(0, pos_x)
        y1_dst = max(0, pos_y)
        x2_dst = min(out_w, pos_x + zoom_w)
        y2_dst = min(out_h, pos_y + zoom_h)
        
        x1_src = max(0, -pos_x)
        y1_src = max(0, -pos_y)
        x2_src = x1_src + (x2_dst - x1_dst)
        y2_src = y1_src + (y2_dst - y1_dst)
        
        canvas[y1_dst:y2_dst, x1_dst:x2_dst] = zoomed_img[y1_src:y2_src, x1_src:x2_src]
        
        # Salva o frame
        frame_file = os.path.join(frames_dir, f"frame_{f:04d}.jpg")
        cv2.imwrite(frame_file, canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
        
        if (f + 1) % 50 == 0 or (f + 1) == total_frames:
            pct = int(((f + 1) / total_frames) * 100)
            print(f"Progresso de renderização: {pct}% ({f+1}/{total_frames} frames)")
            
    print("Renderização de frames concluída com sucesso!")
    
    # 6. Muxing de vídeo e áudio via FFmpeg
    output_video = "static/sample_video.mp4"
    output_backup = "static/video_lider_maconico.mp4"
    
    ffmpeg_cmd = (
        f'ffmpeg -y -framerate {fps} -i {frames_dir}/frame_%04d.jpg '
        f'-i fala_maconica.mp3 -c:v libx264 -preset fast -crf 18 '
        f'-pix_fmt yuv420p -c:a aac -b:a 192k -shortest "{output_video}"'
    )
    
    print("Codificando vídeo final em H.264/AAC...")
    ret = os.system(ffmpeg_cmd)
    
    if ret == 0:
        # Copia backup
        import shutil
        shutil.copy(output_video, output_backup)
        shutil.copy(output_video, "video_lider_maconico.mp4")
        print(f"VÍDEO GERADO COM SUCESSO: {output_video} ({os.path.getsize(output_video)} bytes)")
    else:
        print("Erro durante a codificação FFmpeg!")
        
    # Limpa frames temporários
    import shutil
    shutil.rmtree(frames_dir, ignore_errors=True)
    print("Limpeza concluída!")

if __name__ == "__main__":
    main()
