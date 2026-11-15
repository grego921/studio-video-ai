"""
Servidor API de Ponte Neural - Studio Video AI (Google Colab GPU)
Conecta a GPU NVIDIA T4 do Google Colab diretamente ao seu painel Vercel / Localhost.
Suporta formatação TikTok (9:16), YouTube (16:9), e modo 'full' (sem corte de fundo nem zoom).
"""

import os
import sys
import time
import uuid
import json
import base64
import asyncio
import traceback
import shutil
from PIL import Image, ImageFilter
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Adiciona SadTalker ao PATH
SADTALKER_DIR = os.path.abspath("/content/SadTalker")
if SADTALKER_DIR not in sys.path:
    sys.path.insert(0, SADTALKER_DIR)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100 MB
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
    return response

RESULTS_DIR = "/content/SadTalker/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.route('/', methods=['GET', 'OPTIONS'])
@app.route('/health', methods=['GET', 'OPTIONS'])
@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return jsonify({"ready": True}), 200
    return jsonify({
        "status": "online",
        "ready": True,
        "gpu": "NVIDIA T4 15GB (Google Colab)",
        "engine": "SadTalker Full-Frame 512px (TikTok / YouTube)",
        "bridge": "Studio Video AI Cloud Bridge v3.0"
    })

def prepare_image_aspect_ratio(img_path, target_ratio='original'):
    """
    Ajusta a proporção da imagem para TikTok (9:16) ou YouTube (16:9)
    SEM dar zoom e SEM distorcer o personagem original, preenchendo as laterais
    com desfoque suave da própria imagem de fundo.
    """
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        if target_ratio == 'tiktok':
            target_w, target_h = 720, 1280
        elif target_ratio == 'youtube':
            target_w, target_h = 1280, 720
        else:
            return img_path

        # Cria fundo estendido com desfoque elegante
        scale = max(target_w / w, target_h / h)
        bg_w, bg_h = int(w * scale), int(h * scale)
        bg = img.resize((bg_w, bg_h), Image.LANCZOS).filter(ImageFilter.GaussianBlur(radius=25))
        left = (bg_w - target_w) // 2
        top = (bg_h - target_h) // 2
        bg_cropped = bg.crop((left, top, left + target_w, top + target_h))

        # Posiciona o personagem original centralizado sem nenhum corte
        fg_scale = min(target_w / w, target_h / h)
        fg_w, fg_h = int(w * fg_scale), int(h * fg_scale)
        fg_resized = img.resize((fg_w, fg_h), Image.LANCZOS)

        fg_x = (target_w - fg_w) // 2
        fg_y = (target_h - fg_h) // 2
        bg_cropped.paste(fg_resized, (fg_x, fg_y))

        out_path = img_path + f"_{target_ratio}.jpg"
        bg_cropped.save(out_path, quality=95)
        return out_path
    except Exception as e:
        print(f"Aviso ao ajustar proporção da imagem: {e}")
        return img_path

async def synthesize_text_to_audio(text, voice, out_path):
    try:
        import edge_tts
        v = voice or "pt-BR-AntonioNeural"
        communicate = edge_tts.Communicate(text, v)
        await communicate.save(out_path)
        return True
    except Exception as e:
        print(f"Erro no TTS edge_tts: {e}")
        return False

@app.route('/api/generate_neural_video', methods=['POST', 'OPTIONS'])
def generate_neural_video():
    if request.method == 'OPTIONS':
        return jsonify({"ready": True}), 200
    try:
        req_id = str(uuid.uuid4())[:8]
        work_dir = os.path.join(RESULTS_DIR, f"task_{req_id}")
        os.makedirs(work_dir, exist_ok=True)

        image_path = None
        # 1. Verifica se enviou arquivo real de imagem
        if 'image_file' in request.files and request.files['image_file'].filename:
            f = request.files['image_file']
            image_path = os.path.join(work_dir, "input_face.jpg")
            f.save(image_path)
            print(f"📸 Imagem recebida com sucesso: {f.filename} ({os.path.getsize(image_path)} bytes)")
        elif request.form.get('selected_image'):
            sel = request.form.get('selected_image')
            # Se for um nome conhecido ou caminho
            candidates = [
                sel,
                os.path.join("/content", os.path.basename(sel)),
                "/content/personagem_maconaria.jpg"
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    image_path = c
                    break

        if not image_path or not os.path.exists(image_path):
            candidates = [
                "/content/personagem_maconaria.jpg",
                "/content/SadTalker/examples/source_image/full_body_1.png"
            ]
            for c in candidates:
                if os.path.exists(c):
                    image_path = c
                    break

        if not image_path or not os.path.exists(image_path):
            return jsonify({"success": False, "error": "Imagem de referência não foi enviada"}), 400

        audio_path = None
        # 2. Verifica áudio ou síntese de texto
        if 'speech_file' in request.files and request.files['speech_file'].filename:
            f = request.files['speech_file']
            audio_path = os.path.join(work_dir, "input_speech.mp3")
            f.save(audio_path)
            print(f"🎤 Áudio recebido: {f.filename} ({os.path.getsize(audio_path)} bytes)")
        elif request.form.get('script_text'):
            script_text = request.form.get('script_text').strip()
            if script_text:
                voice = request.form.get('voice', 'pt-BR-AntonioNeural')
                audio_path = os.path.join(work_dir, "speech_tts.mp3")
                print(f"🎤 Sintetizando voz neural via edge_tts ({voice}): {script_text[:40]}...")
                asyncio.run(synthesize_text_to_audio(script_text, voice, audio_path))

        if not audio_path or not os.path.exists(audio_path):
            candidates_audio = [
                "/content/fala_maconica.mp3",
                "/content/SadTalker/examples/driven_audio/bus_chinese.wav"
            ]
            for ca in candidates_audio:
                if os.path.exists(ca):
                    audio_path = ca
                    break

        if not audio_path or not os.path.exists(audio_path):
            return jsonify({"success": False, "error": "Áudio ou texto para fala não foi fornecido"}), 400

        # 3. Configura formato de resolução (TikTok, YouTube ou Original)
        aspect_ratio = request.form.get('aspect_ratio', 'original').lower()
        if aspect_ratio in ('tiktok', 'youtube'):
            print(f"📐 Ajustando proporção para formato {aspect_ratio.upper()}...")
            image_path = prepare_image_aspect_ratio(image_path, aspect_ratio)

        # 4. Configura enquadramento: 'full' mantém 100% da foto original sem cortar o fundo nem dar zoom
        keep_full = request.form.get('keep_full', 'true').lower() in ('true', '1', 'yes')
        preprocess_mode = 'full' if keep_full else 'crop'

        print(f"🎬 Iniciando renderização neural na GPU:")
        print(f"   Foto: {image_path} (Modo: {preprocess_mode}, Formato: {aspect_ratio})")
        print(f"   Áudio: {audio_path}")

        # Copia para local de trabalho para evitar que o SadTalker mova o arquivo original
        temp_img = os.path.join(work_dir, "render_face" + os.path.splitext(image_path)[1])
        shutil.copy2(image_path, temp_img)
        temp_aud = os.path.join(work_dir, "render_audio" + os.path.splitext(audio_path)[1])
        shutil.copy2(audio_path, temp_aud)

        # Cria nova instância do SadTalker para evitar erros de atributos deletados
        from src.gradio_demo import SadTalker
        st = SadTalker(
            checkpoint_path='/content/SadTalker/checkpoints',
            config_path='/content/SadTalker/src/config',
            lazy_load=True
        )

        still_mode = request.form.get('still_mode', 'true').lower() in ('true', '1', 'yes')

        output_video_path = st.test(
            source_image=temp_img,
            driven_audio=temp_aud,
            preprocess=preprocess_mode,
            still_mode=still_mode,
            use_enhancer=False,
            batch_size=2,
            size=512,
            result_dir=work_dir
        )

        # Busca pelo vídeo gerado no diretório
        final_video = None
        if output_video_path and os.path.exists(output_video_path):
            final_video = output_video_path
        else:
            # Procura qualquer mp4 gerado, priorizando arquivos com '_full.mp4' se preprocess='full'
            candidates_mp4 = []
            for root, dirs, files in os.walk(work_dir):
                for file in files:
                    if file.endswith('.mp4'):
                        candidates_mp4.append(os.path.join(root, file))
            if candidates_mp4:
                # Se tem _full.mp4, prefere ele
                full_matches = [c for c in candidates_mp4 if '_full.mp4' in c]
                final_video = full_matches[0] if full_matches else candidates_mp4[-1]

        if not final_video or not os.path.exists(final_video):
            return jsonify({"success": False, "error": "Falha ao gerar o arquivo de vídeo MP4"}), 500

        print(f"🎉 Vídeo gerado com sucesso: {final_video} ({os.path.getsize(final_video)} bytes)")

        with open(final_video, 'rb') as vf:
            video_bytes = vf.read()
            video_b64 = "data:video/mp4;base64," + base64.b64encode(video_bytes).decode('utf-8')

        out_filename = f"video_{aspect_ratio}_{req_id}.mp4"

        return jsonify({
            "success": True,
            "filename": out_filename,
            "video_base64": video_b64,
            "video_url": f"/api/download/{req_id}/{out_filename}",
            "message": f"Vídeo neural gerado com sucesso em modo {aspect_ratio.upper()}!"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/download/<task_id>/<filename>')
def download_video(task_id, filename):
    folder = os.path.join(RESULTS_DIR, f"task_{task_id}")
    for root, dirs, files in os.walk(folder):
        if filename in files or any(f.endswith('.mp4') for f in files):
            for f in files:
                if f.endswith('.mp4'):
                    return send_file(os.path.join(root, f), mimetype="video/mp4", as_attachment=True, download_name=filename)
    return "Vídeo não encontrado", 404

if __name__ == '__main__':
    port = 7860
    print(f"🚀 Iniciando Studio Video AI GPU Bridge v3.0 na porta {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
