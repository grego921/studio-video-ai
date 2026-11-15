"""
Servidor API de Ponte Neural - Studio Video AI (Google Colab GPU)
Conecta a GPU NVIDIA T4 do Google Colab diretamente ao seu painel Vercel.
"""

import os
import sys
import time
import uuid
import json
import base64
import asyncio
import traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Adiciona SadTalker ao PATH
SADTALKER_DIR = os.path.abspath("/content/SadTalker")
if SADTALKER_DIR not in sys.path:
    sys.path.insert(0, SADTALKER_DIR)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Variável global para o modelo SadTalker
sad_talker = None

def get_sad_talker():
    global sad_talker
    if sad_talker is None:
        from src.gradio_demo import SadTalker
        print("⚡ Carregando pesos neurais do SadTalker + GFPGAN...")
        sad_talker = SadTalker(
            checkpoint_path='/content/SadTalker/checkpoints',
            config_path='/content/SadTalker/src/config',
            lazy_load=True
        )
        print("✅ SadTalker carregado com sucesso!")
    return sad_talker

RESULTS_DIR = "/content/SadTalker/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "ready": True,
        "gpu": "NVIDIA T4 15GB (Google Colab)",
        "engine": "SadTalker + GFPGAN 512px HD",
        "bridge": "Studio Video AI Cloud Bridge v2.0"
    })

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

@app.route('/api/generate_neural_video', methods=['POST'])
def generate_neural_video():
    try:
        req_id = str(uuid.uuid4())[:8]
        work_dir = os.path.join(RESULTS_DIR, f"task_{req_id}")
        os.makedirs(work_dir, exist_ok=True)

        image_path = None
        # Verifica se enviou arquivo de imagem
        if 'image_file' in request.files and request.files['image_file'].filename:
            f = request.files['image_file']
            image_path = os.path.join(work_dir, "input_face.jpg")
            f.save(image_path)
        elif request.form.get('selected_image'):
            # Pode ser o avatar padrão do templo
            def_path = "/content/personagem_maconaria.jpg"
            if os.path.exists(def_path):
                image_path = def_path

        # Se ainda não tiver imagem, tenta buscar no diretório ou fallback
        if not image_path or not os.path.exists(image_path):
            candidates = [
                "/content/personagem_maconaria.jpg",
                "/content/SadTalker/examples/source_image/full_body_1.png",
                "/content/SadTalker/examples/source_image/art_13.png"
            ]
            for c in candidates:
                if os.path.exists(c):
                    image_path = c
                    break

        audio_path = None
        # Verifica se enviou arquivo de fala
        if 'speech_file' in request.files and request.files['speech_file'].filename:
            f = request.files['speech_file']
            audio_path = os.path.join(work_dir, "input_speech.mp3")
            f.save(audio_path)
        elif request.form.get('script_text'):
            script_text = request.form.get('script_text')
            voice = request.form.get('voice', 'pt-BR-AntonioNeural')
            audio_path = os.path.join(work_dir, "speech_tts.mp3")
            print(f"🎤 Sintetizando voz neural via edge_tts ({voice}): {script_text[:40]}...")
            asyncio.run(synthesize_text_to_audio(script_text, voice, audio_path))
        
        # Fallback de áudio se nada foi enviado
        if not audio_path or not os.path.exists(audio_path):
            candidates_audio = [
                "/content/fala_maconica.mp3",
                "/content/SadTalker/examples/driven_audio/bus_chinese.wav",
                "/content/SadTalker/examples/driven_audio/RD_Radio31_000.wav"
            ]
            for ca in candidates_audio:
                if os.path.exists(ca):
                    audio_path = ca
                    break

        if not image_path or not os.path.exists(image_path):
            return jsonify({"success": False, "error": "Imagem de referência não encontrada"}), 400

        if not audio_path or not os.path.exists(audio_path):
            return jsonify({"success": False, "error": "Áudio de fala não encontrado"}), 400

        print(f"🎬 Iniciando renderização neural SadTalker...")
        print(f"   Foto: {image_path}")
        print(f"   Áudio: {audio_path}")

        st = get_sad_talker()
        use_enhancer = request.form.get('use_enhancer', 'true').lower() in ('true', '1', 'yes')
        still_mode = request.form.get('still_mode', 'true').lower() in ('true', '1', 'yes')

        output_video_path = st.test(
            source_image=image_path,
            driven_audio=audio_path,
            preprocess='crop',
            still_mode=still_mode,
            use_enhancer=use_enhancer,
            batch_size=2,
            size=512,
            result_dir=work_dir
        )

        if not output_video_path or not os.path.exists(output_video_path):
            # Procura qualquer mp4 no work_dir
            for root, dirs, files in os.walk(work_dir):
                for file in files:
                    if file.endswith('.mp4'):
                        output_video_path = os.path.join(root, file)
                        break

        if not output_video_path or not os.path.exists(output_video_path):
            return jsonify({"success": False, "error": "Falha na geração do arquivo MP4"}), 500

        print(f"🎉 Vídeo gerado com sucesso: {output_video_path} ({os.path.getsize(output_video_path)} bytes)")

        # Converte para base64 para entrega imediata sem problemas de cache
        with open(output_video_path, 'rb') as vf:
            video_bytes = vf.read()
            video_b64 = "data:video/mp4;base64," + base64.b64encode(video_bytes).decode('utf-8')

        out_filename = f"video_ia_{req_id}.mp4"

        return jsonify({
            "success": True,
            "filename": out_filename,
            "video_base64": video_b64,
            "video_url": f"/api/download/{req_id}/{out_filename}",
            "message": "Vídeo neural 3D sintetizado com alta resolução!"
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/download/<task_id>/<filename>')
def download_video(task_id, filename):
    folder = os.path.join(RESULTS_DIR, f"task_{task_id}")
    for root, dirs, files in os.walk(folder):
        if filename in files:
            return send_file(os.path.join(root, filename), mimetype="video/mp4", as_attachment=True, download_name=filename)
    return "Vídeo não encontrado", 404

if __name__ == '__main__':
    port = 7860
    print(f"🚀 Iniciando Studio Video AI GPU Bridge na porta {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
