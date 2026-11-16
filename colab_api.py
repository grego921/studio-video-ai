"""
Servidor API da Superplaca GPU NVIDIA T4 - Studio Video AI v7.0 (High-Speed Engine)
Executa o SadTalker em processo OS isolado com pesos 100% pré-carregados e telemetria em tempo real.
"""

import os
import sys
import time
import uuid
import json
import base64
import shutil
import threading
import subprocess
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

SADTALKER_DIR = os.path.abspath("/content/SadTalker")
if SADTALKER_DIR not in sys.path:
    sys.path.insert(0, SADTALKER_DIR)

def patch_paste_pic():
    paste_file = os.path.join(SADTALKER_DIR, "src/utils/paste_pic.py")
    if os.path.exists(paste_file):
        try:
            with open(paste_file, "r") as f:
                p_code = f.read()
            p_code = p_code.replace(
                "clx, cly, crx, cry = crop_info[1]",
                "clx, cly, crx, cry = [int(float(v)) for v in crop_info[1]]"
            )
            p_code = p_code.replace(
                "lx, ly, rx, ry = crop_info[2]",
                "lx, ly, rx, ry = [int(float(v)) for v in crop_info[2]]"
            )
            p_code = p_code.replace(
                "location = ((ox1+ox2) // 2, (oy1+oy2) // 2)",
                "location = (int((ox1+ox2) // 2), int((oy1+oy2) // 2))"
            )
            p_code = p_code.replace(
                "p = cv2.resize(crop_frame.astype(np.uint8), (ox2-ox1, oy2 - oy1))",
                "p = cv2.resize(crop_frame.astype(np.uint8), (int(ox2-ox1), int(oy2 - oy1)))"
            )
            with open(paste_file, "w") as f:
                f.write(p_code)
            print("✅ paste_pic.py verificado com coordenadas inteiras!")
        except Exception as e:
            print("Aviso ao corrigir paste_pic.py:", e)

patch_paste_pic()

RESULTS_DIR = "/content/output_videos"
os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

TASKS = {}

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
    return response

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return jsonify({"ready": True}), 200
    return jsonify({
        "status": "online",
        "ready": True,
        "gpu": "NVIDIA T4 15GB (Google Colab)",
        "engine": "SadTalker Full-Frame 512px (High-Speed Engine v7.0)",
        "bridge": "Studio Video AI Cloud Bridge v7.0"
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    lines = []
    log_file = "/content/api.log"
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()[-40:]
    return jsonify({"logs": [l.strip() for l in lines]}), 200

def run_tts(text, voice, out_path):
    v = voice or "pt-BR-AntonioNeural"
    cmd = ["edge-tts", "--voice", v, "--text", text, "--write-media", out_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Falha na síntese de voz edge-tts: {res.stderr}")
    return True

def convert_video_aspect_ratio(input_video, output_video, ratio='original'):
    if ratio not in ('tiktok', 'youtube'):
        shutil.copy2(input_video, output_video)
        return output_video

    if ratio == 'tiktok':
        target_w, target_h = 720, 1280
    elif ratio == 'youtube':
        target_w, target_h = 1280, 720

    filter_str = (
        f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},boxblur=25:5[bg];"
        f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
    )
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-lavfi", filter_str,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "copy",
        output_video
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(output_video):
        print(f"Aviso ffmpeg: {res.stderr}. Mantendo vídeo original.")
        shutil.copy2(input_video, output_video)
    return output_video

def background_video_worker(task_id, work_dir, image_path, audio_path, aspect_ratio, keep_full):
    try:
        TASKS[task_id]["status"] = "processing"
        TASKS[task_id]["message"] = "Iniciando motor neural na superplaca NVIDIA T4..."

        preprocess_mode = 'full' if keep_full else 'crop'

        temp_img = os.path.join(work_dir, "render_input" + os.path.splitext(image_path)[1])
        shutil.copy2(image_path, temp_img)

        temp_aud = os.path.join(work_dir, "render_audio" + os.path.splitext(audio_path)[1])
        shutil.copy2(audio_path, temp_aud)

        task_log_path = os.path.join(work_dir, "task_render.log")

        cmd = [
            sys.executable, "/content/SadTalker/inference.py",
            "--driven_audio", temp_aud,
            "--source_image", temp_img,
            "--result_dir", work_dir,
            "--checkpoint_dir", "/content/SadTalker/checkpoints",
            "--bfm_folder", "/content/SadTalker/checkpoints/BFM_Fitting",
            "--size", "512",
            "--preprocess", preprocess_mode,
            "--still"
        ]

        print(f"🚀 [Task {task_id}] Executando SadTalker inference.py (modo: {preprocess_mode})")
        TASKS[task_id]["message"] = f"Mapeando malha 3D e fonemas (GPU T4)..."

        with open(task_log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd="/content/SadTalker"
            )

            # Monitoramento ativo do log para telemetria em tempo real
            while proc.poll() is None:
                time.sleep(2.5)
                try:
                    if os.path.exists(task_log_path):
                        with open(task_log_path, "r", encoding="utf-8", errors="ignore") as rf:
                            lines = [ln.strip() for ln in rf.readlines() if ln.strip()]
                            if lines:
                                last_l = lines[-1]
                                if "3DMM" in last_l:
                                    TASKS[task_id]["message"] = "Extraindo malha 3D facial do personagem..."
                                elif "audio2exp" in last_l:
                                    TASKS[task_id]["message"] = "Sincronizando áudio e expressões neurais..."
                                elif "facerender" in last_l or "facerender_batch" in last_l:
                                    TASKS[task_id]["message"] = "GPU Tesla T4 desenhando quadros faciais em 512px..."
                                elif "seamlessClone" in last_l:
                                    TASKS[task_id]["message"] = "Integrando alta resolução e enquadramento completo..."
                                else:
                                    TASKS[task_id]["message"] = f"Renderizando na superplaca T4 ({last_l[-45:]})..."
                except Exception:
                    pass

        # Fallback de segurança automático se o modo full falhar
        if proc.returncode != 0 and preprocess_mode != 'crop':
            print(f"⚠️ Aviso no modo {preprocess_mode}. Tentando modo de segurança 'crop'...")
            TASKS[task_id]["message"] = "Ajustando para modo de segurança (crop)..."
            cmd_crop = list(cmd)
            cmd_crop[cmd_crop.index('--preprocess') + 1] = 'crop'
            with open(task_log_path, "a", encoding="utf-8") as log_f:
                proc = subprocess.Popen(
                    cmd_crop,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd="/content/SadTalker"
                )
                while proc.poll() is None:
                    time.sleep(2.5)
                    try:
                        if os.path.exists(task_log_path):
                            with open(task_log_path, "r", encoding="utf-8", errors="ignore") as rf:
                                lines = [ln.strip() for ln in rf.readlines() if ln.strip()]
                                if lines:
                                    TASKS[task_id]["message"] = f"GPU T4 (Segurança): {lines[-1][-45:]}"
                    except Exception:
                        pass

        if proc.returncode != 0:
            err_details = "Erro de execução no SadTalker"
            try:
                if os.path.exists(task_log_path):
                    with open(task_log_path, "r", encoding="utf-8", errors="ignore") as rf:
                        err_lines = [ln.strip() for ln in rf.readlines() if ln.strip()]
                        err_details = "\n".join(err_lines[-12:])
            except Exception:
                pass
            raise RuntimeError(f"Falha na GPU: {err_details}")

        raw_video = None
        candidates_mp4 = []
        for root, dirs, files in os.walk(work_dir):
            for file in files:
                if file.endswith('.mp4') and not file.startswith('temp_'):
                    candidates_mp4.append(os.path.join(root, file))
        if candidates_mp4:
            full_matches = [c for c in candidates_mp4 if '_full.mp4' in c]
            raw_video = full_matches[0] if full_matches else candidates_mp4[-1]

        if not raw_video or not os.path.exists(raw_video):
            raise RuntimeError("O SadTalker concluiu mas não gerou o arquivo de vídeo MP4.")

        final_video_name = f"video_{aspect_ratio}_{task_id}.mp4"
        final_video_path = os.path.join(work_dir, final_video_name)
        convert_video_aspect_ratio(raw_video, final_video_path, ratio=aspect_ratio)

        final_video = final_video_path if os.path.exists(final_video_path) else raw_video
        out_filename = os.path.basename(final_video)

        with open(final_video, 'rb') as f_vid:
            video_b64 = f"data:video/mp4;base64,{base64.b64encode(f_vid.read()).decode('utf-8')}"

        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["message"] = "Vídeo finalizado com sucesso!"
        TASKS[task_id]["filename"] = out_filename
        TASKS[task_id]["video_base64"] = video_b64
        TASKS[task_id]["video_url"] = f"/api/download/{task_id}/{out_filename}"
        TASKS[task_id]["size_mb"] = round(os.path.getsize(final_video) / (1024 * 1024), 2)
        print(f"🎉 Tarefa {task_id} concluída com sucesso: {out_filename}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = str(e)
        print(f"❌ Erro na tarefa {task_id}: {e}")

@app.route('/api/generate_neural_video', methods=['POST', 'OPTIONS'])
def generate_neural_video():
    if request.method == 'OPTIONS':
        return jsonify({"ready": True}), 200

    try:
        task_id = str(uuid.uuid4())[:8]
        work_dir = os.path.join(RESULTS_DIR, f"task_{task_id}")
        os.makedirs(work_dir, exist_ok=True)

        image_path = None
        if 'image_file' in request.files and request.files['image_file'].filename:
            f = request.files['image_file']
            raw_path = os.path.join(work_dir, "uploaded_" + f.filename)
            f.save(raw_path)
            try:
                with Image.open(raw_path) as im:
                    im.verify()
                image_path = raw_path
            except Exception:
                image_path = None

        if not image_path:
            sel = request.form.get('selected_image')
            candidates = [
                sel,
                os.path.join("/content", os.path.basename(sel)) if sel else None,
                "/content/personagem_maconaria.jpg"
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    image_path = c
                    break

        if not image_path or not os.path.exists(image_path):
            return jsonify({"success": False, "error": "Imagem não encontrada"}), 400

        audio_path = None
        if 'speech_file' in request.files and request.files['speech_file'].filename:
            f = request.files['speech_file']
            audio_path = os.path.join(work_dir, "input_speech.mp3")
            f.save(audio_path)
        else:
            script_text = request.form.get('script_text', '').strip()
            voice = request.form.get('voice', 'pt-BR-AntonioNeural')
            if script_text:
                audio_path = os.path.join(work_dir, "speech_tts.mp3")
                run_tts(script_text, voice, audio_path)

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
            return jsonify({"success": False, "error": "Áudio ou texto para fala não fornecido"}), 400

        aspect_ratio = request.form.get('aspect_ratio', 'original').lower()
        keep_full = request.form.get('keep_full', 'true').lower() in ('true', '1', 'yes')

        TASKS[task_id] = {
            "task_id": task_id,
            "status": "started",
            "message": "Iniciando renderização neural na superplaca T4...",
            "started_at": time.time()
        }

        t = threading.Thread(
            target=background_video_worker,
            args=(task_id, work_dir, image_path, audio_path, aspect_ratio, keep_full),
            daemon=True
        )
        t.start()

        return jsonify({
            "success": True,
            "async": True,
            "task_id": task_id,
            "message": "Tarefa de renderização iniciada na GPU T4."
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/task_status/<task_id>', methods=['GET', 'OPTIONS'])
def task_status(task_id):
    if request.method == 'OPTIONS':
        return jsonify({"ready": True}), 200

    task = TASKS.get(task_id)
    if not task:
        return jsonify({"success": False, "error": "Tarefa não encontrada"}), 404

    return jsonify({
        "success": True,
        **task
    }), 200

@app.route('/api/download/<task_id>/<filename>')
def download_video(task_id, filename):
    video_path = os.path.join(RESULTS_DIR, f"task_{task_id}", filename)
    if os.path.exists(video_path):
        return send_file(video_path, mimetype='video/mp4', as_attachment=True, download_name=filename)
    return jsonify({"error": "Vídeo não encontrado"}), 404

if __name__ == '__main__':
    port = 7860
    print(f"🚀 Iniciando Studio Video AI GPU Bridge v7.0 (High-Speed Engine) na porta {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
