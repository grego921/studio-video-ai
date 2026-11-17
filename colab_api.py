"""
Servidor API da Superplaca GPU NVIDIA T4 - Studio Video AI v7.3 (Diagnóstico Avançado, Qualidade e Proteção Dinâmica)
Executa o SadTalker em processo OS isolado com telemetria tqdm, proteção contra timeout por ausência de progresso e seleção de qualidade.
"""

import os
import sys
import time
import uuid
import json
import re
import shutil
import threading
import subprocess
from flask import Flask, request, jsonify, send_file
from PIL import Image

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    from flask_cors import CORS
    HAS_CORS = True
except Exception:
    HAS_CORS = False

SADTALKER_DIR = os.environ.get("SADTALKER_DIR", os.path.abspath("/content/SadTalker"))
if SADTALKER_DIR not in sys.path:
    sys.path.insert(0, SADTALKER_DIR)

CHECKPOINTS_DIR = os.environ.get("CHECKPOINTS_DIR", os.path.join(SADTALKER_DIR, "checkpoints"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.abspath("/content/output_videos"))
NO_PROGRESS_TIMEOUT_S = int(os.environ.get("NO_PROGRESS_TIMEOUT_S", 600))  # 10 min sem modificação no log
MAX_TOTAL_TIMEOUT_S = int(os.environ.get("MAX_TOTAL_TIMEOUT_S", 3 * 3600))  # 3 horas de teto absoluto de segurança
WORKER_TIMEOUT_S = NO_PROGRESS_TIMEOUT_S  # Compatibilidade legada
TQDM_REGEX = re.compile(r'(\d+)/(\d+) \[(\d+:\d+)<(\d+:\d+)')
os.makedirs(RESULTS_DIR, exist_ok=True)

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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
if HAS_CORS:
    try:
        CORS(app)
    except Exception:
        pass

TASKS = {}

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With, Accept, Origin'
    return response

def get_log_tail(log_path, n=15):
    if not log_path or not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.rstrip("\r\n") for l in f.readlines() if l.strip()]
            return lines[-n:]
    except Exception:
        return []

def check_environment_health():
    cuda_ok = bool(torch.cuda.is_available()) if HAS_TORCH else False
    inference_ok = os.path.exists(os.path.join(SADTALKER_DIR, "inference.py"))
    ckpt_512_ok = os.path.exists(os.path.join(CHECKPOINTS_DIR, "SadTalker_V0.0.2_512.safetensors"))
    ckpt_mapping_ok = os.path.exists(os.path.join(CHECKPOINTS_DIR, "mapping_00109-model.pth.tar"))
    ckpt_bfm_ok = os.path.exists(os.path.join(CHECKPOINTS_DIR, "BFM_Fitting"))

    ready = bool(cuda_ok and inference_ok and ckpt_512_ok and ckpt_mapping_ok and ckpt_bfm_ok)
    return {
        "ready": ready,
        "cuda_available": cuda_ok,
        "inference_script": inference_ok,
        "checkpoint_512": ckpt_512_ok,
        "checkpoint_mapping": ckpt_mapping_ok,
        "checkpoint_bfm": ckpt_bfm_ok,
        "gpu": torch.cuda.get_device_name(0) if (HAS_TORCH and cuda_ok) else "None (CPU)"
    }

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return jsonify({"ready": True}), 200
    h = check_environment_health()
    return jsonify({
        "status": "online",
        "ready": h["ready"],
        "cuda_available": h["cuda_available"],
        "inference_script": h["inference_script"],
        "checkpoint_512": h["checkpoint_512"],
        "checkpoint_mapping": h["checkpoint_mapping"],
        "checkpoint_bfm": h["checkpoint_bfm"],
        "gpu": h["gpu"],
        "engine": "SadTalker Full-Frame 512px (High-Speed Engine v7.2)",
        "bridge": "Studio Video AI Cloud Bridge v7.2"
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
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(output_video):
            print(f"Aviso ffmpeg: {res.stderr}. Mantendo vídeo original.")
            shutil.copy2(input_video, output_video)
    except Exception as e_ff:
        print(f"Aviso ffmpeg exception: {e_ff}. Mantendo vídeo original.")
        shutil.copy2(input_video, output_video)
    return output_video

def background_video_worker(task_id, work_dir, image_path, audio_path, aspect_ratio, keep_full, quality='high'):
    try:
        TASKS[task_id]["status"] = "processing"
        TASKS[task_id]["stage"] = "init"
        TASKS[task_id]["message"] = "Preparando mídia e ambiente neural..."

        preprocess_mode = 'full' if keep_full else 'crop'
        render_size = "256" if str(quality).lower() == "fast" else "512"

        temp_img = os.path.join(work_dir, "render_input" + os.path.splitext(image_path)[1])
        shutil.copy2(image_path, temp_img)

        # 3. Redimensionar imagem de entrada para no máx. 1024 px no lado maior
        try:
            with Image.open(temp_img) as im:
                w, h = im.size
                max_side = max(w, h)
                if max_side > 1024:
                    scale = 1024.0 / max_side
                    new_w, new_h = int(round(w * scale)), int(round(h * scale))
                    im_resized = im.resize((new_w, new_h), Image.LANCZOS)
                    if im_resized.mode in ("RGBA", "P"):
                        im_resized = im_resized.convert("RGB")
                    im_resized.save(temp_img, quality=95)
                    print(f"🖼️ [Task {task_id}] Imagem redimensionada de {w}x{h} para {new_w}x{new_h} (max 1024px)")
        except Exception as e_resize:
            print(f"⚠️ Aviso ao redimensionar imagem: {e_resize}")

        temp_aud = os.path.join(work_dir, "render_audio" + os.path.splitext(audio_path)[1])
        shutil.copy2(audio_path, temp_aud)

        task_log_path = os.path.join(work_dir, "task_render.log")
        TASKS[task_id]["log_path"] = task_log_path

        cmd = [
            sys.executable, os.path.join(SADTALKER_DIR, "inference.py"),
            "--driven_audio", temp_aud,
            "--source_image", temp_img,
            "--result_dir", work_dir,
            "--checkpoint_dir", CHECKPOINTS_DIR,
            "--bfm_folder", os.path.join(CHECKPOINTS_DIR, "BFM_Fitting"),
            "--size", render_size,
            "--preprocess", preprocess_mode,
            "--still"
        ]

        print(f"🚀 [Task {task_id}] Executando SadTalker inference.py (modo: {preprocess_mode}, size: {render_size}, quality: {quality})")
        TASKS[task_id]["stage"] = "3dmm"
        TASKS[task_id]["message"] = "Mapeando malha 3D e pontos faciais do personagem..."

        sub_start_time = time.time()
        no_prog_limit = int(os.environ.get("NO_PROGRESS_TIMEOUT_S", NO_PROGRESS_TIMEOUT_S))
        max_total_limit = int(os.environ.get("MAX_TOTAL_TIMEOUT_S", MAX_TOTAL_TIMEOUT_S))
        last_mtime = None
        last_size = None
        last_progress_time = sub_start_time
        timed_out = False
        timeout_reason = ""

        with open(task_log_path, "w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=SADTALKER_DIR
            )
            TASKS[task_id]["proc_alive"] = True

            while proc.poll() is None:
                time.sleep(0.3)

                # Verifica se o log foi modificado
                if os.path.exists(task_log_path):
                    try:
                        curr_mtime = os.path.getmtime(task_log_path)
                        curr_size = os.path.getsize(task_log_path)
                        if last_mtime is None or curr_mtime != last_mtime or curr_size != last_size:
                            last_mtime = curr_mtime
                            last_size = curr_size
                            last_progress_time = time.time()
                    except Exception:
                        pass

                now = time.time()
                time_without_prog = now - last_progress_time
                total_elapsed = now - sub_start_time

                # (a) Timeout por ausência de progresso no log
                if time_without_prog > no_prog_limit and proc.poll() is None:
                    print(f"🛑 [Task {task_id}] Ausência de progresso no log por {int(time_without_prog)}s! Matando subprocesso...")
                    proc.kill()
                    proc.wait()
                    timed_out = True
                    timeout_reason = f"ausência de novos logs por {no_prog_limit}s"
                    break

                # (b) Teto absoluto de segurança
                if total_elapsed > max_total_limit and proc.poll() is None:
                    print(f"🛑 [Task {task_id}] Teto de segurança de {max_total_limit}s excedido! Matando subprocesso...")
                    proc.kill()
                    proc.wait()
                    timed_out = True
                    timeout_reason = f"teto absoluto de segurança de {max_total_limit}s excedido"
                    break

                try:
                    if os.path.exists(task_log_path):
                        with open(task_log_path, "r", encoding="utf-8", errors="ignore") as rf:
                            lines = [ln.strip() for ln in rf.readlines() if ln.strip()]
                            if lines:
                                last_l = lines[-1]
                                if "3DMM" in last_l:
                                    TASKS[task_id]["stage"] = "3dmm"
                                    TASKS[task_id]["message"] = "Extraindo malha 3D facial do personagem..."
                                elif "audio2exp" in last_l or "audio2pose" in last_l or "audio2ceoff" in last_l:
                                    TASKS[task_id]["stage"] = "audio2exp"
                                    TASKS[task_id]["message"] = "Sincronizando áudio e expressões neurais..."
                                elif "facerender" in last_l or "facerender_batch" in last_l or "make_animation" in last_l or "Face Renderer" in last_l:
                                    TASKS[task_id]["stage"] = "facerender"
                                    TASKS[task_id]["message"] = f"GPU Tesla T4 desenhando quadros faciais em {render_size}px..."
                                elif "seamlessClone" in last_l or "paste_pic" in last_l:
                                    TASKS[task_id]["stage"] = "paste"
                                    TASKS[task_id]["message"] = "Integrando alta resolução e enquadramento completo..."
                                else:
                                    TASKS[task_id]["message"] = f"Processando GPU T4 ({last_l[-45:]})..."

                                # Parseia a última linha tqdm do log
                                for l in reversed(lines[-15:]):
                                    m_tqdm = TQDM_REGEX.search(l)
                                    if m_tqdm:
                                        c_cur, c_tot, _, c_eta = m_tqdm.groups()
                                        TASKS[task_id]["step_current"] = int(c_cur)
                                        TASKS[task_id]["step_total"] = int(c_tot)
                                        TASKS[task_id]["eta"] = c_eta
                                        TASKS[task_id]["progress_pct"] = round((int(c_cur) / int(c_tot)) * 100, 1) if int(c_tot) > 0 else 0
                                        break
                except Exception:
                    pass

        TASKS[task_id]["proc_alive"] = False

        # Se estourou o timeout: status="error" com últimas linhas do log e NÃO rodar fallback "crop"
        if timed_out:
            TASKS[task_id]["finished_at"] = time.time()
            tail_lines = get_log_tail(task_log_path, 15)
            tail_str = "\n".join(tail_lines) if tail_lines else "Nenhum log gerado pelo processo."
            err_msg = f"Tempo limite excedido no SadTalker ({timeout_reason}).\nÚltimos logs:\n{tail_str}"
            TASKS[task_id]["status"] = "error"
            TASKS[task_id]["error"] = err_msg
            TASKS[task_id]["message"] = f"Timeout no SadTalker ({timeout_reason})"
            print(f"❌ [Task {task_id}] {err_msg}")
            return

        # Fallback de segurança para modo crop apenas se falhou SEM timeout
        if proc.returncode != 0 and preprocess_mode != 'crop':
            print(f"⚠️ Aviso no modo {preprocess_mode}. Tentando modo de segurança 'crop'...")
            TASKS[task_id]["stage"] = "facerender"
            TASKS[task_id]["message"] = "Ajustando para modo de segurança (crop)..."
            cmd_crop = list(cmd)
            cmd_crop[cmd_crop.index('--preprocess') + 1] = 'crop'
            with open(task_log_path, "a", encoding="utf-8") as log_f:
                proc = subprocess.Popen(
                    cmd_crop,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    cwd=SADTALKER_DIR
                )
                TASKS[task_id]["proc_alive"] = True
                sub_crop_start = time.time()
                last_crop_mtime = None
                last_crop_size = None
                last_crop_progress = sub_crop_start
                timed_out_crop = False
                crop_timeout_reason = ""

                while proc.poll() is None:
                    time.sleep(0.5)

                    if os.path.exists(task_log_path):
                        try:
                            curr_m = os.path.getmtime(task_log_path)
                            curr_s = os.path.getsize(task_log_path)
                            if last_crop_mtime is None or curr_m != last_crop_mtime or curr_s != last_crop_size:
                                last_crop_mtime = curr_m
                                last_crop_size = curr_s
                                last_crop_progress = time.time()
                        except Exception:
                            pass

                    now_c = time.time()
                    time_no_prog_crop = now_c - last_crop_progress
                    total_crop_elapsed = now_c - sub_crop_start

                    if time_no_prog_crop > no_prog_limit and proc.poll() is None:
                        print(f"🛑 [Task {task_id}] Ausência de progresso no crop por {int(time_no_prog_crop)}s! Matando...")
                        proc.kill()
                        proc.wait()
                        timed_out_crop = True
                        crop_timeout_reason = f"ausência de novos logs por {no_prog_limit}s"
                        break

                    if total_crop_elapsed > max_total_limit and proc.poll() is None:
                        print(f"🛑 [Task {task_id}] Teto de segurança no crop excedido ({max_total_limit}s)! Matando...")
                        proc.kill()
                        proc.wait()
                        timed_out_crop = True
                        crop_timeout_reason = f"teto absoluto de segurança de {max_total_limit}s excedido"
                        break

                    try:
                        if os.path.exists(task_log_path):
                            with open(task_log_path, "r", encoding="utf-8", errors="ignore") as rf:
                                lines = [ln.strip() for ln in rf.readlines() if ln.strip()]
                                if lines:
                                    TASKS[task_id]["message"] = f"GPU T4 (Segurança): {lines[-1][-45:]}"
                                    for l in reversed(lines[-15:]):
                                        m_tqdm = TQDM_REGEX.search(l)
                                        if m_tqdm:
                                            c_cur, c_tot, _, c_eta = m_tqdm.groups()
                                            TASKS[task_id]["step_current"] = int(c_cur)
                                            TASKS[task_id]["step_total"] = int(c_tot)
                                            TASKS[task_id]["eta"] = c_eta
                                            TASKS[task_id]["progress_pct"] = round((int(c_cur) / int(c_tot)) * 100, 1) if int(c_tot) > 0 else 0
                                            break
                    except Exception:
                        pass

                TASKS[task_id]["proc_alive"] = False

            if timed_out_crop:
                TASKS[task_id]["finished_at"] = time.time()
                tail_lines = get_log_tail(task_log_path, 15)
                tail_str = "\n".join(tail_lines) if tail_lines else ""
                err_msg = f"Tempo limite excedido no SadTalker modo crop ({crop_timeout_reason}).\nÚltimos logs:\n{tail_str}"
                TASKS[task_id]["status"] = "error"
                TASKS[task_id]["error"] = err_msg
                TASKS[task_id]["message"] = f"Timeout no SadTalker modo crop ({crop_timeout_reason})"
                print(f"❌ [Task {task_id}] {err_msg}")
                return

        if proc.returncode != 0:
            tail_lines = get_log_tail(task_log_path, 15)
            err_details = "\n".join(tail_lines) if tail_lines else "Erro de execução no SadTalker"
            raise RuntimeError(f"Falha na execução do SadTalker:\n{err_details}")

        TASKS[task_id]["stage"] = "encode"
        TASKS[task_id]["message"] = "Finalizando codificação e enquadramento do vídeo..."

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
        size_mb = round(os.path.getsize(final_video) / (1024 * 1024), 2)

        # 5. Preencher filename/video_url/size_mb ANTES de setar status="completed"
        TASKS[task_id]["finished_at"] = time.time()
        TASKS[task_id]["stage"] = "done"
        TASKS[task_id]["filename"] = out_filename
        TASKS[task_id]["video_url"] = f"/api/download/{task_id}/{out_filename}"
        TASKS[task_id]["size_mb"] = size_mb
        TASKS[task_id]["proc_alive"] = False
        TASKS[task_id]["message"] = "Vídeo finalizado com sucesso!"
        TASKS[task_id]["status"] = "completed"

        print(f"🎉 Tarefa {task_id} concluída com sucesso: {out_filename} ({size_mb} MB)")

    except Exception as e:
        import traceback
        traceback.print_exc()
        TASKS[task_id]["finished_at"] = time.time()
        TASKS[task_id]["proc_alive"] = False
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = str(e)
        TASKS[task_id]["message"] = f"Erro: {str(e).splitlines()[0]}"
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
                os.path.join(SADTALKER_DIR, "examples/driven_audio/bus_chinese.wav")
            ]
            for ca in candidates_audio:
                if os.path.exists(ca):
                    audio_path = ca
                    break

        if not audio_path or not os.path.exists(audio_path):
            return jsonify({"success": False, "error": "Áudio ou texto para fala não fornecido"}), 400

        aspect_ratio = request.form.get('aspect_ratio', 'original').lower()
        keep_full = request.form.get('keep_full', 'true').lower() in ('true', '1', 'yes')
        quality = request.form.get('quality', 'high').lower().strip()
        if quality not in ('fast', 'high'):
            quality = 'high'

        TASKS[task_id] = {
            "task_id": task_id,
            "status": "started",
            "stage": "init",
            "quality": quality,
            "message": "Iniciando renderização neural na superplaca T4...",
            "started_at": time.time(),
            "finished_at": None,
            "proc_alive": False,
            "progress_pct": 0,
            "step_current": None,
            "step_total": None,
            "eta": None,
            "log_path": os.path.join(work_dir, "task_render.log")
        }

        t = threading.Thread(
            target=background_video_worker,
            args=(task_id, work_dir, image_path, audio_path, aspect_ratio, keep_full, quality),
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

    now = time.time()
    started_at = task.get("started_at", now)
    finished_at = task.get("finished_at")
    elapsed_s = round((finished_at - started_at) if finished_at else (now - started_at), 1)

    log_tail = get_log_tail(task.get("log_path"), 15)

    progress_pct = task.get("progress_pct", 0)
    step_current = task.get("step_current")
    step_total = task.get("step_total")
    eta = task.get("eta")

    if step_current is None and log_tail:
        for l in reversed(log_tail):
            m_tqdm = TQDM_REGEX.search(l)
            if m_tqdm:
                c_cur, c_tot, _, c_eta = m_tqdm.groups()
                step_current = int(c_cur)
                step_total = int(c_tot)
                eta = c_eta
                progress_pct = round((step_current / step_total) * 100, 1) if step_total > 0 else 0
                break

    if task.get("status") == "completed":
        progress_pct = 100
        eta = "00:00"

    resp = {
        "success": True,
        "task_id": task_id,
        "status": task.get("status", "started"),
        "message": task.get("message", ""),
        "elapsed_s": elapsed_s,
        "stage": task.get("stage", "init"),
        "quality": task.get("quality", "high"),
        "progress_pct": progress_pct,
        "step_current": step_current,
        "step_total": step_total,
        "eta": eta,
        "log_tail": log_tail,
        "proc_alive": bool(task.get("proc_alive", False))
    }

    if task.get("error"):
        resp["error"] = task.get("error")

    if task.get("status") == "completed":
        resp["filename"] = task.get("filename")
        resp["video_url"] = task.get("video_url")
        resp["size_mb"] = task.get("size_mb")

    return jsonify(resp), 200

@app.route('/api/download/<task_id>/<filename>')
def download_video(task_id, filename):
    video_path = os.path.join(RESULTS_DIR, f"task_{task_id}", filename)
    if os.path.exists(video_path):
        as_attachment = request.args.get('download', '0') == '1'
        return send_file(video_path, mimetype='video/mp4', as_attachment=as_attachment, download_name=filename)
    return jsonify({"error": "Vídeo não encontrado"}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Iniciando Studio Video AI GPU Bridge v7.2 na porta {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
