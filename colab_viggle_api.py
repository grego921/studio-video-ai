"""
Servidor API do Motor Viggle-Animate (ComfyUI) - Studio Video AI v8.0
Executa animação corporal e transferência de movimento com pesos abertos MiniMax-H3 no Google Colab (GPU A100 / L4).
Mantém estritamente o contrato de API utilizado pelo frontend.
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
import urllib.request
import urllib.error
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

# Diretórios base (configuráveis via variáveis de ambiente)
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", os.path.abspath("/content/ComfyUI"))
COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.abspath("/content/output_videos"))
WORKFLOW_API_PATH = os.environ.get("WORKFLOW_API_PATH", os.path.abspath("workflows/viggle_api.json"))

MODELS_DIR = os.path.join(COMFYUI_DIR, "models")
DIFFUSION_MODEL_PATH = os.environ.get(
    "DIFFUSION_MODEL_PATH",
    os.path.join(MODELS_DIR, "diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors")
)
LORA_MODEL_PATH = os.environ.get(
    "LORA_MODEL_PATH",
    os.path.join(MODELS_DIR, "loras/viggle_animate_dmd_lora_r64.safetensors")
)
TEXT_COND_PATH = os.environ.get(
    "TEXT_COND_PATH",
    os.path.join(MODELS_DIR, "text_cond/fixed_embed_fwd_anyframe.safetensors")
)
VAE_PATH = os.environ.get(
    "VAE_PATH",
    os.path.join(MODELS_DIR, "vae/minimax_h3_video_vae_int8_convrot.safetensors")
)

NO_PROGRESS_TIMEOUT_S = int(os.environ.get("NO_PROGRESS_TIMEOUT_S", 600))  # 10 min sem modificação no log
MAX_TOTAL_TIMEOUT_S = int(os.environ.get("MAX_TOTAL_TIMEOUT_S", 3 * 3600))  # 3 horas de teto absoluto

os.makedirs(RESULTS_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024  # 250 MB para vídeos curtos
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

def append_task_log(log_path, message):
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} {message}\n")
    except Exception:
        pass

def validate_workflow_against_comfyui(workflow_path, comfy_url):
    problems = []
    if not os.path.exists(workflow_path):
        return False, [f"Arquivo de workflow não encontrado: {workflow_path}"]

    try:
        with open(workflow_path, "r", encoding="utf-8") as f:
            raw_wf = json.load(f)
    except Exception as e:
        return False, [f"Erro ao ler JSON do workflow: {str(e)}"]

    workflow = {k: v for k, v in raw_wf.items() if isinstance(v, dict) and "class_type" in v}
    if not workflow:
        return False, ["Nenhum nó válido com 'class_type' encontrado no workflow"]

    # 1. Exigir os 3 títulos obrigatórios: INPUT_IMAGE, INPUT_VIDEO e SEED
    required_titles = ["INPUT_IMAGE", "INPUT_VIDEO", "SEED"]
    found_titles = set()
    for nid, node in workflow.items():
        title = node.get("_meta", {}).get("title")
        if title:
            found_titles.add(title)

    for req_t in required_titles:
        if req_t not in found_titles:
            problems.append(f"Título obrigatório '{req_t}' não encontrado em nenhum nó do workflow (_meta.title)")

    # 2. Consultar GET /object_info no ComfyUI
    object_info = {}
    try:
        req = urllib.request.Request(f"{comfy_url}/object_info", headers={"User-Agent": "StudioVideoAI"})
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            object_info = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        problems.append(f"Não foi possível consultar /object_info no ComfyUI ({comfy_url}): {str(e)}")
        return False, problems

    # 3. Para cada nó do workflow: (a) class_type existe; (b) entradas required presentes
    for nid, node in workflow.items():
        ctype = node.get("class_type")
        if not ctype:
            problems.append(f"Nó [{nid}] não possui 'class_type'")
            continue

        if ctype not in object_info:
            problems.append(f"Nó [{nid}] ({ctype}): 'class_type' não existe no ComfyUI")
            continue

        node_info = object_info[ctype]
        req_inputs = node_info.get("input", {}).get("required", {})
        actual_inputs = node.get("inputs", {})

        for req_key in req_inputs.keys():
            if req_key not in actual_inputs:
                problems.append(f"Nó [{nid}] ({ctype}): entrada obrigatória '{req_key}' ausente no workflow")

    is_valid = (len(problems) == 0)
    return is_valid, problems

def check_environment_health():
    """
    ready=true só se:
    1. ComfyUI responder em /system_stats
    2. GPU tiver 22 GB ou mais de VRAM
    3. Existirem os 4 arquivos de modelo
    4. workflows/viggle_api.json existir e for válido contra o ComfyUI (/object_info)
    """
    comfy_ok = False
    try:
        req = urllib.request.Request(f"{COMFYUI_URL}/system_stats", headers={"User-Agent": "StudioVideoAI"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                comfy_ok = True
    except Exception:
        comfy_ok = False

    cuda_ok = bool(torch.cuda.is_available()) if HAS_TORCH else False
    vram_gb = 0.0
    vram_ok = False
    gpu_name = "None (CPU)"

    if HAS_TORCH and cuda_ok:
        try:
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(total_bytes / (1024 ** 3), 2)
            # Margem para 22 GB (GPUs de 24 GB L4 / 40 GB A100)
            vram_ok = bool(vram_gb >= 21.5)
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

    ckpt_diffusion_ok = os.path.exists(DIFFUSION_MODEL_PATH)
    ckpt_lora_ok = os.path.exists(LORA_MODEL_PATH)
    ckpt_text_cond_ok = os.path.exists(TEXT_COND_PATH)
    ckpt_vae_ok = os.path.exists(VAE_PATH)
    workflow_exists = os.path.exists(WORKFLOW_API_PATH)

    wf_valid, wf_problems = validate_workflow_against_comfyui(WORKFLOW_API_PATH, COMFYUI_URL)

    ready = bool(
        comfy_ok and
        cuda_ok and
        vram_ok and
        ckpt_diffusion_ok and
        ckpt_lora_ok and
        ckpt_text_cond_ok and
        ckpt_vae_ok and
        workflow_exists and
        wf_valid
    )

    return {
        "ready": ready,
        "comfy_online": comfy_ok,
        "cuda_available": cuda_ok,
        "vram_gb": vram_gb,
        "vram_ok": vram_ok,
        "checkpoint_diffusion": ckpt_diffusion_ok,
        "checkpoint_lora": ckpt_lora_ok,
        "checkpoint_text_cond": ckpt_text_cond_ok,
        "checkpoint_vae": ckpt_vae_ok,
        "workflow_exists": workflow_exists,
        "workflow_valid": wf_valid,
        "workflow_problems": wf_problems,
        "gpu": gpu_name,
        "engine": "Viggle-Animate H3 (ComfyUI A100 Engine v1.0)",
        "bridge": "Studio Video AI Viggle Bridge v8.0"
    }

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return jsonify({"ready": True}), 200
    h = check_environment_health()
    return jsonify({
        "status": "online",
        "ready": h["ready"],
        "comfy_online": h["comfy_online"],
        "cuda_available": h["cuda_available"],
        "vram_gb": h["vram_gb"],
        "vram_ok": h["vram_ok"],
        "checkpoint_diffusion": h["checkpoint_diffusion"],
        "checkpoint_lora": h["checkpoint_lora"],
        "checkpoint_text_cond": h["checkpoint_text_cond"],
        "checkpoint_vae": h["checkpoint_vae"],
        "workflow_exists": h["workflow_exists"],
        "workflow_valid": h["workflow_valid"],
        "workflow_problems": h["workflow_problems"],
        "gpu": h["gpu"],
        "engine": h["engine"],
        "bridge": h["bridge"]
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    lines = []
    log_file = "/content/viggle_api.log"
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

def trim_motion_video(input_path, output_path, max_frames=124):
    """
    Verifica e limita o vídeo de movimento a no máximo max_frames (124 quadros, ~5s a 24fps).
    Retorna (caminho_final, foi_cortado, total_frames).
    """
    total_frames = 0
    try:
        import cv2
        cap = cv2.VideoCapture(input_path)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
    except Exception:
        total_frames = 0

    if total_frames > max_frames:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-vframes", str(max_frames),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, True, total_frames

    return input_path, False, total_frames

def upload_file_to_comfyui(file_path, file_type="image", subfolder="viggle"):
    """
    Faz upload de arquivos de imagem ou vídeo para o ComfyUI (/upload/image).
    Se o ComfyUI estiver no mesmo filesystem (/content/ComfyUI/input), também faz cópia direta.
    """
    filename = os.path.basename(file_path)

    # Cópia direta caso filesystem local compartilhado
    comfy_input_dir = os.path.join(COMFYUI_DIR, "input")
    if os.path.exists(comfy_input_dir):
        target_dir = os.path.join(comfy_input_dir, subfolder) if subfolder else comfy_input_dir
        os.makedirs(target_dir, exist_ok=True)
        dest_copy = os.path.join(target_dir, filename)
        shutil.copy2(file_path, dest_copy)

    # Upload via HTTP multipart para garantir registro no servidor ComfyUI
    try:
        import urllib.request
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode("utf-8"))
        content_type = "video/mp4" if filename.lower().endswith(".mp4") else "image/png"
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(file_bytes)
        body.extend(b"\r\n")

        if subfolder:
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="subfolder"\r\n\r\n{subfolder}\r\n'.encode("utf-8"))

        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend('Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'.encode("utf-8"))
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        req = urllib.request.Request(
            f"{COMFYUI_URL}/upload/image",
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("name", filename), data.get("subfolder", subfolder)
    except Exception as e:
        print(f"Aviso no upload HTTP para ComfyUI ({e}); usando arquivo local.")
        return filename, subfolder

def find_node_by_title(workflow, title):
    for nid, node in workflow.items():
        if isinstance(node, dict):
            meta = node.get("_meta", {})
            if meta.get("title") == title:
                return nid, node
    return None, None

def background_viggle_worker(task_id, work_dir, image_path, motion_video_path, audio_path, aspect_ratio):
    task_log_path = os.path.join(work_dir, "task_render.log")
    TASKS[task_id]["log_path"] = task_log_path

    try:
        TASKS[task_id]["status"] = "processing"
        TASKS[task_id]["stage"] = "init"
        TASKS[task_id]["message"] = "Iniciando processamento Viggle-Animate..."
        append_task_log(task_log_path, f"Iniciando tarefa {task_id} no motor Viggle-Animate")

        # 1. Redimensionar foto de entrada para max 1024px no lado maior
        temp_img = os.path.join(work_dir, "input_character.png")
        try:
            with Image.open(image_path) as im:
                w, h = im.size
                max_side = max(w, h)
                if max_side > 1024:
                    scale = 1024.0 / max_side
                    new_w, new_h = int(round(w * scale)), int(round(h * scale))
                    im_resized = im.resize((new_w, new_h), Image.LANCZOS)
                    if im_resized.mode in ("RGBA", "P"):
                        im_resized = im_resized.convert("RGB")
                    im_resized.save(temp_img, "PNG")
                    append_task_log(task_log_path, f"Foto redimensionada de {w}x{h} para {new_w}x{new_h} (max 1024px)")
                else:
                    if im.mode in ("RGBA", "P"):
                        im = im.convert("RGB")
                    im.save(temp_img, "PNG")
        except Exception as e_im:
            append_task_log(task_log_path, f"Aviso ao processar imagem: {e_im}")
            shutil.copy2(image_path, temp_img)

        # 2. Limitar vídeo de movimento a no máximo 124 quadros (~5s)
        temp_motion = os.path.join(work_dir, "input_motion.mp4")
        final_motion_path, was_trimmed, orig_frames = trim_motion_video(motion_video_path, temp_motion, max_frames=124)
        if was_trimmed:
            append_task_log(task_log_path, f"⚠️ Vídeo de movimento cortado: {orig_frames} quadros reduzidos para 124 quadros (~5s).")
            TASKS[task_id]["message"] = "Vídeo de movimento ajustado para 124 quadros (~5s)."
        else:
            append_task_log(task_log_path, f"Vídeo de movimento validado com {orig_frames} quadros.")

        # 3. Enviar foto e vídeo para o ComfyUI (/upload/image)
        TASKS[task_id]["stage"] = "upload"
        TASKS[task_id]["message"] = "Enviando mídias para o ComfyUI..."
        img_name, img_sub = upload_file_to_comfyui(temp_img, "image", subfolder="viggle")
        vid_name, vid_sub = upload_file_to_comfyui(final_motion_path, "video", subfolder="viggle")

        img_comfy_ref = f"{img_sub}/{img_name}" if img_sub else img_name
        vid_comfy_ref = f"{vid_sub}/{vid_name}" if vid_sub else vid_name
        append_task_log(task_log_path, f"Mídias preparadas no ComfyUI: imagem='{img_comfy_ref}', vídeo='{vid_comfy_ref}'")

        # 4. Carregar workflow em formato API
        if not os.path.exists(WORKFLOW_API_PATH):
            raise FileNotFoundError(f"Arquivo de workflow não encontrado: {WORKFLOW_API_PATH}")

        with open(WORKFLOW_API_PATH, "r", encoding="utf-8") as wf_f:
            raw_wf = json.load(wf_f)

        # Remove campos de instruções que não são nós
        workflow = {k: v for k, v in raw_wf.items() if isinstance(v, dict) and "class_type" in v}

        # 5. Localizar nós pelos títulos e preencher entradas
        # INPUT_IMAGE (LoadImage)
        nid_img, node_img = find_node_by_title(workflow, "INPUT_IMAGE")
        if not node_img:
            # Fallback inteligente por class_type se o usuário esqueceu o título
            for nid, n in workflow.items():
                if n.get("class_type") == "LoadImage":
                    nid_img, node_img = nid, n
                    break
        if node_img:
            node_img["inputs"]["image"] = img_comfy_ref
            append_task_log(task_log_path, f"Nó INPUT_IMAGE [{nid_img}] configurado com '{img_comfy_ref}'")
        else:
            raise ValueError("Nó INPUT_IMAGE (LoadImage) não encontrado no workflow API!")

        # INPUT_VIDEO (VHS_LoadVideo)
        nid_vid, node_vid = find_node_by_title(workflow, "INPUT_VIDEO")
        if not node_vid:
            for nid, n in workflow.items():
                if n.get("class_type") in ("VHS_LoadVideo", "VHS_LoadVideoPath"):
                    nid_vid, node_vid = nid, n
                    break
        if node_vid:
            node_vid["inputs"]["video"] = vid_comfy_ref
            node_vid["inputs"]["frame_load_cap"] = 124
            append_task_log(task_log_path, f"Nó INPUT_VIDEO [{nid_vid}] configurado com '{vid_comfy_ref}' (cap: 124)")
        else:
            raise ValueError("Nó INPUT_VIDEO (VHS_LoadVideo) não encontrado no workflow API!")

        # SEED (RandomNoise)
        nid_seed, node_seed = find_node_by_title(workflow, "SEED")
        if not node_seed:
            for nid, n in workflow.items():
                if n.get("class_type") in ("RandomNoise", "KSampler", "KSamplerSelect"):
                    nid_seed, node_seed = nid, n
                    break
        new_seed = int(time.time() * 1000) % (2 ** 32 - 1)
        if node_seed:
            if "noise_seed" in node_seed.get("inputs", {}):
                node_seed["inputs"]["noise_seed"] = new_seed
            elif "seed" in node_seed.get("inputs", {}):
                node_seed["inputs"]["seed"] = new_seed
            append_task_log(task_log_path, f"Nó SEED [{nid_seed}] configurado com semente {new_seed}")

        # 6. Enfileirar prompt no ComfyUI (/prompt)
        client_id = f"viggle_studio_{task_id}"
        prompt_payload = json.dumps({"prompt": workflow, "client_id": client_id}).encode("utf-8")
        req_prompt = urllib.request.Request(
            f"{COMFYUI_URL}/prompt",
            data=prompt_payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req_prompt, timeout=10) as p_resp:
                prompt_res = json.loads(p_resp.read().decode("utf-8"))
        except urllib.error.HTTPError as he:
            err_body = ""
            err_json = {}
            try:
                err_body = he.read().decode("utf-8")
                err_json = json.loads(err_body)
            except Exception:
                pass

            err_main = err_json.get("error", {})
            if isinstance(err_main, dict):
                base_msg = err_main.get("message") or err_main.get("type") or str(he)
            else:
                base_msg = str(err_main) or str(he)

            node_errors = err_json.get("node_errors", {})
            node_err_details = []
            if isinstance(node_errors, dict):
                for nid, nerr in node_errors.items():
                    ctype = nerr.get("class_type", "Desconhecido")
                    for e in nerr.get("errors", []):
                        m = e.get("message", "Erro no nó")
                        d = e.get("details", "")
                        node_err_details.append(f"Nó {nid} ({ctype}): {m} [{d}]" if d else f"Nó {nid} ({ctype}): {m}")

            if node_err_details:
                full_msg = f"{base_msg} - Detalhes: {'; '.join(node_err_details)}"
            elif err_body:
                full_msg = f"{base_msg} - {err_body[:250]}"
            else:
                full_msg = f"Erro HTTP {he.code} ao enfileirar no ComfyUI: {he.reason}"

            append_task_log(task_log_path, f"❌ Erro ao enviar prompt: {full_msg}")
            raise RuntimeError(full_msg)

        prompt_id = prompt_res.get("prompt_id")
        if not prompt_id:
            err_msg = prompt_res.get("error", "ComfyUI não retornou prompt_id")
            raise RuntimeError(f"Erro ao enfileirar no ComfyUI: {err_msg}")

        TASKS[task_id]["prompt_id"] = prompt_id
        TASKS[task_id]["stage"] = "queued"
        TASKS[task_id]["message"] = "Tarefa aguardando execução na fila do ComfyUI..."
        TASKS[task_id]["proc_alive"] = True
        append_task_log(task_log_path, f"Prompt enfileirado no ComfyUI com ID: {prompt_id}")

        # 7. Acompanhamento via polling e WebSocket com proteção de timeout
        start_time = time.time()
        last_progress_time = start_time
        completed = False
        execution_error = None
        outputs = {}

        while not completed:
            time.sleep(1.0)
            now = time.time()

            # (a) Teto absoluto
            if (now - start_time) > MAX_TOTAL_TIMEOUT_S:
                raise TimeoutError(f"Teto absoluto de segurança de {MAX_TOTAL_TIMEOUT_S}s excedido.")

            # (b) Consulta histórico de execução
            try:
                hist_req = urllib.request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
                with urllib.request.urlopen(hist_req, timeout=3.0) as h_resp:
                    hist_data = json.loads(h_resp.read().decode("utf-8"))

                if prompt_id in hist_data:
                    item = hist_data[prompt_id]
                    status_info = item.get("status", {})
                    if status_info.get("completed", False):
                        completed = True
                        outputs = item.get("outputs", {})
                        append_task_log(task_log_path, "ComfyUI informou execução concluída no histórico!")
                        break

                    messages = status_info.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "execution_error":
                            execution_error = str(msg[1])
                            raise RuntimeError(f"Erro na execução dos nós ComfyUI: {execution_error}")
            except urllib.error.URLError:
                pass

            # (c) Consulta fila atual para telemetria de nó e progresso
            try:
                queue_req = urllib.request.Request(f"{COMFYUI_URL}/queue")
                with urllib.request.urlopen(queue_req, timeout=2.0) as q_resp:
                    q_data = json.loads(q_resp.read().decode("utf-8"))
                    running = q_data.get("queue_running", [])
                    if any(r and len(r) > 1 and r[1] == prompt_id for r in running):
                        last_progress_time = now

                # Telemetria estimada dos estágios
                elapsed = now - start_time
                if elapsed < 5:
                    TASKS[task_id]["stage"] = "queued"
                    TASKS[task_id]["message"] = "Carregando pesos e condicionamento 3DMM/H3..."
                    TASKS[task_id]["progress_pct"] = 15.0
                elif elapsed < 15:
                    TASKS[task_id]["stage"] = "sampling"
                    TASKS[task_id]["message"] = "Amostrando difusão neural Viggle (DMD 4-step)..."
                    TASKS[task_id]["progress_pct"] = min(75.0, 15.0 + (elapsed - 5) * 4.0)
                else:
                    TASKS[task_id]["stage"] = "decoding"
                    TASKS[task_id]["message"] = "Decodificando latentes com VAE de vídeo..."
                    TASKS[task_id]["progress_pct"] = min(92.0, 75.0 + (elapsed - 15) * 1.5)

            except Exception:
                pass

            # (d) Timeout por ausência de progresso
            if (now - last_progress_time) > NO_PROGRESS_TIMEOUT_S:
                # Tenta interromper ComfyUI
                try:
                    int_req = urllib.request.Request(f"{COMFYUI_URL}/interrupt", data=b"", headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(int_req, timeout=2.0)
                except Exception:
                    pass
                raise TimeoutError(f"Ausência de progresso no ComfyUI por mais de {NO_PROGRESS_TIMEOUT_S} segundos.")

        # 8. Extrair arquivo de vídeo gerado
        TASKS[task_id]["stage"] = "saving"
        TASKS[task_id]["progress_pct"] = 95.0
        TASKS[task_id]["message"] = "Localizando e finalizando vídeo de saída..."

        generated_video_filename = None
        generated_subfolder = ""

        for nid, nout in outputs.items():
            if "gifs" in nout and nout["gifs"]:
                first_item = nout["gifs"][0]
                generated_video_filename = first_item.get("filename")
                generated_subfolder = first_item.get("subfolder", "")
                break
            elif "videos" in nout and nout["videos"]:
                first_item = nout["videos"][0]
                generated_video_filename = first_item.get("filename")
                generated_subfolder = first_item.get("subfolder", "")
                break
            elif "images" in nout and nout["images"]:
                for im_item in nout["images"]:
                    fname = im_item.get("filename", "")
                    if fname.lower().endswith(".mp4"):
                        generated_video_filename = fname
                        generated_subfolder = im_item.get("subfolder", "")
                        break

        # Baixar do ComfyUI (/view) ou copiar do diretório output/
        task_out_mp4 = os.path.join(work_dir, "viggle_raw_output.mp4")
        if generated_video_filename:
            # Tenta cópia direta do filesystem local
            comfy_output_path = os.path.join(COMFYUI_DIR, "output", generated_subfolder, generated_video_filename)
            if os.path.exists(comfy_output_path):
                shutil.copy2(comfy_output_path, task_out_mp4)
                append_task_log(task_log_path, f"Vídeo copiado diretamente de: {comfy_output_path}")
            else:
                view_url = f"{COMFYUI_URL}/view?filename={generated_video_filename}&subfolder={generated_subfolder}&type=output"
                append_task_log(task_log_path, f"Baixando vídeo via rota ComfyUI /view: {view_url}")
                urllib.request.urlretrieve(view_url, task_out_mp4)

        if not os.path.exists(task_out_mp4) or os.path.getsize(task_out_mp4) == 0:
            # Fallback procurando arquivos mp4 gerados recentemente no output/
            comfy_out_dir = os.path.join(COMFYUI_DIR, "output")
            if os.path.exists(comfy_out_dir):
                all_mp4s = []
                for root, _, files in os.walk(comfy_out_dir):
                    for f in files:
                        if f.lower().endswith(".mp4"):
                            p = os.path.join(root, f)
                            all_mp4s.append((os.path.getmtime(p), p))
                if all_mp4s:
                    all_mp4s.sort(reverse=True)
                    shutil.copy2(all_mp4s[0][1], task_out_mp4)

        if not os.path.exists(task_out_mp4) or os.path.getsize(task_out_mp4) == 0:
            raise FileNotFoundError("O ComfyUI concluiu o workflow, mas nenhum arquivo MP4 foi gerado ou retornado.")

        # 9. Se o usuário enviou áudio ou texto, mixar áudio no vídeo final (sem lip-sync)
        final_video_path = os.path.join(work_dir, f"video_viggle_{task_id}.mp4")
        has_audio = bool(audio_path and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0)

        if has_audio:
            append_task_log(task_log_path, "🎵 Mixando áudio no vídeo final com ffmpeg (sem sincronia labial)...")
            cmd_audio = [
                "ffmpeg", "-y", "-i", task_out_mp4, "-i", audio_path,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                final_video_path
            ]
            res_aud = subprocess.run(cmd_audio, capture_output=True, text=True)
            if res_aud.returncode != 0 or not os.path.exists(final_video_path):
                append_task_log(task_log_path, f"Aviso ao mixar áudio: {res_aud.stderr}. Mantendo sem áudio.")
                shutil.copy2(task_out_mp4, final_video_path)
        else:
            shutil.copy2(task_out_mp4, final_video_path)

        # 10. Ajuste de Aspect Ratio (TikTok / YouTube) se solicitado
        if aspect_ratio in ('tiktok', 'youtube'):
            append_task_log(task_log_path, f"Ajustando formato para {aspect_ratio}...")
            ratio_path = os.path.join(work_dir, f"video_formatted_{task_id}.mp4")
            convert_video_aspect_ratio(final_video_path, ratio_path, ratio=aspect_ratio)
            if os.path.exists(ratio_path):
                final_video_path = ratio_path

        out_filename = os.path.basename(final_video_path)
        size_mb = round(os.path.getsize(final_video_path) / (1024 * 1024), 2)

        # Preenchimento ANTES de marcar como completed
        TASKS[task_id]["filename"] = out_filename
        TASKS[task_id]["video_url"] = f"/api/download/{task_id}/{out_filename}"
        TASKS[task_id]["size_mb"] = size_mb
        TASKS[task_id]["finished_at"] = time.time()
        TASKS[task_id]["proc_alive"] = False
        TASKS[task_id]["stage"] = "done"
        TASKS[task_id]["progress_pct"] = 100.0
        TASKS[task_id]["message"] = "Vídeo finalizado com sucesso pelo Viggle-Animate!"
        TASKS[task_id]["aviso_audio"] = has_audio
        TASKS[task_id]["status"] = "completed"

        append_task_log(task_log_path, f"🎉 Concluído com sucesso! Vídeo: {out_filename} ({size_mb} MB)")

    except Exception as e:
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = str(e)
        TASKS[task_id]["message"] = f"Falha na renderização Viggle: {str(e)}"
        TASKS[task_id]["finished_at"] = time.time()
        TASKS[task_id]["proc_alive"] = False
        append_task_log(task_log_path, f"❌ ERRO na tarefa {task_id}: {str(e)}")

@app.route('/api/generate_neural_video', methods=['POST'])
def generate_neural_video():
    try:
        task_id = uuid.uuid4().hex[:8]
        work_dir = os.path.join(RESULTS_DIR, f"task_{task_id}")
        os.makedirs(work_dir, exist_ok=True)

        # 1. Foto do personagem (image_file ou selected_image)
        image_path = None
        if 'image_file' in request.files and request.files['image_file'].filename:
            f = request.files['image_file']
            image_path = os.path.join(work_dir, "input_character" + os.path.splitext(f.filename)[1])
            f.save(image_path)
        else:
            selected_image = request.form.get('selected_image')
            candidates = [
                selected_image,
                os.path.join("/content/personagens", selected_image) if selected_image else None,
                os.path.join(os.getcwd(), "static/images", selected_image) if selected_image else None,
                "/content/personagem_maconaria.jpg"
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    image_path = c
                    break

        if not image_path or not os.path.exists(image_path):
            return jsonify({"success": False, "error": "Foto do personagem (image_file) não fornecida ou não encontrada."}), 400

        # 2. Vídeo de movimento (motion_video obrigatório)
        motion_video_path = None
        # Aceita 'motion_video' (principal) ou 'ref_video' (compatibilidade)
        motion_field = 'motion_video' if 'motion_video' in request.files else ('ref_video' if 'ref_video' in request.files else None)

        if motion_field and request.files[motion_field].filename:
            f_vid = request.files[motion_field]
            motion_video_path = os.path.join(work_dir, "input_motion" + os.path.splitext(f_vid.filename)[1])
            f_vid.save(motion_video_path)

        if not motion_video_path or not os.path.exists(motion_video_path):
            return jsonify({"success": False, "error": "Vídeo de movimento MP4 (motion_video) é obrigatório para o motor Viggle-Animate."}), 400

        # 3. Áudio opcional (speech_file ou script_text com TTS)
        audio_path = None
        if 'speech_file' in request.files and request.files['speech_file'].filename:
            f_aud = request.files['speech_file']
            audio_path = os.path.join(work_dir, "input_speech" + os.path.splitext(f_aud.filename)[1])
            f_aud.save(audio_path)
        else:
            script_text = request.form.get('script_text', '').strip()
            voice = request.form.get('voice', 'pt-BR-AntonioNeural')
            if script_text:
                audio_path = os.path.join(work_dir, "speech_tts.mp3")
                try:
                    run_tts(script_text, voice, audio_path)
                except Exception as e_tts:
                    print(f"Aviso TTS: {e_tts}")
                    audio_path = None

        aspect_ratio = request.form.get('aspect_ratio', 'original').lower()

        TASKS[task_id] = {
            "task_id": task_id,
            "status": "started",
            "stage": "init",
            "engine": "viggle",
            "message": "Iniciando animação corporal Viggle-Animate no ComfyUI...",
            "started_at": time.time(),
            "finished_at": None,
            "proc_alive": False,
            "progress_pct": 0,
            "step_current": None,
            "step_total": None,
            "eta": None,
            "aviso_audio": bool(audio_path),
            "log_path": os.path.join(work_dir, "task_render.log")
        }

        t = threading.Thread(
            target=background_viggle_worker,
            args=(task_id, work_dir, image_path, motion_video_path, audio_path, aspect_ratio),
            daemon=True
        )
        t.start()

        return jsonify({
            "success": True,
            "async": True,
            "task_id": task_id,
            "aviso_audio": bool(audio_path),
            "message": "Tarefa de animação corporal Viggle iniciada na GPU."
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
        "engine": "viggle",
        "progress_pct": progress_pct,
        "step_current": step_current,
        "step_total": step_total,
        "eta": eta,
        "log_tail": log_tail,
        "proc_alive": bool(task.get("proc_alive", False)),
        "aviso_audio": bool(task.get("aviso_audio", False))
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
    print(f"🚀 Iniciando Studio Video AI Viggle GPU Bridge v8.0 na porta {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
