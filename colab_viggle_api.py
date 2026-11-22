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
import socket
import struct
import threading
import subprocess
import urllib.request
import urllib.error
import http.client
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
WORKFLOW_LONGO_API_PATH = os.environ.get("WORKFLOW_LONGO_API_PATH", os.path.abspath("workflows/viggle_longo_api.json"))

MODELS_DIR = os.path.join(COMFYUI_DIR, "models")
DRIVE_MODELS_DIR = os.environ.get("DRIVE_MODELS_DIR", os.path.abspath("/content/drive/MyDrive/StudioVideoAI/models"))
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

def verify_and_ensure_symlink(comfy_path: str, drive_rel_path: str) -> bool:
    """Verifica se o arquivo do modelo existe no caminho do ComfyUI (inclusive via link simbólico)
    ou no diretório persistente do Google Drive, restabelecendo o link simbólico caso necessário."""
    if os.path.exists(comfy_path) and os.path.getsize(comfy_path) > 1024:
        return True
    drive_path = os.path.join(DRIVE_MODELS_DIR, drive_rel_path)
    if os.path.exists(drive_path) and os.path.getsize(drive_path) > 1024:
        try:
            os.makedirs(os.path.dirname(comfy_path), exist_ok=True)
            if os.path.islink(comfy_path) or os.path.exists(comfy_path):
                try:
                    os.remove(comfy_path)
                except Exception:
                    pass
            os.symlink(drive_path, comfy_path)
            return True
        except Exception:
            return True
    return False

NO_PROGRESS_TIMEOUT_S = int(os.environ.get("NO_PROGRESS_TIMEOUT_S", 600))  # 10 min sem modificação no log
FINAL_STAGE_TIMEOUT_S = int(os.environ.get("FINAL_STAGE_TIMEOUT_S", 1800))  # 30 min para etapa final de decodificação
MAX_TOTAL_TIMEOUT_S = int(os.environ.get("MAX_TOTAL_TIMEOUT_S", 3 * 3600))  # 3 horas de teto absoluto
HISTORY_POLL_TIMEOUT_S = float(os.environ.get("HISTORY_POLL_TIMEOUT_S", 10.0))  # 10s de timeout na consulta a /history

COMFYUI_LOG_PATH = os.environ.get("COMFYUI_LOG_PATH", "/content/comfyui.log")
DRIVE_LOGS_DIR = os.environ.get("DRIVE_LOGS_DIR", "/content/drive/MyDrive/StudioVideoAI/logs")

HEALTH_CHECK_INTERVAL_S = float(os.environ.get("HEALTH_CHECK_INTERVAL_S", 10.0))
HEALTH_CHECK_TIMEOUT_S = float(os.environ.get("HEALTH_CHECK_TIMEOUT_S", 3.0))
HEALTH_CHECK_MAX_FAILURES = int(os.environ.get("HEALTH_CHECK_MAX_FAILURES", 3))
COMFYUI_CRASH_MESSAGE = "O ComfyUI caiu durante o processamento (provável falta de memória). Tente uma duração menor ou qualidade Rápido."

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

def dump_comfyui_log_on_timeout(task_log_path: str = None):
    """
    Antes de qualquer interrupt por timeout, lê as últimas 80 linhas de COMFYUI_LOG_PATH
    e anexa ao log da tarefa e a um arquivo datado em DRIVE_LOGS_DIR.
    """
    from datetime import datetime
    if not os.path.exists(COMFYUI_LOG_PATH):
        msg = f"Arquivo de log do ComfyUI ({COMFYUI_LOG_PATH}) não encontrado para dump de diagnóstico."
        if task_log_path:
            append_task_log(task_log_path, f"⚠️ {msg}")
        return

    try:
        with open(COMFYUI_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        last_lines = lines[-80:] if len(lines) >= 80 else lines
        log_content = "".join(last_lines).strip()

        if task_log_path:
            append_task_log(task_log_path, f"📋 --- DUMP DIAGNÓSTICO COMFYUI.LOG (ÚLTIMAS {len(last_lines)} LINHAS) ---")
            append_task_log(task_log_path, log_content)
            append_task_log(task_log_path, "📋 --- FIM DO DUMP COMFYUI.LOG ---")

        try:
            os.makedirs(DRIVE_LOGS_DIR, exist_ok=True)
            now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            dump_filename = f"comfyui_timeout_{now_str}.log"
            dump_path = os.path.join(DRIVE_LOGS_DIR, dump_filename)
            with open(dump_path, "w", encoding="utf-8") as df:
                df.write(f"Dump automático do comfyui.log gerado em {datetime.now().isoformat()}\n")
                df.write("Motivo: Timeout no processamento da tarefa\n")
                df.write("=" * 70 + "\n")
                df.write(log_content + "\n")
            if task_log_path:
                append_task_log(task_log_path, f"💾 Dump do log salvo no Google Drive: {dump_path}")
        except Exception as de:
            if task_log_path:
                append_task_log(task_log_path, f"⚠️ Erro ao salvar dump do log no Google Drive ({de})")
    except Exception as e:
        if task_log_path:
            append_task_log(task_log_path, f"⚠️ Falha ao ler log do ComfyUI para diagnóstico ({e})")

def read_dmesg_oom_lines(max_lines=30) -> list:
    """Tenta ler linhas de OOM do kernel via dmesg (se houver permissão). Ignora erros."""
    try:
        res = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=3)
        if res.returncode != 0:
            res = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=3)
        if res.returncode == 0 and res.stdout:
            keywords = ["out of memory", "oom-killer", "killed process", "invoked oom-killer", "oom_reaper"]
            lines = [l.strip() for l in res.stdout.splitlines() if any(k in l.lower() for k in keywords)]
            return lines[-max_lines:]
    except Exception:
        pass
    return []

def log_dmesg_oom_if_available(task_log_path: str = None):
    """Registra linhas de OOM do dmesg no log da tarefa se disponíveis."""
    oom_lines = read_dmesg_oom_lines()
    if oom_lines:
        msg_header = f"⚠️ Registros de OOM detectados no kernel (dmesg) ({len(oom_lines)} linhas):"
        if task_log_path:
            append_task_log(task_log_path, msg_header)
            for line in oom_lines:
                append_task_log(task_log_path, f"  {line}")
        print(f"{msg_header}\n" + "\n".join(oom_lines))

def is_link_value(val) -> bool:
    """Verifica se um valor de entrada é uma conexão/link para outro nó [node_id, slot]."""
    return isinstance(val, list) and len(val) == 2 and isinstance(val[1], int) and isinstance(val[0], (str, int))

def _extract_combo_items(raw_items) -> list:
    """Normaliza itens de uma lista de opções de combo (strings, números, ou dicts com 'key')."""
    res = []
    for item in raw_items:
        if isinstance(item, (str, int, float, bool)):
            res.append(item)
        elif isinstance(item, dict):
            if "key" in item:
                res.append(item["key"])
            elif "value" in item:
                res.append(item["value"])
    return res

def get_combo_options(param_spec):
    """
    Retorna a lista de opções permitidas se o parâmetro for do tipo combo (lista),
    ou None caso contrário.

    Trata os dois formatos de combo do /object_info do ComfyUI:
    1. Lista direta: [[opções], {metadados}] ou [[opções]]
    2. Formato COMBO nomeado: ["COMBO", {"options": [opções], ...}] ou ["COMBO", [opções]]
    """
    if not isinstance(param_spec, (list, tuple)) or len(param_spec) == 0:
        return None

    first = param_spec[0]

    # Formato 2A: Tipo nomeado ("COMBO", "COMFY_DYNAMICCOMBO_V3", etc.)
    if isinstance(first, str) and first.upper() in ("COMBO", "COMFY_DYNAMICCOMBO_V3"):
        if len(param_spec) > 1:
            if isinstance(param_spec[1], dict):
                opts = param_spec[1].get("options")
                if opts is None:
                    opts = param_spec[1].get("values")
                if isinstance(opts, (list, tuple)):
                    return _extract_combo_items(opts)
            elif isinstance(param_spec[1], (list, tuple)):
                return _extract_combo_items(param_spec[1])
        return None

    # Formato 1: param_spec[0] é diretamente uma lista de opções
    if isinstance(first, list):
        # Tratar caso em que ["COMBO"] foi aninhado em lista com dict
        if len(first) == 1 and isinstance(first[0], str) and first[0].upper() == "COMBO":
            if len(param_spec) > 1 and isinstance(param_spec[1], dict):
                opts = param_spec[1].get("options") or param_spec[1].get("values")
                if isinstance(opts, (list, tuple)):
                    return _extract_combo_items(opts)
        return _extract_combo_items(first)

    # Formato 2B: Qualquer especificação com dict contendo chave "options"
    if len(param_spec) > 1 and isinstance(param_spec[1], dict):
        opts = param_spec[1].get("options")
        if isinstance(opts, (list, tuple)):
            return _extract_combo_items(opts)

    return None

def is_runtime_upload_field(node: dict, ctype: str, param_name: str, param_spec) -> bool:
    """
    Verifica se um campo de combo representa um arquivo de upload que será
    sobrescrito em tempo de execução (ex: INPUT_IMAGE, INPUT_VIDEO, LoadImage, VHS_LoadVideo).
    Esses campos não devem ser validados estritamente contra as listas de arquivos
    presentes na pasta de input do ComfyUI durante a inicialização / verificação de integridade.
    """
    title = (node.get("_meta", {}).get("title") or "") if isinstance(node, dict) else ""

    # 1. Nós com títulos obrigatórios de entrada no workflow
    if title == "INPUT_IMAGE" and param_name == "image":
        return True
    if title == "INPUT_VIDEO" and param_name == "video":
        return True

    # 2. Classes conhecidas de nós de upload de imagem e vídeo
    if ctype in ("LoadImage", "LoadImageMask") and param_name == "image":
        return True
    if ctype in ("VHS_LoadVideo", "VHS_LoadVideoPath", "VHS_LoadImages", "LoadVideo") and param_name in ("video", "images", "file"):
        return True

    # 3. Metadados de upload na especificação do ComfyUI (ex: image_upload=True, video_upload=True)
    if isinstance(param_spec, (list, tuple)) and len(param_spec) > 1 and isinstance(param_spec[1], dict):
        meta = param_spec[1]
        if meta.get("image_upload") or meta.get("video_upload") or meta.get("upload"):
            return True

    return False

QUALITY_MAP = {
    "rapido": 0.4,
    "padrao": 0.7,
    "alta": 0.7  # Mantido apenas por compatibilidade, tratado como 'padrao' (0.7 MP)
}
DEFAULT_QUALITY = "padrao"
OOM_ERROR_MESSAGE = "Memória da GPU insuficiente nesta qualidade. Tente 'Padrão' ou 'Rápido'."
OOM_ERROR_MESSAGE_LONGO = "Memória da GPU insuficiente nesta qualidade ou duração. Tente reduzir a duração máxima ou usar qualidade 'Rápido'."

def parse_quality(val):
    """
    Normaliza a qualidade de renderização para uma chave válida ('rapido', 'padrao')
    e seu respectivo valor em megapixels (0.4, 0.7).
    'alta' é aceita por compatibilidade, mapeada para 'padrao' (0.7 MP).
    Padrão quando ausente ou inválido: 'padrao' (0.7 MP).
    """
    if not val or not isinstance(val, str):
        return DEFAULT_QUALITY, QUALITY_MAP[DEFAULT_QUALITY]
    q_norm = val.strip().lower()
    if q_norm == "alta":
        return "padrao", 0.7
    if q_norm in QUALITY_MAP:
        return q_norm, QUALITY_MAP[q_norm]
    return DEFAULT_QUALITY, QUALITY_MAP[DEFAULT_QUALITY]

def is_oom_error(err_data) -> bool:
    """
    Identifica se a mensagem ou exceção retornada é decorrente de falta de memória de GPU (OOM).
    Utiliza busca de palavra inteira com limites (\boom\b) para evitar falsos positivos
    com palavras como 'zoom' e 'room'.
    """
    if not err_data:
        return False
    txt = str(err_data).lower()

    # Busca de palavra inteira para 'oom'
    if re.search(r'\boom\b', txt):
        return True

    patterns = [
        "out of memory",
        "outofmemoryerror",
        "cuda out of memory",
        "cuda error: out of memory",
        "memória da gpu insuficiente",
        "not enough memory",
        "allocate memory",
        "allocation on device"
    ]
    return any(p in txt for p in patterns)

def get_video_resolution(video_path):
    """
    Obtém a resolução 'LxA' (ex: 720x1280) de um arquivo de vídeo gerado
    utilizando ffprobe, cv2 ou fallback ffmpeg.
    """
    if not video_path or not os.path.exists(video_path):
        return "Desconhecida"

    # 1. ffprobe
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
            video_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            d = res.stdout.strip().splitlines()[0].strip()
            if "x" in d:
                return d
    except Exception:
        pass

    # 2. cv2
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w > 0 and h > 0:
                return f"{w}x{h}"
    except Exception:
        pass

    # 3. ffmpeg -i
    try:
        cmd = ["ffmpeg", "-i", video_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        m = re.search(r'Stream #.*?: Video:.*? (\d{2,5})x(\d{2,5})', res.stderr)
        if m:
            return f"{m.group(1)}x{m.group(2)}"
    except Exception:
        pass

    return "Desconhecida"

def get_media_duration(file_path: str) -> float:
    """
    Obtém a duração em segundos de um arquivo de mídia (áudio ou vídeo)
    utilizando ffprobe, cv2 ou fallback ffmpeg.
    """
    if not file_path or not os.path.exists(file_path):
        return 0.0

    # 1. ffprobe format=duration
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            val = float(res.stdout.strip().splitlines()[0].strip())
            if val > 0:
                return val
    except Exception:
        pass

    # 2. cv2 para vídeo
    try:
        import cv2
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            fc = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap.release()
            if fps > 0 and fc > 0:
                return float(fc / fps)
    except Exception:
        pass

    # 3. ffmpeg -i parsing duration
    try:
        cmd = ["ffmpeg", "-i", file_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        m = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', res.stderr)
        if m:
            h, m_min, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
            return h * 3600 + m_min * 60 + s
    except Exception:
        pass

    return 0.0

def calculate_target_duration(audio_path, video_path, min_duration_raw=5.0, max_duration_raw=30.0):
    """
    Regra de duração mandatória:
    1. Parse de min_duration e max_duration:
       - Padrão: 5.0 e 30.0 segundos.
       - Valores inválidos (None, <= 0, não-numéricos) usam o padrão.
       - Se min > max, min = max.
    2. Determinação do alvo:
       - Alvo = duração do áudio do usuário, se enviado e válido (> 0s).
       - Senão, a duração do vídeo de referência.
       - Se ambos forem ausentes/inválidos, alvo cai em min_duration.
    3. Clamping do alvo:
       - Limitado estritamente ao intervalo [min_duration, max_duration].
    4. Conversão para 24 fps:
       - target_frames = int(round(alvo_clamped * 24)).
       - Mínimo de 1 quadro.
    Retorna: (alvo_clamped, target_frames, min_d, max_d, is_long_mode)
    onde is_long_mode = (target_frames > 124).
    """
    try:
        min_d = float(min_duration_raw)
        if min_d <= 0:
            min_d = 5.0
    except (TypeError, ValueError):
        min_d = 5.0

    try:
        max_d = float(max_duration_raw)
        if max_d <= 0:
            max_d = 30.0
    except (TypeError, ValueError):
        max_d = 30.0

    if min_d > max_d:
        min_d = max_d

    audio_dur = get_media_duration(audio_path) if audio_path else 0.0
    if audio_dur > 0.0:
        raw_target = audio_dur
    else:
        video_dur = get_media_duration(video_path) if video_path else 0.0
        raw_target = video_dur if video_dur > 0.0 else min_d

    clamped_target = max(min_d, min(raw_target, max_d))
    target_frames = max(1, int(round(clamped_target * 24)))
    is_long_mode = (target_frames > 124)

    return clamped_target, target_frames, min_d, max_d, is_long_mode

class ComfyUIProgressTracker:
    """
    Escuta eventos do ComfyUI via WebSocket (/ws?clientId=...) em background thread.
    Extrai informações de progresso de blocos ("bloco X de Y") e reseta o timestamp
    de atividade na virada de cada bloco.
    """
    def __init__(self, comfy_url: str, client_id: str, prompt_id: str, sampler_node_id: str = None):
        self.comfy_url = comfy_url.rstrip("/")
        self.client_id = client_id
        self.prompt_id = prompt_id
        self.sampler_node_id = str(sampler_node_id) if sampler_node_id is not None else None
        self.current_chunk = None
        self.total_chunks = None
        self.chunk_text = ""
        self.has_received_chunk_progress = False
        self.last_progress_time = time.time()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self):
        import urllib.parse
        try:
            parsed = urllib.parse.urlparse(self.comfy_url)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            path = f"{parsed.path or ''}/ws?clientId={self.client_id}"

            s = socket.create_connection((host, port), timeout=3.0)
            if parsed.scheme == "https":
                import ssl
                ctx = ssl.create_default_context()
                s = ctx.wrap_socket(s, server_hostname=host)

            s.settimeout(2.0)
            key = "dGhlIHNhbXBsZSBub25jZQ=="
            req = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Upgrade: websocket\r\n"
                f"Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                f"Sec-WebSocket-Version: 13\r\n\r\n"
            )
            s.sendall(req.encode("utf-8"))

            header_data = b""
            while b"\r\n\r\n" not in header_data:
                chunk = s.recv(1024)
                if not chunk:
                    return
                header_data += chunk

            if b"101 " not in header_data:
                return

            buf = header_data.split(b"\r\n\r\n", 1)[1]

            while not self._stop_event.is_set():
                try:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                except socket.timeout:
                    continue
                except Exception:
                    break

                while len(buf) >= 2:
                    b1, b2 = buf[0], buf[1]
                    opcode = b1 & 0x0F
                    payload_len = b2 & 0x7F
                    offset = 2
                    if payload_len == 126:
                        if len(buf) < 4:
                            break
                        payload_len = struct.unpack("!H", buf[2:4])[0]
                        offset = 4
                    elif payload_len == 127:
                        if len(buf) < 10:
                            break
                        payload_len = struct.unpack("!Q", buf[2:10])[0]
                        offset = 10

                    if len(buf) < offset + payload_len:
                        break

                    payload = buf[offset:offset + payload_len]
                    buf = buf[offset + payload_len:]

                    if opcode == 0x01:
                        try:
                            msg = json.loads(payload.decode("utf-8", errors="ignore"))
                            self._handle_ws_msg(msg)
                        except Exception:
                            pass
                    elif opcode == 0x08:
                        return
        except Exception:
            pass
        finally:
            if 's' in locals() and s:
                try:
                    s.close()
                except Exception:
                    pass

    def _handle_ws_msg(self, msg: dict):
        mtype = msg.get("type")
        data = msg.get("data", {})
        if not isinstance(data, dict):
            return

        msg_prompt_id = data.get("prompt_id")
        if msg_prompt_id and msg_prompt_id != self.prompt_id:
            return

        now = time.time()
        # Eventos do ComfyUI contam como atividade para o timeout de 600 s
        self.last_progress_time = now

        if mtype == "viggle.chunk_progress":
            text = str(data.get("text", ""))
            self.chunk_text = text
            m = re.search(r'Chunk\s+(\d+)\s+of\s+(\d+)', text, re.IGNORECASE)
            if m:
                c = int(m.group(1))
                tot = int(m.group(2))
                # Se o total de blocos for implausível (> 100), ignorar
                if tot <= 100:
                    self.current_chunk = c
                    self.total_chunks = tot
                    self.has_received_chunk_progress = True
        elif mtype == "progress":
            node_id = data.get("node")
            if node_id is None:
                node_id = data.get("display_node")
            node_str = str(node_id) if node_id is not None else None

            # Só aceitar eventos cujo nó seja o ViggleChunkedSampler (SEED)
            if not self.sampler_node_id or node_str != str(self.sampler_node_id):
                return

            # Assim que chegar qualquer evento "viggle.chunk_progress" válido, ele passa a ser a
            # ÚNICA fonte de current_chunk e total_chunks, e os eventos "progress" do sampler deixam de alterar o bloco.
            if self.has_received_chunk_progress:
                return

            v = data.get("value")
            m = data.get("max")
            if v is not None and m is not None:
                try:
                    c = int(v)
                    tot = int(m)
                    # Se o total de blocos for implausível (> 100), ignorar
                    if tot <= 100:
                        # Antes do primeiro "viggle.chunk_progress", ainda é aceitável usar "progress" do sampler,
                        # mas o bloco nunca pode recuar e o total nunca pode diminuir.
                        if (self.current_chunk is None or c >= self.current_chunk) and (self.total_chunks is None or tot >= self.total_chunks):
                            self.current_chunk = c
                            self.total_chunks = tot
                except (ValueError, TypeError):
                    pass
        elif mtype in ("executing", "executed", "status"):
            self.last_progress_time = now

def extract_tail_exception_message(err_data, max_chars=800) -> str:
    """
    Extrai o final da mensagem de exceção (os últimos max_chars caracteres),
    evitando que cabeçalhos extensos de ferramentas como ffmpeg ocultem a causa raiz.
    Retorna uma mensagem formatada com identificação do nó e o final da mensagem.
    """
    if isinstance(err_data, dict):
        node_id = err_data.get("node_id", "?")
        node_type = err_data.get("node_type", "Desconhecido")
        exc_type = err_data.get("exception_type", "Exception")
        raw_msg = str(err_data.get("exception_message") or "")

        if len(raw_msg) > max_chars:
            tail = raw_msg[-max_chars:]
            formatted_msg = f"...{tail}"
        else:
            formatted_msg = raw_msg

        return f"Nó [{node_id}] ({node_type} - {exc_type}): {formatted_msg}"

    raw_str = str(err_data)
    if len(raw_str) > max_chars:
        return f"...{raw_str[-max_chars:]}"
    return raw_str

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

    # 3. Para cada nó do workflow: (a) class_type existe; (b) entradas required presentes; (c) combos válidos
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
        opt_inputs = node_info.get("input", {}).get("optional", {})
        actual_inputs = node.get("inputs", {})

        for req_key in req_inputs.keys():
            if req_key not in actual_inputs:
                problems.append(f"Nó [{nid}] ({ctype}): entrada obrigatória '{req_key}' ausente no workflow")

        # Validar valores de campos do tipo lista (combo) contra as opções reais do object_info
        all_specs = {}
        all_specs.update(req_inputs)
        if isinstance(opt_inputs, dict):
            all_specs.update(opt_inputs)

        for param_name, param_spec in all_specs.items():
            # Pula campos sobrescritos em tempo de execução (upload de imagem/vídeo)
            if is_runtime_upload_field(node, ctype, param_name, param_spec):
                continue

            allowed_opts = get_combo_options(param_spec)
            if allowed_opts is not None and param_name in actual_inputs:
                val = actual_inputs[param_name]
                if not is_link_value(val):
                    if val not in allowed_opts and str(val) not in [str(o) for o in allowed_opts]:
                        problems.append(
                            f"Nó [{nid}] ({ctype}), campo '{param_name}': valor '{val}' "
                            f"não existe nas opções permitidas pelo ComfyUI. Opções válidas: {allowed_opts}"
                        )

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

    ckpt_diffusion_ok = verify_and_ensure_symlink(
        DIFFUSION_MODEL_PATH,
        "diffusion_models/minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors"
    )
    ckpt_lora_ok = verify_and_ensure_symlink(
        LORA_MODEL_PATH,
        "loras/viggle_animate_dmd_lora_r64.safetensors"
    )
    ckpt_text_cond_ok = verify_and_ensure_symlink(
        TEXT_COND_PATH,
        "text_cond/fixed_embed_fwd_anyframe.safetensors"
    )
    ckpt_vae_ok = verify_and_ensure_symlink(
        VAE_PATH,
        "vae/minimax_h3_video_vae_int8_convrot.safetensors"
    )
    workflow_exists = os.path.exists(WORKFLOW_API_PATH)

    wf_valid, wf_problems = validate_workflow_against_comfyui(WORKFLOW_API_PATH, COMFYUI_URL)

    longo_exists = os.path.exists(WORKFLOW_LONGO_API_PATH)
    longo_valid = False
    if not longo_exists:
        wf_problems.append(f"[Longo] arquivo não encontrado: {WORKFLOW_LONGO_API_PATH}")
    else:
        longo_valid, longo_probs = validate_workflow_against_comfyui(WORKFLOW_LONGO_API_PATH, COMFYUI_URL)
        if not longo_valid:
            wf_problems.extend([f"[Longo] {p}" for p in longo_probs])

    ready = bool(
        comfy_ok and
        cuda_ok and
        vram_ok and
        ckpt_diffusion_ok and
        ckpt_lora_ok and
        ckpt_text_cond_ok and
        ckpt_vae_ok and
        workflow_exists and
        longo_exists and
        wf_valid and
        longo_valid
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
        "workflow_exists": bool(workflow_exists and longo_exists),
        "workflow_valid": bool(wf_valid and longo_valid),
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
    Processa o vídeo de movimento para coincidir com max_frames (a 24 fps):
    1. Normaliza para 24 fps.
    2. Se tiver mais quadros que max_frames: corta para max_frames.
    3. Se tiver menos quadros que max_frames: repete em vai-e-volta (ping-pong)
       até atingir max_frames.
    Retorna (output_path, foi_ajustado, total_frames_originais).
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

    work_dir = os.path.dirname(os.path.abspath(output_path))
    norm_path = os.path.join(work_dir, f"temp_norm24_{uuid.uuid4().hex[:8]}.mp4")

    cmd_norm = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "fps=24",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        norm_path
    ]
    res_norm = subprocess.run(cmd_norm, capture_output=True, text=True)
    if res_norm.returncode != 0 or not os.path.exists(norm_path) or os.path.getsize(norm_path) == 0:
        return input_path, False, total_frames

    norm_frames = 0
    try:
        cmd_count = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", norm_path
        ]
        res_c = subprocess.run(cmd_count, capture_output=True, text=True, timeout=10)
        if res_c.returncode == 0 and res_c.stdout.strip():
            norm_frames = int(res_c.stdout.strip().splitlines()[0].strip())
    except Exception:
        pass

    if norm_frames <= 0:
        try:
            import cv2
            cap = cv2.VideoCapture(norm_path)
            if cap.isOpened():
                norm_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
        except Exception:
            pass

    if norm_frames <= 0:
        norm_frames = total_frames

    # Caso 1: Quadros suficientes ou em excesso -> Cortar para max_frames a 24fps
    if norm_frames >= max_frames:
        cmd_trim = [
            "ffmpeg", "-y", "-i", norm_path,
            "-vframes", str(max_frames),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", "24",
            output_path
        ]
        subprocess.run(cmd_trim, capture_output=True, text=True)

        if os.path.exists(norm_path):
            try:
                os.remove(norm_path)
            except Exception:
                pass

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path, "trimmed", norm_frames
        return input_path, False, total_frames

    # Caso 2: Menos quadros que o alvo -> Repetir em vai-e-volta (ping-pong)
    rev_path = os.path.join(work_dir, f"temp_rev24_{uuid.uuid4().hex[:8]}.mp4")
    cmd_rev = [
        "ffmpeg", "-y", "-i", norm_path,
        "-vf", "reverse",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        rev_path
    ]
    subprocess.run(cmd_rev, capture_output=True, text=True)

    concat_txt = os.path.join(work_dir, f"concat_{uuid.uuid4().hex[:8]}.txt")
    try:
        items = []
        forward = True
        accum = 0
        while accum < max_frames:
            fpath = norm_path if forward else rev_path
            fpath_esc = fpath.replace("\\", "/").replace("'", "'\\''")
            items.append(f"file '{fpath_esc}'")
            forward = not forward
            accum += norm_frames

        with open(concat_txt, "w", encoding="utf-8") as cf:
            cf.write("\n".join(items))

        cmd_concat = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_txt,
            "-vframes", str(max_frames),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", "24",
            output_path
        ]
        subprocess.run(cmd_concat, capture_output=True, text=True)
    finally:
        for p in (norm_path, rev_path, concat_txt):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path, "pingpong", norm_frames

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

def interrupt_comfyui():
    """Envia POST /interrupt ao ComfyUI para cancelar qualquer execução pendente ou em andamento."""
    try:
        req = urllib.request.Request(
            f"{COMFYUI_URL}/interrupt",
            data=b"",
            headers={"Content-Type": "application/json", "User-Agent": "StudioVideoAI"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            pass
    except Exception:
        pass

def clear_comfyui_queue():
    """Envia POST /queue com {"clear": true} ao ComfyUI para limpar a fila pendente."""
    try:
        body = json.dumps({"clear": True}).encode("utf-8")
        req = urllib.request.Request(
            f"{COMFYUI_URL}/queue",
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "StudioVideoAI"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            pass
    except Exception:
        pass

def init_comfyui_session():
    """Limpa fila e interrompe execuções residuais no ComfyUI na inicialização."""
    interrupt_comfyui()
    clear_comfyui_queue()

def background_viggle_worker(
    task_id, work_dir, image_path, motion_video_path, audio_path, aspect_ratio,
    quality="padrao", min_duration=5.0, max_duration=30.0
):
    was_alta = isinstance(quality, str) and quality.strip().lower() == "alta"
    quality_key, target_megapixels = parse_quality(quality)
    task_log_path = os.path.join(work_dir, "task_render.log")
    TASKS[task_id]["log_path"] = task_log_path
    TASKS[task_id]["quality"] = quality_key
    TASKS[task_id]["megapixels"] = target_megapixels

    # Regra de duração
    clamped_target, target_frames, min_d, max_d, is_long_mode = calculate_target_duration(
        audio_path, motion_video_path, min_duration, max_duration
    )
    workflow_mode = "longo" if is_long_mode else "curto"
    oom_msg = OOM_ERROR_MESSAGE_LONGO if is_long_mode else OOM_ERROR_MESSAGE
    TASKS[task_id]["workflow_mode"] = workflow_mode
    TASKS[task_id]["target_duration"] = clamped_target
    TASKS[task_id]["target_frames"] = target_frames

    try:
        TASKS[task_id]["status"] = "processing"
        TASKS[task_id]["stage"] = "init"
        TASKS[task_id]["message"] = "Iniciando processamento Viggle-Animate..."
        if was_alta:
            append_task_log(task_log_path, "Qualidade 'alta' desativada, usando 'padrao'.")
        append_task_log(
            task_log_path,
            f"Iniciando tarefa {task_id} no motor Viggle-Animate "
            f"(Qualidade: '{quality_key}', {target_megapixels} MP, Modo: '{workflow_mode}')"
        )
        append_task_log(
            task_log_path,
            f"⏱️ Regra de duração: alvo={clamped_target:.2f}s ({target_frames} quadros a 24fps) "
            f"[min={min_d}s, max={max_d}s]. Workflow: {workflow_mode.upper()}."
        )

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

        # 2. Processar vídeo de movimento para coincidir com a duração-alvo
        temp_motion = os.path.join(work_dir, "input_motion.mp4")
        final_motion_path, action, orig_frames = trim_motion_video(
            motion_video_path, temp_motion, max_frames=target_frames
        )
        if action == "trimmed":
            append_task_log(
                task_log_path,
                f"⚠️ Vídeo de movimento cortado: {orig_frames} quadros reduzidos para "
                f"{target_frames} quadros ({clamped_target:.1f}s a 24fps)."
            )
            TASKS[task_id]["message"] = f"Vídeo de movimento cortado para {target_frames} quadros ({clamped_target:.1f}s)."
        elif action == "pingpong":
            append_task_log(
                task_log_path,
                f"🔁 Vídeo de movimento repetido em vai-e-volta (ping-pong): {orig_frames} quadros expandidos "
                f"para {target_frames} quadros ({clamped_target:.1f}s a 24fps)."
            )
            TASKS[task_id]["message"] = f"Vídeo de movimento repetido em vai-e-volta até {target_frames} quadros."
        else:
            append_task_log(task_log_path, f"Vídeo de movimento validado com {orig_frames} quadros a 24fps.")

        # 3. Enviar foto e vídeo para o ComfyUI (/upload/image)
        TASKS[task_id]["stage"] = "upload"
        TASKS[task_id]["message"] = "Enviando mídias para o ComfyUI..."
        img_name, img_sub = upload_file_to_comfyui(temp_img, "image", subfolder="viggle")
        vid_name, vid_sub = upload_file_to_comfyui(final_motion_path, "video", subfolder="viggle")

        img_comfy_ref = f"{img_sub}/{img_name}" if img_sub else img_name
        vid_comfy_ref = f"{vid_sub}/{vid_name}" if vid_sub else vid_name
        append_task_log(task_log_path, f"Mídias preparadas no ComfyUI: imagem='{img_comfy_ref}', vídeo='{vid_comfy_ref}'")

        # 4. Carregar workflow correspondente ao modo
        wf_path = WORKFLOW_LONGO_API_PATH if is_long_mode else WORKFLOW_API_PATH
        if not os.path.exists(wf_path):
            raise FileNotFoundError(f"Arquivo de workflow não encontrado: {wf_path}")

        with open(wf_path, "r", encoding="utf-8") as wf_f:
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
            node_vid["inputs"]["frame_load_cap"] = target_frames
            node_vid["inputs"]["skip_first_frames"] = 0
            append_task_log(task_log_path, f"Nó INPUT_VIDEO [{nid_vid}] configurado com '{vid_comfy_ref}' (cap: {target_frames})")
        else:
            raise ValueError("Nó INPUT_VIDEO (VHS_LoadVideo) não encontrado no workflow API!")

        # SEED (RandomNoise ou ViggleChunkedSampler)
        nid_seed, node_seed = find_node_by_title(workflow, "SEED")
        if not node_seed:
            for nid, n in workflow.items():
                if n.get("class_type") in ("RandomNoise", "KSampler", "KSamplerSelect", "ViggleChunkedSampler"):
                    nid_seed, node_seed = nid, n
                    break
        new_seed = int(time.time() * 1000) % (2 ** 32 - 1)
        if node_seed:
            if "noise_seed" in node_seed.get("inputs", {}):
                node_seed["inputs"]["noise_seed"] = new_seed
            elif "seed" in node_seed.get("inputs", {}):
                node_seed["inputs"]["seed"] = new_seed
            append_task_log(task_log_path, f"Nó SEED [{nid_seed}] configurado com semente {new_seed}")

        # Configurações específicas de modo (chunk_frames no longo, length no curto)
        if is_long_mode:
            for nid, n in workflow.items():
                if isinstance(n, dict) and n.get("class_type") == "ViggleAnimateConditioningWindowed":
                    n.setdefault("inputs", {})["chunk_frames"] = 124
                    append_task_log(task_log_path, f"Nó ViggleAnimateConditioningWindowed [{nid}] configurado com chunk_frames=124")
        else:
            for nid, n in workflow.items():
                if isinstance(n, dict) and n.get("class_type") == "ViggleAnimateConditioning":
                    if "length" in n.get("inputs", {}):
                        n["inputs"]["length"] = min(target_frames, 124)
                        append_task_log(task_log_path, f"Nó ViggleAnimateConditioning [{nid}] configurado com length={min(target_frames, 124)}")

        # JWImageResizeToMegapixels (Qualidade / Megapixels)
        resize_node_id = None
        for nid, n in workflow.items():
            if isinstance(n, dict) and n.get("class_type") == "JWImageResizeToMegapixels":
                resize_node_id = nid
                n.setdefault("inputs", {})["megapixels"] = target_megapixels
                append_task_log(task_log_path, f"Nó JWImageResizeToMegapixels [{nid}] configurado com megapixels={target_megapixels} (qualidade: {quality_key})")
        if not resize_node_id:
            append_task_log(task_log_path, f"Aviso: Nó com class_type 'JWImageResizeToMegapixels' não encontrado no workflow para aplicar qualidade '{quality_key}' ({target_megapixels} MP)")

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

            if is_oom_error(full_msg):
                append_task_log(task_log_path, f"❌ Erro ao enviar prompt: {oom_msg}")
                raise RuntimeError(oom_msg)

            append_task_log(task_log_path, f"❌ Erro ao enviar prompt: {full_msg}")
            raise RuntimeError(full_msg)

        prompt_id = prompt_res.get("prompt_id")
        if not prompt_id:
            err_msg = prompt_res.get("error", "ComfyUI não retornou prompt_id")
            if is_oom_error(err_msg):
                raise RuntimeError(oom_msg)
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
        min_vram_free_gb = None
        last_vram_poll_time = 0.0

        # Checagem inicial de VRAM via /system_stats
        try:
            init_sreq = urllib.request.Request(f"{COMFYUI_URL}/system_stats", headers={"User-Agent": "StudioVideoAI"})
            with urllib.request.urlopen(init_sreq, timeout=5.0) as is_resp:
                is_data = json.loads(is_resp.read().decode("utf-8"))
                for dev in is_data.get("devices", []):
                    vf = dev.get("vram_free") if dev.get("vram_free") is not None else dev.get("torch_vram_free")
                    if vf is not None:
                        free_gb = round(float(vf) / (1024 ** 3), 2)
                        if min_vram_free_gb is None or free_gb < min_vram_free_gb:
                            min_vram_free_gb = free_gb
        except (TimeoutError, socket.timeout, OSError, urllib.error.URLError, json.JSONDecodeError, http.client.HTTPException):
            pass

        # Identificar ID do nó sampler pelo título SEED no workflow carregado
        sampler_nid, _ = find_node_by_title(workflow, "SEED")
        if not sampler_nid:
            for nid, n in workflow.items():
                if isinstance(n, dict) and n.get("class_type") == "ViggleChunkedSampler":
                    sampler_nid = nid
                    break

        tracker = ComfyUIProgressTracker(COMFYUI_URL, client_id, prompt_id, sampler_node_id=sampler_nid)
        tracker.start()
        last_logged_chunk = None
        is_finalizing = False
        finalizing_start_time = None
        last_finalizing_log_time = 0.0
        last_health_check_time = 0.0
        consecutive_health_failures = 0

        try:
            while not completed:
                time.sleep(1.0)
                now = time.time()

                # (a) Teto absoluto
                if (now - start_time) > MAX_TOTAL_TIMEOUT_S:
                    dump_comfyui_log_on_timeout(task_log_path)
                    interrupt_comfyui()
                    raise TimeoutError(f"Teto absoluto de segurança de {MAX_TOTAL_TIMEOUT_S}s excedido.")

                # Verificação de cancelamento da tarefa
                if TASKS.get(task_id, {}).get("status") == "cancelled" or TASKS.get(task_id, {}).get("cancelled"):
                    interrupt_comfyui()
                    append_task_log(task_log_path, "Tarefa cancelada. Interrompendo execução no ComfyUI.")
                    raise RuntimeError("Tarefa cancelada.")

                # Sincronização de atividade e bloco via WebSocket
                if tracker.last_progress_time > last_progress_time:
                    last_progress_time = tracker.last_progress_time

                # Detecção do estágio "finalizando" (último bloco iniciado: current_chunk == total_chunks)
                if tracker.current_chunk is not None and tracker.total_chunks is not None:
                    if tracker.current_chunk >= tracker.total_chunks:
                        if not is_finalizing:
                            is_finalizing = True
                            finalizing_start_time = now
                            last_finalizing_log_time = now
                            last_progress_time = now
                            append_task_log(task_log_path, "Entrando na etapa final: decodificando latentes e montando vídeo...")

                if is_finalizing:
                    TASKS[task_id]["stage"] = "finalizing"
                    TASKS[task_id]["message"] = "Finalizando: decodificando e montando o vídeo..."
                    TASKS[task_id]["progress_pct"] = 94.0
                    if (now - last_finalizing_log_time) >= 60.0:
                        elapsed_mins = max(1, int((now - finalizing_start_time) / 60.0))
                        last_finalizing_log_time = now
                        append_task_log(task_log_path, f"Finalizando há {elapsed_mins} min")
                elif tracker.current_chunk is not None:
                    curr_c = tracker.current_chunk
                    tot_c = tracker.total_chunks or "?"
                    TASKS[task_id]["stage"] = "sampling"
                    TASKS[task_id]["message"] = f"Renderizando bloco {curr_c} de {tot_c}..."
                    TASKS[task_id]["step_current"] = curr_c
                    TASKS[task_id]["step_total"] = tracker.total_chunks
                    if curr_c != last_logged_chunk:
                        last_logged_chunk = curr_c
                        last_progress_time = now
                        append_task_log(task_log_path, f"Progresso: renderizando bloco {curr_c} de {tot_c}...")

                # (b) Consulta histórico de execução
                try:
                    hist_req = urllib.request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
                    with urllib.request.urlopen(hist_req, timeout=HISTORY_POLL_TIMEOUT_S) as h_resp:
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
                                tail_err = extract_tail_exception_message(msg[1], max_chars=800)
                                if is_oom_error(tail_err) or is_oom_error(str(msg[1])):
                                    append_task_log(task_log_path, f"❌ Erro na execução dos nós ComfyUI: {oom_msg}")
                                    raise RuntimeError(oom_msg)
                                append_task_log(task_log_path, f"❌ Erro na execução dos nós ComfyUI: {tail_err}")
                                raise RuntimeError(f"Erro na execução dos nós ComfyUI: {tail_err}")

                        if status_info.get("status_str") == "error":
                            for msg in messages:
                                if isinstance(msg, list) and len(msg) >= 2 and msg[0] == "execution_error":
                                    tail_err = extract_tail_exception_message(msg[1], max_chars=800)
                                    if is_oom_error(tail_err) or is_oom_error(str(msg[1])):
                                        append_task_log(task_log_path, f"❌ Erro na execução dos nós ComfyUI: {oom_msg}")
                                        raise RuntimeError(oom_msg)
                                    append_task_log(task_log_path, f"❌ Erro na execução dos nós ComfyUI: {tail_err}")
                                    raise RuntimeError(f"Erro na execução dos nós ComfyUI: {tail_err}")
                            raise RuntimeError(f"ComfyUI reportou erro na execução do prompt {prompt_id}")
                except (TimeoutError, socket.timeout, OSError, urllib.error.URLError, json.JSONDecodeError, http.client.HTTPException):
                    pass

                # (c) Verificação de saúde periódica a cada 10 s (GET /system_stats com timeout curto)
                if (now - last_health_check_time) >= HEALTH_CHECK_INTERVAL_S:
                    last_health_check_time = now
                    stats_ok = False
                    try:
                        sreq = urllib.request.Request(f"{COMFYUI_URL}/system_stats", headers={"User-Agent": "StudioVideoAI"})
                        with urllib.request.urlopen(sreq, timeout=HEALTH_CHECK_TIMEOUT_S) as s_resp:
                            if s_resp.status == 200:
                                s_data = json.loads(s_resp.read().decode("utf-8"))
                                stats_ok = True
                                consecutive_health_failures = 0
                                for dev in s_data.get("devices", []):
                                    vf = dev.get("vram_free") if dev.get("vram_free") is not None else dev.get("torch_vram_free")
                                    if vf is not None:
                                        free_gb = round(float(vf) / (1024 ** 3), 2)
                                        if min_vram_free_gb is None or free_gb < min_vram_free_gb:
                                            min_vram_free_gb = free_gb
                    except Exception:
                        stats_ok = False

                    if not stats_ok:
                        consecutive_health_failures += 1
                        append_task_log(task_log_path, f"⚠️ Verificação de resposta do ComfyUI falhou ({consecutive_health_failures}/{HEALTH_CHECK_MAX_FAILURES}).")
                        if consecutive_health_failures >= HEALTH_CHECK_MAX_FAILURES:
                            dump_comfyui_log_on_timeout(task_log_path)
                            log_dmesg_oom_if_available(task_log_path)
                            interrupt_comfyui()
                            append_task_log(task_log_path, f"❌ {COMFYUI_CRASH_MESSAGE}")
                            raise RuntimeError(COMFYUI_CRASH_MESSAGE)

                # (d) Consulta fila atual para telemetria de nó e progresso
                try:
                    queue_req = urllib.request.Request(f"{COMFYUI_URL}/queue")
                    with urllib.request.urlopen(queue_req, timeout=5.0) as q_resp:
                        q_data = json.loads(q_resp.read().decode("utf-8"))
                        running = q_data.get("queue_running", [])
                        pending = q_data.get("queue_pending", [])
                        in_running = any(r and len(r) > 1 and r[1] == prompt_id for r in running)
                        in_pending = any(p and len(p) > 1 and p[1] == prompt_id for p in pending)
                        if in_running and not is_finalizing:
                            last_progress_time = now

                        # Se no estágio "finalizando" o prompt_id não estiver mais em /queue (nem running nem pending)
                        # e não houver resultado em /history, falhar rápido com mensagem clara, sem esperar o limite.
                        if is_finalizing and not in_running and not in_pending and not completed:
                            hist_found = False
                            try:
                                h_chk_req = urllib.request.Request(f"{COMFYUI_URL}/history/{prompt_id}")
                                with urllib.request.urlopen(h_chk_req, timeout=HISTORY_POLL_TIMEOUT_S) as h_chk_resp:
                                    h_chk_data = json.loads(h_chk_resp.read().decode("utf-8"))
                                    if prompt_id in h_chk_data:
                                        h_item = h_chk_data[prompt_id]
                                        if h_item.get("status", {}).get("completed", False):
                                            completed = True
                                            outputs = h_item.get("outputs", {})
                                            hist_found = True
                            except Exception:
                                pass

                            if not hist_found and not completed:
                                dump_comfyui_log_on_timeout(task_log_path)
                                interrupt_comfyui()
                                err_msg = "ComfyUI finalizou o processamento do prompt sem registrar histórico ou saída de vídeo."
                                append_task_log(task_log_path, f"❌ {err_msg}")
                                raise RuntimeError(err_msg)

                    # Telemetria estimada dos estágios quando não há evento de bloco do WebSocket
                    if tracker.current_chunk is None:
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
                    elif not is_finalizing:
                        tot_denom = tracker.total_chunks or max(1, tracker.current_chunk)
                        TASKS[task_id]["progress_pct"] = min(94.0, round((tracker.current_chunk / tot_denom) * 80.0 + 15.0, 1))

                except RuntimeError:
                    raise
                except Exception:
                    pass

                # (e) Timeout por ausência de progresso
                timeout_limit = FINAL_STAGE_TIMEOUT_S if is_finalizing else NO_PROGRESS_TIMEOUT_S
                if (now - last_progress_time) > timeout_limit:
                    dump_comfyui_log_on_timeout(task_log_path)
                    interrupt_comfyui()
                    stage_desc = "na etapa de finalização" if is_finalizing else "no ComfyUI"
                    raise TimeoutError(f"Ausência de progresso {stage_desc} por mais de {timeout_limit} segundos.")
        finally:
            tracker.stop()

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
            if res_aud.returncode != 0 or not os.path.exists(final_video_path) or os.path.getsize(final_video_path) == 0:
                # Fallback re-encodando caso a cópia direta de stream falhe
                cmd_fallback = [
                    "ffmpeg", "-y", "-i", task_out_mp4, "-i", audio_path,
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    final_video_path
                ]
                res_fb = subprocess.run(cmd_fallback, capture_output=True, text=True)
                if res_fb.returncode != 0 or not os.path.exists(final_video_path) or os.path.getsize(final_video_path) == 0:
                    append_task_log(task_log_path, f"Aviso ao mixar áudio: {res_aud.stderr or res_fb.stderr}. Mantendo sem áudio.")
                    shutil.copy2(task_out_mp4, final_video_path)
        else:
            # Sem áudio do usuário: vídeo final sai sem áudio e a geração NÃO falha
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
        total_time_s = round(time.time() - start_time, 1)
        final_resolution = get_video_resolution(final_video_path)
        vram_log_str = f"{min_vram_free_gb:.2f} GB" if min_vram_free_gb is not None else "Não disponível"

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
        TASKS[task_id]["resolution"] = final_resolution
        TASKS[task_id]["total_time_s"] = total_time_s
        TASKS[task_id]["min_vram_free_gb"] = min_vram_free_gb
        TASKS[task_id]["status"] = "completed"

        append_task_log(
            task_log_path,
            f"📊 Estatísticas da tarefa: qualidade='{quality_key}' ({target_megapixels} MP), "
            f"resolução={final_resolution}, tempo_total={total_time_s}s, menor_vram_livre={vram_log_str}"
        )
        append_task_log(task_log_path, f"🎉 Concluído com sucesso! Vídeo: {out_filename} ({size_mb} MB)")

    except Exception as e:
        interrupt_comfyui()
        err_raw = str(e)
        if COMFYUI_CRASH_MESSAGE in err_raw or err_raw == COMFYUI_CRASH_MESSAGE:
            user_msg = COMFYUI_CRASH_MESSAGE
        elif is_oom_error(err_raw):
            user_msg = oom_msg
        else:
            user_msg = f"Falha na renderização Viggle: {err_raw}"
        TASKS[task_id]["status"] = "error"
        TASKS[task_id]["error"] = user_msg
        TASKS[task_id]["message"] = user_msg
        TASKS[task_id]["finished_at"] = time.time()
        TASKS[task_id]["proc_alive"] = False
        append_task_log(task_log_path, f"❌ ERRO na tarefa {task_id}: {user_msg}")

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
        quality_raw = request.form.get('quality')
        quality_key, target_megapixels = parse_quality(quality_raw)
        min_duration = request.form.get('min_duration', 5.0)
        max_duration = request.form.get('max_duration', 30.0)

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
            "quality": quality_key,
            "megapixels": target_megapixels,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "log_path": os.path.join(work_dir, "task_render.log")
        }

        t = threading.Thread(
            target=background_viggle_worker,
            args=(task_id, work_dir, image_path, motion_video_path, audio_path, aspect_ratio, quality_raw, min_duration, max_duration),
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

    if "quality" in task:
        resp["quality"] = task["quality"]
    if "megapixels" in task:
        resp["megapixels"] = task["megapixels"]
    if "resolution" in task:
        resp["resolution"] = task["resolution"]
    if "total_time_s" in task:
        resp["total_time_s"] = task["total_time_s"]
    if "min_vram_free_gb" in task:
        resp["min_vram_free_gb"] = task["min_vram_free_gb"]

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

@app.route('/api/cancel_task/<task_id>', methods=['POST', 'OPTIONS'])
@app.route('/api/cancel/<task_id>', methods=['POST', 'OPTIONS'])
def cancel_task(task_id):
    if request.method == 'OPTIONS':
        return jsonify({"status": "ok"}), 200
    if task_id in TASKS:
        TASKS[task_id]["cancelled"] = True
        TASKS[task_id]["status"] = "cancelled"
        TASKS[task_id]["proc_alive"] = False
        TASKS[task_id]["message"] = "Tarefa cancelada."
        interrupt_comfyui()
        return jsonify({"success": True, "task_id": task_id, "message": "Tarefa cancelada e ComfyUI interrompido."}), 200
    return jsonify({"success": False, "error": "Tarefa não encontrada."}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    print(f"🧹 Limpando fila e interrompendo execuções residuais no ComfyUI ({COMFYUI_URL})...")
    init_comfyui_session()
    print(f"🚀 Iniciando Studio Video AI Viggle GPU Bridge v8.0 na porta {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
