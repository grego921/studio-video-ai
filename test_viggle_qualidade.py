"""
Suite de Testes Unitários: Seletor de Qualidade Viggle e Polling GPU (colab_viggle_api.py + Frontend)
Branch: viggle-site-polling
Valida:
1. Padrão ausente e inválido => 'padrao' (0.7 MP)
2. 'alta' => 'padrao' (0.7 MP) com aviso no log: "Qualidade 'alta' desativada, usando 'padrao'."
3. 'rapido' => 0.4 MP
4. Detecção OOM com regex de palavra inteira (não dispara com 'zoom' nem 'room', mas dispara com 'oom' ou 'out of memory')
5. Templates index.html e html_template.py idênticos byte a byte, com 'Padrão' ativo por padrão, 'Alta' removida e aviso atualizado
6. Polling GPU: declaração de netErrorStartTime = null, descarte silencioso de tarefas antigas com URL diferente ou falha inicial (404/rede), ausência de alert() sem mensagem útil
"""

import os
import sys
import time
import json
import uuid
import shutil
import tempfile
import threading
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler

import colab_viggle_api as api
import html_template

# ---------------------------------------------------------------------------
# Servidor HTTP Falso do ComfyUI para Testes
# ---------------------------------------------------------------------------
class MockComfyUIHandler(BaseHTTPRequestHandler):
    last_prompt_workflow = None
    simulate_oom = False
    simulate_zoom_error = False
    simulate_prompt_oom = False
    vram_free_bytes = 21474836480  # 20.0 GB

    def log_message(self, format, *args):
        # Silencia logs normais do HTTP server
        pass

    def do_GET(self):
        if self.path == "/system_stats":
            MockComfyUIHandler.vram_free_bytes -= 536870912  # -0.5 GB
            data = {
                "system": {"os": "linux", "python_version": "3.10"},
                "devices": [
                    {
                        "name": "NVIDIA A100-SXM4-40GB",
                        "type": "cuda",
                        "index": 0,
                        "vram_total": 42949672960,
                        "vram_free": MockComfyUIHandler.vram_free_bytes
                    }
                ]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        if self.path.startswith("/history/"):
            prompt_id = self.path.split("/history/")[-1]
            if MockComfyUIHandler.simulate_oom:
                data = {
                    prompt_id: {
                        "status": {
                            "status_str": "error",
                            "completed": False,
                            "messages": [
                                [
                                    "execution_error",
                                    {
                                        "node_id": "11",
                                        "node_type": "KSampler",
                                        "exception_type": "OutOfMemoryError",
                                        "exception_message": "CUDA out of memory. Tried to allocate 2.00 GiB"
                                    }
                                ]
                            ]
                        }
                    }
                }
            elif MockComfyUIHandler.simulate_zoom_error:
                data = {
                    prompt_id: {
                        "status": {
                            "status_str": "error",
                            "completed": False,
                            "messages": [
                                [
                                    "execution_error",
                                    {
                                        "node_id": "26",
                                        "node_type": "JWImageResizeToMegapixels",
                                        "exception_type": "ValueError",
                                        "exception_message": "Invalid zoom interpolation factor"
                                    }
                                ]
                            ]
                        }
                    }
                }
            else:
                data = {
                    prompt_id: {
                        "status": {
                            "status_str": "success",
                            "completed": True,
                            "messages": []
                        },
                        "outputs": {
                            "21": {
                                "videos": [{"filename": "viggle_test.mp4", "subfolder": ""}]
                            }
                        }
                    }
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        if self.path.startswith("/view"):
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.end_headers()
            self.wfile.write(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42mp41")
            return

        if self.path == "/queue":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"queue_running": [], "queue_pending": []}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""

        if self.path == "/upload/image":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"name": "uploaded_media.mp4", "subfolder": "viggle"}).encode("utf-8"))
            return

        if self.path == "/prompt":
            if MockComfyUIHandler.simulate_prompt_oom:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": {"message": "torch.cuda.OutOfMemoryError: CUDA out of memory"},
                    "node_errors": {}
                }).encode("utf-8"))
                return

            try:
                payload = json.loads(body.decode("utf-8"))
                MockComfyUIHandler.last_prompt_workflow = payload.get("prompt", {})
            except Exception:
                MockComfyUIHandler.last_prompt_workflow = {}

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"prompt_id": "mock_prompt_001"}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def create_synthetic_mp4(path, duration_s=1):
    """Cria um MP4 sintético mínimo usando ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:s=320x240:d={duration_s}",
        "-c:v", "libx264", "-t", str(duration_s),
        "-pix_fmt", "yuv420p",
        path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def create_synthetic_png(path):
    """Cria um PNG sintético mínimo usando ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=256x256:d=1",
        "-frames:v", "1",
        path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def main():
    print("=" * 70)
    print("INICIANDO SUITE DE TESTES: VIGGLE-PADRAO (0.7 MP)")
    print("=" * 70)

    # 1. Iniciar Fake ComfyUI Server em porta livre
    server = HTTPServer(("127.0.0.1", 0), MockComfyUIHandler)
    server_port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[*] Mock ComfyUI rodando em http://127.0.0.1:{server_port}")

    api.COMFYUI_URL = f"http://127.0.0.1:{server_port}"
    
    # Criado estritamente na unidade D:
    d_temp_dir = os.path.abspath("temp_test_padrao")
    os.makedirs(d_temp_dir, exist_ok=True)
    work_base_dir = tempfile.mkdtemp(prefix="viggle_padrao_", dir=d_temp_dir)

    try:
        # Prepara mídias sintéticas
        sample_img = os.path.join(work_base_dir, "face.png")
        sample_vid = os.path.join(work_base_dir, "motion.mp4")
        create_synthetic_png(sample_img)
        create_synthetic_mp4(sample_vid, duration_s=1)

        # -------------------------------------------------------------------
        # TESTE 1: Parser de Qualidade e Mapeamento de Megapixels
        # -------------------------------------------------------------------
        print("\n--- Teste 1: Parser de Qualidade e Mapeamento de Megapixels ---")
        # 1. 'padrao' => 0.7
        assert api.parse_quality("padrao") == ("padrao", 0.7), "padrao deve ser 0.7 MP"
        assert api.parse_quality("PADRAO") == ("padrao", 0.7), "case insensitive deve funcionar"
        assert api.parse_quality("  padrao  ") == ("padrao", 0.7), "whitespace deve ser removido"

        # 2. 'alta' => 0.7 (tratado como padrao)
        assert api.parse_quality("alta") == ("padrao", 0.7), "alta deve ser tratada como padrao (0.7 MP)"
        assert api.parse_quality("ALTA") == ("padrao", 0.7), "ALTA deve ser tratada como padrao (0.7 MP)"

        # 3. 'rapido' => 0.4
        assert api.parse_quality("rapido") == ("rapido", 0.4), "rapido deve ser 0.4 MP"

        # 4. Padrão ausente e inválido => 'padrao' (0.7 MP)
        assert api.parse_quality(None) == ("padrao", 0.7), "None deve cair em padrao (0.7 MP)"
        assert api.parse_quality("") == ("padrao", 0.7), "vazio deve cair em padrao (0.7 MP)"
        assert api.parse_quality("invalido_xyz") == ("padrao", 0.7), "inválido deve cair em padrao (0.7 MP)"
        assert api.parse_quality(12345) == ("padrao", 0.7), "tipo inesperado deve cair em padrao (0.7 MP)"
        print(" [OK] parse_quality validado: ausente/inválido/alta => 0.7 MP; rapido => 0.4 MP.")

        # -------------------------------------------------------------------
        # TESTE 2: Execução com cada Preset no Workflow ComfyUI
        # -------------------------------------------------------------------
        print("\n--- Teste 2: Injeção de Presets no Workflow ComfyUI ---")
        mock_output_dir = os.path.join(work_base_dir, "output")
        os.makedirs(mock_output_dir, exist_ok=True)
        sample_out_vid = os.path.join(mock_output_dir, "viggle_test.mp4")
        create_synthetic_mp4(sample_out_vid, duration_s=1)
        api.COMFYUI_DIR = work_base_dir

        # Teste 2A: 'padrao' => 0.7 MP
        t_padrao_id = f"test_padrao_{uuid.uuid4().hex[:4]}"
        t_padrao_dir = os.path.join(work_base_dir, f"task_{t_padrao_id}")
        os.makedirs(t_padrao_dir, exist_ok=True)
        api.TASKS[t_padrao_id] = {"task_id": t_padrao_id, "status": "started"}
        MockComfyUIHandler.last_prompt_workflow = None
        api.background_viggle_worker(t_padrao_id, t_padrao_dir, sample_img, sample_vid, None, "original", "padrao")
        
        res_node_padrao = [n for n in MockComfyUIHandler.last_prompt_workflow.values() if n.get("class_type") == "JWImageResizeToMegapixels"][0]
        assert res_node_padrao["inputs"]["megapixels"] == 0.7, "padrao deve configurar megapixels=0.7"
        print(" [OK] Preset 'padrao' -> megapixels=0.7 configurado com sucesso.")

        # Teste 2B: 'alta' => 0.7 MP com aviso no log: "Qualidade 'alta' desativada, usando 'padrao'."
        t_alta_id = f"test_alta_{uuid.uuid4().hex[:4]}"
        t_alta_dir = os.path.join(work_base_dir, f"task_{t_alta_id}")
        os.makedirs(t_alta_dir, exist_ok=True)
        api.TASKS[t_alta_id] = {"task_id": t_alta_id, "status": "started"}
        MockComfyUIHandler.last_prompt_workflow = None
        api.background_viggle_worker(t_alta_id, t_alta_dir, sample_img, sample_vid, None, "original", "alta")

        res_node_alta = [n for n in MockComfyUIHandler.last_prompt_workflow.values() if n.get("class_type") == "JWImageResizeToMegapixels"][0]
        assert res_node_alta["inputs"]["megapixels"] == 0.7, "alta deve ser tratada como 0.7 MP"

        alta_log_file = os.path.join(t_alta_dir, "task_render.log")
        with open(alta_log_file, "r", encoding="utf-8") as lf:
            alta_log = lf.read()
        assert "Qualidade 'alta' desativada, usando 'padrao'." in alta_log, (
            "Log deve registrar exatamente: \"Qualidade 'alta' desativada, usando 'padrao'.\""
        )
        print(" [OK] Preset 'alta' -> megapixels=0.7 com aviso registrado no log.")

        # Teste 2C: 'rapido' => 0.4 MP
        t_rapido_id = f"test_rapido_{uuid.uuid4().hex[:4]}"
        t_rapido_dir = os.path.join(work_base_dir, f"task_{t_rapido_id}")
        os.makedirs(t_rapido_dir, exist_ok=True)
        api.TASKS[t_rapido_id] = {"task_id": t_rapido_id, "status": "started"}
        MockComfyUIHandler.last_prompt_workflow = None
        api.background_viggle_worker(t_rapido_id, t_rapido_dir, sample_img, sample_vid, None, "original", "rapido")

        res_node_rapido = [n for n in MockComfyUIHandler.last_prompt_workflow.values() if n.get("class_type") == "JWImageResizeToMegapixels"][0]
        assert res_node_rapido["inputs"]["megapixels"] == 0.4, "rapido deve configurar megapixels=0.4"
        print(" [OK] Preset 'rapido' -> megapixels=0.4 configurado com sucesso.")

        # Teste 2D: ausente / inválido => 0.7 MP
        t_inv_id = f"test_inv_{uuid.uuid4().hex[:4]}"
        t_inv_dir = os.path.join(work_base_dir, f"task_{t_inv_id}")
        os.makedirs(t_inv_dir, exist_ok=True)
        api.TASKS[t_inv_id] = {"task_id": t_inv_id, "status": "started"}
        MockComfyUIHandler.last_prompt_workflow = None
        api.background_viggle_worker(t_inv_id, t_inv_dir, sample_img, sample_vid, None, "original", "invalido_qualquer")

        res_node_inv = [n for n in MockComfyUIHandler.last_prompt_workflow.values() if n.get("class_type") == "JWImageResizeToMegapixels"][0]
        assert res_node_inv["inputs"]["megapixels"] == 0.7, "inválido deve cair em padrao (0.7 MP)"
        print(" [OK] Valor inválido/ausente -> cai em padrao (0.7 MP).")

        # -------------------------------------------------------------------
        # TESTE 3: Detecção de Erro OOM vs 'zoom' / 'room'
        # -------------------------------------------------------------------
        print("\n--- Teste 3: Detecção de OOM com Limite de Palavra (Sem Falso Positivo com 'zoom') ---")
        # 3A: Checagens unitárias de is_oom_error
        assert api.is_oom_error("zoom") is False, "'zoom' NAO pode disparar OOM"
        assert api.is_oom_error("room") is False, "'room' NAO pode disparar OOM"
        assert api.is_oom_error("sem dar zoom no rosto") is False, "'zoom no rosto' NAO pode disparar OOM"
        assert api.is_oom_error("zoom_factor = 2.0") is False, "'zoom_factor' NAO pode disparar OOM"
        assert api.is_oom_error("room acoustics") is False, "'room' NAO pode disparar OOM"

        assert api.is_oom_error("oom") is True, "'oom' isolado DEVE disparar OOM"
        assert api.is_oom_error("CUDA OOM occurred") is True, "'CUDA OOM' DEVE disparar OOM"
        assert api.is_oom_error("Error: oom in allocator") is True, "'oom' DEVE disparar OOM"
        assert api.is_oom_error("torch.cuda.OutOfMemoryError: CUDA out of memory") is True, "'out of memory' DEVE disparar OOM"
        assert api.is_oom_error("CUDA error: out of memory") is True, "'cuda error: out of memory' DEVE disparar OOM"
        print(" [OK] is_oom_error: regex de palavra inteira validada.")

        # 3B: Erro contendo 'zoom' no ComfyUI NÃO dispara mensagem de memória
        zoom_task_id = f"test_zoom_{uuid.uuid4().hex[:4]}"
        zoom_task_dir = os.path.join(work_base_dir, f"task_{zoom_task_id}")
        os.makedirs(zoom_task_dir, exist_ok=True)
        api.TASKS[zoom_task_id] = {"task_id": zoom_task_id, "status": "started"}
        MockComfyUIHandler.simulate_oom = False
        MockComfyUIHandler.simulate_zoom_error = True
        api.background_viggle_worker(zoom_task_id, zoom_task_dir, sample_img, sample_vid, None, "original", "padrao")

        assert api.TASKS[zoom_task_id]["status"] == "error", "Status deve ser error"
        assert "Memória da GPU insuficiente" not in api.TASKS[zoom_task_id]["error"], (
            "Erro com 'zoom' NÃO deve gerar mensagem de falta de memória!"
        )
        assert "Invalid zoom interpolation factor" in api.TASKS[zoom_task_id]["error"], (
            "Erro real com 'zoom' deve ser preservado."
        )
        print(" [OK] Erro com a palavra 'zoom' NÃO dispara falso positivo de OOM.")

        # 3C: Erro real de OOM no ComfyUI dispara a mensagem em português
        oom_task_id = f"test_oom_{uuid.uuid4().hex[:4]}"
        oom_task_dir = os.path.join(work_base_dir, f"task_{oom_task_id}")
        os.makedirs(oom_task_dir, exist_ok=True)
        api.TASKS[oom_task_id] = {"task_id": oom_task_id, "status": "started"}
        MockComfyUIHandler.simulate_oom = True
        MockComfyUIHandler.simulate_zoom_error = False
        api.background_viggle_worker(oom_task_id, oom_task_dir, sample_img, sample_vid, None, "original", "padrao")

        expected_oom_msg = "Memória da GPU insuficiente nesta qualidade. Tente 'Padrão' ou 'Rápido'."
        assert api.TASKS[oom_task_id]["error"] == expected_oom_msg, "Erro real de OOM deve mostrar a mensagem exata"
        print(" [OK] Erro real de OOM exibe mensagem em português com sucesso.")

        # -------------------------------------------------------------------
        # TESTE 4: Frontend (index.html e html_template.py)
        # -------------------------------------------------------------------
        print("\n--- Teste 4: Frontend index.html e html_template.py ---")
        with open("templates/index.html", "r", encoding="utf-8") as f:
            index_html = f.read()

        assert index_html == html_template.HTML_PAGE, (
            "templates/index.html e html_template.py DEVEM ser rigorosamente idênticos byte a byte!"
        )
        print(" [OK] templates/index.html e html_template.py são idênticos byte a byte.")

        # 'Padrão' ativo e checked por padrão
        assert 'class="quality-card active" data-quality="padrao"' in index_html, (
            "Padrão deve ter a classe active por padrão"
        )
        assert 'name="videoQuality" value="padrao" checked' in index_html, (
            "Radio de Padrão deve estar checked por padrão"
        )

        # 'Alta' REMOVIDA da tela
        assert 'data-quality="alta"' not in index_html, "Opção 'alta' DEVE estar removida do HTML"
        assert 'value="alta"' not in index_html, "Radio 'alta' DEVE estar removido do HTML"

        # Novo texto explicativo
        expected_text = "Padrão dá mais nitidez no rosto; Rápido é mais veloz."
        assert expected_text in index_html, f"Texto '{expected_text}' deve estar presente abaixo do seletor"

        print(" [OK] Frontend: Padrão ativo por padrão, Alta removida e texto explicativo atualizado.")
        assert "neuralFormData.append('min_duration', minTimeSlider.value);" in index_html
        assert "neuralFormData.append('max_duration', maxTimeSlider.value);" in index_html
        print(" [OK] Frontend: min_duration e max_duration injetados no neuralFormData.")

        # -------------------------------------------------------------------
        # TESTE 5: Polling da GPU, Declaração de Variáveis e Descarte Silencioso
        # -------------------------------------------------------------------
        print("\n--- Teste 5: Polling da GPU, Declaração de Variáveis e Descarte Silencioso ---")

        # 5A: Declaração de netErrorStartTime = null junto de consecutiveNetErrors
        assert "let consecutiveNetErrors = 0;" in index_html, "consecutiveNetErrors deve estar declarado"
        assert "let netErrorStartTime = null;" in index_html, "netErrorStartTime DEVE estar declarado com 'let netErrorStartTime = null;'"

        decl_idx = index_html.find("let netErrorStartTime = null;")
        usage_idx = index_html.find("if (netErrorStartTime === null)")
        assert decl_idx != -1 and usage_idx != -1 and decl_idx < usage_idx, (
            "netErrorStartTime deve ser declarado antes de qualquer uso para evitar ReferenceError!"
        )
        print(" [OK] netErrorStartTime declarado com 'let netErrorStartTime = null;' junto de consecutiveNetErrors.")

        # 5B: Assinatura de runGpuTaskPolling aceita isInitialRestore
        assert "async function runGpuTaskPolling(taskId, cleanGpuUrl, isInitialRestore = false)" in index_html, (
            "runGpuTaskPolling deve aceitar isInitialRestore com padrão false"
        )
        print(" [OK] runGpuTaskPolling aceita isInitialRestore com valor padrão false.")

        # 5C: Descarte ao abrir a página se URL salva for diferente da URL atual do motor GPU
        assert "if (!cleanPendingTaskUrl || !currentGpuEngineUrl || cleanPendingTaskUrl !== currentGpuEngineUrl)" in index_html, (
            "Frontend deve comparar cleanPendingTaskUrl com currentGpuEngineUrl ao abrir a página"
        )
        assert "localStorage.removeItem('studio_active_task_id');" in index_html
        assert "localStorage.removeItem('studio_active_task_url');" in index_html
        print(" [OK] Descarte de tarefa antiga ao abrir a página quando a URL salva diferir da URL atual.")

        # 5D: Descarte silencioso na 1ª consulta de restauração em caso de 404 ou erro de rede
        assert "if (isInitialRestore && isFirstPoll)" in index_html, (
            "runGpuTaskPolling deve tratar a primeira consulta durante restauração inicial"
        )
        assert "if (netFailed || (pollRes && pollRes.status === 404))" in index_html, (
            "Primeira consulta de restauração deve verificar netFailed e status 404"
        )
        print(" [OK] Descarte silencioso na 1ª consulta em caso de 404 ou erro de rede.")

        # 5E: Nenhuma exceção de JavaScript do polling pode virar alert() sem mensagem útil
        fn_start = index_html.find("async function runGpuTaskPolling")
        fn_end = index_html.find("form.addEventListener('submit'", fn_start)
        polling_fn_code = index_html[fn_start:fn_end]
        assert "alert(" not in polling_fn_code, (
            "Nenhuma exceção do polling pode virar alert()! Mensagem deve ser exibida na tela."
        )
        assert "progressStatusMsg.innerHTML = " in polling_fn_code, (
            "Erro inesperado no polling deve ser exibido diretamente na tela em progressStatusMsg"
        )
        assert "logBox.textContent = rawMsg;" in polling_fn_code, (
            "Detalhes do erro devem ser exibidos no logBox na tela"
        )
        print(" [OK] Polling sem chamadas de alert(): erros exibidos com clareza na tela.")

        # 5F: Simulação completa da lógica de restauração e descarte
        class LocalStorageMock:
            def __init__(self, data=None):
                self.store = dict(data or {})
            def getItem(self, key):
                return self.store.get(key)
            def setItem(self, key, val):
                self.store[key] = str(val)
            def removeItem(self, key):
                self.store.pop(key, None)

        def simulate_page_open(ls, current_gpu_url):
            pending_task_id = ls.getItem("studio_active_task_id")
            pending_task_url = ls.getItem("studio_active_task_url")
            cur_url = (current_gpu_url or "").strip().rstrip("/")
            saved_url = (pending_task_url or "").strip().rstrip("/")
            if pending_task_id:
                if not saved_url or not cur_url or saved_url != cur_url:
                    ls.removeItem("studio_active_task_id")
                    ls.removeItem("studio_active_task_url")
                    return "discarded_url_mismatch"
                else:
                    return "resume_polling"
            return "no_task"

        def simulate_first_poll(is_initial_restore, poll_status, net_failed, ls):
            is_first_poll = True
            if is_initial_restore and is_first_poll:
                if net_failed or poll_status == 404:
                    ls.removeItem("studio_active_task_id")
                    ls.removeItem("studio_active_task_url")
                    return {"action": "discard_silent", "submit_enabled": True, "progress_hidden": True, "alert_called": False}
            if poll_status == 404:
                ls.removeItem("studio_active_task_id")
                ls.removeItem("studio_active_task_url")
                return {"action": "throw_restarted_error"}
            return {"action": "continue_polling"}

        # Caso 1: URL diferente => descartar silenciosamente sem polling
        ls1 = LocalStorageMock({"studio_active_task_id": "task_123", "studio_active_task_url": "https://colab-old.trycloudflare.com"})
        res1 = simulate_page_open(ls1, "https://colab-new.trycloudflare.com")
        assert res1 == "discarded_url_mismatch"
        assert ls1.getItem("studio_active_task_id") is None
        assert ls1.getItem("studio_active_task_url") is None

        # Caso 2: URL vazia no motor atual => descartar silenciosamente
        ls2 = LocalStorageMock({"studio_active_task_id": "task_123", "studio_active_task_url": "https://colab-old.trycloudflare.com"})
        res2 = simulate_page_open(ls2, "")
        assert res2 == "discarded_url_mismatch"
        assert ls2.getItem("studio_active_task_id") is None

        # Caso 3: URL idêntica => retoma polling com isInitialRestore=True
        ls3 = LocalStorageMock({"studio_active_task_id": "task_123", "studio_active_task_url": "https://colab-same.trycloudflare.com"})
        res3 = simulate_page_open(ls3, "https://colab-same.trycloudflare.com/")
        assert res3 == "resume_polling"

        # Caso 4: Primeira consulta da restauração retorna 404 => descarta em silêncio, reativa formulário normal
        poll_res4 = simulate_first_poll(True, 404, False, ls3)
        assert poll_res4["action"] == "discard_silent"
        assert poll_res4["submit_enabled"] is True
        assert poll_res4["progress_hidden"] is True
        assert poll_res4["alert_called"] is False
        assert ls3.getItem("studio_active_task_id") is None

        # Caso 5: Primeira consulta da restauração tem erro de rede => descarta em silêncio, reativa formulário normal
        ls5 = LocalStorageMock({"studio_active_task_id": "task_456", "studio_active_task_url": "https://colab-same.trycloudflare.com"})
        poll_res5 = simulate_first_poll(True, None, True, ls5)
        assert poll_res5["action"] == "discard_silent"
        assert poll_res5["submit_enabled"] is True
        assert poll_res5["progress_hidden"] is True
        assert ls5.getItem("studio_active_task_id") is None

        # Caso 6: Tarefa nova criada pelo formulário (is_initial_restore=False) com 404 => lança erro de reinício
        ls6 = LocalStorageMock({"studio_active_task_id": "task_789", "studio_active_task_url": "https://colab-same.trycloudflare.com"})
        poll_res6 = simulate_first_poll(False, 404, False, ls6)
        assert poll_res6["action"] == "throw_restarted_error"

        print(" [OK] Simulação de descarte silencioso e restauração validada em todos os cenários.")

        print("\n" + "=" * 70)
        print("TODOS OS TESTES FORAM CONCLUÍDOS COM SUCESSO!")
        print("=" * 70)

    finally:
        server.shutdown()
        shutil.rmtree(d_temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
