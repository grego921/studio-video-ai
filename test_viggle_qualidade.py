"""
Suite de Testes Unitários: Seletor de Qualidade Viggle (colab_viggle_api.py + Frontend)
Branch: viggle-padrao
Valida:
1. Padrão ausente e inválido => 'padrao' (0.7 MP)
2. 'alta' => 'padrao' (0.7 MP) com aviso no log: "Qualidade 'alta' desativada, usando 'padrao'."
3. 'rapido' => 0.4 MP
4. Detecção OOM com regex de palavra inteira (não dispara com 'zoom' nem 'room', mas dispara com 'oom' ou 'out of memory')
5. Templates index.html e html_template.py idênticos byte a byte, com 'Padrão' ativo por padrão, 'Alta' removida e aviso atualizado
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

        print("\n" + "=" * 70)
        print("TODOS OS TESTES FORAM CONCLUÍDOS COM SUCESSO!")
        print("=" * 70)

    finally:
        server.shutdown()
        shutil.rmtree(d_temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
