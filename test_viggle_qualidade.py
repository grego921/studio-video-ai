"""
Suite de Testes Unitários: Seletor de Qualidade Viggle (colab_viggle_api.py + Frontend)
Branch: viggle-qualidade
Valida:
1. Presets de qualidade ('rapido' -> 0.4, 'padrao' -> 0.7, 'alta' -> 1.0) alteram apenas megapixels do nó certo
2. Valores ausentes ou inválidos caem em 'rapido' (0.4 MP)
3. Erros de falta de memória (OOM) no ComfyUI geram a mensagem: "Memória da GPU insuficiente nesta qualidade. Tente 'Padrão' ou 'Rápido'."
4. Registro completo no log da tarefa: qualidade, megapixels, resolução final, tempo total e menor VRAM livre
5. Templates index.html e html_template.py idênticos byte a byte, com seletor e aviso presentes
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
    simulate_prompt_oom = False
    vram_free_bytes = 21474836480  # 20.0 GB

    def log_message(self, format, *args):
        # Silencia logs normais do HTTP server
        pass

    def do_GET(self):
        if self.path == "/system_stats":
            # Reduz um pouco o VRAM livre para simular oscilação observável
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
            # Devolve bytes de vídeo falso
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
    print("INICIANDO SUITE DE TESTES: SELETOR DE QUALIDADE (VIGGLE)")
    print("=" * 70)

    # 1. Iniciar Fake ComfyUI Server em porta livre
    server = HTTPServer(("127.0.0.1", 0), MockComfyUIHandler)
    server_port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[*] Mock ComfyUI rodando em http://127.0.0.1:{server_port}")

    api.COMFYUI_URL = f"http://127.0.0.1:{server_port}"
    work_base_dir = tempfile.mkdtemp(prefix="viggle_quality_test_")

    try:
        # Prepara mídias de teste sintéticas
        sample_img = os.path.join(work_base_dir, "face.png")
        sample_vid = os.path.join(work_base_dir, "motion.mp4")
        create_synthetic_png(sample_img)
        create_synthetic_mp4(sample_vid, duration_s=1)

        # -------------------------------------------------------------------
        # TESTE 1: Parser de Qualidade e Presets
        # -------------------------------------------------------------------
        print("\n--- Teste 1: Parser de Qualidade e Mapeamento de Megapixels ---")
        assert api.parse_quality("rapido") == ("rapido", 0.4), "rapido deve ser 0.4 MP"
        assert api.parse_quality("padrao") == ("padrao", 0.7), "padrao deve ser 0.7 MP"
        assert api.parse_quality("alta") == ("alta", 1.0), "alta deve ser 1.0 MP"
        assert api.parse_quality("ALTA") == ("alta", 1.0), "case insensitive deve funcionar"
        assert api.parse_quality("  padrao  ") == ("padrao", 0.7), "whitespace deve ser removido"
        assert api.parse_quality("invalido") == ("rapido", 0.4), "inválido deve cair em rapido (0.4 MP)"
        assert api.parse_quality(None) == ("rapido", 0.4), "None deve cair em rapido (0.4 MP)"
        assert api.parse_quality("") == ("rapido", 0.4), "vazio deve cair em rapido (0.4 MP)"
        assert api.parse_quality(12345) == ("rapido", 0.4), "tipo inesperado deve cair em rapido (0.4 MP)"
        print(" [OK] parse_quality validado para todos os presets e fallbacks.")

        # -------------------------------------------------------------------
        # TESTE 2: Execução com cada Preset no Workflow ComfyUI
        # -------------------------------------------------------------------
        print("\n--- Teste 2: Injeção de Presets no Workflow ComfyUI ---")
        presets_to_test = [
            ("rapido", 0.4),
            ("padrao", 0.7),
            ("alta", 1.0),
            ("opcao_inexistente", 0.4)  # Fallback em execução
        ]

        # Salva o arquivo mp4 no ComfyUI output simulado para cópia direta
        mock_output_dir = os.path.join(work_base_dir, "output")
        os.makedirs(mock_output_dir, exist_ok=True)
        sample_out_vid = os.path.join(mock_output_dir, "viggle_test.mp4")
        create_synthetic_mp4(sample_out_vid, duration_s=1)
        api.COMFYUI_DIR = work_base_dir

        for q_input, expected_mp in presets_to_test:
            task_id = f"test_{q_input}_{uuid.uuid4().hex[:4]}"
            task_dir = os.path.join(work_base_dir, f"task_{task_id}")
            os.makedirs(task_dir, exist_ok=True)
            api.TASKS[task_id] = {"task_id": task_id, "status": "started"}

            MockComfyUIHandler.simulate_oom = False
            MockComfyUIHandler.simulate_prompt_oom = False
            MockComfyUIHandler.last_prompt_workflow = None

            # Executa worker
            api.background_viggle_worker(
                task_id=task_id,
                work_dir=task_dir,
                image_path=sample_img,
                motion_video_path=sample_vid,
                audio_path=None,
                aspect_ratio="original",
                quality=q_input
            )

            assert api.TASKS[task_id]["status"] == "completed", f"Tarefa {task_id} falhou: {api.TASKS[task_id].get('error')}"
            wf_sent = MockComfyUIHandler.last_prompt_workflow
            assert wf_sent is not None, "Workflow não foi enviado ao ComfyUI"

            # Encontra o nó JWImageResizeToMegapixels no workflow enviado
            resize_nodes = [
                (nid, n) for nid, n in wf_sent.items()
                if isinstance(n, dict) and n.get("class_type") == "JWImageResizeToMegapixels"
            ]
            assert len(resize_nodes) >= 1, "Nó JWImageResizeToMegapixels não encontrado no workflow enviado"
            res_nid, res_node = resize_nodes[0]

            # 1. Valida megapixels do nó
            actual_mp = res_node.get("inputs", {}).get("megapixels")
            assert actual_mp == expected_mp, f"Preset '{q_input}': esperado {expected_mp} MP, obtido {actual_mp}"

            # 2. Valida que os outros parâmetros do nó continuam intactos
            assert res_node.get("inputs", {}).get("divisible_by") == 32, "divisible_by deve permanecer 32"
            assert res_node.get("inputs", {}).get("interpolation_mode") == "nearest", "interpolation_mode intacto"

            # 3. Valida que CustomSigmas (nó 40) NÃO foi modificado
            sigmas_node = wf_sent.get("40")
            assert sigmas_node is not None, "Nó 40 (CustomSigmas) deve existir"
            assert sigmas_node.get("inputs", {}).get("interpolate_to_steps") == 3, "CustomSigmas passos não podem ser alterados"

            # 4. Valida log da tarefa: qualidade, megapixels, resolução, tempo e menor VRAM livre
            log_file = os.path.join(task_dir, "task_render.log")
            assert os.path.exists(log_file), "Log da tarefa não foi criado"
            with open(log_file, "r", encoding="utf-8") as lf:
                log_content = lf.read()

            assert f"{expected_mp} MP" in log_content, f"Log deve registrar {expected_mp} MP"
            assert "Estatísticas da tarefa" in log_content, "Log deve registrar as estatísticas da tarefa"
            assert "resolução=" in log_content, "Log deve registrar a resolução final do vídeo"
            assert "tempo_total=" in log_content, "Log deve registrar o tempo total"
            assert "menor_vram_livre=" in log_content, "Log deve registrar a menor VRAM livre observada"

            print(f" [OK] Preset '{q_input}' -> megapixels={actual_mp} aplicado exclusivamente no nó JWImageResizeToMegapixels.")

        # -------------------------------------------------------------------
        # TESTE 3: Detecção de Erro de Falta de Memória (OOM) e Mensagem
        # -------------------------------------------------------------------
        print("\n--- Teste 3: Tratamento de Erro OOM (Out of Memory) ---")
        oom_task_id = f"test_oom_{uuid.uuid4().hex[:4]}"
        oom_task_dir = os.path.join(work_base_dir, f"task_{oom_task_id}")
        os.makedirs(oom_task_dir, exist_ok=True)
        api.TASKS[oom_task_id] = {"task_id": oom_task_id, "status": "started"}

        MockComfyUIHandler.simulate_oom = True
        api.background_viggle_worker(
            task_id=oom_task_id,
            work_dir=oom_task_dir,
            image_path=sample_img,
            motion_video_path=sample_vid,
            audio_path=None,
            aspect_ratio="original",
            quality="alta"
        )

        expected_oom_msg = "Memória da GPU insuficiente nesta qualidade. Tente 'Padrão' ou 'Rápido'."
        assert api.TASKS[oom_task_id]["status"] == "error", "Status da tarefa deve ser 'error'"
        assert api.TASKS[oom_task_id]["error"] == expected_oom_msg, (
            f"Mensagem de erro esperada '{expected_oom_msg}', obtida '{api.TASKS[oom_task_id]['error']}'"
        )
        assert api.TASKS[oom_task_id]["message"] == expected_oom_msg, (
            f"Mensagem da tarefa esperada '{expected_oom_msg}', obtida '{api.TASKS[oom_task_id]['message']}'"
        )

        # Checa registro no log da tarefa
        oom_log_file = os.path.join(oom_task_dir, "task_render.log")
        with open(oom_log_file, "r", encoding="utf-8") as lf:
            oom_log_text = lf.read()
        assert expected_oom_msg in oom_log_text, "Mensagem de OOM em português deve estar presente no task log"
        print(" [OK] Erro de OOM capturado e mensagem em português validada com sucesso.")

        # -------------------------------------------------------------------
        # TESTE 4: Sincronização e Validação do Frontend (index.html e html_template.py)
        # -------------------------------------------------------------------
        print("\n--- Teste 4: Frontend index.html e html_template.py ---")
        with open("templates/index.html", "r", encoding="utf-8") as f:
            index_html = f.read()

        assert index_html == html_template.HTML_PAGE, (
            "templates/index.html e html_template.py DEVEM ser idênticos byte a byte!"
        )
        print(" [OK] templates/index.html e html_template.py são idênticos byte a byte.")

        # Verifica elementos obrigatórios no HTML
        assert 'id="qualityGrid"' in index_html, "Seletor qualityGrid ausente no HTML"
        assert 'data-quality="rapido"' in index_html, "Opção rapido ausente no seletor de qualidade"
        assert 'data-quality="padrao"' in index_html, "Opção padrao ausente no seletor de qualidade"
        assert 'data-quality="alta"' in index_html, "Opção alta ausente no seletor de qualidade"
        assert "Alta demora mais e usa mais memória." in index_html, "Texto de aviso ausente no HTML"

        # Verifica que Rápido é selecionado por padrão
        assert 'class="quality-card active" data-quality="rapido"' in index_html, "Rápido deve ser active por padrão"
        assert 'name="videoQuality" value="rapido" checked' in index_html, "Radio rapido deve estar checked por padrão"

        # Verifica que o FormData envia o campo 'quality'
        assert "neuralFormData.append('quality'" in index_html, "neuralFormData deve anexar o campo 'quality'"
        print(" [OK] Elementos HTML e envio no FormData verificados com sucesso.")

        print("\n" + "=" * 70)
        print("TODOS OS TESTES FORAM CONCLUÍDOS COM SUCESSO!")
        print("=" * 70)

    finally:
        server.shutdown()
        shutil.rmtree(work_base_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
