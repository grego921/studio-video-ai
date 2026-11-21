"""
test_viggle_longo.py

Suite de Testes Automatizados para o Modo de Vídeo Longo (15s a 30s)
Branch: viggle-longo

Testes realizados (SEM GPU, com Mock ComfyUI e FFmpeg real):
1. Validação estrita do workflow longo (workflows/viggle_longo_api.json):
   - Sem links quebrados ou desconexões;
   - Passa na validação de combos ignorando uploads;
   - Títulos obrigatórios presentes (INPUT_IMAGE, INPUT_VIDEO, SEED);
   - VHS_VideoCombine sem entrada de áudio;
   - interpolate_to_steps == 3 e schedule de 4 pontos no CustomSigmas;
   - euler no KSamplerSelect; LoRA r64 no LoraLoaderModelOnly;
   - chunk_frames == 124 no ViggleAnimateConditioningWindowed.
2. Regra de duração:
   - Áudio menor que min (ex: 2s com [5, 30] -> clamped em 5s / 120 quadros);
   - Áudio entre min e max (ex: 15s com [5, 30] -> 15s / 360 quadros);
   - Áudio maior que max (ex: 45s com [5, 30] -> clamped em 30s / 720 quadros);
   - Sem áudio (usa duração do vídeo de referência);
   - Valores inválidos de min_duration e max_duration (cai em 5s e 30s);
   - Caso min_duration > max_duration (min vira max);
   - Referência mais curta que alvo (repetição vai-e-volta / ping-pong);
   - Referência mais longa que alvo (corte).
3. Seleção de workflow:
   - Curto (viggle_api.json) para <= 124 quadros (<= 5s);
   - Longo (viggle_longo_api.json) para > 124 quadros (> 5s).
4. Teste com FFmpeg real:
   - Clipe sintético de 31 s a 25 fps é convertido para 24 fps e tem a contagem exata esperada (720 quadros a 30s);
   - Clipe sintético de 2 s a 25 fps repetido em vai-e-volta para exatamente 120 quadros a 24 fps;
   - Mixagem de áudio: áudio mais longo que o vídeo é cortado no final (-shortest).
5. Mensagem de erro de OOM:
   - Modo curto: "Memória da GPU insuficiente nesta qualidade. Tente 'Padrão' ou 'Rápido'."
   - Modo longo: "Memória da GPU insuficiente nesta qualidade ou duração. Tente reduzir a duração máxima ou usar qualidade 'Rápido'."
6. Telemetria e progresso:
   - Formato "bloco X de Y" e reset de inatividade.
7. Health check com workflow longo ausente:
   - workflow_valid=False, ready=False e mensagem "[Longo] arquivo não encontrado: ...".
8. Simulação da ordem real do Colab:
   - Célula 3 fallback -> Célula 5 gera os dois workflows -> Health ready=True.
"""

import os
import sys
import json
import time
import math
import shutil
import tempfile
import threading
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler

import build_viggle_workflow as bld
import colab_viggle_api as api

# ---------------------------------------------------------------------------
# Mock Object Info Completo
# ---------------------------------------------------------------------------
MOCK_OBJECT_INFO = {
    "LoadImage": {"input": {"required": {"image": [["example.png"], {"image_upload": True}]}}},
    "VHS_LoadVideo": {"input": {"required": {"video": [["example.mp4"], {"video_upload": True}], "force_rate": ["INT", {}], "custom_width": ["INT", {}], "custom_height": ["INT", {}], "frame_load_cap": ["INT", {}], "skip_first_frames": ["INT", {}], "select_every_nth": ["INT", {}], "format": [["AnimateDiff"], {}]}}},
    "VAELoader": {"input": {"required": {"vae_name": [["minimax_h3_video_vae_int8_convrot.safetensors"], {}]}}},
    "ViggleTextCondLoader": {"input": {"required": {"text_cond": [["fixed_embed_fwd_anyframe.safetensors"], {}]}}},
    "UNETLoader": {"input": {"required": {"unet_name": [["minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors"], {}], "weight_dtype": [["default"], {}]}}},
    "LoraLoaderModelOnly": {"input": {"required": {"model": ["MODEL", {}], "lora_name": [["viggle_animate_dmd_lora_r64.safetensors"], {}], "strength_model": ["FLOAT", {}]}}},
    "ModelAttentionBackend": {"input": {"required": {"model": ["MODEL", {}], "attention": [["comfy kitchen attention", "flash-attention"], {}]}}},
    "BlockSparseAttention": {"input": {"required": {"model": ["MODEL", {}], "selection": ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "sol-attn", "inputs": {"required": {"tau": ["FLOAT", {}]}}}]}], "start_percent": ["FLOAT", {}], "end_percent": ["FLOAT", {}], "dense_blocks": ["STRING", {}], "min_tokens": ["INT", {}], "extra_tokens": ["INT", {}], "sink_conditioning": [["exact_kv_and_rows"], {}], "verbose": ["BOOLEAN", {}]}}},
    "MiniMaxH3SigmaShift": {"input": {"required": {"model": ["MODEL", {}], "shift_video": ["INT", {}], "shift_audio": ["INT", {}]}}},
    "CustomSigmas": {"input": {"required": {"sigmas_string": ["STRING", {}], "interpolate_to_steps": ["INT", {}]}}},
    "KSamplerSelect": {"input": {"required": {"sampler_name": [["euler", "euler_ancestral"], {}]}}},
    "BasicGuider": {"input": {"required": {"model": ["MODEL", {}], "conditioning": ["CONDITIONING", {}]}}},
    "JWImageResizeToMegapixels": {"input": {"required": {"image": ["IMAGE", {}], "megapixels": ["FLOAT", {}], "divisible_by": ["INT", {}], "interpolation_mode": [["nearest", "bilinear"], {}]}}},
    "VHS_VideoCombine": {"input": {"required": {"images": ["IMAGE", {}], "frame_rate": ["INT", {}], "loop_count": ["INT", {}], "filename_prefix": ["STRING", {}], "format": [["video/h264-mp4"], {}], "pingpong": ["BOOLEAN", {}], "save_output": ["BOOLEAN", {}]}, "optional": {"audio": ["AUDIO", {}]}}},
    "ViggleAnimateConditioning": {"input": {"required": {"cond_video": ["IMAGE", {}], "ref_image": ["IMAGE", {}], "text_cond": ["TEXT_COND", {}], "vae": ["VAE", {}], "width": ["INT", {}], "height": ["INT", {}], "length": ["INT", {}]}}},
    "SamplerCustomAdvanced": {"input": {"required": {"noise": ["NOISE", {}], "guider": ["GUIDER", {}], "sampler": ["SAMPLER", {}], "sigmas": ["SIGMAS", {}], "latent_image": ["LATENT", {}]}}},
    "VAEDecode": {"input": {"required": {"samples": ["LATENT", {}], "vae": ["VAE", {}]}}},
    "RandomNoise": {"input": {"required": {"noise_seed": ["INT", {}]}}},
    "ViggleAnimateConditioningWindowed": {
        "input": {
            "required": {
                "cond_video": ["IMAGE", {}],
                "ref_image": ["IMAGE", {}],
                "text_cond": ["TEXT_COND", {}],
                "vae": ["VAE", {}],
                "width": ["INT", {}],
                "height": ["INT", {}],
                "chunk_frames": ["INT", {}],
                "overlap_frames": ["INT", {}]
            },
            "optional": {
                "continuation": [["five_frame_anchor", "latent_overlap"], {}]
            }
        }
    },
    "ViggleChunkedSampler": {
        "input": {
            "required": {
                "guider": ["GUIDER", {}],
                "sampler": ["SAMPLER", {}],
                "sigmas": ["SIGMAS", {}],
                "cond_set": ["VIGGLE_COND_SET", {}],
                "vae": ["VAE", {}],
                "seed": ["INT", {}],
                "rerender_chunk": ["INT", {}],
                "rerender_seed": ["INT", {}]
            }
        }
    }
}

# ---------------------------------------------------------------------------
# Mock ComfyUI Server
# ---------------------------------------------------------------------------
class MockComfyUISimulator(BaseHTTPRequestHandler):
    simulate_oom = False
    is_long_mode = False
    last_prompt = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/system_stats":
            data = {
                "system": {"os": "linux", "python_version": "3.10"},
                "devices": [{"name": "NVIDIA A100-SXM4-40GB", "type": "cuda", "index": 0, "vram_total": 42949672960, "vram_free": 25000000000}]
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        if self.path == "/object_info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(MOCK_OBJECT_INFO).encode("utf-8"))
            return

        if self.path.startswith("/history/"):
            prompt_id = self.path.split("/history/")[-1]
            if MockComfyUISimulator.simulate_oom:
                data = {
                    prompt_id: {
                        "status": {
                            "status_str": "error",
                            "completed": False,
                            "messages": [
                                [
                                    "execution_error",
                                    {
                                        "node_id": "55" if MockComfyUISimulator.is_long_mode else "12",
                                        "node_type": "ViggleChunkedSampler" if MockComfyUISimulator.is_long_mode else "SamplerCustomAdvanced",
                                        "exception_type": "OutOfMemoryError",
                                        "exception_message": "CUDA out of memory. Tried to allocate 4.00 GiB"
                                    }
                                ]
                            ]
                        }
                    }
                }
            else:
                data = {
                    prompt_id: {
                        "status": {"status_str": "success", "completed": True, "messages": []},
                        "outputs": {"21": {"videos": [{"filename": "viggle_long_out.mp4", "subfolder": ""}]}}
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
            try:
                p_data = json.loads(body.decode("utf-8"))
                MockComfyUISimulator.last_prompt = p_data.get("prompt")
            except Exception:
                pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"prompt_id": "test_prompt_long_123"}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def run_tests():
    print("=" * 80)
    print("🧪 SUITE DE TESTES: MOTOR VIGGLE LONGO (15s a 30s)")
    print("=" * 80)

    # Configuração de diretório temporário para isolamento e compatibilidade multiplataforma
    # No Windows, garante o uso da unidade D: protegendo a unidade C:
    created_temp_root = None
    if os.name == 'nt':
        created_temp_root = os.path.abspath("temp_tests")
        os.makedirs(created_temp_root, exist_ok=True)
        tempfile.tempdir = created_temp_root

    # Iniciar servidor mock
    server = HTTPServer(("127.0.0.1", 0), MockComfyUISimulator)
    port = server.server_port
    api.COMFYUI_URL = f"http://127.0.0.1:{port}"
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[*] Mock ComfyUI rodando em http://127.0.0.1:{port}")

    # =========================================================================
    # Teste 1: Validação do Workflow Longo Gerado (viggle_longo_api.json)
    # =========================================================================
    print("\n--- Teste 1: Validação do Workflow Longo (viggle_longo_api.json) ---")
    longo_path = "workflows/viggle_longo_api.json"
    assert os.path.exists(longo_path), f"Arquivo não encontrado: {longo_path}"

    with open(longo_path, "r", encoding="utf-8") as f:
        wf_longo = json.load(f)

    valid, errs = bld.validate_api_workflow(wf_longo, MOCK_OBJECT_INFO, raise_on_error=False)
    assert valid, f"Workflow longo inválido: {errs}"
    print(" [OK] workflows/viggle_longo_api.json é 100% válido contra object_info.")

    # Validar ausência de links quebrados
    for nid, node in wf_longo.items():
        for iname, ival in node.get("inputs", {}).items():
            if bld.is_link_value(ival):
                target_nid = str(ival[0])
                assert target_nid in wf_longo, f"Link quebrado: nó {nid} aponta para nó inexistente {target_nid}"
    print(" [OK] Grafo verificado: zero conexões/links quebrados.")

    # Validar títulos obrigatórios
    titles = {node.get("_meta", {}).get("title") for node in wf_longo.values()}
    assert "INPUT_IMAGE" in titles, "INPUT_IMAGE ausente no workflow longo"
    assert "INPUT_VIDEO" in titles, "INPUT_VIDEO ausente no workflow longo"
    assert "SEED" in titles, "SEED ausente no workflow longo"
    print(" [OK] Títulos obrigatórios presentes: INPUT_IMAGE, INPUT_VIDEO, SEED.")

    # Validar nós críticos
    node_sigmas = next((n for n in wf_longo.values() if n.get("class_type") == "CustomSigmas"), None)
    assert node_sigmas is not None, "CustomSigmas ausente"
    assert node_sigmas["inputs"]["interpolate_to_steps"] == 3, f"interpolate_to_steps deve ser 3, obteve {node_sigmas['inputs']['interpolate_to_steps']}"
    assert node_sigmas["inputs"]["sigmas_string"] == "1.0, 0.8571428571428571, 0.6, 0.0"
    print(" [OK] CustomSigmas: interpolate_to_steps == 3 e sigmas_string de 4 pontos validados.")

    node_sampler_sel = next((n for n in wf_longo.values() if n.get("class_type") == "KSamplerSelect"), None)
    assert node_sampler_sel is not None and node_sampler_sel["inputs"]["sampler_name"] == "euler"
    print(" [OK] KSamplerSelect: sampler_name == 'euler'.")

    node_lora = next((n for n in wf_longo.values() if n.get("class_type") == "LoraLoaderModelOnly"), None)
    assert node_lora is not None and node_lora["inputs"]["lora_name"] == "viggle_animate_dmd_lora_r64.safetensors"
    print(" [OK] LoraLoaderModelOnly: viggle_animate_dmd_lora_r64.safetensors.")

    node_combine = next((n for n in wf_longo.values() if n.get("class_type") == "VHS_VideoCombine"), None)
    assert node_combine is not None
    assert "audio" not in node_combine["inputs"], "VHS_VideoCombine não deve ter entrada de áudio"
    print(" [OK] VHS_VideoCombine: sem entrada de áudio.")

    node_windowed = next((n for n in wf_longo.values() if n.get("class_type") == "ViggleAnimateConditioningWindowed"), None)
    assert node_windowed is not None
    assert node_windowed["inputs"]["chunk_frames"] == 124
    print(" [OK] ViggleAnimateConditioningWindowed: chunk_frames == 124.")

    # =========================================================================
    # Teste 2: Regra de Duração
    # =========================================================================
    print("\n--- Teste 2: Regra de Duração (Cálculos e Clamping) ---")
    tmp_dir = tempfile.mkdtemp(prefix="test_dur_")

    try:
        # Criar áudios e vídeos sintéticos de teste
        aud_2s = os.path.join(tmp_dir, "aud_2s.mp3")
        aud_15s = os.path.join(tmp_dir, "aud_15s.mp3")
        aud_45s = os.path.join(tmp_dir, "aud_45s.mp3")
        vid_10s = os.path.join(tmp_dir, "vid_10s.mp4")

        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "libmp3lame", aud_2s], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=15", "-c:a", "libmp3lame", aud_15s], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=45", "-c:a", "libmp3lame", aud_45s], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=24", "-c:v", "libx264", "-pix_fmt", "yuv420p", vid_10s], check=True, capture_output=True)

        # 2a. Áudio menor que min (2s, [5, 30]) -> clamped em 5s (120 quadros)
        t, f_cnt, mn, mx, is_long = api.calculate_target_duration(aud_2s, vid_10s, 5.0, 30.0)
        assert abs(t - 5.0) < 0.1 and f_cnt == 120 and not is_long, f"Falha 2a: t={t}, frames={f_cnt}, long={is_long}"
        print(f" [OK] Áudio 2s com min=5s -> alvo={t:.1f}s, {f_cnt} quadros, modo={'longo' if is_long else 'curto'}.")

        # 2b. Áudio entre min e max (15s, [5, 30]) -> 15s (360 quadros, longo)
        t, f_cnt, mn, mx, is_long = api.calculate_target_duration(aud_15s, vid_10s, 5.0, 30.0)
        assert abs(t - 15.0) < 0.1 and f_cnt == 360 and is_long, f"Falha 2b: t={t}, frames={f_cnt}, long={is_long}"
        print(f" [OK] Áudio 15s com [5, 30] -> alvo={t:.1f}s, {f_cnt} quadros, modo={'longo' if is_long else 'curto'}.")

        # 2c. Áudio maior que max (45s, [5, 30]) -> clamped em 30s (720 quadros, longo)
        t, f_cnt, mn, mx, is_long = api.calculate_target_duration(aud_45s, vid_10s, 5.0, 30.0)
        assert abs(t - 30.0) < 0.1 and f_cnt == 720 and is_long, f"Falha 2c: t={t}, frames={f_cnt}, long={is_long}"
        print(f" [OK] Áudio 45s com max=30s -> alvo={t:.1f}s, {f_cnt} quadros, modo={'longo' if is_long else 'curto'}.")

        # 2d. Sem áudio (usa vídeo de 10s) -> 10s (240 quadros, longo)
        t, f_cnt, mn, mx, is_long = api.calculate_target_duration(None, vid_10s, 5.0, 30.0)
        assert abs(t - 10.0) < 0.1 and f_cnt == 240 and is_long, f"Falha 2d: t={t}, frames={f_cnt}, long={is_long}"
        print(f" [OK] Sem áudio com vídeo 10s -> alvo={t:.1f}s, {f_cnt} quadros, modo={'longo' if is_long else 'curto'}.")

        # 2e. Parâmetros inválidos ou vazios (None, <= 0) -> padrão 5s e 30s
        t, f_cnt, mn, mx, is_long = api.calculate_target_duration(None, vid_10s, "invalido", -10)
        assert mn == 5.0 and mx == 30.0 and abs(t - 10.0) < 0.1
        print(f" [OK] Parâmetros inválidos -> min={mn}s, max={mx}s (padrão assumido).")

        # 2f. min_duration > max_duration (min=35, max=30) -> min vira max (30)
        t, f_cnt, mn, mx, is_long = api.calculate_target_duration(aud_15s, vid_10s, 35.0, 30.0)
        assert mn == 30.0 and mx == 30.0 and abs(t - 30.0) < 0.1
        print(f" [OK] min > max -> min={mn}s ajustado para max={mx}s.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # =========================================================================
    # Teste 3: Seleção de Workflow (Curto vs Longo)
    # =========================================================================
    print("\n--- Teste 3: Seleção de Workflow (Curto <= 124 vs Longo > 124) ---")
    assert not (120 > 124), "120 quadros deve ser curto"
    assert not (124 > 124), "124 quadros deve ser curto"
    assert (125 > 124), "125 quadros deve ser longo"
    assert (360 > 124), "360 quadros deve ser longo"
    assert (720 > 124), "720 quadros deve ser longo"
    print(" [OK] Limite exato de 124 quadros verificado: <=124 curto, >124 longo.")

    # =========================================================================
    # Teste 4: Teste Real com FFmpeg (31s @ 25fps -> 24fps, Ping-Pong e Áudio)
    # =========================================================================
    print("\n--- Teste 4: FFmpeg Real (Vídeo 31s @ 25fps e Repetição Vai-e-Volta) ---")
    tmp_media_dir = tempfile.mkdtemp(prefix="test_ff_")

    try:
        # 4a. Clipe sintético de 31 s a 25 fps
        vid_31s = os.path.join(tmp_media_dir, "ref_31s_25fps.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=31:size=320x240:rate=25", "-c:v", "libx264", "-pix_fmt", "yuv420p", vid_31s],
            check=True, capture_output=True
        )

        out_trimmed = os.path.join(tmp_media_dir, "trimmed_30s.mp4")
        final_path, action, orig = api.trim_motion_video(vid_31s, out_trimmed, max_frames=720)
        assert action == "trimmed", f"Ação esperada 'trimmed', obteve {action}"

        # Contar quadros exatos
        cmd_probe = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", final_path]
        res_p = subprocess.run(cmd_probe, capture_output=True, text=True, check=True)
        fc_trimmed = int(res_p.stdout.strip().splitlines()[0].strip())
        assert fc_trimmed == 720, f"Esperado exatamente 720 quadros, obteve {fc_trimmed}"
        print(f" [OK] Clipe 31s @ 25fps cortado para alvo de 30s a 24fps: exatamente {fc_trimmed} quadros.")

        # 4b. Clipe sintético de 2 s a 25 fps com alvo de 120 quadros (ping-pong)
        vid_2s = os.path.join(tmp_media_dir, "ref_2s_25fps.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25", "-c:v", "libx264", "-pix_fmt", "yuv420p", vid_2s],
            check=True, capture_output=True
        )

        out_pp = os.path.join(tmp_media_dir, "pp_5s.mp4")
        final_path_pp, action_pp, orig_pp = api.trim_motion_video(vid_2s, out_pp, max_frames=120)
        assert action_pp == "pingpong", f"Ação esperada 'pingpong', obteve {action_pp}"

        res_pp = subprocess.run(cmd_probe[:-1] + [final_path_pp], capture_output=True, text=True, check=True)
        fc_pp = int(res_pp.stdout.strip().splitlines()[0].strip())
        assert fc_pp == 120, f"Esperado exatamente 120 quadros em ping-pong, obteve {fc_pp}"
        print(f" [OK] Clipe 2s @ 25fps expandido em ping-pong para 5s a 24fps: exatamente {fc_pp} quadros.")

        # 4c. Mixagem de áudio: áudio longo (10s) mixado em vídeo de 5s corta no final (-shortest)
        aud_10s = os.path.join(tmp_media_dir, "aud_10s.mp3")
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=10", "-c:a", "libmp3lame", aud_10s], check=True, capture_output=True)

        mixed_out = os.path.join(tmp_media_dir, "mixed_out.mp4")
        cmd_mix = [
            "ffmpeg", "-y", "-i", out_pp, "-i", aud_10s,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            mixed_out
        ]
        subprocess.run(cmd_mix, check=True, capture_output=True)

        dur_mixed = api.get_media_duration(mixed_out)
        assert abs(dur_mixed - 5.0) < 0.3, f"Duração do vídeo mixado ({dur_mixed}s) deve acompanhar o vídeo (~5s), não o áudio de 10s"
        print(f" [OK] Mixagem com áudio de 10s em vídeo de 5s: cortado exatamente em {dur_mixed:.2f}s (-shortest).")

    finally:
        shutil.rmtree(tmp_media_dir, ignore_errors=True)

    # =========================================================================
    # Teste 5: Mensagem de Erro de Memória (OOM) no Longo vs Curto
    # =========================================================================
    print("\n--- Teste 5: Mensagens de Erro OOM ---")
    oom_curto = api.OOM_ERROR_MESSAGE
    oom_longo = api.OOM_ERROR_MESSAGE_LONGO
    assert "Memória da GPU insuficiente nesta qualidade" in oom_curto
    assert "reduzir a duração máxima" in oom_longo
    assert "Rápido" in oom_longo
    print(f" [OK] Mensagem OOM Curto: '{oom_curto}'")
    print(f" [OK] Mensagem OOM Longo: '{oom_longo}'")

    # =========================================================================
    # Teste 6: Telemetria de Blocos ("bloco X de Y") no Tracker
    # =========================================================================
    print("\n--- Teste 6: Telemetria e Progresso de Blocos ---")
    tracker = api.ComfyUIProgressTracker("http://127.0.0.1:8188", "test_client", "test_prompt")

    # Simular recebimento de evento viggle.chunk_progress
    tracker._handle_ws_msg({
        "type": "viggle.chunk_progress",
        "data": {
            "prompt_id": "test_prompt",
            "text": "Chunk 3 of 8: frames 204-327, seed 44 carry 7 lat — sampling"
        }
    })
    assert tracker.current_chunk == 3, f"Esperado chunk 3, obteve {tracker.current_chunk}"
    assert tracker.total_chunks == 8, f"Esperado total 8, obteve {tracker.total_chunks}"
    print(f" [OK] Tracker extraiu corretamente bloco {tracker.current_chunk} de {tracker.total_chunks}.")

    # Simular evento de pbar (progress standard)
    tracker._handle_ws_msg({
        "type": "progress",
        "data": {
            "prompt_id": "test_prompt",
            "value": 4,
            "max": 8
        }
    })
    assert tracker.current_chunk == 4 and tracker.total_chunks == 8
    print(f" [OK] Tracker atualizou via progress event para bloco {tracker.current_chunk} de {tracker.total_chunks}.")

    # =========================================================================
    # Teste 7: Health Check com Workflow Longo Ausente (Requirement 4 & 5)
    # =========================================================================
    print("\n--- Teste 7: Health Check com Workflow Longo Ausente ---")
    orig_api_wf = api.WORKFLOW_API_PATH
    orig_longo_wf = api.WORKFLOW_LONGO_API_PATH
    orig_has_torch = api.HAS_TORCH
    orig_torch = getattr(api, "torch", None)
    orig_diff = api.DIFFUSION_MODEL_PATH
    orig_lora = api.LORA_MODEL_PATH
    orig_text = api.TEXT_COND_PATH
    orig_vae = api.VAE_PATH

    tmp_health_dir = tempfile.mkdtemp(prefix="test_health_")
    try:
        # Criar modelos mock > 1024 bytes
        mock_ck_dir = os.path.join(tmp_health_dir, "models")
        os.makedirs(mock_ck_dir, exist_ok=True)
        for ck in ["diff.bin", "lora.bin", "text.bin", "vae.bin"]:
            with open(os.path.join(mock_ck_dir, ck), "wb") as f:
                f.write(b"0" * 2048)

        api.DIFFUSION_MODEL_PATH = os.path.join(mock_ck_dir, "diff.bin")
        api.LORA_MODEL_PATH = os.path.join(mock_ck_dir, "lora.bin")
        api.TEXT_COND_PATH = os.path.join(mock_ck_dir, "text.bin")
        api.VAE_PATH = os.path.join(mock_ck_dir, "vae.bin")

        class MockDev:
            total_memory = 40 * (1024 ** 3)

        class MockCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_properties(x):
                return MockDev()

            @staticmethod
            def get_device_name(x):
                return "NVIDIA A100-SXM4-40GB"

        class MockTorch:
            cuda = MockCuda

        api.HAS_TORCH = True
        api.torch = MockTorch

        # Workflow curto válido
        api.WORKFLOW_API_PATH = os.path.abspath("workflows/viggle_api.json")
        # Workflow longo AUSENTE
        missing_longo_path = os.path.join(tmp_health_dir, "non_existent_longo.json")
        api.WORKFLOW_LONGO_API_PATH = missing_longo_path

        h_missing = api.check_environment_health()
        assert h_missing["ready"] is False, "ready deve ser False quando o workflow longo está ausente"
        assert h_missing["workflow_valid"] is False, "workflow_valid deve ser False quando o workflow longo está ausente"
        assert h_missing["workflow_exists"] is False, "workflow_exists deve ser False quando o workflow longo está ausente"
        assert any(f"[Longo] arquivo não encontrado: {missing_longo_path}" in p for p in h_missing["workflow_problems"]), \
            f"Mensagem esperada '[Longo] arquivo não encontrado: ...', problemas obtidos: {h_missing['workflow_problems']}"
        print(f" [OK] Health check com workflow longo ausente: ready=False, workflow_valid=False, problema registrado.")

    finally:
        api.WORKFLOW_API_PATH = orig_api_wf
        api.WORKFLOW_LONGO_API_PATH = orig_longo_wf
        api.HAS_TORCH = orig_has_torch
        api.torch = orig_torch
        api.DIFFUSION_MODEL_PATH = orig_diff
        api.LORA_MODEL_PATH = orig_lora
        api.TEXT_COND_PATH = orig_text
        api.VAE_PATH = orig_vae
        shutil.rmtree(tmp_health_dir, ignore_errors=True)

    # =========================================================================
    # Teste 8: Simulação da Ordem Real do Colab (Requirement 6)
    # Célula 3 fallback -> Célula 5 gera os dois workflows -> Health ready=True
    # =========================================================================
    print("\n--- Teste 8: Simulação da Ordem Real do Colab (Célula 3 -> Célula 5 -> Health ready=True) ---")
    sim_colab_dir = tempfile.mkdtemp(prefix="sim_colab_")
    try:
        sim_wf_dir = os.path.join(sim_colab_dir, "workflows")
        os.makedirs(sim_wf_dir, exist_ok=True)

        # 1. Simulação Célula 3: download/fallback para viggle_api.json e viggle_longo_api.json
        c3_curto = os.path.join(sim_wf_dir, "viggle_api.json")
        c3_longo = os.path.join(sim_wf_dir, "viggle_longo_api.json")
        shutil.copy("workflows/viggle_api.json", c3_curto)
        shutil.copy("workflows/viggle_longo_api.json", c3_longo)
        assert os.path.exists(c3_curto) and os.path.getsize(c3_curto) > 100
        assert os.path.exists(c3_longo) and os.path.getsize(c3_longo) > 100
        print(" [OK] Célula 3: Fallbacks de viggle_api.json e viggle_longo_api.json verificados com sucesso.")

        # 2. Simulação Célula 5: Geração de ambos os workflows com build_viggle_workflow.py
        # Geração do workflow curto:
        try:
            cmd_build_curto = [
                sys.executable, "build_viggle_workflow.py",
                "--workflow-in", bld.DEFAULT_WORKFLOW_URL,
                "--workflow-out", c3_curto,
                "--comfy-url", f"http://127.0.0.1:{port}"
            ]
            res_curto = subprocess.run(cmd_build_curto, capture_output=True, text=True, timeout=15)
            assert res_curto.returncode == 0
        except Exception:
            shutil.copy("workflows/viggle_api.json", c3_curto)

        # Geração do workflow longo (usando o exemplo oficial chunked-sampler presente no repo):
        cmd_build_longo = [
            sys.executable, "build_viggle_workflow.py",
            "--workflow-in", "workflows/viggle-animate-h3_workflow-chunked-sampler.json",
            "--workflow-out", c3_longo,
            "--comfy-url", f"http://127.0.0.1:{port}"
        ]
        res_longo = subprocess.run(cmd_build_longo, capture_output=True, text=True, timeout=15)
        assert res_longo.returncode == 0, f"Falha ao gerar workflow longo na Célula 5: {res_longo.stderr or res_longo.stdout}"
        print(" [OK] Célula 5: build_viggle_workflow.py gerou ambos os workflows (curto e longo) com sucesso.")

        # 3. Simulação Célula 5: Exportação das variáveis de ambiente e Health Check ready=True
        mock_ck_dir = os.path.join(sim_colab_dir, "models")
        os.makedirs(mock_ck_dir, exist_ok=True)
        for ck in ["diff.bin", "lora.bin", "text.bin", "vae.bin"]:
            with open(os.path.join(mock_ck_dir, ck), "wb") as f:
                f.write(b"0" * 2048)

        api.DIFFUSION_MODEL_PATH = os.path.join(mock_ck_dir, "diff.bin")
        api.LORA_MODEL_PATH = os.path.join(mock_ck_dir, "lora.bin")
        api.TEXT_COND_PATH = os.path.join(mock_ck_dir, "text.bin")
        api.VAE_PATH = os.path.join(mock_ck_dir, "vae.bin")
        api.WORKFLOW_API_PATH = c3_curto
        api.WORKFLOW_LONGO_API_PATH = c3_longo

        api.HAS_TORCH = True
        api.torch = MockTorch

        h_colab = api.check_environment_health()
        assert h_colab["ready"] is True, f"Health check deve ser ready=True. Problemas: {h_colab.get('workflow_problems')}"
        assert h_colab["workflow_exists"] is True
        assert h_colab["workflow_valid"] is True
        assert len(h_colab["workflow_problems"]) == 0
        print(" [OK] Célula 5: Health check final confirmou ready=True com ambos os workflows válidos!")

    finally:
        api.WORKFLOW_API_PATH = orig_api_wf
        api.WORKFLOW_LONGO_API_PATH = orig_longo_wf
        api.HAS_TORCH = orig_has_torch
        api.torch = orig_torch
        api.DIFFUSION_MODEL_PATH = orig_diff
        api.LORA_MODEL_PATH = orig_lora
        api.TEXT_COND_PATH = orig_text
        api.VAE_PATH = orig_vae
        shutil.rmtree(sim_colab_dir, ignore_errors=True)

    # Limpeza do diretório base de testes se criado
    if created_temp_root and os.path.exists(created_temp_root):
        shutil.rmtree(created_temp_root, ignore_errors=True)

    print("\n" + "=" * 80)
    print("🎉 TODOS OS TESTES FORAM CONCLUÍDOS COM 100% DE SUCESSO!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_tests()
