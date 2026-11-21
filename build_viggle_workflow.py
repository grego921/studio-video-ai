#!/usr/bin/env python3
"""
build_viggle_workflow.py

Gera automaticamente workflows/viggle_api.json a partir do workflow oficial de interface
(viggle-animate-h3_workflow-v1.2.0.json) utilizando a API GET /object_info de uma instância
do ComfyUI em execução.

Regras atendidas:
1. Converte do formato de interface (nodes/links) para o formato API do ComfyUI.
2. Bypassa nós 'Reroute', ligando a entrada diretamente ao nó e slot de origem real.
3. Se houver 'PrimitiveNode', transfere seu valor como widget no nó de destino.
4. Ignora nós de interface puros ('Note', 'MarkdownNote').
5. Mapeia widgets_values aos parâmetros de widget definidos em /object_info.
   Trata widgets_values tanto no formato lista quanto dicionário (ex: nós VHS).
   Descarta o controle extra 'control_after_generate' adicionado pela interface após sementes.
   Trata combos dinâmicos (ex: BlockSparseAttention / COMFY_DYNAMICCOMBO_V3).
6. Mantém exclusivamente o caminho do grafo que produz o vídeo final: o nó VHS_VideoCombine
   cuja entrada 'images' vem de 'VAEDecode'. Poda ramificações de comparação/concatenação
   (ImageConcanate, VHS_VideoCombine secundário) e nós orfãos.
7. Ajustes automáticos:
   - LoadImage: _meta.title = 'INPUT_IMAGE'
   - VHS_LoadVideo: _meta.title = 'INPUT_VIDEO', frame_load_cap = 124, skip_first_frames = 0
   - RandomNoise: _meta.title = 'SEED'
   - LoraLoaderModelOnly: lora_name = 'viggle_animate_dmd_lora_r64.safetensors'
   - UNETLoader: unet_name = 'minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors'
   - VAELoader: vae_name = 'minimax_h3_video_vae_int8_convrot.safetensors'
   - ViggleTextCondLoader: text_cond = 'fixed_embed_fwd_anyframe.safetensors'
   - VHS_VideoCombine de saída: save_output = True
8. Validação estrita: se qualquer dependência ou nó não puder ser resolvido com 100% de certeza,
   aborta imediatamente com mensagem clara em português.
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error


DEFAULT_WORKFLOW_URL = (
    "https://raw.githubusercontent.com/Saganaki22/ComfyUI-Viggle-Animate-H3/main/"
    "example_workflows/viggle-animate-h3_workflow-v1.2.0.json"
)

DEFAULT_CANDIDATE_PATHS = [
    "custom_nodes/ComfyUI-Viggle-Animate-H3/example_workflows/viggle-animate-h3_workflow-v1.2.0.json",
    "/content/ComfyUI/custom_nodes/ComfyUI-Viggle-Animate-H3/example_workflows/viggle-animate-h3_workflow-v1.2.0.json",
    "temp_comfyui/custom_nodes/ComfyUI-Viggle-Animate-H3/example_workflows/viggle-animate-h3_workflow-v1.2.0.json",
]

CHUNKED_WORKFLOW_URL = (
    "https://raw.githubusercontent.com/Saganaki22/ComfyUI-Viggle-Animate-H3/main/"
    "example_workflows/viggle-animate-h3_workflow-chunked-sampler.json"
)

CHUNKED_CANDIDATE_PATHS = [
    "workflows/viggle-animate-h3_workflow-chunked-sampler.json",
    "custom_nodes/ComfyUI-Viggle-Animate-H3/example_workflows/viggle-animate-h3_workflow-chunked-sampler.json",
    "/content/ComfyUI/custom_nodes/ComfyUI-Viggle-Animate-H3/example_workflows/viggle-animate-h3_workflow-chunked-sampler.json",
    "temp_comfyui/custom_nodes/ComfyUI-Viggle-Animate-H3/example_workflows/viggle-animate-h3_workflow-chunked-sampler.json",
]


def log_info(msg: str):
    print(f"[build_workflow] {msg}")


def log_err(msg: str):
    print(f"❌ [build_workflow ERRO] {msg}", file=sys.stderr)


def is_widget_type(type_spec) -> bool:
    """Verifica se uma especificação de entrada em object_info é um widget."""
    if isinstance(type_spec, list):
        return True
    if isinstance(type_spec, str):
        u = type_spec.upper()
        if u in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"):
            return True
        if type_spec == "COMFY_DYNAMICCOMBO_V3":
            return True
    return False


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


def fetch_object_info(comfy_url: str) -> dict:
    """Obtém as definições de todos os nós registrados no ComfyUI."""
    url = f"{comfy_url.rstrip('/')}/object_info"
    log_info(f"Consultando definições de nós em {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BuildViggleWorkflow/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            log_info(f"Sucesso: {len(data)} classes de nós retornadas pelo ComfyUI.")
            return data
    except urllib.error.URLError as e:
        log_err(f"Falha de conexão com o ComfyUI em '{url}': {e}")
        log_err("Certifique-se de que o ComfyUI está em execução antes de rodar este script.")
        sys.exit(1)
    except Exception as e:
        log_err(f"Erro ao processar /object_info do ComfyUI: {e}")
        sys.exit(1)


def load_ui_workflow(source_path_or_url: str, is_chunked: bool = False) -> dict:
    """Carrega o JSON do workflow de interface a partir de arquivo local ou URL."""
    # 1. Se o caminho fornecido existir diretamente:
    if source_path_or_url and os.path.exists(source_path_or_url):
        log_info(f"Carregando workflow local: {source_path_or_url}")
        with open(source_path_or_url, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2. Se for uma URL HTTP/HTTPS:
    if source_path_or_url and (source_path_or_url.startswith("http://") or source_path_or_url.startswith("https://")):
        log_info(f"Baixando workflow da URL: {source_path_or_url}")
        req = urllib.request.Request(source_path_or_url, headers={"User-Agent": "BuildViggleWorkflow/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # 3. Detectar se busca chunked/longo
    candidate_paths = CHUNKED_CANDIDATE_PATHS if is_chunked else DEFAULT_CANDIDATE_PATHS
    fallback_url = CHUNKED_WORKFLOW_URL if is_chunked else DEFAULT_WORKFLOW_URL

    # Tentar os caminhos padrão conhecidos:
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            log_info(f"Carregando workflow do caminho padrão: {candidate}")
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)

    # 4. Fallback: baixar do GitHub oficial
    log_info(f"Arquivo local não encontrado. Baixando do repositório oficial: {fallback_url}")
    req = urllib.request.Request(fallback_url, headers={"User-Agent": "BuildViggleWorkflow/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_link_source(link_id: int, node_map: dict, link_map: dict, visited: set = None):
    """
    Resolve a origem de um link recursivamente, atravessando nós 'Reroute'.
    Retorna (origin_node_id, origin_slot_index) ou ('PRIMITIVE', valor).
    """
    if visited is None:
        visited = set()
    if link_id in visited:
        raise ValueError(f"Ciclo detectado ao resolver o link {link_id}")
    visited.add(link_id)

    if link_id not in link_map:
        raise ValueError(f"Link ID {link_id} não encontrado na lista de conexões do workflow.")

    link = link_map[link_id]
    orig_id = link[1]
    orig_slot = link[2]

    orig_node = node_map.get(orig_id)
    if not orig_node:
        raise ValueError(f"Nó de origem {orig_id} apontado pelo link {link_id} não existe no workflow.")

    orig_type = orig_node.get("type")

    # Bypassar nós Reroute religando à sua própria entrada
    if orig_type == "Reroute":
        inputs = orig_node.get("inputs", [])
        if not inputs or inputs[0].get("link") is None:
            raise ValueError(f"Nó Reroute (ID {orig_id}) está desconectado na entrada.")
        return resolve_link_source(inputs[0]["link"], node_map, link_map, visited)

    # Bypassar nós ModelPreviewOverrideKJ religando à sua entrada 'model'
    if orig_type == "ModelPreviewOverrideKJ":
        inputs = orig_node.get("inputs", [])
        model_inputs = [i for i in inputs if i.get("name") == "model"]
        if not model_inputs or model_inputs[0].get("link") is None:
            raise ValueError(f"Nó ModelPreviewOverrideKJ (ID {orig_id}) está desconectado na entrada 'model'.")
        return resolve_link_source(model_inputs[0]["link"], node_map, link_map, visited)

    # PrimitiveNode: repassar o valor fixado
    if orig_type == "PrimitiveNode":
        wv = orig_node.get("widgets_values", [])
        val = wv[0] if wv else None
        return ("PRIMITIVE", val)

    return (orig_id, orig_slot)


def convert_workflow(wf_ui: dict, object_info: dict) -> dict:
    """
    Executa a conversão completa do workflow UI para formato API.
    Aplica resolução de links, poda de ramos secundários e overrides obrigatórios.
    """
    nodes = wf_ui.get("nodes", [])
    links = wf_ui.get("links", [])

    if not nodes:
        raise ValueError("O workflow de interface não contém nós válidos.")

    node_map = {n["id"]: n for n in nodes}
    link_map = {l[0]: l for l in links}

    # -------------------------------------------------------------------------
    # 1. Identificar o nó VHS_VideoCombine de saída primária
    #    (cuja entrada 'images' vem diretamente de um nó VAEDecode)
    # -------------------------------------------------------------------------
    primary_output_id = None
    all_video_combines = []

    for n in nodes:
        if n.get("type") == "VHS_VideoCombine":
            all_video_combines.append(n["id"])
            for inp in n.get("inputs", []):
                if inp.get("name") == "images" and inp.get("link") is not None:
                    try:
                        src_res = resolve_link_source(inp["link"], node_map, link_map)
                    except Exception as e:
                        raise ValueError(f"Erro ao resolver entrada 'images' do VHS_VideoCombine {n['id']}: {e}")

                    if src_res[0] != "PRIMITIVE":
                        src_node = node_map.get(src_res[0])
                        if src_node and src_node.get("type") in ("VAEDecode", "ViggleChunkedSampler"):
                            primary_output_id = n["id"]
                            break

    if primary_output_id is None:
        raise ValueError(
            f"Não foi possível identificar com 100% de certeza o nó VHS_VideoCombine primário. "
            f"Nós VHS_VideoCombine encontrados: {all_video_combines}. "
            f"Nenhum deles possui a entrada 'images' conectada diretamente a um VAEDecode ou ViggleChunkedSampler."
        )

    log_info(f"Nó de saída primária identificado: VHS_VideoCombine ID {primary_output_id}")

    # -------------------------------------------------------------------------
    # 2. Rastreamento e poda de dead-ends (Grafo de dependência reversa)
    # -------------------------------------------------------------------------
    reachable_node_ids = set()
    queue = [primary_output_id]

    while queue:
        curr_id = queue.pop(0)
        if curr_id in reachable_node_ids:
            continue
        reachable_node_ids.add(curr_id)

        curr_node = node_map.get(curr_id)
        if not curr_node:
            continue

        for inp in curr_node.get("inputs", []):
            # Ignorar conexão de áudio no VHS_VideoCombine para não acoplar extração de áudio
            if curr_node.get("type") == "VHS_VideoCombine" and inp.get("name") == "audio":
                continue
            lid = inp.get("link")
            if lid is not None:
                src_res = resolve_link_source(lid, node_map, link_map)
                if src_res[0] != "PRIMITIVE":
                    upstream_id = src_res[0]
                    if upstream_id not in reachable_node_ids:
                        queue.append(upstream_id)

    pruned_node_ids = set(node_map.keys()) - reachable_node_ids
    log_info(
        f"Grafo filtrado: {len(reachable_node_ids)} nós ativos mantidos. "
        f"{len(pruned_node_ids)} nós descartados/podados (ex: comparação, notas, reroutes órfãos)."
    )

    # -------------------------------------------------------------------------
    # 3. Mapeamento de cada nó ativo para o formato API do ComfyUI
    # -------------------------------------------------------------------------
    api_workflow = {}

    for nid in sorted(list(reachable_node_ids)):
        n = node_map[nid]
        ctype = n.get("type")
        title = n.get("title")

        # Ignorar nós de anotação de interface se tiverem entrado no grafo
        if ctype in ("Note", "MarkdownNote"):
            continue

        if ctype not in object_info:
            raise ValueError(
                f"Classe de nó '{ctype}' (ID {nid}) não foi encontrada no /object_info do ComfyUI. "
                f"Verifique se todos os custom nodes necessários estão instalados."
            )

        info = object_info[ctype]
        req_spec = info.get("input", {}).get("required", {})
        opt_spec = info.get("input", {}).get("optional", {})

        api_node = {
            "inputs": {},
            "class_type": ctype,
            "_meta": {
                "title": title or ctype
            }
        }

        # a) Mapear conexões originadas de links
        for inp in n.get("inputs", []):
            iname = inp.get("name")
            # Ignorar conexão de áudio no VHS_VideoCombine para evitar extração de áudio pelo VHS_LoadVideo
            if ctype == "VHS_VideoCombine" and iname == "audio":
                continue
            lid = inp.get("link")
            if lid is not None:
                src_res = resolve_link_source(lid, node_map, link_map)
                if src_res[0] == "PRIMITIVE":
                    api_node["inputs"][iname] = src_res[1]
                else:
                    api_node["inputs"][iname] = [str(src_res[0]), src_res[1]]

        # b) Mapear widgets_values
        wv = n.get("widgets_values")
        if isinstance(wv, dict):
            # Formato de dicionário (ex: VHS_VideoCombine, VHS_LoadVideo)
            for k, v in wv.items():
                if k in req_spec or k in opt_spec:
                    if k not in api_node["inputs"]:
                        api_node["inputs"][k] = v
        elif isinstance(wv, list):
            # Formato de lista sequencial
            wv_idx = 0
            all_specs = []
            for k, v in req_spec.items():
                all_specs.append((k, v, True))
            for k, v in opt_spec.items():
                all_specs.append((k, v, False))

            for param_name, param_spec, is_req in all_specs:
                if wv_idx >= len(wv):
                    break
                if param_name in api_node["inputs"]:
                    # Já suprido por conexão de link
                    continue

                ptype = param_spec[0]
                if is_widget_type(ptype):
                    val = wv[wv_idx]
                    wv_idx += 1
                    api_node["inputs"][param_name] = val

                    # Tratamento de COMFY_DYNAMICCOMBO_V3 (ex: BlockSparseAttention)
                    if ptype == "COMFY_DYNAMICCOMBO_V3" and isinstance(param_spec[1], dict):
                        options = param_spec[1].get("options", [])
                        matched_opt = next((o for o in options if o.get("key") == val), None)
                        if matched_opt:
                            sub_req = matched_opt.get("inputs", {}).get("required", {})
                            for sub_k, sub_v in sub_req.items():
                                if wv_idx < len(wv):
                                    sub_val = wv[wv_idx]
                                    api_node["inputs"][f"{param_name}.{sub_k}"] = sub_val
                                    api_node["inputs"][sub_k] = sub_val
                                    wv_idx += 1

                    # Descarte do valor extra 'control_after_generate' adicionado pela interface
                    if param_name in ("seed", "noise_seed") or param_name.endswith("_seed"):
                        if wv_idx < len(wv) and isinstance(wv[wv_idx], str):
                            if wv[wv_idx] in ("fixed", "increment", "decrement", "randomize"):
                                wv_idx += 1

        # ---------------------------------------------------------------------
        # 4. Ajustes automáticos mandatários (Regra 3)
        # ---------------------------------------------------------------------
        if ctype == "LoadImage":
            api_node["_meta"]["title"] = "INPUT_IMAGE"

        elif ctype == "VHS_LoadVideo":
            api_node["_meta"]["title"] = "INPUT_VIDEO"
            api_node["inputs"].setdefault("frame_load_cap", 124)
            api_node["inputs"]["skip_first_frames"] = 0

        elif ctype == "RandomNoise":
            api_node["_meta"]["title"] = "SEED"

        elif ctype == "ViggleChunkedSampler":
            api_node["_meta"]["title"] = "SEED"
            api_node["inputs"]["rerender_chunk"] = 0
            api_node["inputs"]["rerender_seed"] = 0

        elif ctype == "CustomSigmas":
            api_node["inputs"]["sigmas_string"] = "1.0, 0.8571428571428571, 0.6, 0.0"
            api_node["inputs"]["interpolate_to_steps"] = 3

        elif ctype == "KSamplerSelect":
            api_node["inputs"]["sampler_name"] = "euler"

        elif ctype == "ViggleAnimateConditioningWindowed":
            api_node["inputs"]["chunk_frames"] = 124
            api_node["inputs"]["continuation"] = "five_frame_anchor"
            api_node["inputs"]["overlap_frames"] = 22

        elif ctype == "LoraLoaderModelOnly":
            api_node["inputs"]["lora_name"] = "viggle_animate_dmd_lora_r64.safetensors"

        elif ctype == "UNETLoader":
            api_node["inputs"]["unet_name"] = "minimax_h3_ref2va_viggle_pruned_int8_convrot.safetensors"

        elif ctype == "VAELoader":
            api_node["inputs"]["vae_name"] = "minimax_h3_video_vae_int8_convrot.safetensors"

        elif ctype == "ViggleTextCondLoader":
            # No ComfyUI o parâmetro chama-se 'text_cond'
            if "text_cond" in req_spec:
                api_node["inputs"]["text_cond"] = "fixed_embed_fwd_anyframe.safetensors"
            if "text_cond_name" in req_spec:
                api_node["inputs"]["text_cond_name"] = "fixed_embed_fwd_anyframe.safetensors"

        elif ctype == "VHS_VideoCombine":
            if nid == primary_output_id:
                api_node["inputs"]["save_output"] = True
            api_node["inputs"].pop("audio", None)

        api_workflow[str(nid)] = api_node

    # -------------------------------------------------------------------------
    # 5. Remoção genérica de nós de prévia de interface (ModelPreviewOverrideKJ)
    # -------------------------------------------------------------------------
    api_workflow = remove_model_preview_nodes(api_workflow)

    # -------------------------------------------------------------------------
    # 6. Remoção genérica de conexões de áudio no VHS_VideoCombine
    # -------------------------------------------------------------------------
    api_workflow = remove_video_combine_audio(api_workflow)

    return api_workflow


def remove_video_combine_audio(api_workflow: dict) -> dict:
    """
    Remove genericamente a entrada 'audio' de todos os nós VHS_VideoCombine,
    evitando que o VHS_LoadVideo tente extrair áudio do vídeo de movimento/referência.
    """
    removed_count = 0
    for nid, node in api_workflow.items():
        if node.get("class_type") == "VHS_VideoCombine":
            if "audio" in node.get("inputs", {}):
                del node["inputs"]["audio"]
                removed_count += 1
                log_info(f"Entrada 'audio' removida do nó VHS_VideoCombine [{nid}].")
    if removed_count:
        log_info(f"Total de {removed_count} nó(s) VHS_VideoCombine tiveram a entrada 'audio' removida.")
    return api_workflow


def remove_model_preview_nodes(api_workflow: dict) -> dict:
    """
    Remove genericamente todos os nós ModelPreviewOverrideKJ do workflow API,
    religando os nós consumidores da sua saída 'model' (slot 0) diretamente
    à entrada 'model' do nó de prévia removido.
    """
    while True:
        preview_nid = None
        for nid, node in api_workflow.items():
            if node.get("class_type") == "ModelPreviewOverrideKJ":
                preview_nid = nid
                break
        if preview_nid is None:
            break

        preview_node = api_workflow[preview_nid]
        upstream_model = preview_node.get("inputs", {}).get("model")
        if upstream_model is None:
            raise ValueError(
                f"Nó ModelPreviewOverrideKJ [{preview_nid}] não possui entrada 'model' para religação."
            )

        # Religar consumidores que apontam para [preview_nid, 0]
        relinked_count = 0
        for nid, node in api_workflow.items():
            if nid == preview_nid:
                continue
            for iname, ival in list(node.get("inputs", {}).items()):
                if isinstance(ival, list) and len(ival) == 2:
                    if str(ival[0]) == str(preview_nid) and ival[1] == 0:
                        node["inputs"][iname] = upstream_model
                        relinked_count += 1
                        log_info(
                            f"Religada entrada '{iname}' do nó [{nid}] ({node.get('class_type')}): "
                            f"[{preview_nid}, 0] -> {upstream_model}"
                        )

        del api_workflow[preview_nid]
        log_info(
            f"Nó ModelPreviewOverrideKJ [{preview_nid}] removido com sucesso do workflow "
            f"({relinked_count} conexão(ões) religada(s))."
        )

    return api_workflow


def validate_api_workflow(api_wf: dict, object_info: dict, raise_on_error: bool = True):
    """
    Validação estrita do workflow gerado contra as regras do sistema e o object_info do ComfyUI.
    Garante 100% de certeza antes de salvar.
    Valida:
    1. Presença dos 3 títulos obrigatórios (INPUT_IMAGE, INPUT_VIDEO, SEED)
    2. Existência do class_type no ComfyUI
    3. Presença de todas as entradas obrigatórias (required)
    4. Conformidade dos valores de campos combo contra as opções reais do ComfyUI (/object_info)
    """
    errors = []

    # 1. Validar presença dos 3 títulos obrigatórios
    required_titles = ["INPUT_IMAGE", "INPUT_VIDEO", "SEED"]
    found_titles = set()
    for nid, node in api_wf.items():
        title = node.get("_meta", {}).get("title")
        if title:
            found_titles.add(title)

    for req_title in required_titles:
        if req_title not in found_titles:
            errors.append(f"Título obrigatório '{req_title}' não encontrado em nenhum nó do workflow.")

    # 2. Validar que cada nó possui class_type existente, entradas obrigatórias e combos válidos
    for nid, node in api_wf.items():
        ctype = node.get("class_type")
        if not ctype or ctype not in object_info:
            errors.append(f"Nó [{nid}]: class_type '{ctype}' não existe no ComfyUI.")
            continue

        node_info = object_info[ctype]
        req_inputs = node_info.get("input", {}).get("required", {})
        opt_inputs = node_info.get("input", {}).get("optional", {})
        actual_inputs = node.get("inputs", {})

        for req_k in req_inputs.keys():
            if req_k not in actual_inputs:
                errors.append(f"Nó [{nid}] ({ctype}): entrada obrigatória '{req_k}' está ausente.")

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
                        errors.append(
                            f"Nó [{nid}] ({ctype}), campo '{param_name}': valor '{val}' "
                            f"não existe nas opções permitidas pelo ComfyUI. Opções válidas: {allowed_opts}"
                        )

    if errors:
        log_err("Falhas na validação do workflow gerado:")
        for err in errors:
            log_err(f"  - {err}")
        if raise_on_error:
            raise ValueError(f"Workflow API inválido ({len(errors)} problemas detectados). Abortando.")
        return False, errors

    return True, []


def main():
    parser = argparse.ArgumentParser(
        description="Converte o workflow UI oficial do Viggle-Animate para o formato API do ComfyUI."
    )
    parser.add_argument(
        "--source",
        "--workflow-in",
        dest="workflow_in",
        type=str,
        default="",
        help="Caminho local, URL ou preset ('chunked'/'longo') do workflow de interface",
    )
    parser.add_argument(
        "--output",
        "--workflow-out",
        dest="workflow_out",
        type=str,
        default="",
        help="Caminho de saída para o workflow API (default: workflows/viggle_api.json ou workflows/viggle_longo_api.json)",
    )
    parser.add_argument(
        "--comfy-url",
        type=str,
        default="http://127.0.0.1:8188",
        help="URL base da API do ComfyUI (default: http://127.0.0.1:8188)",
    )

    args = parser.parse_args()

    # Detectar preset chunked/longo
    in_lower = args.workflow_in.lower()
    out_lower = args.workflow_out.lower()
    is_chunked = ("chunked" in in_lower or "longo" in in_lower or "chunked" in out_lower or "longo" in out_lower)

    if not args.workflow_out:
        args.workflow_out = "workflows/viggle_longo_api.json" if is_chunked else "workflows/viggle_api.json"

    print("=" * 75)
    print("🛠️  BUILD VIGGLE WORKFLOW: GERADOR DE WORKFLOW API COMFYUI")
    print(f"🎯 Modo selecionado: {'LONGO (Chunked Sampler)' if is_chunked else 'PADRÃO / CURTO (Single-pass)'}")
    print(f"📂 Arquivo de saída: {args.workflow_out}")
    print("=" * 75)

    # 1. Consultar /object_info
    object_info = fetch_object_info(args.comfy_url)

    # 2. Carregar workflow UI
    try:
        wf_ui = load_ui_workflow(args.workflow_in, is_chunked=is_chunked)
    except Exception as e:
        log_err(f"Erro ao carregar workflow de interface: {e}")
        sys.exit(1)

    # 3. Converter para formato API
    try:
        api_wf = convert_workflow(wf_ui, object_info)
    except Exception as e:
        log_err(f"Erro durante a conversão do workflow: {e}")
        sys.exit(1)

    # 4. Validar integridade
    try:
        validate_api_workflow(api_wf, object_info)
    except Exception as e:
        log_err(f"Erro de validação: {e}")
        sys.exit(1)

    # 5. Salvar arquivo final
    out_dir = os.path.dirname(os.path.abspath(args.workflow_out))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.workflow_out, "w", encoding="utf-8") as f:
        json.dump(api_wf, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print(f"✅ WORKFLOW API GERADO COM SUCESSO: {args.workflow_out}")
    print(f"📊 Total de nós operacionais: {len(api_wf)}")
    print(f"🎯 Nó de Saída Final (Vídeo) : VHS_VideoCombine (save_output = True)")
    print(f"🖼️  Entrada de Foto (Image)   : LoadImage (INPUT_IMAGE)")
    print(f"🎬 Entrada de Movimento      : VHS_LoadVideo (INPUT_VIDEO, cap: 124)")
    print(f"🎲 Semente de Ruído          : RandomNoise (SEED)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
