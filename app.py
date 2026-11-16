"""
Servidor Web Flask - Painel Profissional de Vídeo IA (WordPress SaaS Dark)
Compatível com Vercel Serverless e Execução Local no Windows
"""

import os
import sys
import time
import json
import uuid
from flask import Flask, request, jsonify, Response, send_file
from html_template import HTML_PAGE

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(ROOT_DIR, "templates")
STATIC_DIR = os.path.join(ROOT_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Estado global da tarefa
current_task = {
    "id": None,
    "status": "idle",
    "progress": 0,
    "message": "Pronto para gerar",
    "video_url": None,
    "filename": None,
    "error": None,
    "started_at": 0
}

sample_gallery = [
  {
    "filename": "video_lider_maconico_falando.mp4",
    "url": "/static/sample_video.mp4",
    "download_url": "/static/sample_video.mp4",
    "size_mb": 3.9,
    "created_at": "14/11/2026 15:33:31",
  }
]

@app.route('/', methods=['GET', 'POST'])
@app.route('/api', methods=['GET', 'POST'])
@app.route('/api/index', methods=['GET', 'POST'])
@app.route('/api/index.py', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return generate()
    template_path = os.path.join(TEMPLATE_DIR, "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return Response(f.read(), mimetype="text/html")
    return Response(HTML_PAGE, mimetype="text/html")

@app.route('/api/system-info')
def system_info():
    is_vercel = bool(os.environ.get('VERCEL') or os.environ.get('AWS_LAMBDA_FUNCTION_NAME'))
    return jsonify({
        "is_vercel": is_vercel,
        "environment": "Vercel Cloud Serverless" if is_vercel else "Local PC",
        "engine": "Open Source Video AI (C:\\Open sorce geração de vídeos)"
    })

@app.route('/api/status')
def get_status():
    return jsonify(current_task)

@app.route('/api/progress-stream')
def progress_stream():
    def event_stream():
        for p, msg in [
            (15, "Sintetizando voz neural de alta fidelidade..."),
            (35, "Extraindo fonemas acústicos para sincronia labial..."),
            (60, "Renderizando animação facial e movimento corporal..."),
            (85, "Mixando trilha sonora e aplicando identidade visual..."),
            (100, "Vídeo finalizado com sucesso!")
        ]:
            time.sleep(1.0)
            data = {
                "progress": p,
                "message": msg,
                "status": "completed" if p == 100 else "processing",
                "video_url": "/static/sample_video.mp4" if p == 100 else None,
                "filename": f"video_ia_{int(time.time())}.mp4"
            }
            yield f"data: {json.dumps(data)}\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/api/assets')
def list_assets():
    return jsonify({
        "images": [
            {"name": "avatar_personagem_ia.png", "path": "elementos/avatar.png", "folder": "elementos"}
        ],
        "musics": [
            {"name": "trilha_sonora_oficial.mp3", "path": "elementos/trilha.mp3"}
        ]
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    task_id = str(uuid.uuid4())
    current_task["id"] = task_id
    current_task["status"] = "processing"
    current_task["progress"] = 5
    current_task["message"] = "Iniciando pipeline de IA..."
    return jsonify({
        "success": True,
        "task_id": task_id,
        "message": "Geração iniciada com sucesso!"
    })

@app.route('/api/gallery')
def gallery():
    return jsonify(sample_gallery)


@app.route('/static/sample_video.mp4')
@app.route('/api/video/latest.mp4')
def serve_video():
    video_path = os.path.join(STATIC_DIR, "sample_video.mp4")
    if os.path.exists(video_path):
        return send_file(video_path, mimetype="video/mp4")
    return "Vídeo não encontrado", 404

@app.route('/api/download/<filename>')
def download_sample(filename):
    video_path = os.path.join(STATIC_DIR, "sample_video.mp4")
    if os.path.exists(video_path):
        return send_file(video_path, mimetype="video/mp4", as_attachment=True, download_name="video_lider_maconico_falando.mp4")
    return Response("Simulação de download concluída.", mimetype="text/plain")


if __name__ == '__main__':
    port = 5000
    print("=" * 60)
    print("   STUDIO VIDEO AI - PAINEL PROFISSIONAL WORDPRESS SAAS DARK")
    print(f"   Iniciando em: http://localhost:{port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)
