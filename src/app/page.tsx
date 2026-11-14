"use client";

import React, { useState, useRef } from "react";

export default function StudioVideoAIPage() {
  // Estados do Formulário
  const [activeTab, setActiveTab] = useState<"text" | "audio">("text");
  const [scriptText, setScriptText] = useState("");
  const [voice, setVoice] = useState("pt-BR-AntonioNeural");
  const [minDuration, setMinDuration] = useState(5);
  const [maxDuration, setMaxDuration] = useState(60);
  const [musicVolume, setMusicVolume] = useState(18);
  const [enableMusic, setEnableMusic] = useState(true);
  const [enableSpeech, setEnableSpeech] = useState(true);
  const [enableWalk, setEnableWalk] = useState(true);
  const [enableBranding, setEnableBranding] = useState(true);

  // Estados de Arquivos
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [speechFile, setSpeechFile] = useState<File | null>(null);
  const [musicFile, setMusicFile] = useState<File | null>(null);

  // Estados do Pipeline de IA
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("Pronto para gerar");
  const [activeStep, setActiveStep] = useState(1);
  const [resultVideo, setResultVideo] = useState<{ url: string; filename: string } | null>(null);

  // Galeria de Vídeos
  const [gallery, setGallery] = useState([
    {
      filename: "video_ia_apresentacao_vertical.mp4",
      url: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
      downloadUrl: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
      sizeMb: 15.3,
      createdAt: "14/11/2026 15:33",
    }
  ]);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Manipulação de Upload de Imagem
  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (evt) => {
        setImagePreview(evt.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  // Submissão e Pipeline de Geração
  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();

    if (activeTab === "text" && !scriptText.trim()) {
      alert("Por favor, digite o roteiro que a IA irá falar.");
      return;
    }

    if (activeTab === "audio" && !speechFile) {
      alert("Por favor, selecione um arquivo de áudio de fala.");
      return;
    }

    if (!imagePreview && !imageFile) {
      alert("Por favor, adicione uma imagem de referência do personagem.");
      return;
    }

    if (enableMusic && musicFile) {
      console.log("Trilha sonora personalizada anexada:", musicFile.name);
    }

    setIsGenerating(true);
    setProgress(5);
    setStatusMsg("Iniciando pipeline de inteligência artificial...");
    setActiveStep(1);

    // Simulação reativa do pipeline passo a passo
    const steps = [
      { pct: 15, msg: "Sintetizando voz neural de alta fidelidade...", step: 1, delay: 900 },
      { pct: 32, msg: "Extraindo fonemas acústicos para sincronia labial...", step: 2, delay: 1100 },
      { pct: 58, msg: "Renderizando animação facial e movimento corporal...", step: 3, delay: 1500 },
      { pct: 82, msg: "Mixando trilha sonora e aplicando identidade visual...", step: 4, delay: 1200 },
      { pct: 96, msg: "Codificando vídeo final H.264 1080x1920...", step: 5, delay: 1000 },
      { pct: 100, msg: "Vídeo finalizado com sucesso!", step: 5, delay: 600 },
    ];

    for (const s of steps) {
      await new Promise((res) => setTimeout(res, s.delay));
      setProgress(s.pct);
      setStatusMsg(s.msg);
      setActiveStep(s.step);
    }

    // Cria novo item na galeria
    const newVideo = {
      filename: `video_ia_${Date.now()}.mp4`,
      url: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
      downloadUrl: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
      sizeMb: 14.6,
      createdAt: new Date().toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" }),
    };

    setResultVideo(newVideo);
    setGallery((prev) => [newVideo, ...prev]);
    setIsGenerating(false);
  };

  return (
    <div className="wp-root-canvas">
      {/* =========================================================
          ESTILOS CSS EMBUTIDOS (WORDPRESS SAAS DARK - 100% RESPONSIVO)
          ========================================================= */}
      <style>{`
        :root {
          --bg-main: #090d16;
          --bg-surface: #101726;
          --bg-card: #182234;
          --bg-card-hover: #1e2b42;
          --bg-input: #0b111e;
          
          --primary: #6366f1;
          --primary-glow: rgba(99, 102, 241, 0.35);
          --accent: #06b6d4;
          --accent-glow: rgba(6, 182, 212, 0.35);
          --success: #10b981;
          
          --border: rgba(255, 255, 255, 0.08);
          --text-main: #f9fafb;
          --text-muted: #94a3b8;
          --text-subtle: #64748b;
          
          --radius-sm: 8px;
          --radius-md: 14px;
          --radius-lg: 20px;
        }

        * {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }

        .wp-root-canvas {
          background-color: var(--bg-main);
          color: var(--text-main);
          font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          line-height: 1.5;
        }

        /* Trava fixa de SVG: NUNCA FICARÁ GIGANTE */
        svg {
          display: inline-block;
          vertical-align: middle;
          max-width: 100%;
          height: auto;
          flex-shrink: 0;
        }

        .svg-icon-sm { width: 18px; height: 18px; }
        .svg-icon-md { width: 22px; height: 22px; }
        .svg-icon-lg { width: 36px; height: 36px; }

        /* Navbar */
        .wp-header {
          background: rgba(16, 23, 38, 0.95);
          backdrop-filter: blur(16px);
          border-bottom: 1px solid var(--border);
          position: sticky;
          top: 0;
          z-index: 100;
          padding: 14px 28px;
        }

        .wp-header-inner {
          max-width: 1320px;
          margin: 0 auto;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
        }

        .wp-brand {
          display: flex;
          align-items: center;
          gap: 12px;
          text-decoration: none;
          color: inherit;
        }

        .wp-logo-icon {
          width: 42px;
          height: 42px;
          border-radius: var(--radius-md);
          background: linear-gradient(135deg, var(--primary), var(--accent));
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 0 16px var(--primary-glow);
        }

        .wp-logo-icon svg {
          width: 22px;
          height: 22px;
          fill: #ffffff;
        }

        .wp-brand-title {
          font-size: 1.3rem;
          font-weight: 800;
          letter-spacing: -0.02em;
        }

        .wp-brand-title span {
          background: linear-gradient(90deg, #818cf8, #06b6d4);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .wp-badge-pro {
          font-size: 0.68rem;
          font-weight: 800;
          background: rgba(99, 102, 241, 0.15);
          border: 1px solid var(--primary);
          color: #a5b4fc;
          padding: 2px 8px;
          border-radius: 9999px;
          margin-left: 8px;
          text-transform: uppercase;
        }

        .wp-status-pill {
          display: flex;
          align-items: center;
          gap: 8px;
          background: rgba(16, 185, 129, 0.1);
          border: 1px solid rgba(16, 185, 129, 0.25);
          padding: 6px 14px;
          border-radius: 9999px;
          font-size: 0.82rem;
          font-weight: 600;
          color: #34d399;
        }

        .wp-status-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #34d399;
          box-shadow: 0 0 10px #34d399;
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.2); }
        }

        /* Hero */
        .wp-hero {
          max-width: 1320px;
          margin: 28px auto 0;
          padding: 0 24px;
          text-align: center;
        }

        .wp-hero-badge {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          background: rgba(99, 102, 241, 0.12);
          border: 1px solid rgba(99, 102, 241, 0.25);
          color: #a5b4fc;
          font-size: 0.78rem;
          font-weight: 700;
          padding: 4px 14px;
          border-radius: 9999px;
          margin-bottom: 12px;
          text-transform: uppercase;
        }

        .wp-hero-title {
          font-size: 2.2rem;
          font-weight: 800;
          letter-spacing: -0.03em;
          margin-bottom: 8px;
        }

        .wp-hero-title span {
          background: linear-gradient(90deg, #a5b4fc, #38bdf8);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .wp-hero-sub {
          font-size: 0.98rem;
          color: var(--text-muted);
          max-width: 720px;
          margin: 0 auto;
        }

        /* Layout Grid */
        .wp-main-grid {
          max-width: 1320px;
          width: 100%;
          margin: 32px auto 56px;
          padding: 0 24px;
          display: grid;
          grid-template-columns: 1.15fr 0.85fr;
          gap: 32px;
          align-items: start;
        }

        @media (max-width: 1024px) {
          .wp-main-grid {
            grid-template-columns: 1fr;
          }
          .wp-hero-title {
            font-size: 1.8rem;
          }
        }

        /* Card WordPress */
        .wp-card-box {
          background: var(--bg-surface);
          border: 1px solid var(--border);
          border-radius: var(--radius-lg);
          padding: 24px;
          margin-bottom: 24px;
          box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
          transition: all 0.2s ease;
        }

        .wp-card-box:hover {
          border-color: rgba(255, 255, 255, 0.14);
        }

        .wp-card-header {
          display: flex;
          align-items: center;
          gap: 14px;
          margin-bottom: 20px;
        }

        .wp-step-pill {
          width: 32px;
          height: 32px;
          border-radius: 9px;
          background: linear-gradient(135deg, var(--primary), var(--accent));
          color: #fff;
          font-size: 0.9rem;
          font-weight: 800;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 2px 8px var(--primary-glow);
        }

        .wp-card-title {
          font-size: 1.12rem;
          font-weight: 700;
          color: var(--text-main);
        }

        .wp-card-desc {
          font-size: 0.82rem;
          color: var(--text-subtle);
        }

        /* Dropzone */
        .wp-dropzone {
          border: 2px dashed rgba(255, 255, 255, 0.16);
          background: rgba(11, 17, 30, 0.5);
          border-radius: var(--radius-md);
          padding: 24px 16px;
          text-align: center;
          cursor: pointer;
          position: relative;
          transition: all 0.2s ease;
        }

        .wp-dropzone:hover {
          border-color: var(--accent);
          background: rgba(6, 182, 212, 0.05);
        }

        .wp-dropzone-icon {
          width: 44px;
          height: 44px;
          margin: 0 auto 10px;
          color: var(--accent);
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .wp-dropzone-icon svg {
          width: 34px;
          height: 34px;
        }

        .wp-drop-title {
          font-size: 0.95rem;
          font-weight: 600;
          margin-bottom: 4px;
        }

        .wp-drop-title span {
          color: var(--accent);
        }

        .wp-drop-hint {
          font-size: 0.76rem;
          color: var(--text-subtle);
        }

        .wp-preview-box {
          margin-top: 14px;
          text-align: center;
        }

        .wp-preview-box img {
          max-width: 100%;
          max-height: 240px;
          border-radius: var(--radius-sm);
          border: 1px solid var(--border);
          object-fit: cover;
        }

        /* Tabs */
        .wp-tabs {
          display: flex;
          gap: 8px;
          background: var(--bg-input);
          padding: 4px;
          border-radius: var(--radius-sm);
          margin-bottom: 16px;
          border: 1px solid var(--border);
        }

        .wp-tab-btn {
          flex: 1;
          padding: 8px 14px;
          border: none;
          background: transparent;
          color: var(--text-muted);
          font-size: 0.85rem;
          font-weight: 600;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .wp-tab-btn.active {
          background: var(--primary);
          color: #ffffff;
        }

        /* Inputs */
        .wp-form-group {
          margin-bottom: 14px;
        }

        .wp-label {
          display: block;
          font-size: 0.85rem;
          font-weight: 600;
          color: var(--text-muted);
          margin-bottom: 6px;
        }

        textarea, select, input[type="file"] {
          width: 100%;
          background: var(--bg-input);
          border: 1px solid var(--border);
          border-radius: var(--radius-sm);
          color: var(--text-main);
          padding: 10px 14px;
          font-size: 0.9rem;
          font-family: inherit;
        }

        textarea:focus, select:focus {
          outline: none;
          border-color: var(--primary);
          box-shadow: 0 0 0 3px var(--primary-glow);
        }

        /* Sliders */
        .wp-range-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
        }

        @media (max-width: 600px) {
          .wp-range-grid {
            grid-template-columns: 1fr;
          }
        }

        .wp-range-card {
          background: var(--bg-input);
          border: 1px solid var(--border);
          padding: 14px;
          border-radius: var(--radius-md);
        }

        .wp-range-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }

        .wp-range-badge {
          font-family: monospace;
          font-size: 0.85rem;
          font-weight: 700;
          color: var(--accent);
          background: rgba(6, 182, 212, 0.1);
          padding: 2px 8px;
          border-radius: 6px;
        }

        input[type="range"] {
          width: 100%;
          -webkit-appearance: none;
          height: 6px;
          background: rgba(255, 255, 255, 0.15);
          border-radius: 4px;
          outline: none;
        }

        input[type="range"]::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background: var(--accent);
          cursor: pointer;
        }

        /* Toggles */
        .wp-toggle-row {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 14px;
          background: var(--bg-input);
          border: 1px solid var(--border);
          border-radius: var(--radius-sm);
          margin-bottom: 10px;
        }

        .wp-toggle-info strong {
          display: block;
          font-size: 0.88rem;
          color: var(--text-main);
        }

        .wp-toggle-info span {
          font-size: 0.75rem;
          color: var(--text-subtle);
        }

        .wp-switch {
          position: relative;
          display: inline-block;
          width: 44px;
          height: 24px;
          flex-shrink: 0;
        }

        .wp-switch input {
          opacity: 0;
          width: 0;
          height: 0;
        }

        .wp-switch-slider {
          position: absolute;
          cursor: pointer;
          top: 0; left: 0; right: 0; bottom: 0;
          background-color: rgba(255, 255, 255, 0.2);
          transition: .3s;
          border-radius: 24px;
        }

        .wp-switch-slider:before {
          position: absolute;
          content: "";
          height: 18px;
          width: 18px;
          left: 3px;
          bottom: 3px;
          background-color: white;
          transition: .3s;
          border-radius: 50%;
        }

        input:checked + .wp-switch-slider {
          background: linear-gradient(90deg, var(--primary), var(--accent));
        }

        input:checked + .wp-switch-slider:before {
          transform: translateX(20px);
        }

        /* Botão Gerar */
        .wp-btn-generate {
          width: 100%;
          padding: 16px 24px;
          border: none;
          border-radius: var(--radius-md);
          background: linear-gradient(135deg, var(--primary) 0%, #8b5cf6 50%, var(--accent) 100%);
          color: #ffffff;
          font-size: 1.05rem;
          font-weight: 800;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 12px;
          box-shadow: 0 10px 25px var(--primary-glow);
          transition: all 0.2s ease;
        }

        .wp-btn-generate:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 15px 35px var(--primary-glow);
        }

        .wp-btn-generate:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .wp-btn-generate svg {
          width: 22px;
          height: 22px;
          fill: currentColor;
        }

        /* Barra de Progresso */
        .wp-progress-card {
          background: rgba(16, 23, 38, 0.95);
          border: 1px solid var(--accent);
          border-radius: var(--radius-lg);
          padding: 24px;
          margin-bottom: 24px;
          box-shadow: 0 0 25px var(--accent-glow);
        }

        .wp-progress-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 12px;
        }

        .wp-spin-ring {
          width: 24px;
          height: 24px;
          border: 3px solid rgba(6, 182, 212, 0.2);
          border-top-color: var(--accent);
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .wp-pct-text {
          font-family: monospace;
          font-size: 1.4rem;
          font-weight: 800;
          color: var(--accent);
        }

        .wp-track {
          width: 100%;
          height: 10px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 9999px;
          overflow: hidden;
          margin-bottom: 14px;
        }

        .wp-bar-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--primary), var(--accent), #10b981);
          border-radius: 9999px;
          transition: width 0.35s ease;
        }

        .wp-steps-row {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }

        .wp-step-chip {
          font-size: 0.72rem;
          font-weight: 600;
          padding: 4px 10px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid var(--border);
          border-radius: 6px;
          color: var(--text-subtle);
        }

        .wp-step-chip.active {
          border-color: var(--accent);
          color: var(--accent);
          background: rgba(6, 182, 212, 0.1);
        }

        .wp-step-chip.done {
          border-color: var(--success);
          color: var(--success);
          background: rgba(16, 185, 129, 0.1);
        }

        /* Player & Download */
        .wp-player-box {
          background: #000;
          border-radius: var(--radius-md);
          overflow: hidden;
          aspect-ratio: 9/16;
          max-height: 480px;
          margin: 14px auto;
          border: 1px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .wp-player-box video {
          width: 100%;
          height: 100%;
          object-fit: contain;
        }

        .wp-btn-download {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          width: 100%;
          padding: 14px 20px;
          background: linear-gradient(135deg, #10b981, #059669);
          border: none;
          border-radius: var(--radius-sm);
          color: #ffffff;
          text-decoration: none;
          font-size: 0.95rem;
          font-weight: 700;
          box-shadow: 0 6px 18px rgba(16, 185, 129, 0.3);
          transition: all 0.2s ease;
        }

        .wp-btn-download:hover {
          transform: translateY(-2px);
          box-shadow: 0 10px 25px rgba(16, 185, 129, 0.4);
        }

        .wp-btn-download svg {
          width: 20px;
          height: 20px;
        }

        /* Galeria */
        .wp-gallery-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
          gap: 14px;
        }

        .wp-gallery-item {
          background: var(--bg-input);
          border: 1px solid var(--border);
          border-radius: var(--radius-sm);
          overflow: hidden;
          display: flex;
          flex-direction: column;
          transition: all 0.2s ease;
        }

        .wp-gallery-item:hover {
          border-color: var(--accent);
          transform: translateY(-2px);
        }

        .wp-thumb-wrap {
          aspect-ratio: 9/16;
          background: #000;
          position: relative;
          cursor: pointer;
        }

        .wp-thumb-wrap video {
          width: 100%;
          height: 100%;
          object-fit: cover;
        }

        .wp-play-icon {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          width: 38px;
          height: 38px;
          background: rgba(0, 0, 0, 0.75);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          color: #fff;
        }

        .wp-play-icon svg {
          width: 18px;
          height: 18px;
          fill: currentColor;
        }

        .wp-gallery-meta {
          padding: 10px;
          font-size: 0.75rem;
        }

        .wp-meta-title {
          font-weight: 700;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          margin-bottom: 4px;
        }

        .wp-meta-info {
          color: var(--text-subtle);
          display: flex;
          justify-content: space-between;
          margin-bottom: 8px;
        }

        .wp-btn-item-dl {
          display: block;
          padding: 6px;
          text-align: center;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid var(--border);
          border-radius: 4px;
          color: var(--text-muted);
          text-decoration: none;
          font-weight: 600;
          font-size: 0.72rem;
        }

        .wp-btn-item-dl:hover {
          background: var(--primary);
          color: #fff;
        }

        .wp-footer-bar {
          margin-top: auto;
          border-top: 1px solid var(--border);
          background: var(--bg-surface);
          padding: 20px 24px;
          text-align: center;
          font-size: 0.8rem;
          color: var(--text-subtle);
        }
      `}</style>

      {/* Header Superior */}
      <header className="wp-header">
        <div className="wp-header-inner">
          <div className="wp-brand">
            <div className="wp-logo-icon">
              <svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
            </div>
            <div>
              <div className="wp-brand-title">
                Studio <span>Video AI</span> <span className="wp-badge-pro">v2.0</span>
              </div>
            </div>
          </div>
          <div className="wp-status-pill">
            <span className="wp-status-dot"></span>
            <span>Motor de IA Ativo • Vercel Ready</span>
          </div>
        </div>
      </header>

      {/* Hero Header */}
      <section className="wp-hero">
        <div className="wp-hero-badge">IA de Geração de Vídeo • 1080x1920 Vertical</div>
        <h1 className="wp-hero-title">Painel de Criação e <span>Edição de Vídeo</span></h1>
        <p className="wp-hero-sub">
          Transforme imagens estáticas em vídeos realistas com fala sincronizada, expressões faciais,
          movimento corporal e trilha sonora profissional.
        </p>
      </section>

      {/* Layout Grid Responsivo */}
      <main className="wp-main-grid">
        {/* Coluna Esquerda: Formulário de Configuração */}
        <section>
          <form onSubmit={handleGenerate}>
            {/* 1. Imagem de Referência */}
            <div className="wp-card-box">
              <div className="wp-card-header">
                <div className="wp-step-pill">1</div>
                <div>
                  <h2 className="wp-card-title">Imagem de Referência</h2>
                  <p className="wp-card-desc">Carregue a foto do rosto/personagem para animar</p>
                </div>
              </div>

              <div
                className="wp-dropzone"
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  accept="image/*"
                  style={{ display: "none" }}
                  onChange={handleImageChange}
                />
                {!imagePreview ? (
                  <div>
                    <div className="wp-dropzone-icon">
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                      </svg>
                    </div>
                    <div className="wp-drop-title">Arraste uma foto ou <span>clique para escolher</span></div>
                    <div className="wp-drop-hint">Suporta PNG, JPG, JFIF, WEBP (Ideal: 9:16 portrait)</div>
                  </div>
                ) : (
                  <div className="wp-preview-box">
                    <img src={imagePreview} alt="Pré-visualização" />
                    <p className="wp-drop-hint" style={{ marginTop: "8px" }}>Clique para trocar a imagem</p>
                  </div>
                )}
              </div>
            </div>

            {/* 2. Roteiro ou Áudio de Fala */}
            <div className="wp-card-box">
              <div className="wp-card-header">
                <div className="wp-step-pill">2</div>
                <div>
                  <h2 className="wp-card-title">Roteiro ou Áudio de Fala</h2>
                  <p className="wp-card-desc">Insira o que a IA vai falar ou suba uma gravação</p>
                </div>
              </div>

              <div className="wp-tabs">
                <button
                  type="button"
                  className={`wp-tab-btn ${activeTab === "text" ? "active" : ""}`}
                  onClick={() => setActiveTab("text")}
                >
                  Texto (Voz da IA)
                </button>
                <button
                  type="button"
                  className={`wp-tab-btn ${activeTab === "audio" ? "active" : ""}`}
                  onClick={() => setActiveTab("audio")}
                >
                  Upload de Áudio
                </button>
              </div>

              {activeTab === "text" ? (
                <div>
                  <div className="wp-form-group">
                    <label className="wp-label">Texto do Roteiro / Fala:</label>
                    <textarea
                      rows={4}
                      value={scriptText}
                      onChange={(e) => setScriptText(e.target.value)}
                      placeholder="Digite aqui o que a imagem deve falar... Ex: Não confunda êxito com superioridade. A verdadeira força está na humildade e no respeito mútuo."
                    ></textarea>
                  </div>

                  <div className="wp-form-group">
                    <label className="wp-label">Voz da IA (Neural):</label>
                    <select value={voice} onChange={(e) => setVoice(e.target.value)}>
                      <optgroup label="Português do Brasil (IA)">
                        <option value="pt-BR-AntonioNeural">Antonio (Masculino - Firme e Profissional)</option>
                        <option value="pt-BR-FranciscaNeural">Francisca (Feminino - Clara e Expressiva)</option>
                        <option value="pt-BR-ThalitaNeural">Thalita (Feminino - Suave e Jovem)</option>
                      </optgroup>
                      <optgroup label="Inglês (Internacional)">
                        <option value="en-US-GuyNeural">Guy (Masculino Natural)</option>
                        <option value="en-US-JennyNeural">Jenny (Feminino Confiante)</option>
                      </optgroup>
                    </select>
                  </div>
                </div>
              ) : (
                <div className="wp-form-group">
                  <label className="wp-label">Arquivo de Áudio de Fala (MP3, WAV, M4A):</label>
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={(e) => setSpeechFile(e.target.files ? e.target.files[0] : null)}
                  />
                  <p className="wp-drop-hint" style={{ marginTop: "6px" }}>
                    A IA sincronizará lábios e expressões com base no áudio carregado.
                  </p>
                </div>
              )}
            </div>

            {/* 3. Trilha Sonora de Fundo */}
            <div className="wp-card-box">
              <div className="wp-card-header">
                <div className="wp-step-pill">3</div>
                <div>
                  <h2 className="wp-card-title">Trilha Sonora / Fundo Musical</h2>
                  <p className="wp-card-desc">Mixagem automática da música ambiente com a voz</p>
                </div>
              </div>

              <div className="wp-toggle-row">
                <div className="wp-toggle-info">
                  <strong>Ativar Trilha Sonora de Fundo</strong>
                  <span>Insere fundo musical com ducking automático suave</span>
                </div>
                <label className="wp-switch">
                  <input
                    type="checkbox"
                    checked={enableMusic}
                    onChange={(e) => setEnableMusic(e.target.checked)}
                  />
                  <span className="wp-switch-slider"></span>
                </label>
              </div>

              {enableMusic && (
                <div style={{ marginTop: "14px" }}>
                  <div className="wp-form-group">
                    <label className="wp-label">Arquivo de Música:</label>
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={(e) => setMusicFile(e.target.files ? e.target.files[0] : null)}
                    />
                  </div>

                  <div className="wp-range-card">
                    <div className="wp-range-header">
                      <span className="wp-label" style={{ margin: 0 }}>Volume da Trilha:</span>
                      <span className="wp-range-badge">{musicVolume}%</span>
                    </div>
                    <input
                      type="range"
                      min={5}
                      max={50}
                      value={musicVolume}
                      onChange={(e) => setMusicVolume(Number(e.target.value))}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* 4. Configuração de Tempo (Mínimo e Máximo) */}
            <div className="wp-card-box">
              <div className="wp-card-header">
                <div className="wp-step-pill">4</div>
                <div>
                  <h2 className="wp-card-title">Controle de Duração do Vídeo</h2>
                  <p className="wp-card-desc">Defina o tempo mínimo e máximo em segundos</p>
                </div>
              </div>

              <div className="wp-range-grid">
                <div className="wp-range-card">
                  <div className="wp-range-header">
                    <span className="wp-label" style={{ margin: 0 }}>Tempo Mínimo:</span>
                    <span className="wp-range-badge">{minDuration}s</span>
                  </div>
                  <input
                    type="range"
                    min={2}
                    max={60}
                    value={minDuration}
                    onChange={(e) => {
                      const val = Number(e.target.value);
                      setMinDuration(val);
                      if (val > maxDuration) setMaxDuration(val);
                    }}
                  />
                </div>

                <div className="wp-range-card">
                  <div className="wp-range-header">
                    <span className="wp-label" style={{ margin: 0 }}>Tempo Máximo:</span>
                    <span className="wp-range-badge">{maxDuration}s</span>
                  </div>
                  <input
                    type="range"
                    min={5}
                    max={180}
                    value={maxDuration}
                    onChange={(e) => {
                      const val = Number(e.target.value);
                      setMaxDuration(val);
                      if (val < minDuration) setMinDuration(val);
                    }}
                  />
                </div>
              </div>
            </div>

            {/* 5. Recursos de IA */}
            <div className="wp-card-box">
              <div className="wp-card-header">
                <div className="wp-step-pill">5</div>
                <div>
                  <h2 className="wp-card-title">Recursos de IA e Efeitos</h2>
                  <p className="wp-card-desc">Ative movimentos corporais, sincronia labial e moldura</p>
                </div>
              </div>

              <div className="wp-toggle-row">
                <div className="wp-toggle-info">
                  <strong>Sincronização Labial (Fazer Falar)</strong>
                  <span>Modulação labial baseada nos fonemas acústicos</span>
                </div>
                <label className="wp-switch">
                  <input
                    type="checkbox"
                    checked={enableSpeech}
                    onChange={(e) => setEnableSpeech(e.target.checked)}
                  />
                  <span className="wp-switch-slider"></span>
                </label>
              </div>

              <div className="wp-toggle-row">
                <div className="wp-toggle-info">
                  <strong>Movimento Corporal / Andar e Respiração</strong>
                  <span>Simulação orgânica de passos, oscilação e respiração</span>
                </div>
                <label className="wp-switch">
                  <input
                    type="checkbox"
                    checked={enableWalk}
                    onChange={(e) => setEnableWalk(e.target.checked)}
                  />
                  <span className="wp-switch-slider"></span>
                </label>
              </div>

              <div className="wp-toggle-row">
                <div className="wp-toggle-info">
                  <strong>Aplicar Moldura 9:16 e Logo Oficial</strong>
                  <span>Adiciona moldura vertical e logotipo no canto superior</span>
                </div>
                <label className="wp-switch">
                  <input
                    type="checkbox"
                    checked={enableBranding}
                    onChange={(e) => setEnableBranding(e.target.checked)}
                  />
                  <span className="wp-switch-slider"></span>
                </label>
              </div>
            </div>

            {/* Botão Gerar */}
            <button
              type="submit"
              disabled={isGenerating}
              className="wp-btn-generate"
            >
              <svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
              <span>{isGenerating ? "Processando com IA..." : "Gerar Vídeo Agora"}</span>
            </button>
          </form>
        </section>

        {/* Coluna Direita: Progresso em Tempo Real, Player e Galeria */}
        <section>
          {/* Card de Progresso */}
          {isGenerating && (
            <div className="wp-progress-card">
              <div className="wp-progress-top">
                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                  <div className="wp-spin-ring"></div>
                  <div>
                    <strong style={{ fontSize: "0.95rem", display: "block" }}>Processando Vídeo...</strong>
                    <span style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>{statusMsg}</span>
                  </div>
                </div>
                <div className="wp-pct-text">{progress}%</div>
              </div>

              <div className="wp-track">
                <div className="wp-bar-fill" style={{ width: `${progress}%` }}></div>
              </div>

              <div className="wp-steps-row">
                <span className={`wp-step-chip ${activeStep === 1 ? "active" : activeStep > 1 ? "done" : ""}`}>1. Voz IA</span>
                <span className={`wp-step-chip ${activeStep === 2 ? "active" : activeStep > 2 ? "done" : ""}`}>2. Fonemas</span>
                <span className={`wp-step-chip ${activeStep === 3 ? "active" : activeStep > 3 ? "done" : ""}`}>3. Animação Facial</span>
                <span className={`wp-step-chip ${activeStep === 4 ? "active" : activeStep > 4 ? "done" : ""}`}>4. Mixagem</span>
                <span className={`wp-step-chip ${activeStep === 5 ? "active" : activeStep > 5 ? "done" : ""}`}>5. Renderização</span>
              </div>
            </div>
          )}

          {/* Player de Vídeo Concluído */}
          {resultVideo && (
            <div className="wp-card-box">
              <div style={{ marginBottom: "10px" }}>
                <span className="wp-hero-badge" style={{ background: "rgba(16, 185, 129, 0.15)", color: "#34d399", borderColor: "#10b981", margin: 0 }}>
                  Novo Vídeo Concluído
                </span>
              </div>
              <div className="wp-player-box">
                <video src={resultVideo.url} controls autoPlay playsInline></video>
              </div>
              <a
                href={resultVideo.url}
                download={resultVideo.filename}
                target="_blank"
                rel="noreferrer"
                className="wp-btn-download"
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="7 10 12 15 17 10"></polyline>
                  <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                Baixar Vídeo Finalizado
              </a>
            </div>
          )}

          {/* Galeria de Vídeos Gerados */}
          <div className="wp-card-box">
            <div className="wp-card-header">
              <div>
                <h2 className="wp-card-title">Galeria de Vídeos Gerados</h2>
                <p className="wp-card-desc">{gallery.length} vídeo{gallery.length !== 1 ? "s" : ""} disponível{gallery.length !== 1 ? "is" : ""}</p>
              </div>
            </div>

            <div className="wp-gallery-grid">
              {gallery.map((v, i) => (
                <div key={i} className="wp-gallery-item">
                  <div
                    className="wp-thumb-wrap"
                    onClick={() => setResultVideo({ url: v.url, filename: v.filename })}
                  >
                    <video src={`${v.url}#t=0.5`} preload="metadata"></video>
                    <div className="wp-play-icon">
                      <svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                    </div>
                  </div>
                  <div className="wp-gallery-meta">
                    <div className="wp-meta-title" title={v.filename}>{v.filename}</div>
                    <div className="wp-meta-info">
                      <span>{v.createdAt}</span>
                      <span>{v.sizeMb} MB</span>
                    </div>
                    <a
                      href={v.downloadUrl}
                      download={v.filename}
                      target="_blank"
                      rel="noreferrer"
                      className="wp-btn-item-dl"
                    >
                      Baixar MP4
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      <footer className="wp-footer-bar">
        <p>Studio Video AI • Plataforma de Geração e Animação Audiovisual Open Source</p>
      </footer>
    </div>
  );
}
