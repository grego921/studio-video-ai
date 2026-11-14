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
