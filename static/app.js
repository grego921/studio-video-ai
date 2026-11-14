/**
 * Frontend Studio Video AI - Interações e SSE em Tempo Real
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elementos do DOM
  const videoForm = document.getElementById('videoForm');
  const generateBtn = document.getElementById('generateBtn');
  const imageInput = document.getElementById('imageInput');
  const imageDropZone = document.getElementById('imageDropZone');
  const imageDropContent = document.getElementById('imageDropContent');
  const imagePreviewBox = document.getElementById('imagePreviewBox');
  const imagePreview = document.getElementById('imagePreview');
  const removeImageBtn = document.getElementById('removeImageBtn');
  const presetImagesList = document.getElementById('presetImagesList');
  const presetMusicList = document.getElementById('presetMusicList');
  
  // Duração
  const minDurationInput = document.getElementById('minDurationInput');
  const minDurationSlider = document.getElementById('minDurationSlider');
  const maxDurationInput = document.getElementById('maxDurationInput');
  const maxDurationSlider = document.getElementById('maxDurationSlider');

  // Trilha sonora
  const musicVolume = document.getElementById('musicVolume');
  const musicVolumeValue = document.getElementById('musicVolumeValue');
  const enableMusicCheck = document.getElementById('enableMusicCheck');
  const musicControls = document.getElementById('musicControls');

  // Progresso
  const progressCard = document.getElementById('progressCard');
  const progressPercent = document.getElementById('progressPercent');
  const progressBarFill = document.getElementById('progressBarFill');
  const progressStatusText = document.getElementById('progressStatusText');
  const stepVoice = document.getElementById('stepVoice');
  const stepAudio = document.getElementById('stepAudio');
  const stepMotion = document.getElementById('stepMotion');
  const stepMix = document.getElementById('stepMix');
  const stepRender = document.getElementById('stepRender');

  // Resultado e Galeria
  const resultCard = document.getElementById('resultCard');
  const resultVideo = document.getElementById('resultVideo');
  const downloadResultBtn = document.getElementById('downloadResultBtn');
  const galleryGrid = document.getElementById('galleryGrid');
  const galleryCount = document.getElementById('galleryCount');

  let selectedLocalImage = null;
  let selectedLocalMusic = null;
  let progressSource = null;

  // --- Sincronização de Sliders de Duração ---
  function syncDurationInputs() {
    minDurationSlider.addEventListener('input', (e) => {
      minDurationInput.value = e.target.value;
      if (parseInt(minDurationInput.value) > parseInt(maxDurationInput.value)) {
        maxDurationInput.value = e.target.value;
        maxDurationSlider.value = e.target.value;
      }
    });

    minDurationInput.addEventListener('change', (e) => {
      let val = Math.max(2, Math.min(120, parseInt(e.target.value) || 2));
      minDurationInput.value = val;
      minDurationSlider.value = val;
      if (val > parseInt(maxDurationInput.value)) {
        maxDurationInput.value = val;
        maxDurationSlider.value = val;
      }
    });

    maxDurationSlider.addEventListener('input', (e) => {
      maxDurationInput.value = e.target.value;
      if (parseInt(maxDurationInput.value) < parseInt(minDurationInput.value)) {
        minDurationInput.value = e.target.value;
        minDurationSlider.value = e.target.value;
      }
    });

    maxDurationInput.addEventListener('change', (e) => {
      let val = Math.max(2, Math.min(300, parseInt(e.target.value) || 30));
      maxDurationInput.value = val;
      maxDurationSlider.value = val;
      if (val < parseInt(minDurationInput.value)) {
        minDurationInput.value = val;
        minDurationSlider.value = val;
      }
    });

    // Volume da música
    musicVolume.addEventListener('input', (e) => {
      musicVolumeValue.textContent = `${e.target.value}%`;
    });

    enableMusicCheck.addEventListener('change', (e) => {
      musicControls.style.display = e.target.checked ? 'block' : 'none';
    });
  }

  // --- Abas de Fala (Texto vs Áudio) ---
  const tabBtns = document.querySelectorAll('.tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add('active');
    });
  });

  // --- Upload e Drag-and-Drop de Imagem ---
  function setupImageUpload() {
    imageInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        previewImageFile(e.target.files[0]);
      }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
      imageDropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        imageDropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      imageDropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        imageDropZone.classList.remove('dragover');
      });
    });

    imageDropZone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files && files.length > 0 && files[0].type.startsWith('image/')) {
        imageInput.files = files;
        previewImageFile(files[0]);
      }
    });

    removeImageBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      clearImage();
    });
  }

  function previewImageFile(file) {
    selectedLocalImage = null;
    clearChipSelection(presetImagesList);
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      imageDropContent.classList.add('hidden');
      imagePreviewBox.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
  }

  function clearImage() {
    imageInput.value = '';
    imagePreview.src = '';
    selectedLocalImage = null;
    imageDropContent.classList.remove('hidden');
    imagePreviewBox.classList.add('hidden');
    clearChipSelection(presetImagesList);
  }

  function clearChipSelection(container) {
    container.querySelectorAll('.preset-chip').forEach(c => c.classList.remove('active'));
  }

  // --- Carregamento de Assets Locais Pré-Existentes ---
  async function loadLocalAssets() {
    try {
      const res = await fetch('/api/assets');
      const data = await res.json();

      // Renderiza presets de imagens
      if (data.images && data.images.length > 0) {
        presetImagesList.innerHTML = '';
        data.images.forEach((img, idx) => {
          const chip = document.createElement('button');
          chip.type = 'button';
          chip.className = 'preset-chip';
          chip.textContent = `${img.name} (${img.folder})`;
          chip.title = img.path;

          chip.addEventListener('click', () => {
            clearChipSelection(presetImagesList);
            chip.classList.add('active');
            selectedLocalImage = img.path;
            imageInput.value = '';
            
            // Exibe preview
            const folderUrl = img.folder.replace(' ', '_');
            imagePreview.src = `/api/asset-file/${folderUrl}/${encodeURIComponent(img.name)}`;
            imageDropContent.classList.add('hidden');
            imagePreviewBox.classList.remove('hidden');
          });

          // Seleciona o primeiro por padrão se existir
          if (idx === 0) {
            chip.click();
          }

          presetImagesList.appendChild(chip);
        });
      } else {
        presetImagesList.innerHTML = '<span class="drop-hint">Nenhuma imagem na pasta local.</span>';
      }

      // Renderiza presets de trilhas
      if (data.musics && data.musics.length > 0) {
        presetMusicList.innerHTML = '';
        data.musics.forEach((music, idx) => {
          const chip = document.createElement('button');
          chip.type = 'button';
          chip.className = 'preset-chip';
          chip.textContent = music.name;
          chip.title = music.path;

          chip.addEventListener('click', () => {
            clearChipSelection(presetMusicList);
            chip.classList.add('active');
            selectedLocalMusic = music.path;
            document.getElementById('musicInput').value = '';
          });

          if (idx === 0) {
            chip.click();
          }

          presetMusicList.appendChild(chip);
        });
      } else {
        presetMusicList.innerHTML = '<span class="drop-hint">Nenhuma trilha encontrada em elementos/.</span>';
      }
    } catch (err) {
      console.error('Erro ao carregar assets locais:', err);
    }
  }

  // --- Carregamento e Atualização da Galeria ---
  async function loadGallery() {
    try {
      const res = await fetch('/api/gallery');
      const videos = await res.json();

      galleryCount.textContent = `${videos.length} vídeo${videos.length !== 1 ? 's' : ''}`;

      if (videos.length === 0) {
        galleryGrid.innerHTML = '<div class="gallery-empty">Nenhum vídeo gerado ainda. Configure os parâmetros e clique em "Gerar Vídeo Agora"!</div>';
        return;
      }

      galleryGrid.innerHTML = '';
      videos.forEach(v => {
        const item = document.createElement('div');
        item.className = 'gallery-item';
        item.innerHTML = `
          <div class="gallery-preview" title="Clique para reproduzir">
            <video preload="metadata" src="${v.url}#t=0.5"></video>
            <div class="gallery-overlay-play">
              <svg viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
            </div>
          </div>
          <div class="gallery-meta">
            <div class="gallery-title" title="${v.filename}">${v.filename}</div>
            <div class="gallery-info">
              <span>${v.created_at}</span>
              <span>${v.size_mb} MB</span>
            </div>
            <a href="${v.download_url}" download class="gallery-download-btn">Baixar Vídeo</a>
          </div>
        `;

        // Ao clicar no preview, carrega no player principal
        item.querySelector('.gallery-preview').addEventListener('click', () => {
          showResultVideo(v.url, v.filename);
        });

        galleryGrid.appendChild(item);
      });
    } catch (err) {
      console.error('Erro ao carregar galeria:', err);
    }
  }

  function showResultVideo(url, filename) {
    resultVideo.src = url;
    downloadResultBtn.href = `/api/download/${encodeURIComponent(filename)}`;
    downloadResultBtn.setAttribute('download', filename);
    resultCard.classList.remove('hidden');
    resultCard.scrollIntoView({ behavior: 'smooth' });
    resultVideo.play().catch(() => {});
  }

  // --- Atualização Visual dos Passos do Pipeline ---
  function updatePipelineSteps(pct) {
    const steps = [
      { el: stepVoice, min: 5, max: 20 },
      { el: stepAudio, min: 21, max: 35 },
      { el: stepMotion, min: 36, max: 75 },
      { el: stepMix, min: 76, max: 88 },
      { el: stepRender, min: 89, max: 100 }
    ];

    steps.forEach(s => {
      s.el.classList.remove('active', 'done');
      if (pct >= s.max) {
        s.el.classList.add('done');
      } else if (pct >= s.min) {
        s.el.classList.add('active');
      }
    });
  }

  // --- Conexão SSE de Progresso em Tempo Real ---
  function listenToProgress(taskId) {
    if (progressSource) {
      progressSource.close();
    }

    progressCard.classList.remove('hidden');
    progressCard.scrollIntoView({ behavior: 'smooth' });

    progressSource = new EventSource('/api/progress-stream');

    progressSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const pct = data.progress || 0;

        progressBarFill.style.width = `${pct}%`;
        progressPercent.textContent = `${pct}%`;
        progressStatusText.textContent = data.message || 'Processando...';
        updatePipelineSteps(pct);

        if (data.status === 'completed' && data.video_url) {
          progressSource.close();
          generateBtn.disabled = false;
          generateBtn.querySelector('.btn-text').textContent = 'Gerar Vídeo Agora';
          
          showResultVideo(data.video_url, data.filename);
          loadGallery();
        } else if (data.status === 'error') {
          progressSource.close();
          generateBtn.disabled = false;
          generateBtn.querySelector('.btn-text').textContent = 'Gerar Vídeo Agora';
          alert(`Erro na geração: ${data.error || 'Falha no processo'}`);
        }
      } catch (e) {
        console.error('Erro no parser SSE:', e);
      }
    };

    progressSource.onerror = () => {
      // Polling fallback se SSE falhar
      fallbackPolling(taskId);
    };
  }

  function fallbackPolling(taskId) {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const pct = data.progress || 0;

        progressBarFill.style.width = `${pct}%`;
        progressPercent.textContent = `${pct}%`;
        progressStatusText.textContent = data.message || 'Processando...';
        updatePipelineSteps(pct);

        if (data.status === 'completed' || data.status === 'error') {
          clearInterval(interval);
          generateBtn.disabled = false;
          generateBtn.querySelector('.btn-text').textContent = 'Gerar Vídeo Agora';
          if (data.status === 'completed' && data.video_url) {
            showResultVideo(data.video_url, data.filename);
            loadGallery();
          } else if (data.status === 'error') {
            alert(`Erro na geração: ${data.error}`);
          }
        }
      } catch (e) {
        console.error('Erro no polling:', e);
      }
    }, 1000);
  }

  // --- Submissão do Formulário de Geração ---
  videoForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const activeTab = document.querySelector('.tab-btn.active').dataset.tab;
    const scriptText = document.getElementById('scriptText').value.trim();
    const speechInput = document.getElementById('speechInput');

    if (activeTab === 'tab-text' && !scriptText) {
      alert('Por favor, digite o roteiro ou fala da IA.');
      return;
    }

    if (activeTab === 'tab-audio' && (!speechInput.files || speechInput.files.length === 0)) {
      alert('Por favor, selecione um arquivo de áudio para a fala.');
      return;
    }

    if (!imageInput.files[0] && !selectedLocalImage) {
      alert('Por favor, faça upload ou selecione uma imagem de referência.');
      return;
    }

    const formData = new FormData();

    // Imagem
    if (imageInput.files[0]) {
      formData.append('image_file', imageInput.files[0]);
    } else if (selectedLocalImage) {
      formData.append('selected_image', selectedLocalImage);
    }

    // Roteiro ou áudio
    if (activeTab === 'tab-text') {
      formData.append('script_text', scriptText);
      formData.append('voice', document.getElementById('voiceSelect').value);
    } else if (speechInput.files[0]) {
      formData.append('speech_file', speechInput.files[0]);
    }

    // Trilha sonora
    const enableMusic = enableMusicCheck.checked;
    formData.append('enable_music', enableMusic ? 'true' : 'false');
    if (enableMusic) {
      const musicInput = document.getElementById('musicInput');
      if (musicInput.files[0]) {
        formData.append('music_file', musicInput.files[0]);
      } else if (selectedLocalMusic) {
        formData.append('selected_music', selectedLocalMusic);
      }
      formData.append('music_volume', (parseFloat(musicVolume.value) / 100).toFixed(2));
    }

    // Duração
    formData.append('min_duration', minDurationInput.value);
    formData.append('max_duration', maxDurationInput.value);

    // Funcionalidades
    formData.append('enable_speech', document.getElementById('enableSpeechCheck').checked ? 'true' : 'false');
    formData.append('enable_walk', document.getElementById('enableWalkCheck').checked ? 'true' : 'false');
    formData.append('use_frame', document.getElementById('useBrandingCheck').checked ? 'true' : 'false');
    formData.append('use_logo', document.getElementById('useBrandingCheck').checked ? 'true' : 'false');

    // Desabilita botão e inicia
    generateBtn.disabled = true;
    generateBtn.querySelector('.btn-text').textContent = 'Processando IA...';

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      if (!data.success) {
        alert(data.error || 'Erro ao iniciar geração.');
        generateBtn.disabled = false;
        generateBtn.querySelector('.btn-text').textContent = 'Gerar Vídeo Agora';
        return;
      }

      listenToProgress(data.task_id);
    } catch (err) {
      console.error('Erro na requisição:', err);
      alert('Erro de conexão com o servidor local.');
      generateBtn.disabled = false;
      generateBtn.querySelector('.btn-text').textContent = 'Gerar Vídeo Agora';
    }
  });

  // --- Informações do Sistema e Modo de Execução ---
  async function loadSystemInfo() {
    try {
      const res = await fetch('/api/system-info');
      const info = await res.json();
      const statusLabel = document.querySelector('.status-label');
      const statusDot = document.querySelector('.status-dot');
      if (info.is_vercel) {
        statusLabel.textContent = 'Modo Cloud Vercel (Online)';
        statusDot.style.backgroundColor = '#38bdf8';
        statusDot.style.boxShadow = '0 0 10px #38bdf8';
      } else {
        statusLabel.textContent = 'Motor de IA Local Pronto (GPU/FFmpeg)';
        statusDot.style.backgroundColor = '#34d399';
        statusDot.style.boxShadow = '0 0 10px #34d399';
      }
    } catch (e) {
      console.warn('Status do sistema:', e);
    }
  }

  // Inicializações
  loadSystemInfo();
  syncDurationInputs();
  setupImageUpload();
  loadLocalAssets();
  loadGallery();
});
