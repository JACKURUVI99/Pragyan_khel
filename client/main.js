import { VideoSegmenter } from './segmenter.js';
import { Renderer } from './renderer.js';

export const AppState = {
  IDLE: 'IDLE',
  LOADING_MODEL: 'LOADING_MODEL',
  STREAMING: 'STREAMING',
  SEGMENTING: 'SEGMENTING',
  TRACKING: 'TRACKING',
  MULTI_FOCUS: 'MULTI_FOCUS',
  PRIORITY_FOCUS: 'PRIORITY_FOCUS'
};

class App {
  constructor() {
    this.state = AppState.IDLE;

    // DOM Elements
    this.video = document.getElementById('source-video');
    this.canvas = document.getElementById('output-canvas');

    this.renderer = new Renderer(this.canvas);
    this.segmenter = new VideoSegmenter(this);

    this.fileInput = document.getElementById('file-upload');
    this.webcamBtn = document.getElementById('webcam-btn');
    this.debugBtn = document.getElementById('debug-toggle');

    // UI State Elements
    this.ui = {
      stateText: document.getElementById('state-text'),
      stateDot: document.getElementById('state-dot'),
      loadingOverlay: document.getElementById('loading-overlay'),
      idleView: document.getElementById('idle-view'),
      debugText: document.getElementById('debug-text'),
      rackFocusHint: document.getElementById('rack-focus-hint'),
      rackFocusDotA: document.getElementById('rack-dot-a'),
      rackFocusDotB: document.getElementById('rack-dot-b'),
    };

    this.isDebug = false;
    this.lightingMode = 0; // 0=blur, 1=warm, 2=cool, 3=spotlight, 4=vignette

    // Rack focus click positions (in CSS px relative to canvas container)
    this.rackDotAPos = null;
    this.rackDotBPos = null;

    // Priority focus subject dot colors
    this.priorityDotColors = [
      'bg-amber-400 shadow-amber-400/50',    // #1 Gold
      'bg-slate-300 shadow-slate-300/50',     // #2 Silver
      'bg-amber-700 shadow-amber-700/50',     // #3 Bronze
      'bg-cyan-400 shadow-cyan-400/50',       // #4
      'bg-pink-400 shadow-pink-400/50',       // #5
    ];
    this.priorityDotLabels = ['1st', '2nd', '3rd', '4th', '5th'];

    this.init();
  }

  init() {
    this.bindEvents();
    this.updateUI();

    // Init MediaPipe right away
    this.segmenter.init();
  }

  bindEvents() {
    // Webcam
    this.webcamBtn.addEventListener('click', () => this.startWebcam());

    // File Upload
    this.fileInput.addEventListener('change', (e) => this.handleFileUpload(e));

    // Debug Toggle
    this.debugBtn.addEventListener('click', () => {
      this.isDebug = !this.isDebug;
      this.debugBtn.classList.toggle('bg-zinc-800', !this.isDebug);
      this.debugBtn.classList.toggle('bg-emerald-900', this.isDebug);
      this.debugBtn.classList.toggle('text-emerald-400', this.isDebug);

      if (this.ui.debugText) {
        this.ui.debugText.classList.toggle('hidden', !this.isDebug);
      }
    });

    // Lighting Mode Selector
    const lightingSelect = document.getElementById('lighting-select');
    if (lightingSelect) {
      lightingSelect.addEventListener('change', (e) => {
        this.lightingMode = parseInt(e.target.value, 10);
      });
    }

    // Video events
    this.video.addEventListener('loadedmetadata', () => {
      this.canvas.width = this.video.videoWidth;
      this.canvas.height = this.video.videoHeight;
    });

    this.video.addEventListener('play', () => {
      if (this.state !== AppState.LOADING_MODEL) {
        this.setState(AppState.STREAMING);
      }
      this.startFrameLoop();
    });

    // Canvas Click for Focus
    this.canvas.addEventListener('click', (e) => this.handleCanvasClick(e));
  }

  handleCanvasClick(e) {
    if (this.state === AppState.IDLE || this.state === AppState.LOADING_MODEL) return;

    const rect = this.canvas.getBoundingClientRect();

    // Normalized click coordinates 0.0 to 1.0
    // Calculate based on the actual displayed aspect ratio (object-contain)
    const videoRatio = this.video.videoWidth / this.video.videoHeight;
    const canvasRatio = rect.width / rect.height;

    let renderedWidth = rect.width;
    let renderedHeight = rect.height;
    let xOffset = 0;
    let yOffset = 0;

    if (canvasRatio > videoRatio) {
      // Letterboxed left & right
      renderedWidth = rect.height * videoRatio;
      xOffset = (rect.width - renderedWidth) / 2;
    } else {
      // Letterboxed top & bottom
      renderedHeight = rect.width / videoRatio;
      yOffset = (rect.height - renderedHeight) / 2;
    }

    const clickX = e.clientX - rect.left - xOffset;
    const clickY = e.clientY - rect.top - yOffset;

    if (clickX >= 0 && clickX <= renderedWidth && clickY >= 0 && clickY <= renderedHeight) {
      const px = clickX / renderedWidth;
      const py = clickY / renderedHeight;

      if (e.ctrlKey || e.metaKey) {
        // ── Priority Focus mode (Ctrl+Click) ──
        const subjectId = this.segmenter.addPrioritySubject(px, py);
        if (subjectId !== null) {
          this._renderPriorityPanel();
        }
      } else if (e.shiftKey) {
        // ── Multi-Focus mode ──
        this.segmenter.setMultiFocusPoint(px, py);

        // Store CSS position for the visual dot overlay
        const dotX = e.clientX - rect.left;
        const dotY = e.clientY - rect.top;

        if (this.segmenter.multiFocusClickCount === 1) {
          this.rackDotAPos = { x: dotX, y: dotY };
          this.rackDotBPos = null;
          this._updateRackDots(rect);
        } else if (this.segmenter.multiFocusClickCount === 2) {
          this.rackDotBPos = { x: dotX, y: dotY };
          this._updateRackDots(rect);
        }
      } else {
        // ── Single focus (original) ──
        this.rackDotAPos = null;
        this.rackDotBPos = null;
        this._updateRackDots(rect);
        this._clearPriorityPanel();
        this.segmenter.setTarget(px, py);
      }
    }
  }

  _updateRackDots(rect) {
    const dotA = this.ui.rackFocusDotA;
    const dotB = this.ui.rackFocusDotB;
    if (!dotA || !dotB) return;

    if (this.rackDotAPos) {
      dotA.style.left = `${this.rackDotAPos.x}px`;
      dotA.style.top = `${this.rackDotAPos.y}px`;
      dotA.classList.remove('hidden');
    } else {
      dotA.classList.add('hidden');
    }

    if (this.rackDotBPos) {
      dotB.style.left = `${this.rackDotBPos.x}px`;
      dotB.style.top = `${this.rackDotBPos.y}px`;
      dotB.classList.remove('hidden');
    } else {
      dotB.classList.add('hidden');
    }
  }

  setState(newState) {
    if (this.state === newState) return;
    this.state = newState;
    this.updateUI();
  }

  updateUI() {
    this.ui.stateDot.className = 'w-2 h-2 rounded-full transition-colors duration-300';
    this.ui.loadingOverlay.classList.add('hidden');

    // Update multi-focus hint visibility
    if (this.ui.rackFocusHint) {
      this.ui.rackFocusHint.classList.toggle('hidden',
        this.state === AppState.IDLE || this.state === AppState.LOADING_MODEL);
    }

    switch (this.state) {
      case AppState.IDLE:
        this.ui.stateText.textContent = 'IDLE';
        this.ui.stateDot.classList.add('bg-zinc-500');
        this.ui.idleView.classList.remove('hidden');
        this.canvas.classList.add('hidden');
        break;
      case AppState.LOADING_MODEL:
        this.ui.stateText.textContent = 'LOADING';
        this.ui.stateDot.classList.add('bg-emerald-400', 'animate-pulse');
        this.ui.loadingOverlay.classList.remove('hidden');
        // Don't modify canvas visibility during load if streaming already
        break;
      case AppState.STREAMING:
      case AppState.SEGMENTING:
      case AppState.TRACKING:
        this.ui.stateText.textContent = this.state;
        if (this.state === AppState.STREAMING) this.ui.stateDot.classList.add('bg-emerald-500');
        if (this.state === AppState.SEGMENTING) this.ui.stateDot.classList.add('bg-amber-400', 'animate-pulse');
        if (this.state === AppState.TRACKING) this.ui.stateDot.classList.add('bg-emerald-400');

        this.ui.idleView.classList.add('hidden');
        this.canvas.classList.remove('hidden');
        break;
      case AppState.MULTI_FOCUS:
        this.ui.stateText.textContent = 'MULTI FOCUS';
        this.ui.stateDot.classList.add('bg-violet-400', 'animate-pulse');
        this.ui.idleView.classList.add('hidden');
        this.canvas.classList.remove('hidden');
        break;
      case AppState.PRIORITY_FOCUS:
        this.ui.stateText.textContent = 'PRIORITY';
        this.ui.stateDot.classList.add('bg-amber-400', 'animate-pulse');
        this.ui.idleView.classList.add('hidden');
        this.canvas.classList.remove('hidden');
        break;
    }
  }

  // ── Priority Panel ──

  _renderPriorityPanel() {
    const panel = document.getElementById('priority-panel');
    if (!panel) return;

    const subjects = this.segmenter.subjects;
    if (subjects.length === 0) {
      panel.classList.add('hidden');
      return;
    }

    panel.classList.remove('hidden');
    const list = panel.querySelector('#priority-list');
    list.innerHTML = '';

    subjects.forEach((subject, index) => {
      const colorClass = this.priorityDotColors[index] || 'bg-zinc-400';
      const label = this.priorityDotLabels[index] || `#${index + 1}`;
      const weight = (this.segmenter.priorityWeights[index] * 100).toFixed(0);

      const item = document.createElement('div');
      item.className = 'flex items-center gap-2 px-2 py-1 rounded bg-zinc-800/80 group';
      item.innerHTML = `
        <span class="w-3 h-3 rounded-full ${colorClass} shrink-0"></span>
        <span class="text-zinc-200 text-xs font-mono flex-1">${label} <span class="text-zinc-500">${weight}%</span></span>
        <button class="priority-up text-zinc-500 hover:text-zinc-200 text-xs px-1 ${index === 0 ? 'invisible' : ''}" data-id="${subject.id}" title="Move up">▲</button>
        <button class="priority-down text-zinc-500 hover:text-zinc-200 text-xs px-1 ${index === subjects.length - 1 ? 'invisible' : ''}" data-id="${subject.id}" title="Move down">▼</button>
        <button class="priority-remove text-zinc-500 hover:text-red-400 text-xs px-1" data-id="${subject.id}" title="Remove">✕</button>
      `;
      list.appendChild(item);
    });

    // Bind button events
    list.querySelectorAll('.priority-up').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = parseInt(btn.dataset.id);
        const ids = this.segmenter.subjects.map(s => s.id);
        const idx = ids.indexOf(id);
        if (idx > 0) {
          [ids[idx - 1], ids[idx]] = [ids[idx], ids[idx - 1]];
          this.segmenter.reorderPriorities(ids);
          this._renderPriorityPanel();
        }
      });
    });

    list.querySelectorAll('.priority-down').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = parseInt(btn.dataset.id);
        const ids = this.segmenter.subjects.map(s => s.id);
        const idx = ids.indexOf(id);
        if (idx < ids.length - 1) {
          [ids[idx], ids[idx + 1]] = [ids[idx + 1], ids[idx]];
          this.segmenter.reorderPriorities(ids);
          this._renderPriorityPanel();
        }
      });
    });

    list.querySelectorAll('.priority-remove').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = parseInt(btn.dataset.id);
        this.segmenter.removePrioritySubject(id);
        this._renderPriorityPanel();
      });
    });
  }

  _clearPriorityPanel() {
    const panel = document.getElementById('priority-panel');
    if (panel) {
      panel.classList.add('hidden');
      const list = panel.querySelector('#priority-list');
      if (list) list.innerHTML = '';
    }
  }

  async startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'user' },
        audio: false
      });
      this.video.srcObject = stream;
      this.video.play();
    } catch (err) {
      console.error('Error accessing webcam:', err);
      alert('Failed to access webcam. Please check permissions.');
    }
  }

  handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    if (this.video.src && !this.video.srcObject) {
      URL.revokeObjectURL(this.video.src);
    }

    this.video.srcObject = null;
    const url = URL.createObjectURL(file);
    this.video.src = url;
    this.video.play();
  }

  startFrameLoop() {
    const processFrame = (now, metadata) => {
      if (this.video.paused || this.video.ended) return;

      const timestamp = performance.now();
      const maskData = this.segmenter.processFrame(this.video, timestamp);

      // Phase 4: Render via WebGL
      this.renderer.render(this.video, maskData, this.isDebug, this.lightingMode);

      this.video.requestVideoFrameCallback(processFrame);
    };

    this.video.requestVideoFrameCallback(processFrame);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.appState = new App();
});
