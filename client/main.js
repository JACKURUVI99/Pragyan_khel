import { VideoSegmenter } from './segmenter.js';
import { Renderer } from './renderer.js';

export const AppState = {
  IDLE: 'IDLE',
  LOADING_MODEL: 'LOADING_MODEL',
  STREAMING: 'STREAMING',
  SEGMENTING: 'SEGMENTING',
  TRACKING: 'TRACKING',
  MULTI_FOCUS: 'MULTI_FOCUS'
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

    // Rack focus click positions (in CSS px relative to canvas container)
    this.rackDotAPos = null;
    this.rackDotBPos = null;

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

      if (e.shiftKey) {
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
      this.renderer.render(this.video, maskData, this.isDebug);

      this.video.requestVideoFrameCallback(processFrame);
    };

    this.video.requestVideoFrameCallback(processFrame);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  window.appState = new App();
});
