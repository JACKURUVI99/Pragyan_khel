// ── Transformers.js SAM Segmenter ──
// Uses Hugging Face Transformers.js for proper WebGPU support + simple API
import { SamModel, AutoProcessor, RawImage, Tensor } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';
import { AppState } from './main.js';

// Change this to swap models easily:
// 'Xenova/sam-vit-base'          – best accuracy, ~100MB q8
// 'Xenova/slimsam-77-uniform'    – lighter, ~14MB q8
const MODEL_ID = 'Xenova/slimsam-77-uniform';

export class SAMSegmenter {
  constructor(stateManager) {
    this.app = stateManager;
    this.model = null;
    this.processor = null;
    this.isInitializing = false;

    // ── Cached image embeddings ──
    this.cachedEmbeddings = null;  // { image_embeddings, image_positional_embeddings }
    this.encoderBusy = false;
    this.framesSinceEncode = 0;
    this.encodeInterval = 10; // Re-encode every N frames

    // ── Offscreen canvas for video → RawImage ──
    this.captureCanvas = document.createElement('canvas');
    this.captureCtx = this.captureCanvas.getContext('2d', { willReadFrequently: true });

    // ── Single focus state ──
    this.roi = null;                // {x, y} normalized 0-1
    this.lastMask = null;           // Float32Array (video resolution)
    this.width = 0;
    this.height = 0;
    this.maskFailed = false;

    // ── Focus follow damping ──
    this.previousRoi = null;
    this.initialRoi = null;
    this.followDamping = 0.08;
    this.lastArea = null;

    // ── Multi-Focus state ──
    this.multiFocusMode = false;
    this.roiA = null;
    this.roiB = null;
    this.previousRoiA = null;
    this.previousRoiB = null;
    this.initialRoiA = null;
    this.initialRoiB = null;
    this.maskA = null;
    this.maskB = null;
    this.lastAreaA = null;
    this.lastAreaB = null;
    this.multiFocusClickCount = 0;

    // ── Priority Focus state ──
    this.priorityFocusMode = false;
    this.subjects = [];
    this.subjectIdCounter = 0;
    this.priorityWeights = [1.0, 0.7, 0.5, 0.35, 0.25];
    this.maxSubjects = 5;
    this.compositeMask = null;

    // ── Decode lock ──
    this.decodeBusy = false;
  }

  // ══════════════════════════════════════════
  //  Initialization
  // ══════════════════════════════════════════

  async init() {
    if (this.model || this.isInitializing) return;
    this.isInitializing = true;

    const prevState = this.app.state;
    this.app.setState(AppState.LOADING_MODEL);
    this.updateDebugLog(`Loading ${MODEL_ID}...`);

    try {
      // Transformers.js handles: download, caching, WebGPU/WASM fallback
      const [model, processor] = await Promise.all([
        SamModel.from_pretrained(MODEL_ID, {
          dtype: 'q8',  // quantized for smaller download
          progress_callback: (p) => {
            if (p.progress !== undefined) {
              this.updateDebugLog(`Downloading ${p.file}: ${p.progress.toFixed(0)}%`);
            }
          }
        }),
        AutoProcessor.from_pretrained(MODEL_ID),
      ]);

      this.model = model;
      this.processor = processor;
      this.isInitializing = false;

      if (this.app.state === AppState.LOADING_MODEL) {
        this.app.setState(prevState === AppState.IDLE ? AppState.IDLE : AppState.STREAMING);
      }
      this.updateDebugLog('SAM loaded. Click to segment!');
      console.log(`SAM loaded: ${MODEL_ID} (q8)`);
    } catch (e) {
      console.error('Failed to load SAM:', e);
      this.updateDebugLog('Failed to load SAM: ' + e.message);
      this.isInitializing = false;
    }
  }

  // ══════════════════════════════════════════
  //  Video frame → RawImage
  // ══════════════════════════════════════════

  captureFrame(videoElement) {
    const w = videoElement.videoWidth;
    const h = videoElement.videoHeight;
    if (this.captureCanvas.width !== w) this.captureCanvas.width = w;
    if (this.captureCanvas.height !== h) this.captureCanvas.height = h;
    this.captureCtx.drawImage(videoElement, 0, 0, w, h);
    const imageData = this.captureCtx.getImageData(0, 0, w, h);
    // RawImage(data, width, height, channels)
    return new RawImage(imageData.data, w, h, 4);
  }

  // ══════════════════════════════════════════
  //  Encoder (image → embeddings)
  // ══════════════════════════════════════════

  async runEncoder(videoElement) {
    if (!this.model || !this.processor || this.encoderBusy) return;
    this.encoderBusy = true;

    try {
      const t0 = performance.now();

      const rawImage = this.captureFrame(videoElement);
      const inputs = await this.processor(rawImage);

      const t1 = performance.now();
      const embeddings = await this.model.get_image_embeddings(inputs);
      const t2 = performance.now();

      this.cachedEmbeddings = embeddings;
      console.log(`SAM Encoder: preprocess=${(t1 - t0).toFixed(0)}ms, inference=${(t2 - t1).toFixed(0)}ms`);
    } catch (e) {
      console.error('Encoder error:', e);
    } finally {
      this.encoderBusy = false;
    }
  }

  // ══════════════════════════════════════════
  //  Decoder (embeddings + point → mask)
  // ══════════════════════════════════════════

  async runDecoder(nx, ny) {
    if (!this.model || !this.cachedEmbeddings) return null;

    try {
      // SAM expects coords in processor's resized space (1024x1024)
      // The processor resizes the longest side to 1024 and pads
      const longest = Math.max(this.width, this.height);
      const scale = 1024 / longest;
      const ptX = nx * this.width * scale;
      const ptY = ny * this.height * scale;

      // Transformers.js SAM expects: [batch, num_points_per_image, num_points, 2]
      const input_points = new Tensor('float32', [ptX, ptY], [1, 1, 1, 2]);
      const input_labels = new Tensor('int64', [1n], [1, 1, 1]);

      const outputs = await this.model({
        ...this.cachedEmbeddings,
        input_points,
        input_labels,
      });

      // outputs.pred_masks: [1, 1, num_masks, H, W]
      // outputs.iou_scores: [1, 1, num_masks]
      const masks = outputs.pred_masks;
      const scores = outputs.iou_scores;

      // Pick best mask
      const numMasks = scores.dims[scores.dims.length - 1];
      let bestIdx = 0;
      let bestScore = -Infinity;
      for (let i = 0; i < numMasks; i++) {
        if (scores.data[i] > bestScore) {
          bestScore = scores.data[i];
          bestIdx = i;
        }
      }

      // Extract best mask and apply sigmoid
      const maskH = masks.dims[masks.dims.length - 2];
      const maskW = masks.dims[masks.dims.length - 1];
      const maskSize = maskH * maskW;
      const rawMask = masks.data.slice(bestIdx * maskSize, (bestIdx + 1) * maskSize);

      const sigmoidMask = new Float32Array(maskSize);
      for (let i = 0; i < maskSize; i++) {
        sigmoidMask[i] = 1.0 / (1.0 + Math.exp(-rawMask[i]));
      }

      // Upscale to video resolution
      return this._upscaleMask(sigmoidMask, maskW, maskH, this.width, this.height);
    } catch (e) {
      console.error('Decoder error:', e);
      return null;
    }
  }

  _upscaleMask(src, srcW, srcH, dstW, dstH) {
    const dst = new Float32Array(dstW * dstH);
    const longest = Math.max(dstW, dstH);
    const scale = 1024 / longest;

    for (let y = 0; y < dstH; y++) {
      for (let x = 0; x < dstW; x++) {
        const mx = (x * scale) * srcW / 1024;
        const my = (y * scale) * srcH / 1024;
        const x0 = Math.floor(mx);
        const y0 = Math.floor(my);
        const x1 = Math.min(x0 + 1, srcW - 1);
        const y1 = Math.min(y0 + 1, srcH - 1);
        const fx = mx - x0;
        const fy = my - y0;
        dst[y * dstW + x] =
          src[y0 * srcW + x0] * (1 - fx) * (1 - fy) +
          src[y0 * srcW + x1] * fx * (1 - fy) +
          src[y1 * srcW + x0] * (1 - fx) * fy +
          src[y1 * srcW + x1] * fx * fy;
      }
    }
    return dst;
  }

  // ══════════════════════════════════════════
  //  Frame processing
  // ══════════════════════════════════════════

  _hasActiveTarget() {
    return this.roi || this.multiFocusMode || (this.priorityFocusMode && this.subjects.length > 0);
  }

  processFrame(videoElement, timestamp) {
    if (!this.model) return null;

    this.width = videoElement.videoWidth;
    this.height = videoElement.videoHeight;

    // Only encode when there's a target
    if (this._hasActiveTarget()) {
      this.framesSinceEncode++;
      if (this.framesSinceEncode >= this.encodeInterval) {
        this.framesSinceEncode = 0;
        this.runEncoder(videoElement);
      }
    }

    // ── Priority Focus mode ──
    if (this.priorityFocusMode && this.subjects.length > 0) {
      this._triggerPriorityDecodes();
      this._compositeWeightedMask();
      const top = this.subjects[0];
      return {
        mask: this.compositeMask,
        width: this.width,
        height: this.height,
        priorityFocus: true,
        focusCenter: top ? { x: top.roi.x, y: top.roi.y } : null
      };
    }

    // ── Multi-Focus mode ──
    if (this.multiFocusMode) {
      this._triggerMultiFocusDecodes();
      let fc = null;
      if (this.roiA && this.roiB) fc = { x: (this.roiA.x + this.roiB.x) / 2, y: (this.roiA.y + this.roiB.y) / 2 };
      else if (this.roiA) fc = { x: this.roiA.x, y: this.roiA.y };
      return { multiFocus: true, maskA: this.maskA, maskB: this.maskB, width: this.width, height: this.height, mask: this.maskA, focusCenter: fc };
    }

    // ── Single focus ──
    if (this.maskFailed || !this.roi) return null;
    this._triggerSingleDecode();
    return { mask: this.lastMask, width: this.width, height: this.height, focusCenter: this.roi ? { x: this.roi.x, y: this.roi.y } : null };
  }

  // ── Immediate segment on click ──
  async _immediateSegment(nx, ny) {
    const video = this.app.video;
    if (!video || !this.model) return;
    try {
      const t0 = performance.now();
      this.updateDebugLog('Encoding frame...');
      await this.runEncoder(video);
      const t1 = performance.now();
      this.updateDebugLog(`Encoded in ${(t1 - t0).toFixed(0)}ms, decoding...`);
      if (!this.cachedEmbeddings) return;
      const mask = await this.runDecoder(nx, ny);
      const t2 = performance.now();
      if (mask) {
        this.lastMask = mask;
        const centroid = this.calculateCentroid(mask, this.width, this.height);
        if (centroid) {
          this.lastArea = centroid.area;
          this.roi = { x: centroid.x, y: centroid.y };
          this.previousRoi = { ...this.roi };
          this.app.setState(AppState.TRACKING);
          this.updateDebugLog(`SAM ready in ${(t2 - t0).toFixed(0)}ms (enc=${(t1 - t0).toFixed(0)}, dec=${(t2 - t1).toFixed(0)})`);
        } else {
          this.updateDebugLog('No object found at click point.');
          this.handleMaskFailure();
        }
      }
    } catch (e) {
      console.error('Immediate segment error:', e);
    }
  }

  // ── Async decode triggers ──

  async _triggerSingleDecode() {
    if (this.decodeBusy || !this.roi || !this.cachedEmbeddings) return;
    this.decodeBusy = true;
    try {
      const mask = await this.runDecoder(this.roi.x, this.roi.y);
      if (mask) {
        this.lastMask = mask;
        const centroid = this.calculateCentroid(mask, this.width, this.height);
        if (centroid) {
          this.lastArea = centroid.area;
          if (this.previousRoi) {
            this.roi = { x: SAMSegmenter.lerp(this.previousRoi.x, centroid.x, this.followDamping), y: SAMSegmenter.lerp(this.previousRoi.y, centroid.y, this.followDamping) };
          } else {
            this.roi = { x: centroid.x, y: centroid.y };
          }
          this.previousRoi = { ...this.roi };
          if (this.app.state === AppState.SEGMENTING) this.app.setState(AppState.TRACKING);
          this.updateDebugLog(`Tracking: {${this.roi.x.toFixed(2)}, ${this.roi.y.toFixed(2)}}`);
        } else {
          this.handleMaskFailure();
        }
      }
    } finally { this.decodeBusy = false; }
  }

  async _triggerMultiFocusDecodes() {
    if (this.decodeBusy || !this.cachedEmbeddings) return;
    this.decodeBusy = true;
    try {
      if (this.roiA) {
        const m = await this.runDecoder(this.roiA.x, this.roiA.y);
        if (m) { this.maskA = m; const c = this.calculateCentroid(m, this.width, this.height); if (c) { this.lastAreaA = c.area; this.roiA = this.previousRoiA ? { x: SAMSegmenter.lerp(this.previousRoiA.x, c.x, this.followDamping), y: SAMSegmenter.lerp(this.previousRoiA.y, c.y, this.followDamping) } : { x: c.x, y: c.y }; this.previousRoiA = { ...this.roiA }; } }
      }
      if (this.roiB) {
        const m = await this.runDecoder(this.roiB.x, this.roiB.y);
        if (m) { this.maskB = m; const c = this.calculateCentroid(m, this.width, this.height); if (c) { this.lastAreaB = c.area; this.roiB = this.previousRoiB ? { x: SAMSegmenter.lerp(this.previousRoiB.x, c.x, this.followDamping), y: SAMSegmenter.lerp(this.previousRoiB.y, c.y, this.followDamping) } : { x: c.x, y: c.y }; this.previousRoiB = { ...this.roiB }; } }
      }
    } finally { this.decodeBusy = false; }
  }

  async _triggerPriorityDecodes() {
    if (this.decodeBusy || !this.cachedEmbeddings) return;
    this.decodeBusy = true;
    try {
      for (const subject of this.subjects) {
        const m = await this.runDecoder(subject.roi.x, subject.roi.y);
        if (m) { subject.mask = m; const c = this.calculateCentroid(m, this.width, this.height); if (c) { subject.lastArea = c.area; subject.roi = subject.previousRoi ? { x: SAMSegmenter.lerp(subject.previousRoi.x, c.x, this.followDamping), y: SAMSegmenter.lerp(subject.previousRoi.y, c.y, this.followDamping) } : { x: c.x, y: c.y }; subject.previousRoi = { ...subject.roi }; } }
      }
    } finally { this.decodeBusy = false; }
  }

  // ══════════════════════════════════════════
  //  Public API
  // ══════════════════════════════════════════

  static lerp(a, b, t) { return a + (b - a) * t; }

  setTarget(normalizedX, normalizedY) {
    if (!this.model) return;
    this.clearMultiFocus();
    this.clearPriorityFocus();
    this.initialRoi = { x: normalizedX, y: normalizedY };
    this.roi = { ...this.initialRoi };
    this.previousRoi = null;
    this.maskFailed = false;
    this.lastMask = null;
    this.lastArea = null;
    if ([AppState.STREAMING, AppState.TRACKING, AppState.MULTI_FOCUS, AppState.PRIORITY_FOCUS].includes(this.app.state)) {
      this.app.setState(AppState.SEGMENTING);
    }
    this.updateDebugLog(`Target: {${normalizedX.toFixed(2)}, ${normalizedY.toFixed(2)}}`);
    this._immediateSegment(normalizedX, normalizedY);
  }

  setMultiFocusPoint(normalizedX, normalizedY) {
    if (!this.model) return;
    this.multiFocusClickCount++;
    if (this.multiFocusClickCount === 1) {
      this.clearMultiFocus(); this.multiFocusMode = true; this.multiFocusClickCount = 1;
      this.initialRoiA = { x: normalizedX, y: normalizedY }; this.roiA = { ...this.initialRoiA };
      this.maskA = null; this.maskB = null; this.roi = null; this.lastMask = null;
      this.updateDebugLog(`Multi A: {${normalizedX.toFixed(2)}, ${normalizedY.toFixed(2)}}. Shift+Click for B.`);
      if (this.app.state !== AppState.LOADING_MODEL && this.app.state !== AppState.IDLE) this.app.setState(AppState.SEGMENTING);
      this._immediateSegment(normalizedX, normalizedY);
    } else if (this.multiFocusClickCount === 2) {
      this.initialRoiB = { x: normalizedX, y: normalizedY }; this.roiB = { ...this.initialRoiB };
      this.app.setState(AppState.MULTI_FOCUS);
      this.updateDebugLog(`Multi B: {${normalizedX.toFixed(2)}, ${normalizedY.toFixed(2)}}. Both in focus!`);
    } else {
      this.clearMultiFocus(); this.setMultiFocusPoint(normalizedX, normalizedY);
    }
  }

  clearMultiFocus() { this.multiFocusMode = false; this.multiFocusClickCount = 0; this.roiA = null; this.roiB = null; this.previousRoiA = null; this.previousRoiB = null; this.initialRoiA = null; this.initialRoiB = null; this.maskA = null; this.maskB = null; this.lastAreaA = null; this.lastAreaB = null; }

  addPrioritySubject(normalizedX, normalizedY) {
    if (!this.model) return null;
    if (this.subjects.length >= this.maxSubjects) { this.updateDebugLog(`Max ${this.maxSubjects} subjects.`); return null; }
    if (!this.priorityFocusMode) { this.clearMultiFocus(); this.roi = null; this.lastMask = null; this.previousRoi = null; this.priorityFocusMode = true; }
    const id = ++this.subjectIdCounter;
    this.subjects.push({ id, roi: { x: normalizedX, y: normalizedY }, previousRoi: null, initialRoi: { x: normalizedX, y: normalizedY }, mask: null, lastArea: null, priority: this.subjects.length });
    if (this.app.state !== AppState.LOADING_MODEL && this.app.state !== AppState.IDLE) this.app.setState(AppState.PRIORITY_FOCUS);
    this.updateDebugLog(`Priority #${this.subjects.length}: {${normalizedX.toFixed(2)}, ${normalizedY.toFixed(2)}}`);
    this._immediateSegment(normalizedX, normalizedY);
    return id;
  }

  removePrioritySubject(id) {
    this.subjects = this.subjects.filter(s => s.id !== id);
    this.subjects.forEach((s, i) => { s.priority = i; });
    if (this.subjects.length === 0) { this.clearPriorityFocus(); if (this.app.state === AppState.PRIORITY_FOCUS) this.app.setState(AppState.STREAMING); }
    this.updateDebugLog(`Removed. ${this.subjects.length} remaining.`);
  }

  reorderPriorities(orderedIds) {
    const r = []; for (const id of orderedIds) { const s = this.subjects.find(s => s.id === id); if (s) r.push(s); }
    r.forEach((s, i) => { s.priority = i; }); this.subjects = r;
  }

  clearPriorityFocus() { this.priorityFocusMode = false; this.subjects = []; this.compositeMask = null; }

  handleMaskFailure() {
    this.maskFailed = true; this.roi = null; this.previousRoi = null; this.lastMask = null; this.lastArea = null;
    if (this.app.state === AppState.TRACKING || this.app.state === AppState.SEGMENTING) this.app.setState(AppState.STREAMING);
  }

  // ══════════════════════════════════════════
  //  Utilities
  // ══════════════════════════════════════════

  calculateCentroid(mask, width, height) {
    let sumX = 0, sumY = 0, count = 0;
    const step = 4;
    for (let y = 0; y < height; y += step) for (let x = 0; x < width; x += step) if (mask[y * width + x] > 0.5) { sumX += x; sumY += y; count++; }
    if (count < 10) return null;
    return { x: sumX / count / width, y: sumY / count / height, area: count };
  }

  _compositeWeightedMask() {
    const n = this.width * this.height; if (n === 0) return;
    if (!this.compositeMask || this.compositeMask.length !== n) this.compositeMask = new Float32Array(n);
    else this.compositeMask.fill(0);
    for (const s of this.subjects) {
      if (!s.mask || s.mask.length !== n) continue;
      const w = this.priorityWeights[s.priority] || 0.2;
      for (let i = 0; i < n; i++) { const v = s.mask[i] * w; if (v > this.compositeMask[i]) this.compositeMask[i] = v; }
    }
  }

  updateDebugLog(message) { const el = document.getElementById('debug-text'); if (el) el.innerText = message; }
}
