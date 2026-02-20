import { FilesetResolver, InteractiveSegmenter, ImageSegmenter } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8";
import { AppState } from './main.js';

export class VideoSegmenter {
  constructor(stateManager) {
    this.app = stateManager;
    this.segmenter = null;
    this.isInitializing = false;

    this.framesSinceLastSegment = 0;
    this.segmentationInterval = 1; // Run every frame for true tracking

    this.initialRoi = null; // Original click
    this.roi = null; // {x, y} normalized
    this.lastMask = null; // Float32Array
    this.lastArea = null;
    this.width = 0;
    this.height = 0;

    // ── Multi-Focus state ──
    this.multiFocusMode = false;
    this.roiA = null;       // { keypoint: {x, y} }
    this.roiB = null;
    this.initialRoiA = null;
    this.initialRoiB = null;
    this.maskA = null;      // Float32Array
    this.maskB = null;      // Float32Array
    this.lastAreaA = null;
    this.lastAreaB = null;
    this.multiFocusClickCount = 0;
  }

  async init() {
    if (this.segmenter || this.isInitializing) return;
    this.isInitializing = true;

    const prevState = this.app.state;
    this.app.setState(AppState.LOADING_MODEL);

    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
      );

      // We use InteractiveSegmenter for Click-to-Focus (Magic Touch)
      this.segmenter = await InteractiveSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: "https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite",
          delegate: "GPU"
        },
        outputConfidenceMasks: true,
        outputCategoryMask: false
      });

      this.isInitializing = false;
      if (this.app.state === AppState.LOADING_MODEL) {
        this.app.setState(prevState === AppState.IDLE ? AppState.IDLE : AppState.STREAMING);
      }
      console.log("Segmenter initialized successfully.");
    } catch (e) {
      console.error("Failed to load segmenter model:", e);
      this.isInitializing = false;
    }
  }

  // ── Single-click focus (existing) ──

  setTarget(normalizedX, normalizedY) {
    if (!this.segmenter) return;

    // Clear multi-focus mode on regular click
    this.clearMultiFocus();

    this.initialRoi = { keypoint: { x: normalizedX, y: normalizedY } };
    this.roi = { ...this.initialRoi };
    this.framesSinceLastSegment = 0; // force immediate segment
    this.maskFailed = false;
    this.lastArea = null;

    if (this.app.state === AppState.STREAMING || this.app.state === AppState.TRACKING || this.app.state === AppState.MULTI_FOCUS) {
      this.app.setState(AppState.SEGMENTING);
    }
    this.updateDebugLog(`Target set at ROI: {x: ${normalizedX.toFixed(2)}, y: ${normalizedY.toFixed(2)}}`);
  }

  // ── Rack Focus (two-click cinematic pull) ──

  setMultiFocusPoint(normalizedX, normalizedY) {
    if (!this.segmenter) return;

    this.multiFocusClickCount++;

    if (this.multiFocusClickCount === 1) {
      // First point (A)
      this.clearMultiFocus();
      this.multiFocusMode = true;
      this.multiFocusClickCount = 1; // re-set after clear
      this.initialRoiA = { keypoint: { x: normalizedX, y: normalizedY } };
      this.roiA = { ...this.initialRoiA };
      this.maskA = null;
      this.maskB = null;
      this.lastAreaA = null;
      this.lastAreaB = null;

      // Also clear single-focus state
      this.roi = null;
      this.lastMask = null;

      this.updateDebugLog(`Multi-Focus: Point A set at {x: ${normalizedX.toFixed(2)}, y: ${normalizedY.toFixed(2)}}. Shift+Click to set Point B.`);

      // Start segmenting A immediately
      this.framesSinceLastSegment = 0;
      if (this.app.state !== AppState.LOADING_MODEL && this.app.state !== AppState.IDLE) {
        this.app.setState(AppState.SEGMENTING);
      }

    } else if (this.multiFocusClickCount === 2) {
      // Second point (B) – both objects now in focus
      this.initialRoiB = { keypoint: { x: normalizedX, y: normalizedY } };
      this.roiB = { ...this.initialRoiB };
      this.lastAreaB = null;

      this.framesSinceLastSegment = 0;
      this.app.setState(AppState.MULTI_FOCUS);

      this.updateDebugLog(`Multi-Focus: Point B set at {x: ${normalizedX.toFixed(2)}, y: ${normalizedY.toFixed(2)}}. Both objects in focus!`);

    } else {
      // Third click resets
      this.clearMultiFocus();
      this.setMultiFocusPoint(normalizedX, normalizedY);
    }
  }

  clearMultiFocus() {
    this.multiFocusMode = false;
    this.multiFocusClickCount = 0;
    this.roiA = null;
    this.roiB = null;
    this.initialRoiA = null;
    this.initialRoiB = null;
    this.maskA = null;
    this.maskB = null;
    this.lastAreaA = null;
    this.lastAreaB = null;
  }

  // ── Debug ──

  updateDebugLog(message) {
    const el = document.getElementById('debug-text');
    if (el) el.innerText = message;
  }

  calculateCentroid(mask, width, height) {
    let sumX = 0, sumY = 0, count = 0;
    // Sample every nth pixel to keep it fast
    const step = 4;
    for (let y = 0; y < height; y += step) {
      for (let x = 0; x < width; x += step) {
        const idx = y * width + x;
        if (mask[idx] > 0.5) {
          sumX += x;
          sumY += y;
          count++;
        }
      }
    }
    if (count < 10) return null; // Mask failed
    return { x: sumX / count / width, y: sumY / count / height, area: count };
  }

  // ── Frame processing ──

  processFrame(videoElement, timestamp) {
    if (!this.segmenter) return null;

    this.width = videoElement.videoWidth;
    this.height = videoElement.videoHeight;

    // ── Multi-Focus mode ──
    if (this.multiFocusMode) {
      return this._processMultiFocusFrame(videoElement, timestamp);
    }

    // ── Single focus mode (original) ──
    if (this.maskFailed) return null;
    if (!this.roi) return null;

    if (this.framesSinceLastSegment === 0) {
      try {
        if (this.segmenter.segmentForVideo) {
          this.segmenter.segmentForVideo(videoElement, this.roi, timestamp, (result) => this.handleResult(result));
        } else {
          this.segmenter.segment(videoElement, this.roi, (result) => this.handleResult(result));
        }
      } catch (e) {
        console.error("Segmentation error:", e);
      }
    }

    this.framesSinceLastSegment++;
    if (this.framesSinceLastSegment >= this.segmentationInterval) {
      this.framesSinceLastSegment = 0;
    }

    return {
      mask: this.lastMask,
      width: this.width,
      height: this.height
    };
  }

  _processMultiFocusFrame(videoElement, timestamp) {
    // Segment ROI A
    if (this.roiA && this.framesSinceLastSegment === 0) {
      try {
        const roiACopy = { keypoint: { ...this.roiA.keypoint } };
        if (this.segmenter.segmentForVideo) {
          this.segmenter.segmentForVideo(videoElement, roiACopy, timestamp, (result) => {
            this._handleMultiResult(result, 'A');
          });
        } else {
          this.segmenter.segment(videoElement, roiACopy, (result) => {
            this._handleMultiResult(result, 'A');
          });
        }
      } catch (e) {
        console.error("Multi-focus segmentation A error:", e);
      }
    }

    // Segment ROI B (only if second point is set)
    if (this.roiB && this.framesSinceLastSegment === 0) {
      try {
        const roiBCopy = { keypoint: { ...this.roiB.keypoint } };
        // Use a slightly offset timestamp to avoid collision with the same frame
        const tsB = timestamp + 1;
        if (this.segmenter.segmentForVideo) {
          this.segmenter.segmentForVideo(videoElement, roiBCopy, tsB, (result) => {
            this._handleMultiResult(result, 'B');
          });
        } else {
          this.segmenter.segment(videoElement, roiBCopy, (result) => {
            this._handleMultiResult(result, 'B');
          });
        }
      } catch (e) {
        console.error("Multi-focus segmentation B error:", e);
      }
    }

    this.framesSinceLastSegment++;
    if (this.framesSinceLastSegment >= this.segmentationInterval) {
      this.framesSinceLastSegment = 0;
    }

    // Return multi-focus data
    return {
      multiFocus: true,
      maskA: this.maskA,
      maskB: this.maskB,
      width: this.width,
      height: this.height,
      mask: this.maskA
    };
  }

  _handleMultiResult(result, which) {
    if (result && result.confidenceMasks && result.confidenceMasks.length > 0) {
      const maskData = result.confidenceMasks[0].getAsFloat32Array();
      const centroid = this.calculateCentroid(maskData, this.width, this.height);

      if (which === 'A') {
        this.maskA = maskData;
        if (centroid) {
          this.lastAreaA = centroid.area;
          this.roiA = { keypoint: { x: centroid.x, y: centroid.y } };
        }
      } else {
        this.maskB = maskData;
        if (centroid) {
          this.lastAreaB = centroid.area;
          this.roiB = { keypoint: { x: centroid.x, y: centroid.y } };
        }
      }

      // Update debug log
      if (this.roiA && this.roiB) {
        this.updateDebugLog(
          `Rack Focus | T: ${this.focusT.toFixed(2)} | A: {${this.roiA.keypoint.x.toFixed(2)},${this.roiA.keypoint.y.toFixed(2)}} | B: {${this.roiB.keypoint.x.toFixed(2)},${this.roiB.keypoint.y.toFixed(2)}}`
        );
      }
    }
  }

  // ── Single-focus result handlers (original) ──

  handleResult(result) {
    if (result && result.confidenceMasks && result.confidenceMasks.length > 0) {
      this.lastMask = result.confidenceMasks[0].getAsFloat32Array();

      // Check if masking failed (compute centroid to update ROI for tracking)
      const centroid = this.calculateCentroid(this.lastMask, this.width, this.height);

      if (centroid) {

        let validTrack = true;
        if (this.lastArea !== null && this.app.state === AppState.TRACKING) {
          // Check if area changed drastically (e.g. 50% change)
          const diff = Math.abs(centroid.area - this.lastArea) / this.lastArea;
          if (diff > 0.5) { // Threshold
            this.updateDebugLog(`Contour lost (Area diff ${(diff * 100).toFixed(0)}%). Resegmenting!`);
            validTrack = false;
          }
        }

        if (validTrack) {
          this.lastArea = centroid.area;
          this.roi = { keypoint: { x: centroid.x, y: centroid.y } };

          if (this.app.state === AppState.SEGMENTING) {
            this.app.setState(AppState.TRACKING);
          }
          this.updateDebugLog(`Tracking at ROI: {x: ${centroid.x.toFixed(2)}, y: ${centroid.y.toFixed(2)}}`);
        } else {
          // Contour failed threshold, fallback to original ROI to try a fresh resegment
          this.roi = { ...this.initialRoi }; // Copy original
          this.app.setState(AppState.SEGMENTING);
        }

      } else {
        this.updateDebugLog("Masking failed. Dropping track.");
        this.handleMaskFailure();
      }
    } else {
      this.updateDebugLog("Result empty. Masking failed.");
      this.handleMaskFailure();
    }
  }

  handleMaskFailure() {
    this.maskFailed = true;
    this.roi = null;
    this.lastMask = null;
    this.lastArea = null;
    if (this.app.state === AppState.TRACKING || this.app.state === AppState.SEGMENTING) {
      this.app.setState(AppState.STREAMING);
    }
  }
}
