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

  setTarget(normalizedX, normalizedY) {
    if (!this.segmenter) return;
    this.initialRoi = { keypoint: { x: normalizedX, y: normalizedY } };
    this.roi = { ...this.initialRoi };
    this.framesSinceLastSegment = 0; // force immediate segment
    this.maskFailed = false;
    this.lastArea = null;

    if (this.app.state === AppState.STREAMING || this.app.state === AppState.TRACKING) {
      this.app.setState(AppState.SEGMENTING);
    }
    this.updateDebugLog(`Target set at ROI: {x: ${normalizedX.toFixed(2)}, y: ${normalizedY.toFixed(2)}}`);
  }

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

  processFrame(videoElement, timestamp) {
    if (!this.segmenter || this.maskFailed) return null;

    this.width = videoElement.videoWidth;
    this.height = videoElement.videoHeight;

    if (!this.roi) return null;

    // The user requested: "re segmentation shud happen only when masking fails"
    // So we assume "segmentation" is the heavy process, and "tracking" is holding it?
    // Actually, InteractiveSegmenter requires the ROI to track.
    if (this.framesSinceLastSegment === 0) {
      try {
        // InteractiveSegmenter API
        // If segmentForVideo exists we use it, otherwise fallback to segment
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
