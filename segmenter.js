import { FilesetResolver, InteractiveSegmenter, ImageSegmenter } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8";
import { AppState } from './main.js';

export class VideoSegmenter {
  constructor(stateManager) {
    this.app = stateManager;
    this.segmenter = null;
    this.isInitializing = false;

    this.framesSinceLastSegment = 0;
    this.segmentationInterval = 15; // 15 frames

    this.roi = null; // {x, y} normalized
    this.lastMask = null; // Float32Array
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
    this.roi = { keypoint: { x: normalizedX, y: normalizedY } };
    this.framesSinceLastSegment = 0; // trigger immediate segment

    if (this.app.state === AppState.STREAMING || this.app.state === AppState.TRACKING) {
      this.app.setState(AppState.SEGMENTING);
    }
  }

  processFrame(videoElement, timestamp) {
    if (!this.segmenter) return null;

    // Remember dimensions for WebGL texture creation later
    this.width = videoElement.videoWidth;
    this.height = videoElement.videoHeight;

    if (!this.roi) return null;

    if (this.framesSinceLastSegment === 0) {
      // InteractiveSegmenter API
      try {
        this.segmenter.segment(videoElement, this.roi, (result) => {
          if (result && result.confidenceMasks && result.confidenceMasks.length > 0) {
            this.lastMask = result.confidenceMasks[0].getAsFloat32Array();
            if (this.app.state === AppState.SEGMENTING) {
              this.app.setState(AppState.TRACKING);
            }
          }
        });
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
}
