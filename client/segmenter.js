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
    this.previousRoi = null; // For focus-follow damping
    this.lastMask = null; // Float32Array
    this.lastArea = null;
    this.width = 0;
    this.height = 0;

    // ── Focus Follow damping factor ──
    // Lower = smoother/slower follow, higher = snappier
    this.followDamping = 0.08;

    // ── Multi-Focus state ──
    this.multiFocusMode = false;
    this.roiA = null;       // { keypoint: {x, y} }
    this.roiB = null;
    this.previousRoiA = null; // For focus-follow damping on A
    this.previousRoiB = null; // For focus-follow damping on B
    this.initialRoiA = null;
    this.initialRoiB = null;
    this.maskA = null;      // Float32Array
    this.maskB = null;      // Float32Array
    this.lastAreaA = null;
    this.lastAreaB = null;
    this.multiFocusClickCount = 0;

    // ── Priority Focus state ──
    this.priorityFocusMode = false;
    this.subjects = [];         // Array of { id, roi, previousRoi, initialRoi, mask, lastArea, priority }
    this.subjectIdCounter = 0;
    this.priorityWeights = [1.0, 0.7, 0.5, 0.35, 0.25];
    this.maxSubjects = 5;
    this.compositeMask = null;  // Final weighted Float32Array
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

  // ── Lerp helper for smooth focus follow ──
  static lerp(a, b, t) {
    return a + (b - a) * t;
  }

  // ── Single-click focus (existing) ──

  setTarget(normalizedX, normalizedY) {
    if (!this.segmenter) return;

    // Clear multi-focus and priority mode on regular click
    this.clearMultiFocus();
    this.clearPriorityFocus();

    this.initialRoi = { keypoint: { x: normalizedX, y: normalizedY } };
    this.roi = { ...this.initialRoi };
    this.previousRoi = null; // Reset damping so first frame snaps to click
    this.framesSinceLastSegment = 0; // force immediate segment
    this.maskFailed = false;
    this.lastArea = null;

    if (this.app.state === AppState.STREAMING || this.app.state === AppState.TRACKING || this.app.state === AppState.MULTI_FOCUS || this.app.state === AppState.PRIORITY_FOCUS) {
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
    this.previousRoiA = null;
    this.previousRoiB = null;
    this.initialRoiA = null;
    this.initialRoiB = null;
    this.maskA = null;
    this.maskB = null;
    this.lastAreaA = null;
    this.lastAreaB = null;
  }

  // ── Priority Focus (N-subject with weights) ──

  addPrioritySubject(normalizedX, normalizedY) {
    if (!this.segmenter) return null;
    if (this.subjects.length >= this.maxSubjects) {
      this.updateDebugLog(`Max ${this.maxSubjects} subjects reached.`);
      return null;
    }

    // First subject clears single/multi focus
    if (!this.priorityFocusMode) {
      this.clearMultiFocus();
      this.roi = null;
      this.lastMask = null;
      this.previousRoi = null;
      this.priorityFocusMode = true;
    }

    const id = ++this.subjectIdCounter;
    const priority = this.subjects.length; // 0-indexed priority
    const subject = {
      id,
      roi: { keypoint: { x: normalizedX, y: normalizedY } },
      previousRoi: null,
      initialRoi: { keypoint: { x: normalizedX, y: normalizedY } },
      mask: null,
      lastArea: null,
      priority
    };
    this.subjects.push(subject);
    this.framesSinceLastSegment = 0;

    if (this.app.state !== AppState.LOADING_MODEL && this.app.state !== AppState.IDLE) {
      this.app.setState(AppState.PRIORITY_FOCUS);
    }

    this.updateDebugLog(`Priority Focus: Added subject #${this.subjects.length} at {${normalizedX.toFixed(2)}, ${normalizedY.toFixed(2)}}`);
    return id;
  }

  removePrioritySubject(id) {
    this.subjects = this.subjects.filter(s => s.id !== id);
    // Reassign priorities
    this.subjects.forEach((s, i) => { s.priority = i; });

    if (this.subjects.length === 0) {
      this.clearPriorityFocus();
      if (this.app.state === AppState.PRIORITY_FOCUS) {
        this.app.setState(AppState.STREAMING);
      }
    }
    this.framesSinceLastSegment = 0;
    this.updateDebugLog(`Priority Focus: Removed subject. ${this.subjects.length} remaining.`);
  }

  reorderPriorities(orderedIds) {
    const reordered = [];
    for (const id of orderedIds) {
      const subject = this.subjects.find(s => s.id === id);
      if (subject) reordered.push(subject);
    }
    // Reassign priorities
    reordered.forEach((s, i) => { s.priority = i; });
    this.subjects = reordered;
    this.framesSinceLastSegment = 0;
    this.updateDebugLog(`Priority Focus: Reordered. #1=${orderedIds[0]}`);
  }

  clearPriorityFocus() {
    this.priorityFocusMode = false;
    this.subjects = [];
    this.compositeMask = null;
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

    // ── Priority Focus mode ──
    if (this.priorityFocusMode && this.subjects.length > 0) {
      return this._processPriorityFrame(videoElement, timestamp);
    }

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
      height: this.height,
      focusCenter: this.roi ? { x: this.roi.keypoint.x, y: this.roi.keypoint.y } : null
    };
  }

  // ── Priority Focus frame processing ──

  _processPriorityFrame(videoElement, timestamp) {
    if (this.framesSinceLastSegment === 0) {
      // Segment each subject sequentially with offset timestamps
      for (let i = 0; i < this.subjects.length; i++) {
        const subject = this.subjects[i];
        try {
          const roiCopy = { keypoint: { ...subject.roi.keypoint } };
          const ts = timestamp + i; // offset to avoid collision
          const idx = i;
          if (this.segmenter.segmentForVideo) {
            this.segmenter.segmentForVideo(videoElement, roiCopy, ts, (result) => {
              this._handlePriorityResult(result, idx);
            });
          } else {
            this.segmenter.segment(videoElement, roiCopy, (result) => {
              this._handlePriorityResult(result, idx);
            });
          }
        } catch (e) {
          console.error(`Priority segmentation error for subject ${i}:`, e);
        }
      }
    }

    this.framesSinceLastSegment++;
    if (this.framesSinceLastSegment >= this.segmentationInterval) {
      this.framesSinceLastSegment = 0;
    }

    // Composite all masks
    this._compositeWeightedMask();

    // Priority focus center = highest priority subject's position
    const topSubject = this.subjects[0];
    const fc = topSubject ? { x: topSubject.roi.keypoint.x, y: topSubject.roi.keypoint.y } : null;

    return {
      mask: this.compositeMask,
      width: this.width,
      height: this.height,
      priorityFocus: true,
      focusCenter: fc
    };
  }

  _handlePriorityResult(result, subjectIndex) {
    if (subjectIndex >= this.subjects.length) return;
    const subject = this.subjects[subjectIndex];

    if (result && result.confidenceMasks && result.confidenceMasks.length > 0) {
      const maskData = result.confidenceMasks[0].getAsFloat32Array();
      const centroid = this.calculateCentroid(maskData, this.width, this.height);

      subject.mask = maskData;

      if (centroid) {
        subject.lastArea = centroid.area;
        // Focus Follow: lerp toward centroid
        if (subject.previousRoi) {
          const smoothX = VideoSegmenter.lerp(subject.previousRoi.x, centroid.x, this.followDamping);
          const smoothY = VideoSegmenter.lerp(subject.previousRoi.y, centroid.y, this.followDamping);
          subject.roi = { keypoint: { x: smoothX, y: smoothY } };
          subject.previousRoi = { x: smoothX, y: smoothY };
        } else {
          subject.roi = { keypoint: { x: centroid.x, y: centroid.y } };
          subject.previousRoi = { x: centroid.x, y: centroid.y };
        }
      }
    }

    // Update debug
    const parts = this.subjects.map((s, i) =>
      `#${i + 1}(w=${this.priorityWeights[s.priority]?.toFixed(1) || '?'}): {${s.roi.keypoint.x.toFixed(2)},${s.roi.keypoint.y.toFixed(2)}}`
    );
    this.updateDebugLog(`Priority | ${parts.join(' | ')}`);
  }

  _compositeWeightedMask() {
    const totalPixels = this.width * this.height;
    if (totalPixels === 0) return;

    // Reuse or create composite buffer
    if (!this.compositeMask || this.compositeMask.length !== totalPixels) {
      this.compositeMask = new Float32Array(totalPixels);
    } else {
      this.compositeMask.fill(0);
    }

    // For each subject, blend its mask with priority weight (max wins)
    for (const subject of this.subjects) {
      if (!subject.mask || subject.mask.length !== totalPixels) continue;
      const weight = this.priorityWeights[subject.priority] || 0.2;

      for (let i = 0; i < totalPixels; i++) {
        const weighted = subject.mask[i] * weight;
        if (weighted > this.compositeMask[i]) {
          this.compositeMask[i] = weighted;
        }
      }
    }
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

    // Multi-focus center = midpoint of A and B (or whichever exists)
    let fc = null;
    if (this.roiA && this.roiB) {
      fc = {
        x: (this.roiA.keypoint.x + this.roiB.keypoint.x) / 2,
        y: (this.roiA.keypoint.y + this.roiB.keypoint.y) / 2
      };
    } else if (this.roiA) {
      fc = { x: this.roiA.keypoint.x, y: this.roiA.keypoint.y };
    }

    return {
      multiFocus: true,
      maskA: this.maskA,
      maskB: this.maskB,
      width: this.width,
      height: this.height,
      mask: this.maskA,
      focusCenter: fc
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
          // Focus Follow: lerp ROI A toward centroid
          if (this.previousRoiA) {
            const smoothX = VideoSegmenter.lerp(this.previousRoiA.x, centroid.x, this.followDamping);
            const smoothY = VideoSegmenter.lerp(this.previousRoiA.y, centroid.y, this.followDamping);
            this.roiA = { keypoint: { x: smoothX, y: smoothY } };
            this.previousRoiA = { x: smoothX, y: smoothY };
          } else {
            // First frame: snap directly
            this.roiA = { keypoint: { x: centroid.x, y: centroid.y } };
            this.previousRoiA = { x: centroid.x, y: centroid.y };
          }
        }
      } else {
        this.maskB = maskData;
        if (centroid) {
          this.lastAreaB = centroid.area;
          // Focus Follow: lerp ROI B toward centroid
          if (this.previousRoiB) {
            const smoothX = VideoSegmenter.lerp(this.previousRoiB.x, centroid.x, this.followDamping);
            const smoothY = VideoSegmenter.lerp(this.previousRoiB.y, centroid.y, this.followDamping);
            this.roiB = { keypoint: { x: smoothX, y: smoothY } };
            this.previousRoiB = { x: smoothX, y: smoothY };
          } else {
            this.roiB = { keypoint: { x: centroid.x, y: centroid.y } };
            this.previousRoiB = { x: centroid.x, y: centroid.y };
          }
        }
      }

      // Update debug log
      if (this.roiA && this.roiB) {
        this.updateDebugLog(
          `Multi-Focus | A: {${this.roiA.keypoint.x.toFixed(2)},${this.roiA.keypoint.y.toFixed(2)}} | B: {${this.roiB.keypoint.x.toFixed(2)},${this.roiB.keypoint.y.toFixed(2)}}`
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

          // Focus Follow: lerp ROI smoothly toward centroid
          if (this.previousRoi) {
            const smoothX = VideoSegmenter.lerp(this.previousRoi.x, centroid.x, this.followDamping);
            const smoothY = VideoSegmenter.lerp(this.previousRoi.y, centroid.y, this.followDamping);
            this.roi = { keypoint: { x: smoothX, y: smoothY } };
            this.previousRoi = { x: smoothX, y: smoothY };
          } else {
            // First frame after click: snap directly to centroid
            this.roi = { keypoint: { x: centroid.x, y: centroid.y } };
            this.previousRoi = { x: centroid.x, y: centroid.y };
          }

          if (this.app.state === AppState.SEGMENTING) {
            this.app.setState(AppState.TRACKING);
          }
          this.updateDebugLog(`Tracking at ROI: {x: ${this.roi.keypoint.x.toFixed(2)}, y: ${this.roi.keypoint.y.toFixed(2)}}`);
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
    this.previousRoi = null;
    this.lastMask = null;
    this.lastArea = null;
    if (this.app.state === AppState.TRACKING || this.app.state === AppState.SEGMENTING) {
      this.app.setState(AppState.STREAMING);
    }
  }
}
