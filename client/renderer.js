export class Renderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = this.canvas.getContext('webgl2', { alpha: false, depth: false, antialias: false, stencil: false });
    if (!this.gl) {
      console.error("WebGL2 not supported.");
      alert("Your browser does not support WebGL2.");
      return;
    }

    this.program = null;
    this.videoTexture = null;
    this.maskTexture = null;
    this.maskTextureB = null;
    this.initWebGL();
  }

  initWebGL() {
    const gl = this.gl;

    const vsSource = `#version 300 es
            in vec2 a_position;
            in vec2 a_texCoord;
            out vec2 v_texCoord;
            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
                v_texCoord = a_texCoord;
            }
        `;

    const fsSource = `#version 300 es
            precision mediump float;
            in vec2 v_texCoord;
            uniform sampler2D u_image;
            uniform sampler2D u_mask;
            uniform sampler2D u_maskB;
            uniform bool u_debug;
            uniform bool u_multiFocus;
            uniform bool u_hasMask;    // false = no focus target yet, show raw video
            uniform int u_lightMode;   // 0=blur, 1=warm, 2=cool, 3=spotlight, 4=vignette
            uniform float u_exposure;
            uniform float u_blurStrength;
            uniform vec2 u_focusCenter; // Normalized subject center for depth blur
            out vec4 outColor;

            // ── Depth-aware Gaussian blur ──
            // depthFactor: 0.0 (near subject) → 1.0 (far away)
            vec4 blurBG(vec2 uv, float depthFactor) {
                vec4 color = vec4(0.0);
                vec2 texSize = vec2(textureSize(u_image, 0));
                vec2 texelSize = 1.0 / texSize;
                float totalWeight = 0.0;
                // Reduced max radius so gaps between samples stay small
                float radius = mix(1.0, 3.0, depthFactor) * max(u_blurStrength, 0.0);
                // Gaussian sigma scales with radius for consistent softness
                float sigma = max(radius * 2.0, 1.0);
                float invTwoSigmaSq = 1.0 / (2.0 * sigma * sigma);
                // 13x13 kernel = 169 samples, keeps gaps ≤ 3 texels apart
                for (float x = -6.0; x <= 6.0; x++) {
                    for (float y = -6.0; y <= 6.0; y++) {
                        float w = exp(-(x * x + y * y) * invTwoSigmaSq);
                        color += texture(u_image, uv + vec2(x, y) * texelSize * radius) * w;
                        totalWeight += w;
                    }
                }
                return color / totalWeight;
            }

            // ── Desaturate helper ──
            vec3 desaturate(vec3 c, float amount) {
                float lum = dot(c, vec3(0.299, 0.587, 0.114));
                return mix(c, vec3(lum), amount);
            }

            // ── Apply lighting effect on background color ──
            vec4 applyLighting(vec4 bg, vec2 uv) {
                if (u_lightMode == 0) {
                    // Mode 0: Plain blur (no color change)
                    return bg;
                }
                if (u_lightMode == 1) {
                    // Mode 1: Warm Studio — amber tint + slight brightness lift
                    vec3 warm = mix(bg.rgb, vec3(1.0, 0.82, 0.55), 0.28);
                    warm *= 1.08;
                    return vec4(warm, 1.0);
                }
                if (u_lightMode == 2) {
                    // Mode 2: Cool Night — blue-teal shift + slight dim
                    vec3 cool = mix(bg.rgb, vec3(0.3, 0.55, 0.95), 0.30);
                    cool *= 0.85;
                    return vec4(cool, 1.0);
                }
                if (u_lightMode == 3) {
                    // Mode 3: Spotlight — radial darkening from center
                    float dist = length(uv - vec2(0.5));
                    float falloff = smoothstep(0.15, 0.75, dist);
                    vec3 lit = bg.rgb * mix(1.0, 0.12, falloff);
                    return vec4(lit, 1.0);
                }
                if (u_lightMode == 4) {
                    // Mode 4: Vignette — soft edge darkening + desaturation
                    float dist = length(uv - vec2(0.5));
                    float vig = smoothstep(0.25, 0.85, dist);
                    vec3 c = desaturate(bg.rgb, vig * 0.6);
                    c *= mix(1.0, 0.3, vig);
                    return vec4(c, 1.0);
                }
                return bg;
            }

            void main() {
                vec4 rawColor = texture(u_image, v_texCoord);

                // No focus target set yet → pass through raw video
                if (!u_hasMask) {
                    outColor = rawColor;
                    outColor.rgb *= u_exposure;
                    return;
                }

                float maskValA = texture(u_mask, v_texCoord).r;
                float maskValB = texture(u_maskB, v_texCoord).r;

                // Compute depth factor: distance from focus center, clamped to [0,1]
                float depthDist = length(v_texCoord - u_focusCenter);
                float depthFactor = clamp(depthDist / 0.7, 0.0, 1.0); // normalize so ~0.7 diagonal = max

                if (u_debug) {
                    if (u_multiFocus) {
                        vec4 tinted = rawColor;
                        tinted = mix(tinted, vec4(1.0, 0.0, 0.0, 1.0), maskValA * 0.5);
                        tinted = mix(tinted, vec4(0.0, 0.3, 1.0, 1.0), maskValB * 0.5);
                        outColor = tinted;
                    } else {
                        outColor = mix(rawColor, vec4(1.0, 0.0, 0.0, 1.0), maskValA * 0.5);
                    }
                    outColor.rgb *= u_exposure;
                    return;
                }

                // Combine masks based on mode
                float maskVal = u_multiFocus ? max(maskValA, maskValB) : maskValA;

                // Apply smoothstep to soften the mask edge slightly for a more natural transition
                float alpha = smoothstep(0.0, 0.8, maskVal);

                // Base fallback when not masked
                if (alpha < 0.001) {
                    outColor = applyLighting(blurBG(v_texCoord, depthFactor), v_texCoord);
                    outColor.rgb *= u_exposure;
                    return;
                }
                
                // When fully masked, save some computation
                if (alpha > 0.999) {
                    outColor = rawColor;
                    outColor.rgb *= u_exposure;
                    return;
                }

                // Calculate the original blurred background
                vec4 blurredBg = blurBG(v_texCoord, depthFactor);
                // Calculate the lighted background
                vec4 litBg = applyLighting(blurredBg, v_texCoord);

                // Un-premultiply the foreground color
                // We assume rawColor = fg * alpha + blurredBg * (1 - alpha)
                vec3 fg = (rawColor.rgb - blurredBg.rgb * (1.0 - alpha)) / max(alpha, 0.001);

                // Clamp foreground to prevent artifacting from over/under shooting
                fg = clamp(fg, 0.0, 1.0);

                // Mix the modified background with the un-premultiplied foreground
                outColor = vec4(mix(litBg.rgb, fg, alpha), 1.0);
                
                // Final Exposure Adjustment
                outColor.rgb *= u_exposure;
            }
        `;

    this.program = this.createProgram(gl, vsSource, fsSource);

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
      -1.0, 1.0, 1.0, -1.0, 1.0, 1.0
    ]), gl.STATIC_DRAW);

    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 1.0, 1.0, 0.0
    ]), gl.STATIC_DRAW);

    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    const posLoc = gl.getAttribLocation(this.program, "a_position");
    gl.enableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    const texCoordLoc = gl.getAttribLocation(this.program, "a_texCoord");
    gl.enableVertexAttribArray(texCoordLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 0, 0);

    // Setup textures
    // TEXTURE0 = video
    gl.activeTexture(gl.TEXTURE0);
    this.videoTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

    // WebGL2 Extension for Float32 Textures
    gl.getExtension('EXT_color_buffer_float');

    // TEXTURE1 = mask A (or single mask)
    gl.activeTexture(gl.TEXTURE1);
    this.maskTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    // TEXTURE2 = mask B (rack focus)
    gl.activeTexture(gl.TEXTURE2);
    this.maskTextureB = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.maskTextureB);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    gl.useProgram(this.program);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_image"), 0);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_mask"), 1);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_maskB"), 2);

    this.vao = vao;
  }

  createProgram(gl, vsSource, fsSource) {
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, vsSource);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) console.error('VS:', gl.getShaderInfoLog(vs));

    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fsSource);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) console.error('FS:', gl.getShaderInfoLog(fs));

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    return prog;
  }

  updateMaskTexture(textureUnit, texture, maskArray, width, height) {
    const gl = this.gl;
    gl.activeTexture(textureUnit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    if (maskArray && width > 0 && height > 0) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, maskArray);
    } else {
      const dummy = new Float32Array(1).fill(0.0);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1, 1, 0, gl.RED, gl.FLOAT, dummy);
    }
  }

  render(videoElement, maskData, isDebug, lightingMode = 0, exposure = 1.0, blurStrength = 1.0) {
    if (!this.gl) return;
    const gl = this.gl;

    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    // Upload video texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, videoElement);

    const isMultiFocus = maskData && maskData.multiFocus;

    if (isMultiFocus) {
      // Multi focus: upload both masks
      this.updateMaskTexture(gl.TEXTURE1, this.maskTexture, maskData.maskA, maskData.width, maskData.height);
      this.updateMaskTexture(gl.TEXTURE2, this.maskTextureB, maskData.maskB, maskData.width, maskData.height);
    } else if (maskData) {
      // Single focus: upload mask A only
      this.updateMaskTexture(gl.TEXTURE1, this.maskTexture, maskData.mask, maskData.width, maskData.height);
      this.updateMaskTexture(gl.TEXTURE2, this.maskTextureB, null, 1, 1);
    } else {
      this.updateMaskTexture(gl.TEXTURE1, this.maskTexture, null, 1, 1);
      this.updateMaskTexture(gl.TEXTURE2, this.maskTextureB, null, 1, 1);
    }

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_debug"), isDebug ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_multiFocus"), isMultiFocus ? 1 : 0);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_lightMode"), lightingMode);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_exposure"), exposure);
    gl.uniform1f(gl.getUniformLocation(this.program, "u_blurStrength"), blurStrength);

    // hasMask = true only when we have actual mask data from a click
    const hasMask = maskData && (maskData.mask || maskData.maskA || maskData.multiFocus || maskData.priorityFocus);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_hasMask"), hasMask ? 1 : 0);

    // Upload focus center for depth blur
    const fcX = (maskData && maskData.focusCenter) ? maskData.focusCenter.x : 0.5;
    const fcY = (maskData && maskData.focusCenter) ? maskData.focusCenter.y : 0.5;
    gl.uniform2f(gl.getUniformLocation(this.program, "u_focusCenter"), fcX, fcY);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}