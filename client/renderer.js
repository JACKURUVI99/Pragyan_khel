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
            out vec4 outColor;

            void main() {
                float maskValA = texture(u_mask, v_texCoord).r;
                float maskValB = texture(u_maskB, v_texCoord).r;
                vec4 rawColor = texture(u_image, v_texCoord);
                
                if (u_debug) {
                    if (u_multiFocus) {
                        // Debug: Red for mask A, Blue for mask B
                        vec4 tinted = rawColor;
                        tinted = mix(tinted, vec4(1.0, 0.0, 0.0, 1.0), maskValA * 0.5);
                        tinted = mix(tinted, vec4(0.0, 0.3, 1.0, 1.0), maskValB * 0.5);
                        outColor = tinted;
                    } else {
                        outColor = mix(rawColor, vec4(1.0, 0.0, 0.0, 1.0), maskValA * 0.5);
                    }
                    return;
                }

                // ── Multi-Focus mode: both regions stay sharp ──
                if (u_multiFocus) {
                    float sharpness = max(maskValA, maskValB);

                    if (sharpness > 0.1) {
                        outColor = rawColor;
                    } else {
                        // Blur everything outside both masks
                        vec4 color = vec4(0.0);
                        vec2 texSize = vec2(textureSize(u_image, 0));
                        vec2 texelSize = 1.0 / texSize;
                        float total = 0.0;
                        float radius = 4.0;

                        for (float x = -2.0; x <= 2.0; x++) {
                            for (float y = -2.0; y <= 2.0; y++) {
                                color += texture(u_image, v_texCoord + vec2(x, y) * texelSize * radius);
                                total += 1.0;
                            }
                        }
                        outColor = color / total;
                    }
                    return;
                }

                // ── Single focus mode (original) ──
                if (maskValA > 0.1) {
                    outColor = rawColor;
                } else {
                    // Simple box blur inline
                    vec4 color = vec4(0.0);
                    vec2 texSize = vec2(textureSize(u_image, 0));
                    vec2 texelSize = 1.0 / texSize;
                    float total = 0.0;
                    float radius = 4.0; 

                    for (float x = -2.0; x <= 2.0; x++) {
                        for (float y = -2.0; y <= 2.0; y++) {
                            color += texture(u_image, v_texCoord + vec2(x, y) * texelSize * radius);
                            total += 1.0;
                        }
                    }
                    outColor = color / total;
                }
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

  render(videoElement, maskData, isDebug) {
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

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}
