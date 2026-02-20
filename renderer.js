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
                v_texCoord = vec2(a_texCoord.x, 1.0 - a_texCoord.y); // Flip Y 
            }
        `;

    const fsSource = `#version 300 es
            precision mediump float;
            in vec2 v_texCoord;
            uniform sampler2D u_image;
            uniform sampler2D u_mask;
            uniform bool u_debug;
            out vec4 outColor;

            void main() {
                float maskVal = texture(u_mask, v_texCoord).r;
                vec4 rawColor = texture(u_image, v_texCoord);
                
                if (u_debug) {
                    outColor = mix(rawColor, vec4(1.0, 0.0, 0.0, 1.0), maskVal * 0.5);
                    return;
                }

                if (maskVal > 0.1) {
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
    gl.activeTexture(gl.TEXTURE0);
    this.videoTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

    // WebGL2 Extension for Float32 Textures (EXT_color_buffer_float in WebGL2 allows R32F)
    gl.getExtension('EXT_color_buffer_float');

    gl.activeTexture(gl.TEXTURE1);
    this.maskTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    gl.useProgram(this.program);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_image"), 0);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_mask"), 1);

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

  updateMaskTexture(maskArray, width, height) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTexture);
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

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, videoElement);

    if (maskData) {
      this.updateMaskTexture(maskData.mask, maskData.width, maskData.height);
    } else {
      this.updateMaskTexture(null, 1, 1);
    }

    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.uniform1i(gl.getUniformLocation(this.program, "u_debug"), isDebug ? 1 : 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}
