// Pos-processamento: brilho, efeito maquete, vinheta e tom de cor.
//
// Escrito aqui porque o `EffectComposer` dos exemplos do Three.js importa o pacote por
// nome, o que nao funciona com a biblioteca embutida na pagina.

const VERTICE_TELA = "varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }";

class Pos {
  constructor(renderizador) {
    this.renderizador = renderizador;
    this.tilt = 0;
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.quad.frustumCulled = false;
    this.cenaTela = new THREE.Scene();
    this.cenaTela.add(this.quad);

    const passe = function (fragmentShader, uniforms) {
      return new THREE.ShaderMaterial({ vertexShader: VERTICE_TELA, fragmentShader: fragmentShader, uniforms: uniforms, depthTest: false, depthWrite: false, toneMapped: false });
    };
    this.brilho = passe(`
      uniform sampler2D tCena; varying vec2 vUv;
      void main(){ vec3 c = texture2D(tCena, vUv).rgb; float l = max(c.r, max(c.g, c.b));
        gl_FragColor = vec4(c * smoothstep(1.35, 2.3, l), 1.0); }`, { tCena: { value: null } });
    this.copia = passe(`
      uniform sampler2D tCena; varying vec2 vUv;
      void main(){ gl_FragColor = vec4(min(texture2D(tCena, vUv).rgb, vec3(4.0)), 1.0); }`, { tCena: { value: null } });
    this.borrao = passe(`
      uniform sampler2D tCena; uniform vec2 passo; varying vec2 vUv;
      void main(){
        vec3 c = texture2D(tCena, vUv).rgb * 0.227;
        c += (texture2D(tCena, vUv + passo * 1.385).rgb + texture2D(tCena, vUv - passo * 1.385).rgb) * 0.316;
        c += (texture2D(tCena, vUv + passo * 3.231).rgb + texture2D(tCena, vUv - passo * 3.231).rgb) * 0.070;
        gl_FragColor = vec4(c, 1.0); }`, { tCena: { value: null }, passo: { value: new THREE.Vector2() } });
    this.final = new THREE.ShaderMaterial({
      vertexShader: VERTICE_TELA, depthTest: false, depthWrite: false,
      uniforms: { tCena: { value: null }, tBrilho: { value: null }, tBorrao: { value: null }, tilt: { value: 0 } },
      fragmentShader: `
        uniform sampler2D tCena; uniform sampler2D tBrilho; uniform sampler2D tBorrao; uniform float tilt;
        varying vec2 vUv;
        void main(){
          vec3 c = texture2D(tCena, vUv).rgb;
          float fora = smoothstep(0.1, 0.42, abs(vUv.y - 0.46)) * tilt;
          c = mix(c, texture2D(tBorrao, vUv).rgb, fora);
          c += texture2D(tBrilho, vUv).rgb * 0.45;
          float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
          c = mix(vec3(l), c, 1.1);
          vec2 q = vUv - 0.5;
          c *= 1.0 - dot(q, q) * 0.55;
          gl_FragColor = vec4(c, 1.0);
          #include <tonemapping_fragment>
          #include <colorspace_fragment>
        }`,
    });
  }

  redimensionar(w, h) {
    [this.rtCena, this.rtA, this.rtB, this.rtC].forEach(function (rt) { if (rt) rt.dispose(); });
    const opcoes = { type: THREE.HalfFloatType, depthBuffer: false };
    this.rtCena = new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, samples: 4 });
    const pw = Math.max(2, Math.round(w / 4));
    const ph = Math.max(2, Math.round(h / 4));
    this.rtA = new THREE.WebGLRenderTarget(pw, ph, opcoes);
    this.rtB = new THREE.WebGLRenderTarget(pw, ph, opcoes);
    this.rtC = new THREE.WebGLRenderTarget(pw, ph, opcoes);
    this.passoX = 1 / pw;
    this.passoY = 1 / ph;
  }

  passar(material, destino) {
    this.quad.material = material;
    this.renderizador.setRenderTarget(destino);
    this.renderizador.render(this.cenaTela, this.camera);
  }

  borrar(origem, destino, vezes, escala) {
    let leitura = origem;
    for (let i = 0; i < vezes; i++) {
      this.borrao.uniforms.tCena.value = leitura.texture;
      this.borrao.uniforms.passo.value.set(this.passoX * escala, 0);
      this.passar(this.borrao, this.rtB);
      this.borrao.uniforms.tCena.value = this.rtB.texture;
      this.borrao.uniforms.passo.value.set(0, this.passoY * escala);
      this.passar(this.borrao, destino);
      leitura = destino;
    }
  }

  render(cena, camera) {
    const r = this.renderizador;
    r.setRenderTarget(this.rtCena);
    r.clear();
    r.render(cena, camera);

    this.brilho.uniforms.tCena.value = this.rtCena.texture;
    this.passar(this.brilho, this.rtA);
    this.borrar(this.rtA, this.rtA, 2, 1.4);

    if (this.tilt > 0.01) {
      this.copia.uniforms.tCena.value = this.rtCena.texture;
      this.passar(this.copia, this.rtC);
      this.borrar(this.rtC, this.rtC, 2, 1);
    }

    const u = this.final.uniforms;
    u.tCena.value = this.rtCena.texture;
    u.tBrilho.value = this.rtA.texture;
    u.tBorrao.value = this.rtC.texture;
    u.tilt.value = this.tilt;
    this.passar(this.final, null);
  }
}
