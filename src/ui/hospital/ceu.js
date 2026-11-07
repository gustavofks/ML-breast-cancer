// Ceu, sol e luzes. Uma hora do dia define tudo: as cores vem de `ceuDaHora`.

function criarAmbiente(renderizador) {
  // Reflexos sem arquivo HDR: uma sala branca com paineis de luz, gerada uma vez.
  const cena = new THREE.Scene();
  const fundo = new THREE.Mesh(new THREE.SphereGeometry(20, 24, 16), new THREE.MeshBasicMaterial({ color: "#c9d6dc", side: THREE.BackSide }));
  cena.add(fundo);
  [[0, 12, 0, 16, 16], [12, 5, -8, 8, 8], [-12, 7, 8, 8, 6]].forEach(function (p) {
    const painel = new THREE.Mesh(new THREE.PlaneGeometry(p[3], p[4]), new THREE.MeshBasicMaterial({ color: new THREE.Color(5, 4.8, 4.4), side: THREE.DoubleSide }));
    painel.position.set(p[0], p[1], p[2]);
    painel.lookAt(0, 0, 0);
    cena.add(painel);
  });
  const gerador = new THREE.PMREMGenerator(renderizador);
  const textura = gerador.fromScene(cena, 0.03).texture;
  gerador.dispose();
  return textura;
}

function criarNuvens(cena) {
  const material = new THREE.MeshStandardMaterial({ color: "#ffffff", roughness: 1, flatShading: true });
  const nuvens = [];
  for (let i = 0; i < 11; i++) {
    const nuvem = new THREE.Group();
    const partes = 3 + (i % 3);
    for (let j = 0; j < partes; j++) {
      const m = new THREE.Mesh(new THREE.IcosahedronGeometry(3.4 + ((i + j) % 3) * 1.1, 1), material);
      m.position.set((j - partes / 2) * 4.2, ((j * 7) % 3) * 0.6, ((i + j) % 2) * 2.4);
      m.scale.y = 0.52;
      nuvem.add(m);
    }
    nuvem.position.set(-120 + i * 27, 44 + ((i * 13) % 5) * 3.6, -86 + ((i * 41) % 9) * 21);
    cena.add(nuvem);
    nuvens.push(nuvem);
  }
  return { grupos: nuvens, material: material };
}

function criarEstrelas(cena) {
  const posicoes = [];
  for (let i = 0; i < 520; i++) {
    const a = (i * 2.399963) % (Math.PI * 2);
    const y = 0.08 + ((i * 53) % 100) / 108;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    posicoes.push(Math.cos(a) * r * 280, y * 280, Math.sin(a) * r * 280);
  }
  const geometria = new THREE.BufferGeometry();
  geometria.setAttribute("position", new THREE.Float32BufferAttribute(posicoes, 3));
  const material = new THREE.PointsMaterial({ color: "#ffffff", size: 1.5, sizeAttenuation: false, transparent: true, opacity: 0, fog: false, depthWrite: false });
  cena.add(new THREE.Points(geometria, material));
  return material;
}

// Postes do caminho: a noite, a luz deles passa pelo brilho do pos-processamento.
function criarPostes(cena) {
  const luz = new THREE.MeshStandardMaterial({ color: "#3a3a34", emissive: "#ffd9a0", emissiveIntensity: 0 });
  const haste = material("#59666a", { metalness: 0.4 });
  [[-24.5, 3.6], [-24.5, -3.6], [-30, 3.2], [-30, -3.2], [-35.5, 3.2], [-35.5, -3.2]].forEach(function (p) {
    cena.add(caixa(0.1, 3, 0.1, haste, p[0], 1.5, p[1]));
    const globo = new THREE.Mesh(new THREE.SphereGeometry(0.2, 14, 12), luz);
    globo.position.set(p[0], 3.1, p[1]);
    cena.add(globo);
  });
  return luz;
}

class Ceu {
  constructor(cena, renderizador, tetoComLuz) {
    this.cena = cena;
    cena.environment = criarAmbiente(renderizador);
    cena.fog = new THREE.Fog("#cfe4ea", 90, 250);
    this.hemi = new THREE.HemisphereLight("#fffaf2", "#e9e4da", 1.2);
    this.sol = new THREE.DirectionalLight("#fff3df", 1.5);
    const sol = this.sol;
    sol.target.position.set(6, 0, 0);
    sol.castShadow = true;
    sol.shadow.mapSize.set(4096, 4096);
    sol.shadow.bias = -0.0004;
    sol.shadow.normalBias = 0.03;
    Object.assign(sol.shadow.camera, { left: -48, right: 48, top: 30, bottom: -30, near: 5, far: 130 });
    cena.add(this.hemi, sol, sol.target);

    // Uma luz por sala e tres no corredor, sem sombra.
    const pontos = Object.values(L.SALAS).map(function (s) { return [(s.x0 + s.x1) / 2, (s.z0 + s.z1) / 2]; });
    [[-4.5, 0], [6.5, 0], [17.5, 0]].forEach(function (p) { pontos.push(p); });
    this.luzes = pontos.map(function (p) {
      const luz = new THREE.PointLight("#fff1dc", 6, 12, 1.6);
      luz.position.set(p[0], 3, p[1]);
      cena.add(luz);
      return luz;
    });

    this.domo = new THREE.Mesh(
      new THREE.SphereGeometry(300, 32, 20),
      new THREE.ShaderMaterial({
        side: THREE.BackSide, depthWrite: false, fog: false,
        uniforms: {
          topo: { value: new THREE.Color() }, horizonte: { value: new THREE.Color() },
          corSol: { value: new THREE.Color() }, direcaoSol: { value: new THREE.Vector3(0, 1, 0) }, forcaSol: { value: 1 },
        },
        vertexShader: "varying vec3 vDir; void main(){ vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
        fragmentShader: `
          uniform vec3 topo; uniform vec3 horizonte; uniform vec3 corSol; uniform vec3 direcaoSol; uniform float forcaSol;
          varying vec3 vDir;
          void main() {
            vec3 d = normalize(vDir);
            vec3 c = mix(horizonte, topo, pow(clamp(d.y, 0.0, 1.0), 0.52));
            float s = max(dot(d, direcaoSol), 0.0);
            c += corSol * (pow(s, 420.0) * 3.0 + pow(s, 14.0) * 0.26) * forcaSol;
            gl_FragColor = vec4(c, 1.0);
          }`,
      })
    );
    this.domo.renderOrder = -10;
    cena.add(this.domo);
    this.nuvens = criarNuvens(cena);
    this.estrelas = criarEstrelas(cena);
    this.postes = criarPostes(cena);
    this.tetoComLuz = tetoComLuz;
    this.definirHora(10);
  }

  definirHora(h) {
    this.hora = h;
    const c = L.ceuDaHora(h);
    const u = this.domo.material.uniforms;
    u.topo.value.set(c.topo);
    u.horizonte.value.set(c.horizonte);
    u.corSol.value.set(c.sol);
    this.cena.fog.color.set(c.horizonte);
    this.cena.background = null;

    const a = Math.min(1, Math.max(0, (h - 5.5) / 14)) * Math.PI;
    const direcao = new THREE.Vector3(-Math.cos(a) * 0.85, Math.max(Math.sin(a), 0.1) * 0.82, 0.42).normalize();
    u.direcaoSol.value.copy(direcao);
    u.forcaSol.value = Math.min(1, c.solI);
    this.sol.position.copy(direcao).multiplyScalar(70).add(this.sol.target.position);
    this.sol.color.set(c.sol);
    this.sol.intensity = c.solI * 0.9;
    this.hemi.color.set(c.ceu);
    this.hemi.groundColor.set(c.chao);
    this.hemi.intensity = c.hemi * 0.68;
    this.cena.environmentIntensity = 0.06 + c.hemi * 0.2;

    const luzes = c.luzes;
    this.luzes.forEach(function (luz) { luz.intensity = 2 + luzes * 11; });
    const noite = Math.min(1, Math.max(0, (h - 18) / 2.2));
    this.estrelas.opacity = Math.min(1, Math.max(0, (h - 19.2) / 1.6)) * 0.9;
    this.postes.emissiveIntensity = noite * 5;
    if (VIDRO.material) {
      VIDRO.material.emissiveIntensity = noite * 2.3;
      VIDRO.material.opacity = 0.22 + noite * 0.42;
    }
    this.nuvens.material.color.set(L.misturarCor("#ffffff", c.horizonte, 0.45));
    this.nuvens.material.emissive.set(c.horizonte);
    this.nuvens.material.emissiveIntensity = 0.12;
  }

  avancar(t, dt) {
    this.nuvens.grupos.forEach(function (nuvem) {
      nuvem.position.x += dt * 0.9;
      if (nuvem.position.x > 160) nuvem.position.x = -160;
    });
  }
}
