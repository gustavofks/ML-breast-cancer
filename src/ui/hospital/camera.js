// Camera com tres comportamentos: orbita, primeira pessoa e voo entre os dois.
//
// Escrita aqui em vez de usar os controles dos exemplos do Three.js: aqueles importam
// o pacote por nome, o que nao funciona com a biblioteca embutida na pagina.

const ALTURA_DOS_OLHOS = 1.62;
const RAIO_DO_VISITANTE = 0.34;

class Camera {
  constructor(camera, tela, colisores) {
    this.camera = camera;
    this.tela = tela;
    this.colisores = colisores;
    this.modo = "orbita";
    this.alvo = new THREE.Vector3(6, 0, 0);
    this.teta = -0.7;
    this.fi = 0.9;
    this.raio = 56;
    this.raioMin = 3;
    this.raioMax = 80;
    this.guinada = 0;
    this.arfagem = 0;
    this.teclas = {};
    this.voo = null;
    this.arrastando = false;
    this.ligarEventos();
  }

  ligarEventos() {
    const tela = this.tela;
    let ultimo = null;
    tela.addEventListener("pointerdown", (e) => {
      this.arrastando = true;
      this.moveu = 0;
      ultimo = { x: e.clientX, y: e.clientY };
      tela.setPointerCapture(e.pointerId);
    });
    tela.addEventListener("pointermove", (e) => {
      if (document.pointerLockElement === tela) {
        this.olhar(e.movementX, e.movementY);
        return;
      }
      if (!this.arrastando || !ultimo) return;
      const dx = e.clientX - ultimo.x;
      const dy = e.clientY - ultimo.y;
      this.moveu += Math.abs(dx) + Math.abs(dy);
      ultimo = { x: e.clientX, y: e.clientY };
      if (this.voo) return;
      if (this.modo === "orbita") {
        this.teta -= dx * 0.006;
        this.fi = Math.min(1.5, Math.max(0.12, this.fi - dy * 0.005));
      } else {
        this.olhar(dx * 1.4, dy * 1.4);
      }
    });
    tela.addEventListener("pointerup", () => { this.arrastando = false; ultimo = null; });
    tela.addEventListener("wheel", (e) => {
      e.preventDefault();
      if (this.modo === "orbita" && !this.voo) {
        this.raio = Math.min(this.raioMax, Math.max(this.raioMin, this.raio * (1 + e.deltaY * 0.001)));
      }
    }, { passive: false });
    addEventListener("keydown", (e) => { this.teclas[e.code] = true; });
    addEventListener("keyup", (e) => { this.teclas[e.code] = false; });
    addEventListener("blur", () => { this.teclas = {}; });
  }

  olhar(dx, dy) {
    if (this.modo !== "primeira") return;
    this.guinada -= dx * 0.0022;
    this.arfagem = Math.min(1.35, Math.max(-1.35, this.arfagem - dy * 0.0022));
  }

  poseDaOrbita() {
    const seno = Math.sin(this.fi);
    return new THREE.Vector3(
      this.alvo.x + this.raio * seno * Math.sin(this.teta),
      this.alvo.y + this.raio * Math.cos(this.fi),
      this.alvo.z + this.raio * seno * Math.cos(this.teta)
    );
  }

  // Voa da pose atual ate a nova, com curva suave. Sem movimento quando o visitante
  // pediu menos animacao: a camera salta direto.
  voar(posicao, olharPara, duracao, aoTerminar) {
    const direcao = new THREE.Vector3();
    this.camera.getWorldDirection(direcao);
    const reduzir = matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.voo = {
      t: 0,
      dur: reduzir ? 0.001 : (duracao || 1.6),
      de: this.camera.position.clone(),
      para: posicao.clone(),
      olharDe: this.camera.position.clone().add(direcao.multiplyScalar(this.raioDoVoo(olharPara))),
      olharPara: olharPara.clone(),
      fim: aoTerminar,
    };
  }

  raioDoVoo(olharPara) {
    return this.camera.position.distanceTo(olharPara);
  }

  // Termina em orbita ao redor do ponto olhado: o visitante continua girando dali.
  voarEOrbitar(posicao, olharPara, duracao, aoTerminar) {
    this.modo = "orbita";
    this.voar(posicao, olharPara, duracao, () => {
      this.adotarOrbita(posicao, olharPara);
      if (aoTerminar) aoTerminar();
    });
  }

  adotarOrbita(posicao, olharPara) {
    this.alvo.copy(olharPara);
    const v = posicao.clone().sub(olharPara);
    this.raio = v.length();
    this.fi = Math.acos(v.y / this.raio);
    this.teta = Math.atan2(v.x, v.z);
  }

  entrarEmPrimeiraPessoa(x, z, guinada) {
    this.modo = "primeira";
    this.voo = null;
    this.guinada = guinada;
    this.arfagem = -0.04;
    this.camera.position.set(x, ALTURA_DOS_OLHOS, z);
  }

  atualizar(dt, bloqueado) {
    if (this.voo) {
      const v = this.voo;
      v.t = Math.min(1, v.t + dt / v.dur);
      const s = v.t * v.t * (3 - 2 * v.t);
      this.camera.position.lerpVectors(v.de, v.para, s);
      const olhar = new THREE.Vector3().lerpVectors(v.olharDe, v.olharPara, s);
      this.camera.lookAt(olhar);
      if (v.t >= 1) {
        this.voo = null;
        if (v.fim) v.fim();
      }
      return;
    }
    if (this.modo === "orbita") {
      this.camera.position.copy(this.poseDaOrbita());
      this.camera.lookAt(this.alvo);
      return;
    }
    if (this.modo === "seguir") return;
    if (!bloqueado) this.andar(dt);
    this.camera.rotation.order = "YXZ";
    this.camera.rotation.set(this.arfagem, this.guinada, 0);
  }

  andar(dt) {
    const k = this.teclas;
    const frente = (k.KeyW || k.ArrowUp ? 1 : 0) - (k.KeyS || k.ArrowDown ? 1 : 0);
    const lado = (k.KeyD || k.ArrowRight ? 1 : 0) - (k.KeyA || k.ArrowLeft ? 1 : 0);
    if (!frente && !lado) return;
    const velocidade = (k.ShiftLeft ? 5.4 : 3.2) * dt / Math.hypot(frente, lado);
    const sen = Math.sin(this.guinada);
    const cos = Math.cos(this.guinada);
    const novo = L.colidir({
      x: this.camera.position.x + (-sen * frente + cos * lado) * velocidade,
      z: this.camera.position.z + (-cos * frente - sen * lado) * velocidade,
    }, RAIO_DO_VISITANTE, this.colisores);
    this.camera.position.x = novo.x;
    this.camera.position.z = novo.z;
  }
}
