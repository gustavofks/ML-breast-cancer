// Construcao do predio: cada ala sobe aos poucos, peca a peca.

function saltoFinal(p) {
  // Passa um pouco do ponto e assenta: a peca parece ter peso.
  const c = 1.4;
  const x = p - 1;
  return 1 + (c + 1) * x * x * x + c * x * x;
}

class Obra {
  constructor(pecas) {
    this.pecas = pecas;
    this.progresso = {};
    Object.keys(pecas).forEach((ala) => {
      this.progresso[ala] = 1;
      const lista = pecas[ala];
      lista.forEach(function (peca, i) { peca.atraso = (i / Math.max(1, lista.length)) * 0.6; });
    });
  }

  definir(ala, p) {
    this.progresso[ala] = p;
    this.pecas[ala].forEach(function (peca) {
      const local = Math.min(1, Math.max(0, (p - peca.atraso) / 0.4));
      peca.objeto.visible = local > 0;
      if (peca.tipo === "piso") {
        const s = R.suave(local);
        peca.objeto.scale.set(s, s, 1);
      } else {
        peca.objeto.scale.y = Math.max(0.001, saltoFinal(local));
      }
    });
  }

  pronta(ala) {
    return this.progresso[ala] >= 1;
  }

  terminarTudo() {
    Object.keys(this.pecas).forEach((ala) => this.definir(ala, 1));
  }
}
