// Tomadas de camera dos filmes. Cada uma vai de uma pose a outra durante o plano.
//
// `p` e a posicao da camera, `o` o ponto olhado. `semTeto` abre a maquete, e `tilt`
// liga o efeito de miniatura. Os roteiros, em `src/ui/filmes.py`, citam estes nomes.
const TOMADAS = {
  terreno: { de: { p: [-52, 30, 58], o: [0, 1, 2] }, ate: { p: [-24, 20, 36], o: [0, 0, 0] }, fov: 36, semTeto: true, tilt: 0.8 },
  terreno_tarde: { de: { p: [52, 26, 50], o: [12, 2, 2] }, ate: { p: [34, 19, 33], o: [12, 0, 0] }, fov: 36, semTeto: true, tilt: 0.8 },
  obra_noite: { de: { p: [58, 7, -24], o: [29, 3, 0] }, ate: { p: [52, 13, 16], o: [29, 2, 0] }, fov: 40, semTeto: true, tilt: 0.25 },

  recepcao: { de: { p: [-20.6, 2, 0.2], o: [-17, 1.25, -6.4] }, ate: { p: [-19.2, 1.75, -1.4], o: [-17.2, 1.25, -6.4] } },

  dados_estante: { de: { p: [-8.8, 2.2, -5.2], o: [-6.4, 1, -10.5] }, ate: { p: [-0.8, 2.2, -5.2], o: [-2.6, 1, -10.5] } },
  dados_cofre: { de: { p: [-4.8, 2.3, -3.2], o: [-8, 1.15, -6] }, ate: { p: [-6.2, 1.9, -3], o: [-9.1, 1.05, -3.6] } },

  modelos_podio: { de: { p: [-1.8, 2.2, 2.8], o: [-5.2, 1.1, 8.8] }, ate: { p: [-2.4, 1.9, 4.2], o: [-5.6, 1, 9] } },
  modelos_podio_perto: { de: { p: [-5.6, 1.4, 5.4], o: [-7.6, 1.2, 9.4] }, ate: { p: [-6.4, 1.3, 6.4], o: [-7.8, 1.25, 9.4] } },
  modelos_shap: { de: { p: [-4.4, 1.9, 6.4], o: [0.86, 1.85, 6.4] }, ate: { p: [-0.5, 1.85, 6.4], o: [0.86, 1.85, 6.4] }, velocidade: 2.4 },

  imagem_sala: { de: { p: [3.2, 2.2, -2.8], o: [7.2, 1.3, -8] }, ate: { p: [3.8, 1.9, -3.6], o: [7.6, 1.25, -8] } },
  imagem_gradcam: { de: { p: [8.2, 1.85, -5.2], o: [11.86, 1.85, -6.4] }, ate: { p: [10.4, 1.85, -6.4], o: [11.86, 1.85, -6.4] }, velocidade: 2.4 },

  evolucao_ampla: { de: { p: [10.6, 2.3, 2.8], o: [6.2, 1.25, 7.2] }, ate: { p: [9.6, 2, 3.4], o: [5.6, 1.2, 6.6] } },
  evolucao_ampla_perto: { de: { p: [7, 1.7, 3.2], o: [4.6, 1.1, 5.2] }, ate: { p: [6.4, 1.5, 3.6], o: [4.6, 1.05, 5.2] } },
  enxame_orbita: { orbita: { centro: [7.4, 1.25, 6.8], raio: 3.3, altura: 2, de: -1.2, ate: 0.5 } },
  evolucao_convergencia: { de: { p: [4.2, 1.9, 6.6], o: [4.2, 1.85, 10.86] }, ate: { p: [4.2, 1.85, 9.5], o: [4.2, 1.85, 10.86] }, velocidade: 2.4 },
  regua_limiar: { de: { p: [8.4, 1.9, 6.8], o: [1.17, 1.75, 6.8] }, ate: { p: [5.2, 1.65, 6.8], o: [1.17, 1.5, 6.8] }, velocidade: 3 },

  central_ampla: { de: { p: [13.4, 2.3, 2.8], o: [17.6, 1.5, 8.6] }, ate: { p: [14.2, 2, 3.6], o: [18, 1.45, 8.8] } },
  corredor_portador: { seguir: true },
  laudos_selos: { de: { p: [15.4, 2.1, -3], o: [20.6, 1.6, -6.2] }, ate: { p: [17.6, 1.95, -3.2], o: [21.4, 1.6, -6] } },

  obras_ampla: { de: { p: [24.2, 2.3, -3.4], o: [28.6, 1, 3.2] }, ate: { p: [25, 2, -2.6], o: [28.8, 1, 3.2] } },
  obras_fluxo: { de: { p: [25.4, 2, 0.1], o: [26.4, 1, 4.4] }, ate: { p: [33, 2, 0.1], o: [33.4, 1, 4.4] } },
  obras_validacao: { de: { p: [29.8, 1.9, 0.6], o: [31.7, 1.1, 4.4] }, ate: { p: [30.6, 1.7, 1.4], o: [31.7, 1.1, 4.4] } },
  telao: { de: { p: [28, 2, 0], o: [34.82, 1.85, 0] }, ate: { p: [32.9, 1.85, 0], o: [34.82, 1.85, 0] }, velocidade: 2.4 },

  planta_fase1: { orbita: { centro: [-3, 0, 1], raio: 42, altura: 27, de: 2.05, ate: 1.5 }, fov: 36, semTeto: true, tilt: 0.8 },
  planta_final: { orbita: { centro: [6.5, 0, 1], raio: 46, altura: 30, de: 1.95, ate: 1.45 }, fov: 36, semTeto: true, tilt: 0.8 },
  planta_noite: { orbita: { centro: [8, 0, 0], raio: 54, altura: 26, de: 0.75, ate: 1.3 }, fov: 36, semTeto: true, tilt: 0.45 },
};

function vetor(a) {
  return new THREE.Vector3(a[0], a[1], a[2]);
}

// Pose da camera na fracao `p` do plano. Tomadas com `velocidade` chegam antes do fim:
// a camera mergulha na tela, e o resto do plano fica para a figura.
function poseDaTomada(nome, p, portador) {
  const tomada = TOMADAS[nome];
  const fov = tomada.fov || 50;
  if (tomada.orbita) {
    const o = tomada.orbita;
    const a = o.de + (o.ate - o.de) * p;
    return {
      posicao: new THREE.Vector3(o.centro[0] + Math.cos(a) * o.raio, o.altura, o.centro[2] + Math.sin(a) * o.raio),
      olhar: vetor(o.centro), fov: fov,
    };
  }
  if (tomada.seguir) {
    const alvo = portador.posicao;
    const giro = portador.robo.raiz.rotation.y;
    return {
      posicao: new THREE.Vector3(alvo.x - Math.sin(giro) * 3.1, 2, alvo.z - Math.cos(giro) * 3.1),
      olhar: new THREE.Vector3(alvo.x, 1, alvo.z), fov: fov,
    };
  }
  const s = R.suave(Math.min(1, p * (tomada.velocidade || 1)));
  return {
    posicao: vetor(tomada.de.p).lerp(vetor(tomada.ate.p), s),
    olhar: vetor(tomada.de.o).lerp(vetor(tomada.ate.o), s),
    fov: fov,
  };
}
