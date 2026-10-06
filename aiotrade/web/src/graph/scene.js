/**
 * Graphe 3D AIOTrade (Three.js) : le pipeline de décision en motion design.
 *
 * Marché → filtre d'intégrité χ² (+ Macro Gate) → prévision (essaim d'agents)
 * → TAP (faisceau de trajectoires) → SCG (élagage) → Kelly fractionnaire →
 * exécution maker / taker → décision LONG / SHORT / CASH, chaque décision
 * étant scellée dans le registre XAI (chaîne de blocs SHA-256).
 *
 * - Les ordres candidats sont des particules ; une partie est élaguée par le
 *   SCG (elles rougissent et tombent), le reste atteint la décision.
 * - Le TAP projette un faisceau de trajectoires ; les trajectoires
 *   inadmissibles sortent du cône et rougissent à l'étape SCG.
 * - À l'étape intégrité, un choc gèle le flux et une bulle met la décision
 *   en cash.
 *
 * Performance : pixel ratio adaptatif (cible 60 FPS), rendu suspendu hors
 * écran ou onglet masqué, particules réduites sur mobile.
 * Mémoire : dispose() libère géométries, matériaux, textures, environnement
 * et contexte WebGL.
 */
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

const C = {
  bg: 0x070b14,
  market: 0x60a5fa,
  integrity: 0xa78bfa,
  macro: 0xf472b6,
  forecast: 0x38bdf8,
  agent: 0x7dd3fc,
  tap: 0x2dd4bf,
  scg: 0xfbbf24,
  kelly: 0x93c5fd,
  execution: 0x67e8f9,
  decision: 0x4ade80,
  ledger: 0xe2e8f0,
  reject: 0xff5a5a,
  edge: 0x334155,
};

export const NODES = [
  { id: 'market', label: 'Marché', pos: [-8, 0, 0], color: C.market, size: 0.55, stage: 1, target: '#etape-marche',
    desc: 'Chandeliers OHLCV, spread et profondeur du carnet.' },
  { id: 'integrity', label: 'Intégrité χ²', pos: [-5.6, 0, 0.3], color: C.integrity, size: 0.55, stage: 2, kind: 'shield',
    target: '#etape-integrite', desc: 'Distance de Mahalanobis : gel et passage en cash si le flux est anormal.' },
  { id: 'macro', label: 'Macro Gate', pos: [-5.6, 2.7, -0.6], color: C.macro, size: 0.5, stage: 2, kind: 'clock',
    target: '#etape-integrite', desc: 'Veto calendaire : NFP, IPC, Fed, BCE.' },
  { id: 'forecast', label: 'Prévision', pos: [-3.2, 0, 0], color: C.forecast, size: 0.55, stage: 3, target: '#etape-prevision',
    desc: "Essaim d'agents pondérés en ligne, ou Kronos en option." },
  { id: 'trend', label: 'Trend', pos: [-1.5, 2.0, -0.6], color: C.agent, size: 0.26, stage: 3, target: '#etape-prevision',
    desc: 'Agent de suivi de tendance.' },
  { id: 'meanrev', label: 'Mean-Reversion', pos: [-1.5, 0.7, 0.9], color: C.agent, size: 0.26, stage: 3, target: '#etape-prevision',
    desc: 'Agent de retour à la moyenne.' },
  { id: 'macroagent', label: 'Macro', pos: [-1.5, -0.7, 0.9], color: C.agent, size: 0.26, stage: 3, target: '#etape-prevision',
    desc: 'Agent de dérive de fond.' },
  { id: 'risk', label: 'Risque', pos: [-1.5, -2.0, -0.6], color: C.agent, size: 0.26, stage: 3, target: '#etape-prevision',
    desc: 'Agent défensif à volatilité majorée.' },
  { id: 'tap', label: 'TAP', pos: [0.6, 0, 0], color: C.tap, size: 0.6, stage: 4, target: '#etape-tap',
    desc: '1 024 trajectoires multi-pas sur l’horizon de décision.' },
  { id: 'scg', label: 'SCG', pos: [3.4, 0, 0], color: C.scg, size: 0.9, stage: 5, kind: 'gate', target: '#etape-scg',
    desc: 'Élague les trajectoires qui violent le mandat ; seuil de 75 %.' },
  { id: 'kelly', label: 'Kelly fractionnaire', pos: [5.1, 1.1, 0.3], color: C.kelly, size: 0.42, stage: 6, kind: 'cube',
    target: '#etape-execution', desc: 'Taille λ·f*, plafonnée par le mandat.' },
  { id: 'execution', label: 'Exécution', pos: [6.5, -1.0, 0.3], color: C.execution, size: 0.45, stage: 6, kind: 'cone',
    target: '#etape-execution', desc: 'Impact Almgren-Chriss, ordres maker ou taker.' },
  { id: 'decision', label: 'Décision', pos: [8.3, 0, 0], color: C.decision, size: 0.72, stage: 7, target: '#etape-decision',
    desc: 'LONG, SHORT ou CASH, transmise au courtier.' },
  { id: 'ledger', label: 'XAI Ledger', pos: [7.9, -3.4, 0.5], color: C.ledger, size: 0.3, stage: 7, kind: 'block',
    target: '#etape-decision', desc: 'Registre d’audit immuable, chaîné par SHA-256.' },
];

// Vues caméra par étape : [position, cible]
const VIEWS = [
  [[0.2, 0.6, 28], [0.2, -0.6, 0]],
  [[-7.2, 1.2, 9], [-7.2, 0, 0]],
  [[-5.4, 1.6, 10.5], [-5.4, 1.0, 0]],
  [[-2.3, 0.8, 10.5], [-2.3, 0, 0]],
  [[1.7, 1.2, 9.5], [1.8, 0, 0]],
  [[3.0, 1.0, 9.5], [3.0, 0, 0]],
  [[5.9, 0.6, 9.5], [5.9, 0, 0]],
  [[6.4, -0.8, 12.5], [6.4, -1.5, 0]],
];

const AGENT_IDS = ['trend', 'meanrev', 'macroagent', 'risk'];
const ROUTE = ['market', 'integrity', 'forecast', null, 'tap', 'scg', 'kelly', 'execution', 'decision'];
const SEGMENTS = ROUTE.length - 1;
const T_INTEGRITY = 1 / SEGMENTS;
const T_SCG = 5 / SEGMENTS;

function vec(p) {
  return new THREE.Vector3(p[0], p[1], p[2]);
}

function makeLabelTexture(text) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const font = '600 44px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
  ctx.font = font;
  const width = Math.ceil(ctx.measureText(text).width) + 56;
  canvas.width = width;
  canvas.height = 84;
  ctx.font = font;
  ctx.fillStyle = 'rgba(7, 11, 20, 0.78)';
  ctx.beginPath();
  ctx.roundRect(0, 0, width, 84, 24);
  ctx.fill();
  ctx.fillStyle = '#eef2f8';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 28, 44);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return { texture, aspect: width / 84 };
}

function makeDotTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext('2d');
  const g = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.35, 'rgba(255,255,255,0.85)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 64, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export function createGraph({ canvas, tooltip, reducedMotion = false, onContextLost, onNodeActivate }) {
  const isSmall = window.matchMedia('(max-width: 720px)').matches;
  const maxDpr = Math.min(window.devicePixelRatio || 1, 2);
  let dpr = isSmall ? Math.min(maxDpr, 1.5) : maxDpr;

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(dpr);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.setClearColor(C.bg, 0);

  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(C.bg, 30, 60);

  // Éclairage réaliste : environnement studio pré-filtré (PBR) + lumières douces.
  const pmrem = new THREE.PMREMGenerator(renderer);
  const room = new RoomEnvironment();
  const envTexture = pmrem.fromScene(room, 0.04).texture;
  scene.environment = envTexture;
  room.traverse((o) => {
    o.geometry?.dispose();
    o.material?.dispose?.();
  });
  pmrem.dispose();
  scene.add(new THREE.HemisphereLight(0xbfd7ff, 0x0b1020, 0.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(4, 8, 6);
  scene.add(key);

  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 90);
  const camPos = vec(VIEWS[0][0]);
  const camLook = vec(VIEWS[0][1]);
  camera.position.copy(camPos);

  const root = new THREE.Group();
  scene.add(root);

  /* ------------------------------------------------------------- nœuds */
  const nodeMeshes = new Map();
  const labels = [];
  const sphereGeo = new THREE.IcosahedronGeometry(1, 4);
  const geometries = {
    gate: () => new THREE.TorusGeometry(1, 0.12, 24, 96),
    shield: () => new THREE.OctahedronGeometry(1, 0),
    clock: () => new THREE.CylinderGeometry(1, 1, 0.22, 48),
    cube: () => new THREE.BoxGeometry(1.3, 1.3, 1.3),
    cone: () => new THREE.ConeGeometry(0.9, 1.7, 32),
    block: () => new THREE.BoxGeometry(1.5, 1.5, 1.5),
  };
  const hands = [];
  NODES.forEach((node) => {
    const geometry = node.kind ? geometries[node.kind]() : sphereGeo;
    const material = new THREE.MeshStandardMaterial({
      color: node.color,
      emissive: node.color,
      emissiveIntensity: 0.35,
      metalness: 0.35,
      roughness: 0.28,
      flatShading: node.kind === 'shield',
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.copy(vec(node.pos));
    mesh.scale.setScalar(node.size);
    if (node.kind === 'gate') mesh.rotation.y = Math.PI / 2;
    if (node.kind === 'clock') mesh.rotation.x = Math.PI / 2;
    if (node.kind === 'cone') mesh.rotation.z = -Math.PI / 2;
    if (node.kind === 'cube') mesh.rotation.set(0.5, 0.6, 0);
    if (node.kind === 'clock') {
      // Aiguille du calendrier macro : tourne en continu, s'emballe pendant un blackout.
      const hand = new THREE.Mesh(
        new THREE.BoxGeometry(0.1, 0.12, 0.8),
        new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.6 }),
      );
      hand.position.set(0, 0.16, -0.32);
      const pivot = new THREE.Group();
      pivot.add(hand);
      mesh.add(pivot);
      hands.push(pivot);
    }
    mesh.userData = { node, baseScale: node.size, emissive: 0.35, hover: 0, flash: 0 };
    root.add(mesh);
    nodeMeshes.set(node.id, mesh);

    const { texture, aspect } = makeLabelTexture(node.label);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }));
    const h = isSmall ? 0.42 : node.size < 0.3 ? 0.3 : 0.36;
    sprite.scale.set(h * aspect, h, 1);
    const above = node.id === 'risk' || node.id === 'ledger' ? -(node.size + 0.42) : node.size + 0.45;
    sprite.position.copy(vec(node.pos)).add(new THREE.Vector3(0, above, 0));
    sprite.renderOrder = 10;
    sprite.userData = { node };
    root.add(sprite);
    labels.push(sprite);
  });
  const posOf = (id) => vec(NODES.find((n) => n.id === id).pos);

  // Disque intérieur du garde-fou (effet de porte filtrante).
  const gateDisc = new THREE.Mesh(
    new THREE.CircleGeometry(0.85, 48),
    new THREE.MeshBasicMaterial({ color: C.scg, transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false }),
  );
  gateDisc.position.copy(posOf('scg'));
  gateDisc.rotation.y = Math.PI / 2;
  root.add(gateDisc);

  // Bulle de mise en cash autour de la décision (gel d'intégrité).
  const bubble = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.5, 3),
    new THREE.MeshStandardMaterial({
      color: C.integrity, emissive: C.integrity, emissiveIntensity: 0.4, transparent: true, opacity: 0,
      wireframe: true, depthWrite: false,
    }),
  );
  bubble.position.copy(posOf('decision'));
  root.add(bubble);

  /* ------------------------------------------------------------ liaisons */
  const edgeMaterial = new THREE.MeshStandardMaterial({ color: C.edge, emissive: 0x1e293b, emissiveIntensity: 0.6, roughness: 0.6 });
  const tube = (points, radius = 0.025) => {
    const curve = new THREE.CatmullRomCurve3(points);
    root.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 48, radius, 8, false), edgeMaterial));
    return curve;
  };
  tube([posOf('market'), posOf('integrity'), posOf('forecast')]);
  AGENT_IDS.forEach((id) => {
    tube([posOf('forecast'), posOf(id)], 0.018);
    tube([posOf(id), posOf('tap')], 0.018);
  });
  tube([posOf('scg'), posOf('kelly'), posOf('execution'), posOf('decision')]);

  // Liaisons de contrôle en pointillés : gel d'intégrité, veto macro, écriture au registre.
  const dashed = (points, color, opacity) => {
    const curve = new THREE.CatmullRomCurve3(points);
    const material = new THREE.LineDashedMaterial({ color, dashSize: 0.18, gapSize: 0.14, transparent: true, opacity });
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(160)), material);
    line.computeLineDistances();
    root.add(line);
    return { curve, material };
  };
  const freezeLink = dashed(
    [posOf('integrity'), new THREE.Vector3(-3, -3.0, 0.5), new THREE.Vector3(3.5, -2.4, 0.5), posOf('decision')],
    C.integrity, 0.4,
  );
  const macroLink = dashed(
    [posOf('macro'), new THREE.Vector3(-1, 3.8, -0.4), new THREE.Vector3(5.5, 3.0, -0.2), posOf('decision')],
    C.macro, 0.35,
  );

  /* -------------------------------------------- registre XAI : chaîne de blocs */
  const blockGeo = new THREE.BoxGeometry(0.34, 0.34, 0.34);
  const chain = [];
  const CHAIN = 6;
  for (let i = 0; i < CHAIN; i += 1) {
    const p = new THREE.Vector3(2.2 + i * 0.95, -3.4, 0.5);
    const block = new THREE.Mesh(
      blockGeo,
      new THREE.MeshStandardMaterial({ color: C.ledger, emissive: C.ledger, emissiveIntensity: 0.15, metalness: 0.5, roughness: 0.3 }),
    );
    block.position.copy(p);
    block.rotation.set(0.4, 0.5, 0);
    root.add(block);
    chain.push(block);
    if (i > 0) tube([chain[i - 1].position, p], 0.012);
  }
  tube([chain[CHAIN - 1].position, posOf('ledger')], 0.012);
  const ledgerLinks = [
    dashed([posOf('scg'), new THREE.Vector3(3.2, -1.8, 0.4), chain[1].position], C.ledger, 0.18),
    dashed([posOf('decision'), new THREE.Vector3(8.4, -1.9, 0.4), posOf('ledger')], C.ledger, 0.18),
  ];

  /* ----------------------------------------- faisceau de trajectoires TAP */
  const rand = mulberry32(44);
  const admissible = [];
  const pruned = [];
  const admissibleMaterial = new THREE.LineBasicMaterial({ color: C.tap, transparent: true, opacity: 0.32, depthWrite: false, blending: THREE.AdditiveBlending });
  const prunedMaterial = new THREE.LineBasicMaterial({ color: C.tap, transparent: true, opacity: 0.2, depthWrite: false, blending: THREE.AdditiveBlending });
  const nLines = isSmall ? 18 : 34;
  const start = posOf('tap');
  const end = posOf('scg');
  for (let i = 0; i < nLines; i += 1) {
    const isPruned = i % 4 === 3; // environ un quart des trajectoires sort du cône admissible
    const angle = rand() * Math.PI * 2;
    const radius = isPruned ? 1.1 + rand() * 0.9 : rand() * 0.62;
    const pts = [];
    for (let k = 0; k <= 8; k += 1) {
      const t = k / 8;
      const p = start.clone().lerp(end, t);
      const spread = radius * Math.pow(t, 1.25); // l'incertitude s'ouvre avec l'horizon
      const wobble = (rand() - 0.5) * 0.18 * Math.sin(Math.PI * t);
      p.y += Math.sin(angle) * spread + wobble;
      p.z += Math.cos(angle) * spread + wobble;
      pts.push(p);
    }
    const curve = new THREE.CatmullRomCurve3(pts);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(40)), isPruned ? prunedMaterial : admissibleMaterial);
    line.userData.count = 41;
    line.userData.delay = rand() * 0.6;
    root.add(line);
    (isPruned ? pruned : admissible).push(line);
  }
  const bundle = admissible.concat(pruned);

  /* ------------------------------------------------------ particules (ordres) */
  const dotTexture = makeDotTexture();
  const routeCurves = AGENT_IDS.map((agent) => new THREE.CatmullRomCurve3(ROUTE.map((id) => posOf(id ?? agent))));
  const COUNT = isSmall ? 110 : 220;
  const particles = Array.from({ length: COUNT }, () => ({ alive: false }));
  const pGeometry = new THREE.BufferGeometry();
  const pPositions = new Float32Array(COUNT * 3);
  const pColors = new Float32Array(COUNT * 3);
  pPositions.fill(9999); // hors champ tant qu'aucun ordre n'est émis
  pGeometry.setAttribute('position', new THREE.BufferAttribute(pPositions, 3));
  pGeometry.setAttribute('color', new THREE.BufferAttribute(pColors, 3));
  const pMaterial = new THREE.PointsMaterial({
    size: isSmall ? 0.2 : 0.17, map: dotTexture, vertexColors: true, transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const points = new THREE.Points(pGeometry, pMaterial);
  points.frustumCulled = false;
  root.add(points);

  // Capteurs du filtre d'intégrité sur la liaison de gel.
  const SENSORS = isSmall ? 18 : 32;
  const sGeometry = new THREE.BufferGeometry();
  const sPositions = new Float32Array(SENSORS * 3);
  sGeometry.setAttribute('position', new THREE.BufferAttribute(sPositions, 3));
  const sMaterial = new THREE.PointsMaterial({
    size: 0.12, map: dotTexture, color: C.integrity, transparent: true, opacity: 0.8,
    depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const sensors = new THREE.Points(sGeometry, sMaterial);
  sensors.frustumCulled = false;
  root.add(sensors);
  const sensorPhase = Array.from({ length: SENSORS }, (_, i) => i / SENSORS);

  const colGood = new THREE.Color(C.tap);
  const colOut = new THREE.Color(C.decision);
  const colReject = new THREE.Color(C.reject);
  const tmp = new THREE.Vector3();
  const tmpColor = new THREE.Color();
  const shockColor = new THREE.Color();

  function spawn(p) {
    p.alive = true;
    p.route = Math.floor(rand() * routeCurves.length);
    p.t = 0;
    p.speed = 0.075 + rand() * 0.05;
    p.rejected = rand() < (state.stage === 5 ? 0.4 : 0.22);
    p.falling = false;
    p.fade = 1;
    p.jitter = new THREE.Vector3((rand() - 0.5) * 0.25, (rand() - 0.5) * 0.25, (rand() - 0.5) * 0.25);
    p.vel = new THREE.Vector3();
  }

  /* --------------------------------------------------------------- état */
  const state = {
    stage: 0,
    running: false,
    disposed: false,
    time: 0,
    stageTime: 0,
    spawnAcc: 0,
    shock: 0, // 0..1, intensité du gel d'intégrité
    blackout: 0, // 0..1, veto macro
    ledgerFlash: 0,
    pointer: new THREE.Vector2(0, 0),
    parallax: new THREE.Vector2(0, 0),
    hovered: null,
  };

  function updateParticles(dt) {
    const spawnRate = state.shock > 0.5 ? 0 : isSmall ? 18 : 34; // ordres / seconde
    state.spawnAcc += spawnRate * dt;
    for (const p of particles) {
      if (state.spawnAcc < 1) break;
      if (!p.alive) {
        spawn(p);
        state.spawnAcc -= 1;
      }
    }
    state.spawnAcc = Math.min(state.spawnAcc, 2);

    let flash = 0;
    particles.forEach((p, i) => {
      const o = i * 3;
      if (!p.alive) {
        pPositions[o] = pPositions[o + 1] = pPositions[o + 2] = 9999;
        return;
      }
      if (p.falling) {
        p.vel.y -= 4.5 * dt;
        tmp.set(pPositions[o], pPositions[o + 1], pPositions[o + 2]).addScaledVector(p.vel, dt);
        p.fade -= dt * 0.9;
        tmpColor.copy(colReject).multiplyScalar(Math.max(p.fade, 0));
        if (p.fade <= 0) p.alive = false;
      } else {
        // Gel d'intégrité : les ordres pas encore passés par le SCG sont annulés (fondu).
        if (state.shock > 0.5 && p.t < T_SCG) {
          p.fade -= dt * 2;
          if (p.fade <= 0) p.alive = false;
        }
        p.t += p.speed * dt;
        if (p.rejected && p.t >= T_SCG) {
          p.falling = true;
          p.vel.set((rand() - 0.2) * 0.8, 0.6 + rand() * 0.6, (rand() - 0.5) * 1.2);
          flash = 1;
        }
        if (p.t >= 1) {
          p.alive = false;
          state.ledgerFlash = 1; // chaque décision est scellée dans le registre
          return;
        }
        routeCurves[p.route].getPoint(p.t, tmp).add(p.jitter);
        const k = THREE.MathUtils.smoothstep(p.t, T_SCG, 1);
        tmpColor.copy(colGood).lerp(colOut, k).multiplyScalar(Math.max(p.fade, 0));
      }
      pPositions[o] = tmp.x;
      pPositions[o + 1] = tmp.y;
      pPositions[o + 2] = tmp.z;
      pColors[o] = tmpColor.r;
      pColors[o + 1] = tmpColor.g;
      pColors[o + 2] = tmpColor.b;
    });
    pGeometry.attributes.position.needsUpdate = true;
    pGeometry.attributes.color.needsUpdate = true;
    if (flash) nodeMeshes.get('scg').userData.flash = 1;

    const sensorSpeed = 0.05 + state.shock * 0.12;
    sensorPhase.forEach((ph, i) => {
      sensorPhase[i] = (ph + sensorSpeed * dt) % 1;
      freezeLink.curve.getPoint(sensorPhase[i], tmp);
      sPositions[i * 3] = tmp.x;
      sPositions[i * 3 + 1] = tmp.y;
      sPositions[i * 3 + 2] = tmp.z;
    });
    sGeometry.attributes.position.needsUpdate = true;
  }

  const isFocused = (node, s) => s === 0 || node.stage === s || (s === 2 && node.id === 'market') || (s === 2 && node.id === 'decision');

  function updateNodes(dt) {
    const s = state.stage;
    nodeMeshes.forEach((mesh) => {
      const { node } = mesh.userData;
      const focused = isFocused(node, s);
      const targetEmissive = (s === 0 ? 0.75 : focused ? 1.3 : 0.12) + mesh.userData.hover * 0.8;
      mesh.userData.emissive += (targetEmissive - mesh.userData.emissive) * Math.min(1, dt * 4);
      let emissive = mesh.userData.emissive;
      shockColor.setHex(node.color);
      if (node.id === 'scg' && mesh.userData.flash > 0) {
        mesh.userData.flash = Math.max(0, mesh.userData.flash - dt * 3);
        emissive += mesh.userData.flash * 0.9;
      }
      if (node.id === 'ledger') emissive += state.ledgerFlash * 0.8;
      if (node.id === 'market' && state.shock > 0.01) {
        const pulse = 0.5 + 0.5 * Math.sin(state.time * 9);
        shockColor.lerp(colReject, state.shock * pulse);
      }
      mesh.material.emissive.copy(shockColor);
      if (node.id === 'market') mesh.material.color.copy(shockColor);
      mesh.material.emissiveIntensity = emissive;
      const breathe = 1 + Math.sin(state.time * 1.6 + node.pos[0]) * 0.03;
      const scale = mesh.userData.baseScale * breathe * (1 + mesh.userData.hover * 0.18 + (focused && s !== 0 ? 0.08 : 0));
      mesh.scale.setScalar(scale);
      if (node.kind === 'gate') mesh.rotation.x += dt * 0.25;
      if (node.kind === 'shield') mesh.rotation.y += dt * (0.6 + state.shock * 2.5);
      if (node.kind === 'cube' || node.kind === 'block') mesh.rotation.y += dt * 0.35;
      mesh.userData.hover += ((state.hovered === mesh ? 1 : 0) - mesh.userData.hover) * Math.min(1, dt * 10);
    });
    hands.forEach((h) => {
      h.rotation.y -= dt * (0.5 + state.blackout * 4);
    });
    labels.forEach((sprite) => {
      const target = isFocused(sprite.userData.node, s) ? 1 : 0.35;
      sprite.material.opacity += (target - sprite.material.opacity) * Math.min(1, dt * 5);
    });

    // Faisceau TAP : se dessine à l'étape 4 ; les trajectoires inadmissibles rougissent à l'étape SCG.
    bundle.forEach((line) => {
      const local = s === 4 ? THREE.MathUtils.clamp((state.stageTime - line.userData.delay) / 1.2, 0, 1) : 1;
      line.geometry.setDrawRange(0, Math.max(2, Math.floor(line.userData.count * local)));
    });
    const bundleTarget = s === 4 || s === 5 ? 0.55 : s === 0 ? 0.28 : 0.14;
    admissibleMaterial.opacity += (bundleTarget - admissibleMaterial.opacity) * Math.min(1, dt * 4);
    const prunedTarget = s === 5 ? 0.75 : bundleTarget * 0.7;
    prunedMaterial.opacity += (prunedTarget - prunedMaterial.opacity) * Math.min(1, dt * 4);
    const redness = s === 5 ? Math.min(1, state.stageTime / 1.2) : 0;
    prunedMaterial.color.setHex(C.tap).lerp(colReject, redness);

    // Gel d'intégrité (étape 2) : choc périodique, retour au nominal entre deux chocs.
    const cycle = reducedMotion ? 1 : (state.stageTime % 6) > 2.6 ? 1 : 0;
    const shockTarget = s === 2 ? cycle : 0;
    state.shock += (shockTarget - state.shock) * Math.min(1, dt * 3);
    state.blackout += ((s === 2 ? 1 : 0) - state.blackout) * Math.min(1, dt * 2);
    bubble.material.opacity = state.shock * 0.35;
    bubble.rotation.y += dt * 0.3;
    bubble.scale.setScalar(1 + Math.sin(state.time * 2.2) * 0.03 * state.shock);
    freezeLink.material.opacity = 0.3 + state.shock * 0.6;
    macroLink.material.opacity = 0.25 + state.blackout * 0.5;
    sMaterial.opacity = 0.55 + state.shock * 0.45;
    gateDisc.material.opacity = 0.05 + nodeMeshes.get('scg').userData.flash * 0.25;

    // Registre : une impulsion parcourt la chaîne de blocs, le dernier s'illumine à chaque décision.
    state.ledgerFlash = Math.max(0, state.ledgerFlash - dt * 2.5);
    const ledgerFocus = s === 7 ? 1 : s === 0 ? 0.5 : 0.15;
    const head = (state.time * 1.5) % CHAIN;
    chain.forEach((block, i) => {
      const pulse = Math.max(0, 1 - Math.abs(head - i));
      block.material.emissiveIntensity = 0.12 + ledgerFocus * (0.35 + pulse * 0.9);
      block.rotation.y += dt * 0.3;
    });
    ledgerLinks.forEach((l) => {
      l.material.opacity = 0.12 + ledgerFocus * 0.4;
    });
  }

  function updateCamera(dt, immediate = false) {
    const [p, l] = VIEWS[state.stage];
    const k = immediate ? 1 : 1 - Math.exp(-dt * 2.2);
    state.parallax.lerp(state.pointer, immediate ? 1 : Math.min(1, dt * 3));
    const idle = reducedMotion ? 0 : 1;
    tmp.set(
      p[0] + state.parallax.x * 0.6 + Math.sin(state.time * 0.21) * 0.25 * idle,
      p[1] + state.parallax.y * 0.4 + Math.sin(state.time * 0.17) * 0.18 * idle,
      p[2],
    );
    camPos.lerp(tmp, k);
    camLook.lerp(vec(l), k);
    camera.position.copy(camPos);
    camera.lookAt(camLook);
  }

  /* --------------------------------------------------------- redimension */
  function resize() {
    const w = canvas.clientWidth || window.innerWidth;
    const h = canvas.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    // Grand écran : le texte occupe la gauche, le graphe est décalé vers la droite.
    // Mobile : les cartes de texte occupent le centre, le graphe remonte dans le tiers haut.
    if (w >= 960) camera.setViewOffset(w, h, -w * 0.2, 0, w, h);
    else camera.setViewOffset(w, h, 0, h * 0.24, w, h);
    camera.fov = w < 720 ? 55 : 42;
    camera.updateProjectionMatrix();
    if (!state.running) renderOnce();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(canvas);

  /* ----------------------------------------------------------- survol */
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const pickables = [...nodeMeshes.values()];
  const BLOCKERS = '.card, a, button, input, select, textarea, label, .site-header, .consent, .hero-copy';

  function pick(event) {
    const rect = canvas.getBoundingClientRect();
    ndc.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    return raycaster.intersectObjects(pickables, false)[0]?.object ?? null;
  }

  function onPointerMove(event) {
    state.pointer.set((event.clientX / window.innerWidth) * 2 - 1, -(event.clientY / window.innerHeight) * 2 + 1);
    if (event.pointerType === 'touch' || !state.running) return;
    const blocked = event.target instanceof Element && event.target.closest(BLOCKERS);
    const hit = blocked ? null : pick(event);
    if (hit !== state.hovered) {
      state.hovered = hit;
      document.documentElement.classList.toggle('graph-hover', Boolean(hit));
    }
    if (tooltip) {
      if (hit) {
        const { node } = hit.userData;
        tooltip.innerHTML = `<strong>${node.label}</strong><span>${node.desc}</span>`;
        tooltip.style.transform = `translate(${event.clientX + 16}px, ${event.clientY + 16}px)`;
        tooltip.hidden = false;
      } else {
        tooltip.hidden = true;
      }
    }
  }

  function onClick(event) {
    if (!state.hovered) return;
    const blocked = event.target instanceof Element && event.target.closest(BLOCKERS);
    if (blocked) return;
    onNodeActivate?.(state.hovered.userData.node);
  }

  window.addEventListener('pointermove', onPointerMove, { passive: true });
  window.addEventListener('click', onClick);

  function onContextLostEvent(event) {
    event.preventDefault();
    onContextLost?.();
  }
  canvas.addEventListener('webglcontextlost', onContextLostEvent);

  /* ------------------------------------------------------------- boucle */
  let lastTime = performance.now();
  let raf = 0;
  let frames = 0;
  let frameTime = 0;
  let goodStreak = 0;

  function adaptQuality(dt) {
    frames += 1;
    frameTime += dt;
    if (frames < 90) return;
    const fps = frames / frameTime;
    frames = 0;
    frameTime = 0;
    if (fps < 52 && dpr > 0.75) {
      dpr = Math.max(0.75, dpr - 0.25);
      renderer.setPixelRatio(dpr);
      goodStreak = 0;
    } else if (fps > 58.5 && dpr < maxDpr) {
      goodStreak += 1;
      if (goodStreak >= 4) {
        dpr = Math.min(maxDpr, dpr + 0.25);
        renderer.setPixelRatio(dpr);
        goodStreak = 0;
      }
    }
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    const dt = Math.min(Math.max((now - lastTime) / 1000, 0), 1 / 20);
    lastTime = now;
    state.time += dt;
    state.stageTime += dt;
    updateParticles(dt);
    updateNodes(dt);
    updateCamera(dt);
    renderer.render(scene, camera);
    adaptQuality(dt);
  }

  function renderOnce() {
    if (state.disposed) return;
    // Mouvement réduit : un rendu statique et complet de l'étape courante.
    state.stageTime = 10;
    updateNodes(1);
    updateCamera(0, true);
    renderer.render(scene, camera);
  }

  function setActive(active) {
    if (state.disposed) return;
    const shouldRun = active && !reducedMotion && document.visibilityState === 'visible';
    if (shouldRun === state.running) return;
    state.running = shouldRun;
    if (shouldRun) {
      lastTime = performance.now();
      raf = requestAnimationFrame(frame);
    } else {
      cancelAnimationFrame(raf);
      if (tooltip) tooltip.hidden = true;
      state.hovered = null;
      document.documentElement.classList.remove('graph-hover');
    }
  }

  function setStage(stage) {
    const next = THREE.MathUtils.clamp(stage | 0, 0, VIEWS.length - 1);
    if (next === state.stage) return;
    state.stage = next;
    state.stageTime = 0;
    if (reducedMotion) {
      state.shock = next === 2 ? 1 : 0;
      state.blackout = state.shock;
      bubble.material.opacity = state.shock * 0.35;
      renderOnce();
    }
  }

  function dispose() {
    if (state.disposed) return;
    setActive(false);
    state.disposed = true;
    resizeObserver.disconnect();
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('click', onClick);
    canvas.removeEventListener('webglcontextlost', onContextLostEvent);
    const textures = new Set();
    scene.traverse((obj) => {
      obj.geometry?.dispose();
      const materials = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
      materials.forEach((m) => {
        if (m.map) textures.add(m.map);
        m.dispose();
      });
    });
    textures.forEach((t) => t.dispose());
    sphereGeo.dispose();
    envTexture.dispose();
    scene.clear();
    renderer.dispose();
    renderer.forceContextLoss();
  }

  resize();
  renderOnce();
  return { setStage, setActive, dispose, get stage() { return state.stage; } };
}

/** Générateur pseudo-aléatoire déterministe (rendu identique à chaque visite). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
