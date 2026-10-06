/**
 * Graphe 3D AIOTrade (Three.js) : visualisation en motion design du flux
 * Marché → Moteur TAP → scénarios → Garde-fou SCG → Portefeuille, sous la
 * surveillance du filtre χ².
 *
 * - Les « ordres » sont des particules ; une partie est rejetée par SCG
 *   (elles rougissent et tombent), le reste atteint le portefeuille.
 * - Le moteur TAP émet un faisceau de trajectoires vers chaque scénario.
 * - À l'étape χ², un choc fige le flux et une bulle de sécurité entoure le
 *   portefeuille.
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
  tap: 0x2dd4bf,
  scenario: 0x5eead4,
  scg: 0xfbbf24,
  chi2: 0xa78bfa,
  portfolio: 0x4ade80,
  reject: 0xff5a5a,
  edge: 0x334155,
};

export const NODES = [
  { id: 'market', label: 'Marché', pos: [-6, 0, 0], color: C.market, size: 0.55, stage: 1, target: '#etape-marche',
    desc: 'Prix, volumes et spreads en continu.' },
  { id: 'tap', label: 'Moteur TAP', pos: [-3, 0, 0], color: C.tap, size: 0.65, stage: 2, target: '#etape-tap',
    desc: 'Génère les scénarios et leurs faisceaux de trajectoires.' },
  { id: 'trend', label: 'Tendance', pos: [0, 2.2, -0.8], color: C.scenario, size: 0.34, stage: 2, target: '#etape-tap',
    desc: 'Scénario momentum ajusté du risque.' },
  { id: 'meanrev', label: 'Retour à la moyenne', pos: [0, 0, 0.9], color: C.scenario, size: 0.34, stage: 2, target: '#etape-tap',
    desc: "Scénario qui joue l'écart à la moyenne." },
  { id: 'hedge', label: 'Couverture', pos: [0, -2.2, -0.4], color: C.scenario, size: 0.34, stage: 2, target: '#etape-tap',
    desc: 'Scénario à variance minimale, exposition réduite.' },
  { id: 'scg', label: 'Garde-fou SCG', pos: [3, 0, 0], color: C.scg, size: 0.9, stage: 3, target: '#etape-scg', kind: 'gate',
    desc: 'Rejette tout ordre violant drawdown, VaR ou marge.' },
  { id: 'chi2', label: 'Filtre χ²', pos: [0.6, -4.2, 0.6], color: C.chi2, size: 0.55, stage: 4, target: '#etape-chi2', kind: 'shield',
    desc: 'Détecte les ruptures de marché et met en sécurité.' },
  { id: 'portfolio', label: 'Portefeuille', pos: [6, 0, 0], color: C.portfolio, size: 0.7, stage: 5, target: '#etape-portefeuille',
    desc: 'Allocation retenue, journalisée pour l’audit.' },
];

// Vues caméra par étape : [position, cible]
const VIEWS = [
  [[-0.6, 1.0, 23], [-0.6, -0.6, 0]],
  [[-4.5, 1.2, 9], [-4.5, 0, 0]],
  [[-1.5, 2.0, 10.5], [-1.5, 0, 0]],
  [[3.2, 1.2, 10.5], [3.2, 0, 0]],
  [[2.0, -0.8, 15], [2.0, -1.6, 0]],
  [[5.0, 1.4, 9.5], [5.0, 0, 0]],
];

const SCENARIO_IDS = ['trend', 'meanrev', 'hedge'];
const SCG_T = 0.75; // paramètre de la courbe au passage du garde-fou

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
  scene.fog = new THREE.Fog(C.bg, 24, 48);

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

  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 80);
  const camPos = vec(VIEWS[0][0]);
  const camLook = vec(VIEWS[0][1]);
  camera.position.copy(camPos);

  const root = new THREE.Group();
  scene.add(root);

  /* ------------------------------------------------------------- nœuds */
  const nodeMeshes = new Map();
  const labels = [];
  const sphereGeo = new THREE.IcosahedronGeometry(1, 4);
  NODES.forEach((node) => {
    let geometry = sphereGeo;
    if (node.kind === 'gate') geometry = new THREE.TorusGeometry(1, 0.12, 24, 96);
    if (node.kind === 'shield') geometry = new THREE.OctahedronGeometry(1, 0);
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
    mesh.userData = { node, baseScale: node.size, emissive: 0.35, hover: 0, flash: 0 };
    root.add(mesh);
    nodeMeshes.set(node.id, mesh);

    const { texture, aspect } = makeLabelTexture(node.label);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }));
    const h = isSmall ? 0.42 : 0.36;
    sprite.scale.set(h * aspect, h, 1);
    sprite.position.copy(vec(node.pos)).add(new THREE.Vector3(0, node.size + 0.45, 0));
    sprite.renderOrder = 10;
    sprite.userData = { node };
    root.add(sprite);
    labels.push(sprite);
  });

  // Anneau intérieur du garde-fou (effet de porte filtrante).
  const gateDisc = new THREE.Mesh(
    new THREE.CircleGeometry(0.85, 48),
    new THREE.MeshBasicMaterial({ color: C.scg, transparent: true, opacity: 0.06, side: THREE.DoubleSide, depthWrite: false }),
  );
  gateDisc.position.copy(vec(NODES[5].pos));
  gateDisc.rotation.y = Math.PI / 2;
  root.add(gateDisc);

  // Bulle de sécurité autour du portefeuille (mode sécurité χ²).
  const bubble = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.5, 3),
    new THREE.MeshStandardMaterial({
      color: C.chi2, emissive: C.chi2, emissiveIntensity: 0.4, transparent: true, opacity: 0,
      wireframe: true, depthWrite: false,
    }),
  );
  bubble.position.copy(vec(NODES[7].pos));
  root.add(bubble);

  /* ------------------------------------------------------------ liaisons */
  const posOf = (id) => vec(NODES.find((n) => n.id === id).pos);
  const edgeMaterial = new THREE.MeshStandardMaterial({ color: C.edge, emissive: 0x1e293b, emissiveIntensity: 0.6, roughness: 0.6 });
  const tube = (points, radius = 0.025) => {
    const curve = new THREE.CatmullRomCurve3(points);
    const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, 48, radius, 8, false), edgeMaterial);
    root.add(mesh);
    return curve;
  };
  tube([posOf('market'), posOf('tap')]);
  SCENARIO_IDS.forEach((id) => {
    const mid = posOf(id).clone().lerp(posOf('scg'), 0.5).add(new THREE.Vector3(0, 0, 0.2));
    tube([posOf(id), mid, posOf('scg')]);
  });
  tube([posOf('scg'), posOf('portfolio')]);

  // Liaison de surveillance χ² : marché → filtre → portefeuille.
  const chiCurve = new THREE.CatmullRomCurve3([
    posOf('market'),
    new THREE.Vector3(-4, -3.2, 0.4),
    posOf('chi2'),
    new THREE.Vector3(4.6, -3.0, 0.4),
    posOf('portfolio'),
  ]);
  const chiMaterial = new THREE.LineDashedMaterial({ color: C.chi2, dashSize: 0.18, gapSize: 0.14, transparent: true, opacity: 0.45 });
  const chiLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(chiCurve.getPoints(160)), chiMaterial);
  chiLine.computeLineDistances();
  root.add(chiLine);

  /* ----------------------------------------- faisceau de trajectoires TAP */
  const rand = mulberry32(44);
  const bundle = [];
  const bundleMaterial = new THREE.LineBasicMaterial({ color: C.tap, transparent: true, opacity: 0.32, depthWrite: false, blending: THREE.AdditiveBlending });
  const linesPerScenario = isSmall ? 8 : 14;
  SCENARIO_IDS.forEach((id) => {
    const start = posOf('tap');
    const end = posOf(id);
    for (let i = 0; i < linesPerScenario; i += 1) {
      const pts = [];
      const spread = 0.9;
      for (let k = 0; k <= 6; k += 1) {
        const t = k / 6;
        const p = start.clone().lerp(end, t);
        const env = Math.sin(Math.PI * t); // l'incertitude s'ouvre puis converge
        p.x += 0;
        p.y += (rand() - 0.5) * spread * env;
        p.z += (rand() - 0.5) * spread * env;
        pts.push(p);
      }
      const curve = new THREE.CatmullRomCurve3(pts);
      const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(40));
      const line = new THREE.Line(geometry, bundleMaterial);
      line.userData.count = 41;
      line.userData.delay = rand() * 0.6;
      root.add(line);
      bundle.push(line);
    }
  });

  /* ------------------------------------------------------ particules (ordres) */
  const dotTexture = makeDotTexture();
  const routeCurves = SCENARIO_IDS.map((id) =>
    new THREE.CatmullRomCurve3([posOf('market'), posOf('tap'), posOf(id), posOf('scg'), posOf('portfolio')]),
  );
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

  // Capteurs du filtre χ² sur la liaison de surveillance.
  const SENSORS = isSmall ? 18 : 32;
  const sGeometry = new THREE.BufferGeometry();
  const sPositions = new Float32Array(SENSORS * 3);
  sGeometry.setAttribute('position', new THREE.BufferAttribute(sPositions, 3));
  const sMaterial = new THREE.PointsMaterial({
    size: 0.12, map: dotTexture, color: C.chi2, transparent: true, opacity: 0.8,
    depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const sensors = new THREE.Points(sGeometry, sMaterial);
  sensors.frustumCulled = false;
  root.add(sensors);
  const sensorPhase = Array.from({ length: SENSORS }, (_, i) => i / SENSORS);
  sensorPhase.forEach((ph, i) => {
    const p = chiCurve.getPoint(ph);
    sPositions.set([p.x, p.y, p.z], i * 3);
  });

  const colGood = new THREE.Color(C.tap);
  const colOut = new THREE.Color(C.portfolio);
  const colReject = new THREE.Color(C.reject);
  const tmp = new THREE.Vector3();
  const tmpColor = new THREE.Color();
  const shockColor = new THREE.Color();

  function spawn(p) {
    p.alive = true;
    p.route = Math.floor(rand() * routeCurves.length);
    p.t = 0;
    p.speed = 0.09 + rand() * 0.06;
    p.rejectChance = state.stage === 3 ? 0.45 : 0.25;
    p.rejected = rand() < p.rejectChance;
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
    shock: 0, // 0..1, intensité du mode sécurité
    pointer: new THREE.Vector2(0, 0),
    parallax: new THREE.Vector2(0, 0),
    hovered: null,
  };

  function updateParticles(dt) {
    const spawnRate = state.stage === 4 ? 0 : isSmall ? 18 : 34; // ordres / seconde
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
        // En mode sécurité, les ordres en vol avant SCG sont annulés (fondu).
        if (state.shock > 0.5 && p.t < SCG_T) {
          p.fade -= dt * 2;
          if (p.fade <= 0) p.alive = false;
        }
        p.t += p.speed * dt;
        if (p.rejected && p.t >= SCG_T) {
          p.falling = true;
          p.vel.set((rand() - 0.2) * 0.8, 0.6 + rand() * 0.6, (rand() - 0.5) * 1.2);
          flash = 1;
        }
        if (p.t >= 1) {
          p.alive = false;
          return;
        }
        routeCurves[p.route].getPoint(p.t, tmp).add(p.jitter);
        const k = THREE.MathUtils.smoothstep(p.t, SCG_T, 1);
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

    // Capteurs χ²
    const sensorSpeed = 0.05 + state.shock * 0.12;
    sensorPhase.forEach((ph, i) => {
      sensorPhase[i] = (ph + sensorSpeed * dt) % 1;
      chiCurve.getPoint(sensorPhase[i], tmp);
      sPositions[i * 3] = tmp.x;
      sPositions[i * 3 + 1] = tmp.y;
      sPositions[i * 3 + 2] = tmp.z;
    });
    sGeometry.attributes.position.needsUpdate = true;
  }

  function updateNodes(dt) {
    const s = state.stage;
    nodeMeshes.forEach((mesh) => {
      const { node } = mesh.userData;
      const focused = s === 0 || node.stage === s || (s === 4 && node.id === 'portfolio') || (s === 4 && node.id === 'market');
      const targetEmissive = (s === 0 ? 0.75 : focused ? 1.3 : 0.12) + mesh.userData.hover * 0.8;
      mesh.userData.emissive += (targetEmissive - mesh.userData.emissive) * Math.min(1, dt * 4);
      let emissive = mesh.userData.emissive;
      shockColor.setHex(node.color);
      if (node.id === 'scg' && mesh.userData.flash > 0) {
        mesh.userData.flash = Math.max(0, mesh.userData.flash - dt * 3);
        emissive += mesh.userData.flash * 0.9;
      }
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
      mesh.userData.hover += ((state.hovered === mesh ? 1 : 0) - mesh.userData.hover) * Math.min(1, dt * 10);
    });
    labels.forEach((sprite) => {
      const { node } = sprite.userData;
      const focused = s === 0 || node.stage === s || (s === 4 && ['portfolio', 'market'].includes(node.id));
      const target = focused ? 1 : 0.35;
      sprite.material.opacity += (target - sprite.material.opacity) * Math.min(1, dt * 5);
    });

    // Faisceau TAP : se dessine à l'étape 2, reste visible ensuite.
    bundle.forEach((line) => {
      const local = s === 2 ? THREE.MathUtils.clamp((state.stageTime - line.userData.delay) / 1.2, 0, 1) : 1;
      line.geometry.setDrawRange(0, Math.max(2, Math.floor(line.userData.count * local)));
    });
    const bundleTarget = s === 2 ? 0.55 : s === 0 ? 0.28 : s === 4 ? 0.06 : 0.16;
    bundleMaterial.opacity += (bundleTarget - bundleMaterial.opacity) * Math.min(1, dt * 4);

    // Mode sécurité χ²
    const shockTarget = s === 4 ? 1 : 0;
    state.shock += (shockTarget - state.shock) * Math.min(1, dt * 3);
    bubble.material.opacity = state.shock * 0.35;
    bubble.rotation.y += dt * 0.3;
    bubble.scale.setScalar(1 + Math.sin(state.time * 2.2) * 0.03 * state.shock);
    chiMaterial.opacity = 0.35 + state.shock * 0.5;
    sMaterial.opacity = 0.55 + state.shock * 0.45;
    gateDisc.material.opacity = 0.05 + nodeMeshes.get('scg').userData.flash * 0.25;
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
      state.shock = next === 4 ? 1 : 0;
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
