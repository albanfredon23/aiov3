/**
 * Démo interactive : appelle POST /api/simulate (API v2) et trace les courbes
 * de valeur des quatre bras du protocole, la distance d² du filtre
 * d'intégrité, les gels et les blackouts macro, puis affiche le dernier
 * enregistrement du registre XAI.
 */
import { track } from './analytics.js';

const ARMS = [
  { id: 'D', color: '#2dd4bf', width: 2.6, dash: [] },
  { id: 'B', color: '#fbbf24', width: 1.6, dash: [6, 4] },
  { id: 'A', color: '#94a3b8', width: 1.4, dash: [2, 3] },
  { id: 'C', color: '#f472b6', width: 1.4, dash: [8, 3, 2, 3] },
];
const COLORS = {
  freeze: 'rgba(167, 139, 250, 0.2)',
  blackout: 'rgba(244, 114, 182, 0.22)',
  shock: 'rgba(255, 90, 90, 0.9)',
  grid: 'rgba(148, 163, 184, 0.16)',
  text: '#a9b4c8',
  d2: '#a78bfa',
  alert: 'rgba(251, 191, 36, 0.8)',
  freezeLine: 'rgba(255, 122, 122, 0.85)',
};

const fmt = (v, digits = 1) => v.toLocaleString('fr-FR', { maximumFractionDigits: digits, minimumFractionDigits: digits });
const pct = (v, digits = 1) => `${fmt(v * 100, digits)} %`;
const signedPct = (v, digits = 2) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${pct(Math.abs(v), digits)}`;

export function initDemo({ reducedMotion }) {
  const form = document.getElementById('demo-form');
  const canvas = document.getElementById('demo-chart');
  if (!form || !canvas) return;
  const status = document.getElementById('demo-status');
  const kpis = document.getElementById('demo-kpis');
  const table = document.getElementById('demo-table');
  const record = document.getElementById('demo-record');
  const lev = document.getElementById('demo-lev');
  const levOut = document.getElementById('demo-lev-out');
  const adm = document.getElementById('demo-adm');
  const admOut = document.getElementById('demo-adm-out');
  const button = form.querySelector('button[type="submit"]');

  let data = null;
  let progress = 1;
  let controller = null;

  lev.addEventListener('input', () => {
    levOut.value = `${fmt(Number(lev.value), 1)}×`;
  });
  adm.addEventListener('input', () => {
    admOut.value = `${adm.value} %`;
  });

  function shade(ctx, flags, x, top, height, color) {
    ctx.fillStyle = color;
    let start = null;
    flags.forEach((on, i) => {
      if (on && start === null) start = i;
      if ((!on || i === flags.length - 1) && start !== null) {
        ctx.fillRect(x(start), top, Math.max(2, x(i) - x(start)), height);
        start = null;
      }
    });
  }

  function draw() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(320, Math.round(rect.width));
    const h = Math.round(w * 0.56);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!data) return;

    const { equity, d2, status: integrity, gate } = data.series;
    const n = d2.length;
    const pad = { l: 60, r: 12, t: 14, b: 26 };
    const split = Math.round((h - pad.t - pad.b) * 0.66);
    const top1 = pad.t;
    const h1 = split - 10;
    const top2 = pad.t + split + 8;
    const h2 = h - pad.b - top2;
    const x = (i) => pad.l + (i / (n - 1)) * (w - pad.l - pad.r);

    const all = ARMS.flatMap((a) => equity[a.id]).filter((v) => v !== null);
    const lo = Math.min(...all, 1);
    const hi = Math.max(...all, 1);
    const span = Math.max(hi - lo, 0.002);
    const min = lo - span * 0.08;
    const max = hi + span * 0.08;
    const y1 = (v) => top1 + (1 - (v - min) / (max - min)) * h1;
    const logOf = (v) => Math.log10(Math.max(v ?? 0.1, 0.1));
    const dMax = Math.max(logOf(Math.max(...d2.filter((v) => v !== null))), logOf(data.thresholds.freeze) + 0.2);
    const y2 = (v) => top2 + (1 - (logOf(v) + 1) / (dMax + 1)) * h2;

    // Zones : gels d'intégrité et blackouts macro, sur les deux panneaux.
    shade(ctx, integrity.map((s) => s === 'FREEZE'), x, top1, h - pad.b - top1, COLORS.freeze);
    shade(ctx, gate.map((s) => s === 'BLACKOUT'), x, top1, h - pad.b - top1, COLORS.blackout);

    ctx.font = '12px system-ui, sans-serif';
    ctx.strokeStyle = COLORS.grid;
    ctx.fillStyle = COLORS.text;
    ctx.lineWidth = 1;
    for (let k = 0; k <= 4; k += 1) {
      const v = min + ((max - min) * k) / 4;
      const yy = y1(v);
      ctx.beginPath();
      ctx.moveTo(pad.l, yy);
      ctx.lineTo(w - pad.r, yy);
      ctx.stroke();
      ctx.fillText(signedPct(v - 1, 2), 4, yy + 4);
    }
    ctx.fillText('d² (échelle log)', 4, top2 + 12);
    ctx.fillText('Une semaine hors échantillon, barres de 15 minutes →', pad.l, h - 6);

    // Seuils du filtre d'intégrité
    const thresholds = [[data.thresholds.alert, COLORS.alert, 'alerte', 14], [data.thresholds.freeze, COLORS.freezeLine, 'gel', -4]];
    thresholds.forEach(([v, color]) => {
      ctx.strokeStyle = color;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(pad.l, y2(v));
      ctx.lineTo(w - pad.r, y2(v));
      ctx.stroke();
      ctx.setLineDash([]);
    });

    // Début du choc
    if (data.shock_window) {
      const xs = x(data.shock_window[0]);
      ctx.strokeStyle = COLORS.shock;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xs, top1);
      ctx.lineTo(xs, h - pad.b);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.shock;
      ctx.fillText('choc', Math.min(xs + 4, w - 40), top1 + 12);
    }

    const last = Math.max(1, Math.floor((n - 1) * progress));
    const line = (series, y, color, width, dash) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i <= last; i += 1) {
        if (series[i] === null) continue;
        if (!started) {
          ctx.moveTo(x(i), y(series[i]));
          started = true;
        } else ctx.lineTo(x(i), y(series[i]));
      }
      ctx.stroke();
      ctx.setLineDash([]);
    };
    [...ARMS].reverse().forEach((a) => line(equity[a.id], y1, a.color, a.width, a.dash));
    line(d2, y2, COLORS.d2, 1.2, []);

    // Étiquettes des seuils, au-dessus de la courbe d².
    thresholds.forEach(([v, color, label, dy]) => {
      const text = `${label} ${fmt(v, 1)}`;
      const tw = ctx.measureText(text).width;
      ctx.fillStyle = 'rgba(11, 16, 29, 0.85)';
      ctx.fillRect(w - pad.r - tw - 8, y2(v) + dy - 11, tw + 6, 15);
      ctx.fillStyle = color;
      ctx.fillText(text, w - pad.r - tw - 5, y2(v) + dy);
    });
  }

  function animate() {
    if (reducedMotion) {
      progress = 1;
      draw();
      return;
    }
    const start = performance.now();
    const tick = (now) => {
      progress = Math.min(1, (now - start) / 1600);
      draw();
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function render(result) {
    data = result;
    const s = result.summary;
    const rows = table.tBodies[0].rows;
    ['D', 'B', 'A', 'C'].forEach((arm, i) => {
      const cells = rows[i].cells;
      cells[1].textContent = signedPct(s[arm].total_return);
      cells[2].textContent = pct(s[arm].max_drawdown, 2);
      cells[3].textContent = String(s[arm].trades);
    });
    const frozen = result.series.status.filter((v) => v === 'FREEZE').length;
    const tested = result.scg.tested;
    const b = result.ledger.breakdown || {};
    const parts = [
      [b.exposed, 'exposées'],
      [b.freeze, 'gel intégrité'],
      [b.macro, 'blackout macro'],
      [b.edge, 'sans avantage net'],
      [b.scg, 'refus SCG'],
    ].filter(([v]) => v > 0);
    const scgNote = tested ? ` (SCG : ${pct(result.scg.scg_pruning_rate, 1)} de ${tested.toLocaleString('fr-FR')} trajectoires élaguées)` : '';
    const values = [
      `${frozen} sur ${result.series.status.length}`,
      `${parts.map(([v, label]) => `${v} ${label}`).join(' · ') || '–'}${scgNote}`,
      `${result.ledger.records} décisions, chaîne ${result.ledger.verified ? 'vérifiée' : 'INVALIDE'}`,
    ];
    kpis.querySelectorAll('dd').forEach((dd, i) => {
      dd.textContent = values[i];
    });
    kpis.querySelectorAll('dd')[2].classList.toggle('kpi-ok', result.ledger.verified);
    record.textContent = JSON.stringify(result.ledger.latest, null, 2);

    canvas.setAttribute(
      'aria-label',
      `Courbes de valeur sur une semaine simulée. AIOTrade : ${signedPct(s.D.total_return)}, drawdown maximal ${pct(s.D.max_drawdown, 2)}. ` +
        `Prédictif non contraint : ${signedPct(s.B.total_return)}, drawdown ${pct(s.B.max_drawdown, 2)}. ` +
        `Momentum : ${signedPct(s.A.total_return)}. TAP sans SCG : ${signedPct(s.C.total_return)}. ` +
        `${frozen} barres gelées par le filtre d'intégrité.`,
    );
    animate();
  }

  async function run() {
    controller?.abort();
    controller = new AbortController();
    const body = {
      seed: Number(form.elements.seed.value) || 0,
      shock: form.elements.shock.value,
      mandate: { max_leverage: Number(lev.value), min_admissibility: Number(adm.value) / 100 },
    };
    button.disabled = true;
    form.setAttribute('aria-busy', 'true');
    status.textContent = 'Calibrage sur 12 mois puis simulation de la semaine…';
    try {
      const res = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      render(await res.json());
      status.textContent = 'Simulation terminée.';
      track('demo_simulation', { shock: body.shock });
    } catch (error) {
      if (error.name === 'AbortError') return;
      status.textContent =
        "La démo n'a pas pu joindre l'API AIOTrade. Lancez-la avec « docker compose up » puis réessayez.";
    } finally {
      button.disabled = false;
      form.removeAttribute('aria-busy');
    }
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    run();
  });
  new ResizeObserver(() => draw()).observe(canvas);

  // Première simulation lancée quand la section approche de l'écran.
  const io = new IntersectionObserver(
    (entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        io.disconnect();
        run();
      }
    },
    { rootMargin: '200px 0px' },
  );
  io.observe(form);
}
