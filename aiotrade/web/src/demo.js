/**
 * Démo interactive : appelle l'API /api/simulate et trace la courbe de valeur
 * d'AIOTrade face à un portefeuille équipondéré sans garde-fou.
 */
import { track } from './analytics.js';

const COLORS = {
  aiotrade: '#2dd4bf',
  benchmark: '#94a3b8',
  safe: 'rgba(167, 139, 250, 0.18)',
  shock: 'rgba(255, 90, 90, 0.85)',
  grid: 'rgba(148, 163, 184, 0.16)',
  text: '#a9b4c8',
};

const pct = (v, digits = 1) => `${(v * 100).toLocaleString('fr-FR', { maximumFractionDigits: digits, minimumFractionDigits: digits })}\u00a0%`;
const signedPct = (v) => `${v >= 0 ? '+' : ''}${pct(v)}`;

export function initDemo({ reducedMotion }) {
  const form = document.getElementById('demo-form');
  const canvas = document.getElementById('demo-chart');
  if (!form || !canvas) return;
  const status = document.getElementById('demo-status');
  const kpis = document.getElementById('demo-kpis');
  const events = document.getElementById('demo-events');
  const ddInput = document.getElementById('demo-dd');
  const ddOut = document.getElementById('demo-dd-out');
  const button = form.querySelector('button[type="submit"]');

  let data = null;
  let progress = 1;
  let controller = null;

  ddInput.addEventListener('input', () => {
    ddOut.value = `${ddInput.value}\u00a0%`;
  });

  function draw() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(320, Math.round(rect.width));
    const h = Math.round(w * 0.44);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!data) return;

    const { equity, benchmark, safe_mode: safe, step } = data.series;
    const e0 = equity[0];
    const b0 = benchmark[0];
    const eq = equity.map((v) => v / e0);
    const bm = benchmark.map((v) => v / b0);
    const all = eq.concat(bm);
    const min = Math.min(...all) * 0.98;
    const max = Math.max(...all) * 1.02;
    const pad = { l: 56, r: 12, t: 14, b: 26 };
    const x = (i) => pad.l + (i / (eq.length - 1)) * (w - pad.l - pad.r);
    const y = (v) => pad.t + (1 - (v - min) / (max - min)) * (h - pad.t - pad.b);

    // Grille et axe des ordonnées
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillStyle = COLORS.text;
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (let k = 0; k <= 4; k += 1) {
      const v = min + ((max - min) * k) / 4;
      const yy = y(v);
      ctx.beginPath();
      ctx.moveTo(pad.l, yy);
      ctx.lineTo(w - pad.r, yy);
      ctx.stroke();
      ctx.fillText(signedPct(v - 1), 4, yy + 4);
    }
    ctx.fillText('Temps (pas de simulation) →', pad.l, h - 6);

    // Zones de mode sécurité
    ctx.fillStyle = COLORS.safe;
    let startSafe = null;
    safe.forEach((on, i) => {
      if (on && startSafe === null) startSafe = i;
      if ((!on || i === safe.length - 1) && startSafe !== null) {
        ctx.fillRect(x(startSafe), pad.t, Math.max(3, x(i) - x(startSafe)), h - pad.t - pad.b);
        startSafe = null;
      }
    });

    // Début du choc
    if (data.shock_window) {
      const idx = step.findIndex((s) => s >= data.shock_window[0]);
      if (idx >= 0) {
        ctx.strokeStyle = COLORS.shock;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x(idx), pad.t);
        ctx.lineTo(x(idx), h - pad.b);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COLORS.shock;
        ctx.fillText('choc', Math.min(x(idx) + 4, w - 40), pad.t + 12);
      }
    }

    const last = Math.max(1, Math.floor((eq.length - 1) * progress));
    const line = (series, color, width, dash = []) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.beginPath();
      for (let i = 0; i <= last; i += 1) {
        if (i === 0) ctx.moveTo(x(i), y(series[i]));
        else ctx.lineTo(x(i), y(series[i]));
      }
      ctx.stroke();
      ctx.setLineDash([]);
    };
    line(bm, COLORS.benchmark, 1.5, [5, 4]);
    line(eq, COLORS.aiotrade, 2.5);
  }

  function animate() {
    if (reducedMotion) {
      progress = 1;
      draw();
      return;
    }
    const start = performance.now();
    const tick = (now) => {
      progress = Math.min(1, (now - start) / 1400);
      draw();
      if (progress < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function render(result) {
    data = result;
    const s = result.summary;
    const values = [
      pct(s.max_drawdown),
      pct(s.benchmark_max_drawdown),
      signedPct(s.final_return),
      String(s.rejected_allocations),
      String(s.safe_mode_activations),
    ];
    kpis.querySelectorAll('dd').forEach((dd, i) => {
      dd.textContent = values[i];
    });
    const kpiDd = kpis.querySelectorAll('dd')[0];
    kpiDd.classList.toggle('kpi-ok', s.max_drawdown <= s.drawdown_limit);

    const interesting = result.events.filter((e) => e.kind !== 'rebalance');
    const lastRebalance = [...result.events].reverse().find((e) => e.kind === 'rebalance');
    const shown = interesting.slice(-6);
    if (lastRebalance) shown.push(lastRebalance);
    events.replaceChildren(
      ...shown.map((e) => {
        const li = document.createElement('li');
        li.className = `event event-${e.kind}`;
        const stepEl = document.createElement('span');
        stepEl.className = 'event-step';
        stepEl.textContent = `pas ${e.step}`;
        li.append(stepEl, document.createTextNode(` ${e.message}`));
        return li;
      }),
    );

    canvas.setAttribute(
      'aria-label',
      `Courbe de valeur simulée. AIOTrade : performance ${signedPct(s.final_return)}, drawdown maximal ${pct(s.max_drawdown)} ` +
        `pour une limite de ${pct(s.drawdown_limit, 0)}. Portefeuille équipondéré sans garde-fou : performance ` +
        `${signedPct(s.benchmark_return)}, drawdown maximal ${pct(s.benchmark_max_drawdown)}. ` +
        `${s.safe_mode_activations} mise(s) en sécurité par le filtre χ².`,
    );
    animate();
  }

  async function run() {
    controller?.abort();
    controller = new AbortController();
    const body = {
      seed: Number(form.elements.seed.value) || 0,
      steps: 600,
      shock: form.elements.shock.value,
      limits: { max_drawdown: Number(ddInput.value) / 100 },
    };
    button.disabled = true;
    form.setAttribute('aria-busy', 'true');
    status.textContent = 'Simulation en cours…';
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

  // Première simulation lancée quand la section devient visible.
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
