# AIOTrade · Garde-fou neuro-symbolique pour le trading algorithmique

> La rentabilité est une conséquence. La trajectoire admissible sous contraintes de risque est l'objectif.

AIOTrade n'est pas un bot prédictif : c'est une garantie algorithmique d'auditabilité (XAI) et de non-violation
du capital pour prop desks, fonds quantitatifs et traders en challenge prop-firm. Chaque décision est testée sur
un faisceau de trajectoires, élaguée par le mandat de risque, dimensionnée par un Kelly prudent et scellée dans
un registre d'audit chaîné.

Ce dossier contient le moteur (Python), son API (FastAPI), un connecteur courtier (Binance Spot, testnet par
défaut), le site vitrine 3D (Vite + Three.js) et la conteneurisation (Docker Compose).

**Par défaut, rien n'est envoyé à un marché réel** : la démo et le benchmark tournent sur un marché synthétique,
le mode `trade` est en papier, et le live exige un double verrou explicite (voir plus bas).

## Démarrage en une commande

```bash
cd aiotrade
docker compose up --build
```

Puis ouvrir <http://localhost:8088> (port modifiable : `AIOTRADE_PORT=9000 docker compose up --build`).
La documentation interactive de l'API est servie sur <http://localhost:8088/api/docs>.

## Le pipeline

```text
 Barres OHLCV ─▶ 1. Filtre d'intégrité χ² / Mahalanobis ── FREEZE ─────────────────────────────▶ CASH
                 2. Macro Gate (calendrier NFP, IPC, Fed, BCE) ── BLACKOUT ──────────────────▶ CASH
                                                              └─ DERISK : exposition × 0,5
                 3. Prévision : essaim d'agents (Trend, Mean-Reversion, Macro, Risque) ou Kronos
                    → N = 1 024 trajectoires sur H = 12 barres
                 4. TAP : faisceau par direction, stop au quantile 80 % de l'excursion adverse
                 5. Kelly fractionnaire prudent, plafonné par le mandat, arrondi au lot inférieur
                 6. SCG : E = 1 + w·R − frictions ; admis si ≥ 75 % des trajectoires survivent
                    et VaR / CVaR dans le budget, sinon taille ÷ 2 (4 fois au plus), puis CASH
                 7. LONG / SHORT / CASH ─▶ exécution maker / taker ─▶ registre XAI chaîné SHA-256
```

| Composant | Fichier | Rôle |
|---|---|---|
| Marché synthétique | `engine/aiotrade/market.py` | GARCH(1,1) à chocs de Student, régimes, calendrier macro, chocs `flash_crash`, `liquidity_drop`, `spoofing` |
| Filtre d'intégrité | `engine/aiotrade/integrity.py` | Distance de Mahalanobis d² sur (rendement, spread, profondeur, volume) contre une référence EWMA ; seuil d'alerte = max(χ²(4) à 99 % = 13,28, quantile empirique), gel immédiat au-delà de 99,99 % (23,51 en théorie) ou après 3 alertes sur 5 barres ; calibré hors fenêtres d'annonces |
| Macro Gate | `engine/aiotrade/macro_gate.py` | Veto calendaire déterministe sur les événements HIGH : exposition × 0,5 de −45 à −15 min, blackout et réduction des positions ouvertes de −15 à +10 min ; tous les événements sont évalués, horodatages UTC conscients |
| Prévision | `engine/aiotrade/forecast.py` | Essaim de 4 agents dont les coefficients sont estimés (OLS ridge) sur la fenêtre de calibration ; pondération en ligne par régime de volatilité (log-vraisemblance actualisée) ; adaptateur Kronos (`T`, `top_p`) en option |
| TAP | `engine/aiotrade/tap.py` | Faisceau de N trajectoires multi-pas, contrôle de forme et de finitude, stop = max(q80 de l'excursion adverse, 2σ) |
| SCG | `engine/aiotrade/scg.py` | Courbes d'équité avec stop (gap compris) et frictions aller-retour ; rejets exclusifs par trajectoire : levier → drawdown → perte → CVaR → VaR ; budgets VaR / CVaR journaliers ramenés à l'horizon (√(H·Δt/1 jour)) |
| Dimensionnement | `engine/aiotrade/sizing.py` | f* = λ(pb − (1 − p))/b, 0 < λ ≤ 0,25 (0,20 par défaut), p remplacé par sa borne basse p − z√(p(1 − p)/N) ; prix et taille de contrat pris en compte, arrondi vers le bas au pas de lot, aucune division par zéro |
| Exécution | `engine/aiotrade/execution.py` | Impact Almgren-Chriss γσ(Q/V)^α (γ = 0,8, α = 0,6), spread, slippage, commissions ; ordre maker rempli si la barre suivante traverse la limite (ou la touche, avec probabilité de file 35 %), sinon repli taker |
| Registre XAI | `engine/aiotrade/ledger.py` | Un JSON par décision (y compris les refus), chaîné par SHA-256 du JSON canonique, fichier en ajout seul avec `fsync` ; `verify_chain` localise la première ligne altérée |
| Co-pilote | `engine/aiotrade/copilot.py` | Enchaîne les étapes 1 à 7 et produit l'enregistrement au format demandé (`market_integrity_d2`, `chi2_threshold`, `status`, `tap_trajectories_tested`, `scg_rejections`, `admissibility_ratio`, `selected_allocation`, `active_constraints`, plus Kelly, coûts, VaR, CVaR et motifs) |
| Backtest | `engine/aiotrade/backtest.py` | Walk-forward 12 mois / 3 mois, 4 bras, exécution à la barre suivante, mêmes frictions pour tous |
| Courtiers | `engine/aiotrade/broker/` | Interface commune ; `PaperBroker` (données publiques, aucun ordre) et `BinanceSpot` (testnet par défaut, signature HMAC-SHA256) |
| Boucle live | `engine/aiotrade/live.py` | Décision à chaque clôture de barre, recalibrage, enregistrement XAI, ordre du delta de position |
| API | `engine/aiotrade/api.py` | `GET /api/health`, `GET /api/config`, `POST /api/simulate`, `POST /api/scg/check`, `GET /api/benchmark`, `POST /api/ledger/verify`, `POST /api/contact` |

Enregistrement réel du benchmark (extrait de `engine/reports/benchmark.json`) :

```json
{
  "timestamp_utc": "2026-04-03T20:00:00.000Z",
  "market_integrity_d2": 2.62,
  "chi2_threshold": 20.5,
  "status": "NOMINAL",
  "macro_gate": "NOMINAL",
  "tap_trajectories_tested": 1024,
  "scg_rejections": {"leverage_violation": 0, "drawdown_violation": 0, "loss_violation": 0, "cvar_violation": 0, "var_violation": 22},
  "admissibility_ratio": 0.9785,
  "selected_allocation": -0.7428,
  "decision": "SHORT",
  "active_constraints": ["MAX_LEVERAGE_1.5X", "MAX_LOSS_PER_TRADE_1PCT", "STOP_LOSS_1.08009"],
  "seq": 236,
  "prev_hash": "…",
  "hash": "…"
}
```

## Résultats mesurés

Protocole : marché synthétique EURUSD en barres de 15 minutes, calibration sur 12 mois puis test sur les 3 mois
suivants, 4 fois (12 mois hors échantillon), décision toutes les 4 barres, frais complets pour les 4 bras.
Rapport complet : `engine/reports/benchmark.json` (`python -m aiotrade benchmark`).

| Bras (graine 11) | Perf. nette | Drawdown max | Sortino | Calmar |
|---|---|---|---|---|
| A · Momentum classique | −22,9 % | 26,1 % | −4,14 | −0,88 |
| B · Prédictif non contraint | −27,6 % | 30,4 % | −3,43 | −0,91 |
| C · TAP sans SCG | −39,0 % | 40,2 % | −9,06 | −0,97 |
| **D · AIOTrade complet** | **+0,05 %** | **0,31 %** | **0,22** | **0,15** |

| Objectif du protocole | Cible | Mesuré | Statut |
|---|---|---|---|
| Réduction du drawdown max vs B | ≥ 40 % | 99,0 % | Atteint |
| Sortino net | > 2,0 | 0,22 | Non atteint |
| Calmar net | > 1,5 | 0,15 | Non atteint |
| Taux d'élagage du SCG | 15 à 45 % | 5,8 % | Non atteint |

Robustesse (graines 12 et 13) : le drawdown de D reste entre 0,1 et 0,2 % ; celui de B va de 8,8 à 21,6 %.

Lecture honnête : le marché synthétique n'offre pas d'avantage exploitable net de frais à 3 heures. AIOTrade le
détecte, le Kelly prudent reste nul et le portefeuille demeure en cash 99,7 % du temps. Le garde-fou fait son
travail (aucun dépassement du mandat), mais les objectifs de Sortino, de Calmar et d'élagage ne peuvent être
jugés que sur données réelles, avec un modèle de prévision qui porte un vrai signal.

## Développement local

```bash
# Moteur + tests
cd aiotrade/engine
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest                                  # 112 tests (1 ignoré sans pandas)
python -m aiotrade simulate --shock flash_crash --ledger
python -m aiotrade benchmark --seeds 11 12 13 --out reports/benchmark.json
python -m aiotrade ledger-verify ledger/decisions.jsonl
python -m aiotrade serve --port 8000

# Site (proxy /api vers le port 8000)
cd aiotrade/web
npm ci
npm run dev
```

Les tests s'exécutent aussi dans le conteneur : `docker build --target test aiotrade/engine`.
La CI GitHub Actions (`.github/workflows/aiotrade.yml`) lance les tests, le build du site et la stack Docker.

Kronos (optionnel) : `pip install -e ".[kronos]"` installe torch, pandas et huggingface_hub ; le dépôt Kronos
doit être cloné et ajouté au `PYTHONPATH`. Sans lui, l'essaim d'agents est utilisé.

## Courtiers : papier, testnet, live

**XTB** : l'API xAPI d'XTB a été fermée le 14 mars 2025, il n'existe plus d'accès programmatique. Le connecteur
livré est Binance Spot ; tout autre courtier se branche en implémentant l'interface `broker/base.py`.

**Papier (par défaut)** : données publiques Binance, exécution simulée, aucune clé.

```bash
python -m aiotrade trade --broker paper --symbol BTCUSDT --loop
```

**Testnet Binance Spot** : créer une clé sur <https://testnet.binance.vision>, puis la passer uniquement par
variables d'environnement (jamais dans le code, un commit ou un journal) :

```bash
cp .env.example .env        # puis renseigner AIOTRADE_BINANCE_API_KEY et AIOTRADE_BINANCE_API_SECRET
python -m aiotrade broker-check --symbol BTCUSDT
python -m aiotrade trade --broker binance --symbol BTCUSDT --loop
# ou en conteneur :
docker compose --profile trading up -d trader
```

Avec `AIOTRADE_SEND_ORDERS=0` (défaut), les ordres sont validés par Binance via `/api/v3/order/test` sans être
exécutés ; passer à `1` pour les exécuter sur le testnet. Le spot ne permet pas la vente à découvert : en mode
courtier, AIOTrade n'émet que LONG ou CASH. Un stop `STOP_LOSS_LIMIT` est posé après chaque achat.

**Live : check-list avant d'activer**

1. Plusieurs semaines de testnet sans anomalie, registre vérifié (`ledger-verify`) et relu.
2. Clé live dédiée : trading seul, retraits désactivés, liste blanche d'IP de votre serveur.
3. Double verrou : `AIOTRADE_BINANCE_ENV=live` **et** `AIOTRADE_LIVE_TRADING=JE_COMPRENDS_LES_RISQUES`, avec
   les clés séparées `AIOTRADE_BINANCE_LIVE_API_KEY` / `AIOTRADE_BINANCE_LIVE_API_SECRET`.
4. Plafond de notionnel par ordre bas (`AIOTRADE_MAX_ORDER_NOTIONAL`, par exemple 50 USDT) et capital limité.
5. Tourne sur votre infrastructure, sous votre responsabilité ; surveillance du conteneur et du registre.

Sans ces deux variables, le connecteur refuse de s'adresser à l'API live.

## Site web : exigences couvertes

| Exigence | Mise en œuvre |
|---|---|
| 3D Three.js à 60 FPS | Graphe du pipeline en 7 étapes : flux de marché, bouclier d'intégrité et horloge macro, essaim d'agents, faisceau TAP, anneau SCG qui élague les trajectoires en rouge, cube Kelly, cône d'exécution, chaîne de blocs du registre ; pixel ratio adaptatif selon le FPS mesuré ; rendu suspendu hors écran ; Three.js chargé à la demande |
| Rendu PBR | `MeshStandardMaterial`, environnement studio pré-filtré (`RoomEnvironment` + PMREM), tone mapping ACES |
| Repli 2D | SVG animé en CSS par étape si WebGL absent, appareil modeste (`deviceMemory`, cœurs, `saveData`) ou perte de contexte |
| Mémoire | `dispose()` des géométries, matériaux, textures, environnement et contexte WebGL à la sortie de page |
| Scroll natif | Aucune capture du défilement : les étapes HTML pilotent la caméra via `IntersectionObserver` ; survol des nœuds avec info-bulle, clic = défilement vers l'étape |
| Mouvement réduit | `prefers-reduced-motion` : rendu statique par étape, aucune animation |
| Accessibilité | HTML sémantique sous le canvas (`aria-hidden`), lien d'évitement, focus visible, contrastes AA, formulaires étiquetés, erreurs annoncées, tableaux avec légendes ; audit axe-core WCAG 2.1 AA : 0 violation sur les 4 pages |
| Consentement CNIL | « Tout accepter » et « Tout refuser » de même style et même niveau, personnalisation, aucun traceur avant choix, choix conservé 6 mois, lien « Gérer mes cookies » |
| Pages légales | `mentions-legales.html`, `confidentialite.html` (finalités, bases légales, durées, droits), `cookies.html` |
| Formulaire | Case de consentement non pré-cochée, mention d'information, validation serveur, limitation de débit, aucune donnée conservée dans cette démo |
| SEO | Open Graph + image, JSON-LD (`Organization`, `WebSite`, `SoftwareApplication` avec offres, `FAQPage`), `robots.txt`, `sitemap.xml`, URL canoniques |
| Analytics éthiques | Plausible / Matomo optionnel, chargé uniquement après consentement (`meta name="aiotrade:analytics-src"`) ; aucun appel tiers par défaut |
| Sécurité | CSP stricte (`script-src 'self'`, `style-src 'self'`), `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy` ; API servie sur la même origine |
| Conversion | CTA « Demander une démo » répétés, démo interactive à 4 bras avec registre en direct, résultats et limites, compatibilité des marchés, offres (SaaS, licence API / FIX, cession), indicateurs de confiance, FAQ |

## Avant la mise en ligne

- Remplacer le domaine `www.aiotrade.example` (balises canoniques, Open Graph, JSON-LD, `robots.txt`, `sitemap.xml`).
- Compléter les champs surlignés `[…]` des pages légales (éditeur, hébergeur, contacts).
- Remplacer les deux emplacements de témoignages par des témoignages réels et autorisés.
- Brancher `POST /api/contact` sur votre CRM ou votre messagerie (la démo valide puis ignore la demande).
- Servir le site en HTTPS et activer HSTS (`web/nginx/security-headers.conf`).
- Pour activer la mesure d'audience, renseigner `aiotrade:analytics-src` et ajouter le domaine de l'outil à la CSP.
- Remplacer `engine/data/macro_calendar.example.json` par un vrai calendrier économique.

## Avertissement

AIOTrade est un outil d'aide à la décision et de contrôle des risques. Il ne constitue pas un conseil en
investissement. Les résultats présentés proviennent de simulations sur données synthétiques et ne préjugent pas
des performances futures. Le trading à effet de levier peut entraîner la perte de tout le capital engagé.
