# AIOTrade · Gestion des risques et trading algorithmique

> Ne pas prédire le marché : générer un faisceau de trajectoires d'allocation et n'exécuter que celles qui
> respectent des contraintes strictes.

AIOTrade est un co-pilote de conformité, de surveillance et de décision pour desks de prop trading et gérants
de portefeuille. Ce dossier contient le moteur (Python), son API (FastAPI), le site vitrine 3D (Vite + Three.js)
et la conteneurisation (Docker Compose).

**Simulation uniquement** : marché synthétique avec chocs injectables, aucune connexion broker, aucun ordre réel,
aucune clé d'API.

## Démarrage en une commande

```bash
cd aiotrade
docker compose up --build
```

Puis ouvrir <http://localhost:8088> (port modifiable : `AIOTRADE_PORT=9000 docker compose up --build`).
La documentation interactive de l'API est servie sur <http://localhost:8088/api/docs>.

## Architecture

```text
                       ┌─────────────────────────────┐
  Flux de marché ────▶ │ Filtre χ²  (surveillance)   │──── alerte ──▶ mode sécurité (cash)
  prix · volumes ·     └─────────────────────────────┘
  spreads    │
             ▼
   ┌──────────────────┐   tendance        ┌────────────────────┐
   │   Moteur TAP     │── retour moyenne ─▶│  Garde-fou SCG     │── admissibles ──▶ meilleur score ──▶ Portefeuille
   │ faisceaux de     │   couverture       │ drawdown · VaR ·   │                                        │
   │ trajectoires     │   cash             │ marge              │── rejets ──▶ journal d'audit ◀─────────┘
   └──────────────────┘                    └────────────────────┘
```

| Composant | Fichier | Rôle |
|---|---|---|
| Marché synthétique | `engine/aiotrade/market.py` | 4 actifs corrélés, régimes tendance / range, chocs `flash_crash`, `liquidity_drop`, `manipulation` |
| Moteur TAP | `engine/aiotrade/tap.py` | Scénarios tendance, retour à la moyenne, couverture (variance minimale) et cash ; 256 trajectoires par scénario (bootstrap par blocs) ; score moyenne-variance |
| Garde-fou SCG | `engine/aiotrade/scg.py` | Rejet immédiat si drawdown projeté > limite, VaR historique 99 % > budget, ou marge requise > marge disponible ; vérification d'ordres unitaires |
| Filtre χ² | `engine/aiotrade/chi2.py` | Test de Pearson sur les classes de quantiles (rendements, log-volumes, log-spreads) ; p-value calibrée par Monte-Carlo ; mode sécurité avec temporisation de sortie |
| Co-pilote | `engine/aiotrade/copilot.py` | Boucle χ² → TAP → SCG → exécution, surveillance continue de la position, journal d'audit |
| API | `engine/aiotrade/api.py` | `GET /api/health`, `GET /api/limits`, `POST /api/simulate`, `POST /api/guard/check`, `POST /api/contact` |

### Pourquoi une p-value Monte-Carlo pour le χ² ?

Les classes de queue (2 % de chaque côté) attendent 0,2 observation par fenêtre : l'approximation asymptotique
du χ² déclenchait environ 10 % de fausses alertes. La p-value de décision est donc calculée sur 100 000 tirages
multinomiaux sous l'hypothèse nulle ; la p-value asymptotique reste exposée à titre indicatif.

## Résultats mesurés (simulation)

Sur 80 simulations (20 graines × 4 types de marché, 600 pas, limite de drawdown 10 %) :

| Marché | Drawdown moyen AIOTrade | Drawdown max AIOTrade | Drawdown moyen équipondéré |
|---|---|---|---|
| Sans choc | 6,4 % | 9,9 % | 12,9 % |
| Flash-crash | 7,0 % | 9,8 % | 15,2 % |
| Décrochage de liquidité | 5,9 % | 9,8 % | 13,4 % |
| Manipulation | 6,2 % | 9,9 % | 12,7 % |

Aucun dépassement de la limite sur ces 80 runs. Le filtre χ² détecte un flash-crash ou un décrochage de liquidité
en 1 pas, une manipulation en 2 à 3 pas, avec environ 0,1 % de fausses alertes par pas en marché calme.

Limite connue : le garde-fou raisonne sur des risques projetés. Un gap de marché survenant avant la détection
(le premier pas d'un krach) peut, rarement, faire dépasser la limite ; c'est un risque de gap, pas un ordre
accepté à tort. Les trajectoires TAP sont rééchantillonnées sur l'historique récent : elles décrivent la dispersion
plausible, pas une prévision.

## Développement local

```bash
# Moteur + tests
cd aiotrade/engine
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest                       # 45 tests
python -m aiotrade --shock flash_crash --events
uvicorn aiotrade.api:app --reload --port 8000

# Site (proxy /api vers le port 8000)
cd aiotrade/web
npm ci
npm run dev
```

Les tests s'exécutent aussi dans le conteneur : `docker build --target test aiotrade/engine`.
La CI GitHub Actions (`.github/workflows/aiotrade.yml`) lance les tests, le build du site et la stack Docker.

## Site web : exigences couvertes

| Exigence | Mise en œuvre |
|---|---|
| 3D Three.js à 60 FPS | Graphe du flux de décision en motion design (particules = ordres, faisceaux TAP, rejets SCG, bulle de sécurité χ²) ; pixel ratio adaptatif selon le FPS mesuré ; rendu suspendu hors écran et onglet masqué ; Three.js chargé à la demande après le premier rendu |
| Rendu PBR | `MeshStandardMaterial`, environnement studio pré-filtré (`RoomEnvironment` + PMREM), tone mapping ACES |
| Repli 2D | SVG animé en CSS si WebGL absent, appareil modeste (`deviceMemory`, cœurs, `saveData`) ou perte de contexte |
| Mémoire | `dispose()` des géométries, matériaux, textures, environnement et contexte WebGL à la sortie de page |
| Scroll natif | Aucune capture du défilement : les étapes HTML pilotent la caméra via `IntersectionObserver` ; survol des nœuds avec info-bulle, clic = défilement vers l'étape |
| Mouvement réduit | `prefers-reduced-motion` : rendu statique par étape, aucune animation |
| Accessibilité | HTML sémantique sous le canvas (`aria-hidden`), lien d'évitement, focus visible, contrastes AA, formulaires étiquetés, erreurs annoncées ; audit axe-core WCAG 2.1 AA : 0 violation sur les 4 pages |
| Consentement CNIL | « Tout accepter » et « Tout refuser » de même style et même niveau, personnalisation, aucun traceur avant choix, choix conservé 6 mois, lien « Gérer mes cookies » |
| Pages légales | `mentions-legales.html`, `confidentialite.html` (finalités, bases légales, durées, droits), `cookies.html` |
| Formulaire | Case de consentement non pré-cochée, mention d'information, validation serveur, limitation de débit, aucune donnée conservée dans cette démo |
| SEO | Open Graph + image, JSON-LD (`Organization`, `WebSite`, `SoftwareApplication`, `FAQPage`), `robots.txt`, `sitemap.xml`, URL canoniques |
| Analytics éthiques | Plausible / Matomo optionnel, chargé uniquement après consentement (`meta name="aiotrade:analytics-src"`) ; aucun appel tiers par défaut |
| Sécurité | CSP stricte (`script-src 'self'`, `style-src 'self'`), `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy` ; API servie sur la même origine |
| Conversion | CTA « Demander une démo » répétés, démo interactive, avantages, cas d'usage, indicateurs de confiance chiffrés, FAQ, fiche technique en lead magnet |

## Avant la mise en ligne

- Remplacer le domaine `www.aiotrade.example` (balises canoniques, Open Graph, JSON-LD, `robots.txt`, `sitemap.xml`).
- Compléter les champs surlignés `[…]` des pages légales (éditeur, hébergeur, contacts).
- Remplacer les deux emplacements de témoignages par des témoignages réels et autorisés.
- Brancher `POST /api/contact` sur votre CRM ou votre messagerie (la démo valide puis ignore la demande).
- Servir le site en HTTPS et activer HSTS (`web/nginx/security-headers.conf`).
- Pour activer la mesure d'audience, renseigner `aiotrade:analytics-src` et ajouter le domaine de l'outil à la CSP.

## Avertissement

AIOTrade est un outil d'aide à la décision et de contrôle des risques. Il ne constitue pas un conseil en
investissement. Les résultats présentés proviennent de simulations sur données synthétiques et ne préjugent pas
des performances futures.
