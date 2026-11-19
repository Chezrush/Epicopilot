# EpiCopilot

> Faire voler un Tiny Whoop FPV grâce à une IA : vol en mode Acro (non stabilisé) piloté par l'IA, suivi de personne, plans cinématiques et détection d'obstacles.

Projet professionnel PGE3 — Epitech Paris, 2026-2027 — réalisé en binôme.
Énoncé complet : [`EpiCopilot_Tracker_Final.pptx`](./EpiCopilot_Tracker_Final.pptx)

---

## Sommaire

- [Le projet](#le-projet)
- [Objectifs](#objectifs)
- [Matériel](#matériel)
- [Architecture du dépôt](#architecture-du-dépôt)
- [Feuille de route](#feuille-de-route)
- [Installation](#installation)
- [Sécurité](#sécurité)
- [Données personnelles et RGPD](#données-personnelles-et-rgpd)
- [Usage de l'IA](#usage-de-lia)
- [Équipe](#équipe)

---

## Le projet

Le but est de faire voler notre Tiny Whoop avec une IA. Un ordinateur récupère en temps réel le flux vidéo du drone et envoie les commandes à la radiocommande. L'IA garde le drone stable alors qu'il est en mode **Acro** (aucune stabilisation automatique par défaut). Elle repère une personne, la suit et exécute des figures et des plans cinématiques commandés depuis une station web.

L'IA s'entraîne d'abord sur le simulateur **Uncrashed** à partir de vols enregistrés, puis passe sur le vrai drone.

## Objectifs

| Objectif | Description |
|---|---|
| **Vol stabilisé par l'IA** | Garder le drone stable en mode Acro sans intervention humaine. |
| **Suivi de personne** | Verrouiller une personne dans le champ de la caméra et la suivre en gardant une distance de sécurité. |
| **Figures et plans cinématiques** | 2 ou 3 manœuvres, par exemple tourner autour de la personne (orbite 360°), la survoler, ou un plan de drone classique comme le travelling arrière (« dronie »). |
| **Détection d'objets** | Détecter les obstacles qui apparaissent dans le champ de la caméra, puis les éviter ou s'arrêter. |
| **Station de contrôle web** | Retour vidéo avec HUD, choix des modes de vol, envoi des ordres et arrêt d'urgence. |

## Matériel

- **Drone** : Tiny Whoop caréné de moins de 50 g (type Meteor75 Pro)
- **Radiocommande** : ELRS (protocole CRSF), avec un interrupteur physique d'arrêt d'urgence (*kill switch*)
- **Récepteur vidéo** : récepteur FPV branché en USB sur le PC
- **Simulateur** : [Uncrashed: FPV Drone Simulator](https://store.steampowered.com/app/1682970/Uncrashed__FPV_Drone_Simulator/)
- **PC** avec un GPU : toute l'inférence tourne en local

## Architecture du dépôt

> L'arborescence est encore en construction et va évoluer.

```
Epicopilot/
├── ia/                 # Toute la partie intelligence artificielle
│   ├── perception/     # Vision : détection et suivi de personne, profondeur, obstacles
│   ├── control/        # Contrôle : stabilisation Acro, boucle temps réel, sécurité
│   └── training/       # Dataset, scripts d'entraînement et de fine-tuning
├── sdk/                # Kit de comportements de vol (suivi, orbite, survol…)
├── bridge/             # Pont matériel : radio (CRSF / manette virtuelle) et capture vidéo, simulateur et vrai drone
├── web/                # Station de contrôle web (HUD, ordres, débriefing)
├── test/               # Tests unitaires, fonctionnels et E2E
├── docs/               # Documentation utilisateur, technique et guide du SDK
│   └── adr/            # Journal des décisions d'architecture
└── EpiCopilot_Tracker_Final.pptx   # Énoncé du projet
```

## Feuille de route

Le projet suit quatre étapes, calées sur les suivis obligatoires de l'énoncé.

### Étape 1 — Setup *(novembre 2026, Suivi 1)*

- [x] Simuler la radiocommande : manette virtuelle qui envoie les 4 axes (Roll, Pitch, Yaw, Throttle) au simulateur
- [ ] Capturer le flux vidéo du simulateur
- [ ] Choisir le modèle de base et préparer son fine-tuning
- [ ] Mettre en place toute l'infrastructure IA pour pouvoir lancer l'entraînement : environnement, pipeline de données, scripts
- [ ] Mettre en place le dépôt, le tableau GitHub Projects et la CI

### Étape 2 — Entraînement *(décembre 2026, Suivi 2)*

- [ ] Créer le dataset : enregistrer l'écran pendant des vols manuels, en capturant à la fois la vidéo et les commandes des sticks, pour que l'IA apprenne à partir de ces vols
- [ ] Nettoyer et annoter le dataset
- [ ] Lancer l'entraînement sur le simulateur Uncrashed à partir des vols enregistrés
- [ ] Obtenir un premier vol stable en mode Acro dans le simulateur
- [ ] Mesurer la latence de la boucle de contrôle (objectif : moins de 30 ms, entre 30 et 60 Hz)

### Étape 3 — Test *(janvier 2027, Suivi 3)*

- [ ] Tester sur le vrai drone avec la vraie radiocommande
- [ ] Entraîner l'IA sur les figures à apprendre
- [ ] Développer la partie web : station de contrôle, HUD, panneau d'ordres
- [ ] Ajouter les fonctionnalités supplémentaires :
  - [ ] Tourner autour de la personne (orbite)
  - [ ] Plans classiques de drone (survol, travelling arrière…)
  - [ ] Détection et évitement d'obstacles

### Étape 4 — Validation *(février 2027, Suivi 4 puis soutenance)*

- [ ] Terminer l'entraînement sur toutes les fonctionnalités
- [ ] Tester le projet en conditions réelles
- [ ] Rendre le projet conforme au RGPD et à la réglementation
- [ ] Lancer les tests de charge et de performance (FPS, latence) et l'audit d'accessibilité
- [ ] Finaliser la documentation et préparer la soutenance

## Installation

> 🚧 À compléter dès que la stack sera fixée.

Cette section décrira pas à pas :

1. Les prérequis (Python, GPU/CUDA, Node, simulateur)
2. L'installation des dépendances
3. Le lancement de la station de contrôle
4. Le mode développeur, qui permet de tester sans drone physique
5. Le lancement des tests

## Sécurité

- Tous les essais réels se font avec un micro-drone caréné de moins de 50 g.
- Un **kill switch** sur la radiocommande coupe immédiatement le contrôle du PC et rend la main au pilote.
- Le logiciel interdit au drone d'approcher à **moins de 1,5 m** de la personne suivie.
- **Fail-safe** : les gaz sont coupés automatiquement si le flux vidéo est perdu, si la radio se déconnecte ou si le modèle IA se bloque.

## Données personnelles et RGPD

- Toute l'inférence se fait **en local**, sans service cloud.
- La station s'utilise **sans créer de compte**.
- Aucune image ni vidéo de personne n'est conservée en dehors des sessions de test auxquelles les personnes filmées ont explicitement consenti.
- Les datasets vidéo ne sont jamais versionnés dans le dépôt.

## Usage de l'IA

> Déclaration obligatoire (charte IA du groupe IONIS), à compléter au fil du projet.

| Outil / modèle | Utilisation |
|---|---|
| *Model maison de réseau de neurone* | *code, tests, documentation, poids de modèles…* |

## Équipe

| Membre |
|---|
| Jérémie Cousin |
| Valentin Minè |
