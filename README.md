# Astro Starter Kit: Minimal

```sh
npm create astro@latest -- --template minimal
```

> 🧑‍🚀 **Seasoned astronaut?** Delete this file. Have fun!

## 🚀 Project Structure

Inside of your Astro project, you'll see the following folders and files:

```text
/
├── public/
├── src/
│   └── pages/
│       └── index.astro
└── package.json
```

Astro looks for `.astro` or `.md` files in the `src/pages/` directory. Each page is exposed as a route based on its file name.

There's nothing special about `src/components/`, but that's where we like to put any Astro/React/Vue/Svelte/Preact components.

Any static assets, like images, can be placed in the `public/` directory.

## 🧞 Commands

All commands are run from the root of the project, from a terminal:

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `npm install`             | Installs dependencies                            |
| `npm run dev`             | Starts local dev server at `localhost:4321`      |
| `npm run build`           | Build your production site to `./dist/`          |
| `npm run preview`         | Preview your build locally, before deploying     |
| `npm run astro ...`       | Run CLI commands like `astro add`, `astro check` |
| `npm run astro -- --help` | Get help using the Astro CLI                     |

## Vision

> **Vision:** Eine adaptive Web-Architektur, die die Grenzen zwischen statischem Web-Interface (2D) und interaktiver Traumwelt (3D) auflöst.

---

## 1. Das Konzept

Die Website ist kein Fenster, durch das man schaut, sondern ein Raum, den man betritt. Der Benutzer navigiert nicht durch "Seiten", sondern reist durch Zustände einer Welt.

Die Immersion entsteht durch die **nahtlose Brücke**: Ein Klick, ein Scrollen oder ein Ton beeinflusst gleichzeitig die HTML-Oberfläche (DOM) und die physikalische 3D-Engine.

## 2. Technische Architektur

Wir nutzen einen hybriden Stack für maximale Performance, Typsicherheit und Kontrolle:

* **UI-Layer (Astro JS):** Verwaltet das DOM, Layouts, SVGs und die *View Transitions*.
* **Engine-Layer (Rust / WGPU):** Eine via WASM geladene High-Performance-Engine für Rendering, Physik und prozedurale Generierung.
* **Nervensystem (`sceneStore`):** Ein zentraler State-Store (Nano Stores), der die 2D- und 3D-Welt synchronisiert.

## 3. Die Performance-Leiter (Progressive Immersion)

Das Erlebnis skaliert automatisch mit der Hardware des Nutzers. Wir unterscheiden zwei Qualitätsstufen:

| Feature | **Tier 1: Standard** (WebGL 2 / Mobile) | **Tier 2: High-End** (WebGPU / Desktop) |
| :--- | :--- | :--- |
| **Rendering** | Standard PBR Shading, Statische Texturen | Compute Shader, Prozedurale Geometrie, Raytracing-Schatten |
| **Physik** | Einfache CPU-Physik (WASM) | GPU-gestützte Partikel & Simulationen |
| **Audio** | Spatial Audio Wiedergabe | Echtzeit-Synthese & Granular-Effekte |
| **Interaktion**| Reaktive UI & Licht | Physikalische Simulationen & Mesh-Verformung |

---

## 4. Die Immersions-Achsen (Core Features)

Wir definieren das Erlebnis entlang von vier Hauptachsen, die Audio, Zeit und Raum verbinden.

### Achse I: ZEIT (Der "Chrono-Synthesizer")

**Konzept:** Scrollen ist keine Bewegung im Raum, sondern eine Manipulation der Zeitachse.

* **Tier 1 (Scrubbing):** Der Scroll-Fortschritt ist direkt an die `currentTime` des Audios und der Animationen gekoppelt. Vorwärts-Scrollen spielt die Welt ab, Rückwärts-Scrollen spult sie zurück.
* **Tier 2 (Granular-Riss):** Beim Rückwärts-Scrollen wird das Audio nicht einfach gepitcht. Ein Granular-Synthesizer (in Rust) "zerreißt" das Audio in mikroskopische Fragmente. Visuell dekonstruiert sich das 3D-Modell zeitgleich in seine Polygone. Die Zeit fühlt sich "zerbrochen" an.

### Achse II: AUDIO-MODELL (Der "Tesserakt")

**Konzept:** Audio wackelt nicht an der ganzen Welt, sondern manifestiert sich in *einem* zentralen, abstrakten Artefakt.

* **Tier 1 (Skulptur):** Ein komplexes 3D-Objekt, dessen Oberfläche (Vertex Shader) durch Bass und Höhen verformt wird. Es pulsiert organisch im Zentrum des Bildschirms.
* **Tier 2 (4D-Hyperkugel):** Ein prozedural generiertes Objekt (Compute Shader), das sich physikalisch unmöglich verhält. Es ist die visuelle Seele des Sounds.

### Achse III: PHYSIK (Die "Atmende Welt")

**Konzept:** Seitenwechsel und Musik beeinflussen die physikalischen Gesetze der Welt.

* **Tier 1 (Audio-Schwerkraft):** Die Musik steuert die Gravitation. Bei intensiven Passagen oder Drops verliert die Welt an Schwerkraft – Objekte beginnen sanft zu schweben ("Floating State"). In ruhigen Passagen landen sie wieder.
* **Tier 2 (Transition-Druckwelle):** Ein Seitenwechsel (Astro View Transition) ist kein harter Schnitt, sondern ein physikalisches Ereignis.
    1.  Der Klick löst eine **Schockwelle** in der Physik-Engine aus.
    2.  Die schwebenden Objekte werden sanft nach außen gedrückt ("Ausatmen").
    3.  Der neue Inhalt erscheint, die Objekte strömen zurück ("Einatmen").

### Achse IV: RAUM (Unified Light & Crystal UI)

**Konzept:** Licht und Schatten existieren konsistent über beide Welten (2D/3D) hinweg.

* **Kristall-UI:** HTML-Elemente schweben als halbtransparente, unscharfe Scheiben (`backdrop-filter`) über der 3D-Welt. Sie wirken physisch präsent.
* **Unified Light:** Die Maus (User) ist die Lichtquelle. Sie beleuchtet das 3D-Modell (WGPU) und wirft gleichzeitig korrekte Schlagschatten der 2D-HTML-Elemente (berechnet via CSS-Variablen aus der Rust-Engine).

---

## 5. Roadmap

### Phase 1: Basis & Immersion (Aktuell)

- [x] WGPU Renderer Setup & Loop
- [x] Basic Interaktion (Maus-Licht)
- [ ] **Nächster Schritt:** Umbau auf Fullscreen-Hintergrund & transparente UI.

### Phase 2: Verbindung der Welten

- [ ] Implementierung "Unified Light" (Theme Toggle & Schattenwurf).
- [ ] Bau des "Audio-Modells" (Start mit einfacher Sphere).
- [ ] Anbindung der Audio-Daten (AnalyserNode) an den Rust-Shader.

### Phase 3: High-End & Physik

- [ ] Integration der Physik-Engine (`rapier`) in Rust.
- [ ] Implementierung der View-Transition-Logik (Druckwelle).
- [ ] Compute Shader für Tier 2 Effekte (Tesserakt).
