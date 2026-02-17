"""
notion_full_sync.py — Comprehensive Notion Project Synchronization
===================================================================

Populates the Notion workspace with the *complete* state of the
Hybrid Digital Twin project:

    1. Updates the **main project page** with architecture, metrics,
       technology stack, publication progress, and current status.
    2. Updates the **Roadmap DB** with actual milestone statuses
       derived from the Git history.
    3. Syncs the **Simulation DB** with NLTHA campaign results
       from ``data/raw/factory_summary.csv``.

Usage::

    python -m src.utils.notion_full_sync          # full sync
    python -m src.utils.notion_full_sync --dry-run # preview only

Author: Mikisbell
"""

from __future__ import annotations

import csv
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
ROADMAP_DB_ID = "a14c5ebc-6b4d-47a7-ab47-edc371618f83"
SIMULATION_DB_ID = "1313f3f3-1b89-4fe7-a286-64919bb20560"
MAIN_PAGE_ID = "30351e60-d3c2-800e-9dba-e8dcc2643cec"
REPO_URL = "https://github.com/Mikisbell/Hybrid-Digital-Twin-Seismic-RC"

# ═══════════════════════════════════════════════════════════════════════════
# Notion block helpers
# ═══════════════════════════════════════════════════════════════════════════


def _rich(text: str, bold: bool = False, code: bool = False, color: str = "default") -> dict:
    """Build a single Notion rich_text element."""
    return {
        "type": "text",
        "text": {"content": text},
        "annotations": {
            "bold": bold,
            "italic": False,
            "strikethrough": False,
            "underline": False,
            "code": code,
            "color": color,
        },
    }


def heading1(text: str) -> dict:
    return {"object": "block", "type": "heading_1", "heading_1": {"rich_text": [_rich(text)]}}


def heading2(text: str) -> dict:
    return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [_rich(text)]}}


def heading3(text: str) -> dict:
    return {"object": "block", "type": "heading_3", "heading_3": {"rich_text": [_rich(text)]}}


def paragraph(*parts: tuple[str, bool, bool]) -> dict:
    """paragraph(("hello", bold, code), ("world", False, False))"""
    rt = []
    for p in parts:
        if isinstance(p, str):
            rt.append(_rich(p))
        else:
            rt.append(
                _rich(p[0], bold=p[1] if len(p) > 1 else False, code=p[2] if len(p) > 2 else False)
            )
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rt}}


def bullet(*parts: tuple[str, bool, bool]) -> dict:
    rt = []
    for p in parts:
        if isinstance(p, str):
            rt.append(_rich(p))
        else:
            rt.append(
                _rich(p[0], bold=p[1] if len(p) > 1 else False, code=p[2] if len(p) > 2 else False)
            )
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": rt},
    }


def numbered(*parts: tuple[str, bool, bool]) -> dict:
    rt = []
    for p in parts:
        if isinstance(p, str):
            rt.append(_rich(p))
        else:
            rt.append(
                _rich(p[0], bold=p[1] if len(p) > 1 else False, code=p[2] if len(p) > 2 else False)
            )
    return {
        "object": "block",
        "type": "numbered_list_item",
        "numbered_list_item": {"rich_text": rt},
    }


def todo(text: str, checked: bool = False) -> dict:
    return {
        "object": "block",
        "type": "to_do",
        "to_do": {"rich_text": [_rich(text)], "checked": checked},
    }


def callout(text: str, emoji: str = "💡") -> dict:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "rich_text": [_rich(text)],
            "icon": {"type": "emoji", "emoji": emoji},
        },
    }


def divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def code_block(code: str, language: str = "python") -> dict:
    return {
        "object": "block",
        "type": "code",
        "code": {"rich_text": [_rich(code)], "language": language},
    }


def quote_block(text: str) -> dict:
    return {"object": "block", "type": "quote", "quote": {"rich_text": [_rich(text)]}}


def toggle(title: str, children: list[dict]) -> dict:
    return {
        "object": "block",
        "type": "toggle",
        "toggle": {"rich_text": [_rich(title)], "children": children},
    }


def table_of_contents() -> dict:
    return {
        "object": "block",
        "type": "table_of_contents",
        "table_of_contents": {"color": "default"},
    }


def bookmark(url: str, caption: str = "") -> dict:
    block: dict = {"object": "block", "type": "bookmark", "bookmark": {"url": url}}
    if caption:
        block["bookmark"]["caption"] = [_rich(caption)]
    return block


# ═══════════════════════════════════════════════════════════════════════════
# Git history helper
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class GitCommit:
    sha: str
    date: str
    message: str


def get_git_history() -> list[GitCommit]:
    """Return list of commits oldest-first."""
    result = subprocess.run(
        ["git", "log", "--format=%H|%aI|%s", "--reverse"],
        capture_output=True,
        text=True,
        check=True,
    )
    commits = []
    for line in result.stdout.strip().split("\n"):
        if "|" in line:
            parts = line.split("|", 2)
            commits.append(GitCommit(sha=parts[0][:7], date=parts[1][:10], message=parts[2]))
    return commits


# ═══════════════════════════════════════════════════════════════════════════
# Campaign results loader
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class NLTHAResult:
    name: str
    source: str
    scale_factor: float
    duration: float
    pga: float
    converged: bool
    wall_clock: float
    n_steps: int
    idr: list[float]
    max_idr: float
    peak_base_shear: float


def load_campaign_results(summary_path: str = "data/raw/factory_summary.csv") -> list[NLTHAResult]:
    path = Path(summary_path)
    if not path.exists():
        return []
    results: list[NLTHAResult] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(
                NLTHAResult(
                    name=row["record_name"],
                    source=row["source"],
                    scale_factor=float(row["scale_factor"]),
                    duration=float(row["duration_s"]),
                    pga=float(row["pga_g"]),
                    converged=row["converged"] == "True",
                    wall_clock=float(row["wall_clock_s"]),
                    n_steps=int(row["n_steps"]),
                    idr=[
                        float(row["max_idr_1"]),
                        float(row["max_idr_2"]),
                        float(row["max_idr_3"]),
                        float(row["max_idr_4"]),
                        float(row["max_idr_5"]),
                    ],
                    max_idr=float(row["max_idr_overall"]),
                    peak_base_shear=float(row["peak_base_shear_kN"]),
                )
            )
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Section builders — main project page
# ═══════════════════════════════════════════════════════════════════════════


def _build_header_section() -> list[dict]:
    """Title, description, and ToC."""
    return [
        heading1("🏗️ Hybrid Digital Twin for Seismic RC Buildings"),
        paragraph(
            "Framework de Gemelo Digital Híbrido que combina simulación de "
            "alta fidelidad (OpenSeesPy) con redes neuronales informadas por "
            "física (PINN) para predicción en tiempo real de daño sísmico en "
            "edificios de concreto reforzado."
        ),
        callout(
            "Última sincronización: "
            + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            + "  •  Generado automáticamente por notion_full_sync.py",
            "🔄",
        ),
        divider(),
        table_of_contents(),
        divider(),
    ]


def _build_project_info() -> list[dict]:
    return [
        heading2("📋 Información del Proyecto"),
        bullet(("Investigador: ", True), "Mikisbell"),
        bullet(("Inicio: ", True), "16 de febrero de 2026"),
        bullet(("Journal objetivo: ", True), "HRPUB — Horizon Research Publishing"),
        bullet(("Repositorio: ", True), REPO_URL),
        bullet(("Idioma del código/publicación: ", True), "Inglés"),
        bullet(("Idioma de Notion/documentación: ", True), "Español"),
        bullet(("Licencia: ", True), "MIT"),
        divider(),
    ]


def _build_architecture() -> list[dict]:
    """3-layer framework architecture."""
    return [
        heading2("🧠 Arquitectura del Framework — 3 Capas"),
        paragraph(
            "El framework integra tres capas especializadas que operan "
            "en secuencia durante el entrenamiento y en paralelo durante "
            "la inferencia en tiempo real:"
        ),
        numbered(
            ("Capa de Simulación (OpenSeesPy): ", True),
            "Modelo no-lineal de pórtico RC de 5 pisos, 3 vanos. "
            "forceBeamColumn con Concrete02 (Mander), Steel02 (Menegotto-Pinto), "
            "amortiguamiento Rayleigh 5%. Ejecuta NLTHA batch para generar la "
            '"Verdad Sintética".',
        ),
        numbered(
            ("Capa de Inteligencia (Hybrid-PINN): ", True),
            "Red neuronal de 603,653 parámetros con 6 capas ocultas × 128 "
            "neuronas, activación Swish, normalización por capa. Función de "
            "pérdida multi-objetivo: L = λ_data·L_data + λ_phys·L_phys + "
            "λ_bc·L_bc + λ_reg·L_reg con pesos auto-adaptativos.",
        ),
        numbered(
            ("Capa de Documentación (Notion + GitHub): ", True),
            "Sincronización automática bidireccional: GitHub Actions → Notion "
            "Roadmap (cada push), Data Factory → Simulation DB (cada NLTHA), "
            "trazabilidad completa HRPUB.",
        ),
        divider(),
    ]


def _build_tech_stack() -> list[dict]:
    """Technology stack with versions."""
    return [
        heading2("🔧 Stack Tecnológico"),
        heading3("Simulación Estructural"),
        bullet(("OpenSeesPy 3.7.1", True, False), " — Motor de análisis no-lineal"),
        bullet(("NumPy 2.2.6", True, False), " — Álgebra matricial y procesamiento de señales"),
        bullet(("SciPy", True, False), " — Integración numérica, espectros de respuesta"),
        heading3("Machine Learning"),
        bullet(("PyTorch (CPU)", True, False), " — Framework de deep learning para PINN"),
        bullet(("Arquitectura:", True, False), " 6×128 MLP + Swish + LayerNorm = 603,653 params"),
        bullet(("Optimizador:", True, False), " AdamW (lr=1e-3) + Cosine Annealing Warm Restarts"),
        heading3("Datos y Preprocesamiento"),
        bullet(("PEER NGA-West2", True, False), " — Base de datos de registros sísmicos reales"),
        bullet(("Nigam-Jennings (1969)", True, False), " — Espectro de respuesta piecewise-exact"),
        bullet(("ASCE 7-22 §16.2", True, False), " — Spectral matching de suite sísmica"),
        heading3("Infraestructura"),
        bullet(("Python 3.10.12", True, False), " — Runtime en Ubuntu WSL2"),
        bullet(("Pre-commit", True, False), " — 8 hooks: ruff, ruff-format, isort, etc."),
        bullet(("GitHub Actions", True, False), " — CI/CD: notion_sync + pinn_notify"),
        bullet(("notion-client 3.0.0", True, False), " — API de Notion para sync automático"),
        divider(),
    ]


def _build_model_specs() -> list[dict]:
    """Structural and PINN model specifications."""
    return [
        heading2("📐 Especificaciones del Modelo"),
        heading3("Modelo Estructural RC — 5 Pisos, 3 Vanos"),
        bullet(("Nodos:", True, False), " 24 (4 base fijos + 20 libres)"),
        bullet(("Elementos:", True, False), " 35 (20 columnas + 15 vigas) — forceBeamColumn"),
        bullet(("Columnas:", True, False), " 500×500 mm, As=5,890 mm², ρ=2.36%"),
        bullet(("Vigas:", True, False), " 350×600 mm, As=2,199 mm², ρ=1.05%"),
        bullet(("Concreto:", True, False), " f'c=28 MPa, f'cc=35.4 MPa (Mander, ratio=1.27)"),
        bullet(("Acero:", True, False), " fy=420 MPa, Steel02 (Menegotto-Pinto)"),
        bullet(("Amortiguamiento:", True, False), " Rayleigh ξ=5.0%, modos 1 y 3"),
        bullet(("Período fundamental:", True, False), " T₁ = 1.186 s"),
        bullet(("Períodos modales:", True, False), " [1.186, 0.368, 0.201, 0.132, 0.100] s"),
        heading3("Hybrid-PINN — 603,653 Parámetros"),
        bullet(("Input:", True, False), " 7 features (time, ag, 5×floor coords)"),
        bullet(("Hidden:", True, False), " 6 capas × 128 neuronas"),
        bullet(("Activación:", True, False), " Swish (β entrenables)"),
        bullet(("Normalización:", True, False), " LayerNorm en cada capa oculta"),
        bullet(("Output:", True, False), " 5 desplazamientos (un valor por piso)"),
        bullet(("Loss:", True, False), " L = λ_d·L_data + λ_p·L_phys + λ_bc·L_bc + λ_reg·L_reg"),
        bullet(("Inicialización:", True, False), " Kaiming fan_in (He et al., 2015)"),
        divider(),
    ]


def _build_design_spectrum() -> list[dict]:
    """ASCE 7-22 design spectrum parameters."""
    return [
        heading2("🌊 Espectro de Diseño y Selección Sísmica"),
        heading3("Parámetros del Espectro ASCE 7-22"),
        bullet(("SDS = 1.0 g", True, False), " (aceleración espectral en período corto)"),
        bullet(("SD1 = 0.6 g", True, False), " (aceleración espectral a 1 segundo)"),
        bullet(("TL = 8.0 s", True, False), " (período de transición largo)"),
        bullet(("Rango de matching:", True, False), " [0.2T₁, 2.0T₁] = [0.237, 2.372] s"),
        heading3("Criterios de Selección — PEER NGA-West2"),
        bullet(("Magnitud:", True, False), " 6.0 ≤ Mw ≤ 7.5"),
        bullet(("Distancia:", True, False), " 10 ≤ Rjb ≤ 50 km"),
        bullet(("Clase de sitio:", True, False), " C/D (180 ≤ Vs30 ≤ 760 m/s)"),
        bullet(("Records mínimos:", True, False), " 200+"),
        bullet(("Scale factor máximo:", True, False), " 5.0"),
        heading3("Spectral Matching"),
        bullet(("Método:", True, False), " Nigam-Jennings (1969) piecewise-exact recurrence"),
        bullet(("Validación:", True, False), " ASCE 7-22 §16.2 — media del suite ≥ 90% del target"),
        bullet(("Convergencia:", True, False), " Iterative boost (max 10 iteraciones)"),
        divider(),
    ]


def _build_campaign_results(results: list[NLTHAResult]) -> list[dict]:
    """NLTHA campaign summary statistics."""
    if not results:
        return [
            heading2("🔬 Campaña NLTHA"),
            callout(
                "No se han encontrado resultados en data/raw/factory_summary.csv. "
                "Ejecuta: python -m src.preprocessing.data_factory --synthetic 20",
                "⚠️",
            ),
            divider(),
        ]

    n_total = len(results)
    n_converged = sum(1 for r in results if r.converged)
    pga_min = min(r.pga for r in results)
    pga_max = max(r.pga for r in results)
    pga_mean = sum(r.pga for r in results) / n_total
    idr_min = min(r.max_idr for r in results)
    idr_max = max(r.max_idr for r in results)
    idr_mean = sum(r.max_idr for r in results) / n_total
    total_time = sum(r.wall_clock for r in results)
    total_steps = sum(r.n_steps for r in results)

    # Find worst case
    worst = max(results, key=lambda r: r.max_idr)
    # Find the story with max drift across all
    story_max_drifts = [0.0] * 5
    for r in results:
        for i in range(5):
            if r.idr[i] > story_max_drifts[i]:
                story_max_drifts[i] = r.idr[i]

    critical_story = story_max_drifts.index(max(story_max_drifts)) + 1

    blocks: list[dict] = [
        heading2("🔬 Campaña NLTHA — Resultados"),
        callout(
            f"✅ {n_converged}/{n_total} registros convergieron ({100 * n_converged / n_total:.0f}%) "
            f"en {total_time:.0f} s totales ({total_steps:,} pasos de análisis)",
            "🎯",
        ),
        heading3("Estadísticas Globales"),
        bullet(("PGA:", True), f" min={pga_min:.3f}g, max={pga_max:.3f}g, media={pga_mean:.3f}g"),
        bullet(("Max IDR:", True), f" min={idr_min:.4f}, max={idr_max:.4f}, media={idr_mean:.4f}"),
        bullet(
            ("Piso crítico:", True),
            f" Story {critical_story} (max drift={max(story_max_drifts):.4f})",
        ),
        bullet(
            ("Caso extremo:", True),
            f" {worst.name} — IDR={worst.max_idr:.4f}, PGA={worst.pga:.3f}g",
        ),
        heading3("Distribución de IDR por Piso (máximos en toda la suite)"),
    ]

    for i in range(5):
        bar_length = int(story_max_drifts[i] / max(story_max_drifts) * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        blocks.append(bullet((f"Piso {i + 1}: ", True), f"{bar} {story_max_drifts[i]:.5f}"))

    # Top 5 most severe
    blocks.append(heading3("Top 5 Registros Más Severos"))
    sorted_results = sorted(results, key=lambda r: r.max_idr, reverse=True)
    for _i, r in enumerate(sorted_results[:5]):
        blocks.append(
            numbered(
                (f"{r.name}", True),
                f" — IDR={r.max_idr:.4f}, PGA={r.pga:.3f}g, "
                f"VBase={r.peak_base_shear:.0f} kN, {r.wall_clock:.1f}s",
            )
        )

    blocks.append(divider())
    return blocks


def _build_publication_progress() -> list[dict]:
    """Manuscript writing progress tracker."""
    return [
        heading2("📝 Progreso del Manuscrito HRPUB"),
        callout(
            "Journal: HRPUB — Horizon Research Publishing\n"
            "Formato: Numérico correlativo [1], [2], ...\n"
            "Idioma: Inglés (publicación) / Español (Notion)",
            "📰",
        ),
        todo("§1 Introduction — Contexto, brecha de conocimiento, objetivos", False),
        todo("§2 Literature Review — Estado del arte PINN + Ingeniería Sísmica", False),
        todo("§3.1 Framework Architecture — Tres capas", False),
        todo("§3.2 Ground Motion Selection — PEER NGA-West2, ASCE 7-22", False),
        todo("§3.3 Structural Model — OpenSeesPy RC frame 5 pisos", False),
        todo("§3.4 PINN Formulation — Arquitectura, pérdida, entrenamiento ✅ LISTO", True),
        todo("§3.5 Training Protocol — Modos de entrenamiento", False),
        todo("§4 Results — Figuras de entrenamiento, comparativas, espectros", False),
        todo("§5 Discussion — Contribuciones, limitaciones, trabajo futuro", False),
        todo("§6 Conclusions — Hallazgos clave", False),
        todo("References — 15 entradas definidas [1]-[15]", True),
        heading3("Referencias Clave Implementadas"),
        bullet(("[3] Raissi et al. (2019)", False), " — Physics-Informed Neural Networks"),
        bullet(("[9] Ramachandran et al. (2017)", False), " — Swish activation"),
        bullet(("[11] Loshchilov & Hutter (2019)", False), " — AdamW decoupled weight decay"),
        bullet(("[14] He et al. (2015)", False), " — Kaiming initialization"),
        bullet(("[15] McClenny & Braga-Neto (2023)", False), " — Self-Adaptive PINNs"),
        divider(),
    ]


def _build_file_structure() -> list[dict]:
    """Codebase map."""
    return [
        heading2("📁 Estructura del Repositorio"),
        code_block(
            "Hybrid-Digital-Twin-Seismic-RC/\n"
            "├── src/\n"
            "│   ├── opensees_analysis/\n"
            "│   │   ├── ospy_model.py        # RC frame 5-story model (935 lines)\n"
            "│   │   └── nltha_runner.py       # NLTHA batch runner (659 lines)\n"
            "│   ├── pinn/\n"
            "│   │   ├── model.py              # HybridPINN architecture (352 lines)\n"
            "│   │   ├── loss.py               # Multi-objective loss (407 lines)\n"
            "│   │   ├── trainer.py            # Training engine (657 lines)\n"
            "│   │   └── benchmark_latency.py  # Real-time benchmark (373 lines)\n"
            "│   ├── preprocessing/\n"
            "│   │   ├── data_factory.py       # PEER + NLTHA orchestrator (1314 lines)\n"
            "│   │   └── pipeline.py           # ML data pipeline\n"
            "│   └── utils/\n"
            "│       ├── sync_results.py       # NotionResearchLogger (273 lines)\n"
            "│       ├── notion_full_sync.py   # Full project sync (this file)\n"
            "│       └── figure_manager.py     # HRPUB 300-DPI figures\n"
            "├── manuscript/\n"
            "│   ├── 01_introduction.md\n"
            "│   ├── 02_literature_review.md\n"
            "│   ├── 03_methods.md             # §3.4 PINN fully expanded\n"
            "│   ├── 04_results.md\n"
            "│   ├── 05_discussion.md\n"
            "│   ├── 06_conclusions.md\n"
            "│   └── references.bib            # 15 entries [1]-[15]\n"
            "├── data/\n"
            "│   ├── external/peer_nga/         # PEER NGA-West2 AT2 files\n"
            "│   ├── raw/                       # NLTHA outputs (CSVs + JSONs)\n"
            "│   ├── processed/                 # ML-ready tensors\n"
            "│   └── models/                    # Trained PINN checkpoints\n"
            "├── notebooks/\n"
            "│   └── 01_model_verification.ipynb\n"
            "├── .github/workflows/\n"
            "│   ├── notion_sync.yml            # Push → Roadmap DB\n"
            "│   └── pinn_notify.yml            # PINN → Simulation DB\n"
            "└── requirements.txt               # 25+ dependencies\n",
            "plain text",
        ),
        divider(),
    ]


def _build_commit_timeline(commits: list[GitCommit]) -> list[dict]:
    """Git commit timeline."""
    blocks: list[dict] = [
        heading2("🕐 Línea de Tiempo del Proyecto"),
    ]

    # Group by phase
    phases = {
        "Infraestructura": [],
        "Simulación": [],
        "Inteligencia": [],
        "Datos": [],
        "Documentación": [],
    }

    for c in commits:
        msg = c.message.lower()
        if any(
            k in msg
            for k in (
                "initial",
                "structure",
                "requirements",
                "tooling",
                "pre-commit",
                "hardening",
                "notion sync workflow",
                "merge",
            )
        ):
            phases["Infraestructura"].append(c)
        elif any(k in msg for k in ("opensees", "model", "nltha", "verification")):
            phases["Simulación"].append(c)
        elif any(k in msg for k in ("pinn", "hybrid-pinn")):
            phases["Inteligencia"].append(c)
        elif any(k in msg for k in ("data factory", "spectral", "asce", "peer")):
            phases["Datos"].append(c)
        elif any(k in msg for k in ("docs", "manuscript", "§")):
            phases["Documentación"].append(c)
        else:
            phases["Infraestructura"].append(c)

    emoji_map = {
        "Infraestructura": "🔩",
        "Simulación": "🏗️",
        "Inteligencia": "🧠",
        "Datos": "🏭",
        "Documentación": "📝",
    }

    for phase_name, phase_commits in phases.items():
        if not phase_commits:
            continue
        emoji = emoji_map.get(phase_name, "📌")
        children = []
        for c in phase_commits:
            children.append(
                bullet(
                    (f"[{c.sha}] ", False, True),
                    (f"{c.date} — ", True),
                    c.message[:100],
                )
            )
        blocks.append(toggle(f"{emoji} {phase_name} ({len(phase_commits)} commits)", children))

    blocks.append(divider())
    return blocks


def _build_next_steps() -> list[dict]:
    """What's pending."""
    return [
        heading2("🚀 Próximos Pasos"),
        callout(
            "El pipeline de simulación está validado (20/20 synthetic converged). "
            "La próxima fase es integrar datos reales PEER NGA-West2.",
            "🎯",
        ),
        todo("Descargar 200+ registros AT2 de PEER NGA-West2", False),
        todo("Ejecutar campaña completa: python -m src.preprocessing.data_factory", False),
        todo("Validar spectral matching ASCE 7-22 §16.2 con datos reales", False),
        todo("Ejecutar pipeline ML: python -m src.preprocessing.pipeline", False),
        todo("Entrenar Hybrid-PINN (3 modos: data-only → physics-only → hybrid)", False),
        todo("Benchmark latencia real-time (< 10ms target)", False),
        todo("Generar figuras de entrenamiento (300 DPI)", False),
        todo("Completar §4 Results del manuscrito", False),
        todo("Completar §5 Discussion y §6 Conclusions", False),
        todo("Enviar manuscrito a HRPUB", False),
        divider(),
    ]


def _build_equations() -> list[dict]:
    """Governing equations section."""
    return [
        heading2("📐 Ecuaciones Gobernantes"),
        paragraph("El framework PINN minimiza una función de pérdida multi-objetivo:"),
        quote_block("L_total = λ_data · L_data + λ_phys · L_phys + λ_bc · L_bc + λ_reg · L_reg"),
        paragraph(
            "donde λ_i son pesos auto-adaptativos (McClenny & Braga-Neto, 2023) "
            "y cada término penaliza:"
        ),
        bullet(("L_data:", True), " MSE entre predicción PINN y respuesta OpenSeesPy"),
        bullet(
            ("L_phys:", True), " Residuo de la ecuación de movimiento Mü + Cu̇ + f_int(u) = −M·1·ag"
        ),
        bullet(("L_bc:", True), " Condiciones iniciales u(0)=0, u̇(0)=0"),
        bullet(("L_reg:", True), " Regularización L2 de pesos (weight decay = 1e-4)"),
        divider(),
    ]


def _build_metrics() -> list[dict]:
    """Damage metrics."""
    return [
        heading2("📊 Métricas de Daño"),
        bullet(("IDR", True), " — Inter-story Drift Ratio (razón de deriva de entrepiso)"),
        bullet(("PGA", True), " — Peak Ground Acceleration (aceleración pico del suelo, g)"),
        bullet(
            ("Park-Ang", True), " — Índice de daño combinado (deformación + energía histerética)"
        ),
        bullet(("Base Shear", True), " — Cortante basal máximo (kN)"),
        divider(),
    ]


def _build_footer() -> list[dict]:
    return [
        callout(
            "🔄 Este proyecto utiliza automatización completa: cada push a "
            "GitHub actualiza la Hoja de Ruta, cada NLTHA registra resultados "
            "en el Log de Simulaciones, y este dashboard se genera con "
            "notion_full_sync.py.\n\n"
            f"Repositorio: {REPO_URL}\n"
            "Última sincronización: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "🤖",
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Main sync orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class NotionProjectSync:
    """Full project synchronization to Notion workspace."""

    def __init__(self, dry_run: bool = False):
        from notion_client import Client

        self.client = Client(auth=NOTION_TOKEN)
        self.dry_run = dry_run
        self.commits = get_git_history()
        self.campaign = load_campaign_results()
        logger.info(
            "NotionProjectSync initialized: %d commits, %d NLTHA results, dry_run=%s",
            len(self.commits),
            len(self.campaign),
            dry_run,
        )

    # -------------------------------------------------------------------
    # 1. Update main project page (replace all blocks)
    # -------------------------------------------------------------------
    def sync_main_page(self) -> None:
        """Rebuild the main project page with full content."""
        logger.info("Syncing main project page: %s", MAIN_PAGE_ID)

        # Build all sections
        blocks: list[dict] = []
        blocks.extend(_build_header_section())
        blocks.extend(_build_project_info())
        blocks.extend(_build_architecture())
        blocks.extend(_build_tech_stack())
        blocks.extend(_build_model_specs())
        blocks.extend(_build_design_spectrum())
        blocks.extend(_build_equations())
        blocks.extend(_build_metrics())
        blocks.extend(_build_campaign_results(self.campaign))
        blocks.extend(_build_publication_progress())
        blocks.extend(_build_file_structure())
        blocks.extend(_build_commit_timeline(self.commits))
        blocks.extend(_build_next_steps())
        blocks.extend(_build_footer())

        if self.dry_run:
            logger.info("[DRY RUN] Would write %d blocks to main page.", len(blocks))
            for i, b in enumerate(blocks):
                btype = b.get("type", "?")
                logger.info("  Block %02d: %s", i, btype)
            return

        # Delete existing blocks (except the two child_databases)
        existing = self.client.blocks.children.list(MAIN_PAGE_ID)
        db_block_ids = []
        for b in existing["results"]:
            if b["type"] == "child_database":
                db_block_ids.append(b["id"])
            else:
                try:
                    self.client.blocks.delete(b["id"])
                except Exception as exc:
                    logger.warning("Could not delete block %s: %s", b["id"], exc)

        # Append new blocks in batches of 100 (Notion limit)
        for i in range(0, len(blocks), 100):
            batch = blocks[i : i + 100]
            self.client.blocks.children.append(MAIN_PAGE_ID, children=batch)
            logger.info("  Appended blocks %d–%d", i, i + len(batch))

        logger.info("✅ Main page updated with %d blocks.", len(blocks))

    # -------------------------------------------------------------------
    # 2. Update Roadmap DB — set statuses based on real progress
    # -------------------------------------------------------------------
    def sync_roadmap(self) -> None:
        """Update Roadmap DB entries with correct completion status."""
        logger.info("Syncing Roadmap DB: %s", ROADMAP_DB_ID)

        if self.dry_run:
            logger.info("[DRY RUN] Would update Roadmap DB milestones.")
            return

        # Fetch all existing pages
        existing_pages = self._get_all_roadmap_pages()
        logger.info("  Found %d existing Roadmap entries.", len(existing_pages))

        # Define milestone completion mapping
        completed_keywords = [
            "investigación inicial",
            "instalación",
            "validación del entorno",
            "creación del repositorio",
            "definición de requirements",
            "diseño de la arquitectura",
            "formulación matemática",
            "github action",
            "script sync_results",
            "script figure_manager",
            "configuración de bases de datos",
            "definición del modelo base",
            "descarga y procesamiento",
            # From git commits
            "notion sync workflow",
            "peer-review hardening",
            "openseespy rc frame",
            "model verification",
            "hybrid-pinn engine",
            "data factory",
            "§3.4 pinn subsection",
            "spectral matching",
            "nigam-jennings",
            "diseño de arquitectura pinn",
        ]

        in_progress_keywords = [
            "generación automatizada de datos",
            "entrenamiento del modelo pinn",
        ]

        pending_keywords = [
            "visualización del gemelo digital",
            "redacción del manuscrito",
            "envío del manuscrito",
        ]

        for page_id, title in existing_pages:
            title_lower = title.lower()
            new_status = None

            if any(kw in title_lower for kw in completed_keywords):
                new_status = "Completada"
            elif any(kw in title_lower for kw in in_progress_keywords):
                new_status = "En Progreso"
            elif any(kw in title_lower for kw in pending_keywords):
                new_status = "Pendiente"

            if new_status:
                try:
                    self.client.pages.update(
                        page_id=page_id,
                        properties={"Estado": {"select": {"name": new_status}}},
                    )
                    logger.info("  Updated: %s → %s", title[:50], new_status)
                except Exception as exc:
                    logger.warning("  Failed to update %s: %s", title[:50], exc)

        # Add missing commit milestones
        existing_titles = {t.lower() for _, t in existing_pages}
        for commit in self.commits:
            commit_short = commit.message[:100]
            if not any(commit_short.lower()[:30] in t for t in existing_titles):
                # Categorize
                msg_lower = commit.message.lower()
                if any(k in msg_lower for k in ("opensees", "nltha", "model")):
                    cat = "Capa de Simulación"
                elif any(k in msg_lower for k in ("pinn", "hybrid")):
                    cat = "Capa de Inteligencia"
                elif any(k in msg_lower for k in ("data", "factory", "spectral", "asce")):
                    cat = "Pipeline de Datos"
                elif any(k in msg_lower for k in ("docs", "manuscript", "§")):
                    cat = "Documentación"
                else:
                    cat = "Infraestructura"

                try:
                    self.client.pages.create(
                        parent={"database_id": ROADMAP_DB_ID},
                        properties={
                            "Tarea": {"title": [{"text": {"content": commit_short}}]},
                            "Estado": {"select": {"name": "Completada"}},
                            "Categoría": {"select": {"name": cat}},
                            "Fase": {"select": {"name": "Métodos"}},
                            "Fecha": {"date": {"start": commit.date}},
                            "Commit": {"url": f"{REPO_URL}/commit/{commit.sha}"},
                        },
                    )
                    logger.info("  Added milestone: %s", commit_short[:50])
                except Exception as exc:
                    logger.warning("  Failed to add %s: %s", commit_short[:50], exc)

        logger.info("✅ Roadmap DB synchronized.")

    # -------------------------------------------------------------------
    # 3. Sync Simulation DB with campaign results
    # -------------------------------------------------------------------
    def sync_simulations(self) -> None:
        """Populate Simulation DB with NLTHA campaign results."""
        logger.info(
            "Syncing Simulation DB: %s (campaign: %d records)", SIMULATION_DB_ID, len(self.campaign)
        )

        if not self.campaign:
            logger.info("  No campaign results to sync.")
            return

        if self.dry_run:
            logger.info("[DRY RUN] Would sync %d NLTHA results.", len(self.campaign))
            return

        # Check existing entries to avoid duplicates
        existing_names = self._get_existing_simulation_names()
        logger.info("  Found %d existing simulation entries.", len(existing_names))

        n_added = 0
        for r in self.campaign:
            if r.name in existing_names:
                logger.debug("  Skipping duplicate: %s", r.name)
                continue

            idr_str = ", ".join(f"{d:.5f}" for d in r.idr)
            notes = (
                f"SF={r.scale_factor:.3f} | Duration={r.duration:.1f}s | "
                f"Steps={r.n_steps} | Time={r.wall_clock:.1f}s | "
                f"VBase={r.peak_base_shear:.0f}kN | "
                f"IDR=[{idr_str}]"
            )

            properties: dict[str, Any] = {
                "Movimiento Sísmico": {"title": [{"text": {"content": r.name}}]},
                "Deriva Máxima": {"number": round(r.max_idr, 6)},
                "PGA (g)": {"number": round(r.pga, 4)},
                "Pisos": {"number": 5},
                "Estado": {"select": {"name": "Convergió" if r.converged else "Divergió"}},
                "Fase": {"select": {"name": "Métodos"}},
                "Fecha": {"date": {"start": datetime.now(timezone.utc).strftime("%Y-%m-%d")}},
                "Notas": {"rich_text": [{"text": {"content": notes[:2000]}}]},
                "Referencia": {"rich_text": [{"text": {"content": r.source}}]},
            }

            try:
                self.client.pages.create(
                    parent={"database_id": SIMULATION_DB_ID},
                    properties=properties,
                )
                n_added += 1
            except Exception as exc:
                logger.warning("  Failed to log %s: %s", r.name, exc)

        logger.info("✅ Simulation DB: %d/%d new records added.", n_added, len(self.campaign))

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def _get_all_roadmap_pages(self) -> list[tuple[str, str]]:
        """Return (page_id, title) for all Roadmap DB entries."""
        pages: list[tuple[str, str]] = []
        search_results = self.client.search(
            query="", filter={"value": "page", "property": "object"}
        )
        for p in search_results.get("results", []):
            parent = p.get("parent", {})
            db_id = parent.get("database_id", "")
            if ROADMAP_DB_ID not in db_id:
                continue
            props = p.get("properties", {})
            title = ""
            for _k, v in props.items():
                if v.get("type") == "title":
                    title = "".join(t.get("plain_text", "") for t in v.get("title", []))
                    break
            pages.append((p["id"], title))

        # Handle pagination
        while search_results.get("has_more"):
            search_results = self.client.search(
                query="",
                filter={"value": "page", "property": "object"},
                start_cursor=search_results["next_cursor"],
            )
            for p in search_results.get("results", []):
                parent = p.get("parent", {})
                db_id = parent.get("database_id", "")
                if ROADMAP_DB_ID not in db_id:
                    continue
                props = p.get("properties", {})
                title = ""
                for _k, v in props.items():
                    if v.get("type") == "title":
                        title = "".join(t.get("plain_text", "") for t in v.get("title", []))
                        break
                pages.append((p["id"], title))

        return pages

    def _get_existing_simulation_names(self) -> set[str]:
        """Return set of Ground Motion names already in Simulation DB."""
        names: set[str] = set()
        search_results = self.client.search(
            query="", filter={"value": "page", "property": "object"}
        )
        for p in search_results.get("results", []):
            parent = p.get("parent", {})
            db_id = parent.get("database_id", "")
            if SIMULATION_DB_ID not in db_id:
                continue
            props = p.get("properties", {})
            for _k, v in props.items():
                if v.get("type") == "title":
                    name = "".join(t.get("plain_text", "") for t in v.get("title", []))
                    if name:
                        names.add(name)
                    break

        while search_results.get("has_more"):
            search_results = self.client.search(
                query="",
                filter={"value": "page", "property": "object"},
                start_cursor=search_results["next_cursor"],
            )
            for p in search_results.get("results", []):
                parent = p.get("parent", {})
                db_id = parent.get("database_id", "")
                if SIMULATION_DB_ID not in db_id:
                    continue
                props = p.get("properties", {})
                for _k, v in props.items():
                    if v.get("type") == "title":
                        name = "".join(t.get("plain_text", "") for t in v.get("title", []))
                        if name:
                            names.add(name)
                        break

        return names

    # -------------------------------------------------------------------
    # Run all
    # -------------------------------------------------------------------
    def run_full_sync(self) -> None:
        """Execute complete project synchronization."""
        logger.info("=" * 60)
        logger.info("NOTION FULL PROJECT SYNC")
        logger.info("=" * 60)

        self.sync_main_page()
        self.sync_roadmap()
        self.sync_simulations()

        logger.info("=" * 60)
        logger.info("SYNC COMPLETE")
        logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    import argparse
    from pathlib import Path as _Path

    # Load .env if present (allows local runs without exporting vars)
    _env = _Path(__file__).resolve().parents[2] / ".env"
    if _env.exists():
        for line in _env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    # Re-read token after loading .env
    global NOTION_TOKEN  # noqa: PLW0603
    if not NOTION_TOKEN:
        NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")

    parser = argparse.ArgumentParser(
        description="Full Notion project synchronization for Hybrid Digital Twin"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview what would be synced without writing"
    )
    parser.add_argument(
        "--page-only", action="store_true", help="Only update the main project page"
    )
    parser.add_argument("--roadmap-only", action="store_true", help="Only update the Roadmap DB")
    parser.add_argument(
        "--simulations-only", action="store_true", help="Only sync NLTHA results to Simulation DB"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    sync = NotionProjectSync(dry_run=args.dry_run)

    if args.page_only:
        sync.sync_main_page()
    elif args.roadmap_only:
        sync.sync_roadmap()
    elif args.simulations_only:
        sync.sync_simulations()
    else:
        sync.run_full_sync()


if __name__ == "__main__":
    main()
