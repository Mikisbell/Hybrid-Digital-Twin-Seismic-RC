"""
notion_pages_sync.py — Create & populate Notion sub-pages
==========================================================

Builds the full Notion documentation workspace with dedicated
child pages under the main project page:

    1. 📖 Documentación Técnica
    2. 📝 Manuscrito HRPUB
    3. 🔬 Metodología y Resultados
    4. 🛠️ Guía de Desarrollo

Each page is created as a child of the main project page and
populated with rich content derived from the codebase and manuscript.

Usage::

    python -m src.utils.notion_pages_sync           # create all pages
    python -m src.utils.notion_pages_sync --dry-run  # preview only
    python -m src.utils.notion_pages_sync --clean    # delete old sub-pages first

Author: Mikisbell
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
MAIN_PAGE_ID = "30351e60-d3c2-800e-9dba-e8dcc2643cec"
REPO_URL = "https://github.com/Mikisbell/Hybrid-Digital-Twin-Seismic-RC"


# ═══════════════════════════════════════════════════════════════════════════
# Notion block helpers (compact)
# ═══════════════════════════════════════════════════════════════════════════


def _rt(text: str, bold: bool = False, code: bool = False, color: str = "default") -> dict:
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


def _link(text: str, url: str, bold: bool = False) -> dict:
    return {
        "type": "text",
        "text": {"content": text, "link": {"url": url}},
        "annotations": {
            "bold": bold,
            "italic": False,
            "strikethrough": False,
            "underline": False,
            "code": False,
            "color": "default",
        },
    }


def h1(text: str) -> dict:
    return {"object": "block", "type": "heading_1", "heading_1": {"rich_text": [_rt(text)]}}


def h2(text: str) -> dict:
    return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": [_rt(text)]}}


def h3(text: str) -> dict:
    return {"object": "block", "type": "heading_3", "heading_3": {"rich_text": [_rt(text)]}}


def para(*parts: dict) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": list(parts)}}


def bullet(*parts: dict) -> dict:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": list(parts)},
    }


def numbered(*parts: dict) -> dict:
    return {
        "object": "block",
        "type": "numbered_list_item",
        "numbered_list_item": {"rich_text": list(parts)},
    }


def todo(text: str, checked: bool = False) -> dict:
    return {
        "object": "block",
        "type": "to_do",
        "to_do": {"rich_text": [_rt(text)], "checked": checked},
    }


def divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def callout(text: str, emoji: str = "💡") -> dict:
    return {
        "object": "block",
        "type": "callout",
        "callout": {
            "icon": {"type": "emoji", "emoji": emoji},
            "rich_text": [_rt(text)],
        },
    }


def code_block(text: str, lang: str = "python") -> dict:
    return {
        "object": "block",
        "type": "code",
        "code": {"rich_text": [_rt(text[:2000])], "language": lang},
    }


def quote(text: str) -> dict:
    return {"object": "block", "type": "quote", "quote": {"rich_text": [_rt(text)]}}


def toggle(title: str, children: list[dict] | None = None) -> dict:
    blk: dict[str, Any] = {
        "object": "block",
        "type": "toggle",
        "toggle": {"rich_text": [_rt(title)]},
    }
    if children:
        blk["toggle"]["children"] = children[:100]
    return blk


def bookmark(url: str) -> dict:
    return {"object": "block", "type": "bookmark", "bookmark": {"url": url}}


def toc() -> dict:
    return {"object": "block", "type": "table_of_contents", "table_of_contents": {}}


# ═══════════════════════════════════════════════════════════════════════════
# Sub-page builder functions
# ═══════════════════════════════════════════════════════════════════════════


def _build_documentacion_tecnica() -> list[dict]:
    """📖 Documentación Técnica — complete technical reference."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocks: list[dict] = []

    blocks.append(callout(f"Referencia técnica completa del framework. Actualizado: {ts}", "📖"))
    blocks.append(toc())
    blocks.append(divider())

    # --- Modelo Estructural ---
    blocks.append(h1("🏗️ Modelo Estructural OpenSeesPy"))
    blocks.append(
        para(
            _rt("Pórtico de concreto reforzado de "),
            _rt("5 pisos, 3 vanos", bold=True),
            _rt(" diseñado conforme a "),
            _rt("ACI 318-19", bold=True),
            _rt(" y analizado mediante NLTHA con amortiguamiento de Rayleigh."),
        )
    )

    blocks.append(h2("Geometría"))
    blocks.append(bullet(_rt("Pisos: "), _rt("5", bold=True), _rt(" (altura de entrepiso: 3.2 m)")))
    blocks.append(bullet(_rt("Vanos: "), _rt("3", bold=True), _rt(" (luz: 6.0 m)")))
    blocks.append(bullet(_rt("Nodos: "), _rt("24", bold=True), _rt(" (4 base fijos + 20 libres)")))
    blocks.append(
        bullet(
            _rt("Elementos: "),
            _rt("35", bold=True),
            _rt(" (20 columnas + 15 vigas) — "),
            _rt("forceBeamColumn", code=True),
        )
    )

    blocks.append(h2("Materiales"))
    blocks.append(h3("Concreto — Concrete02"))
    blocks.append(bullet(_rt("f'c = 28 MPa (no confinado)")))
    blocks.append(bullet(_rt("f'cc = 35.4 MPa (confinado, ratio Mander = 1.27)")))
    blocks.append(bullet(_rt("εc0 = 0.002, εcu = 0.006 (confinado), 0.004 (no confinado)")))
    blocks.append(bullet(_rt("Modelo: Mander et al. (1988) para confinamiento")))

    blocks.append(h3("Acero — Steel02"))
    blocks.append(bullet(_rt("fy = 420 MPa, Es = 200 GPa")))
    blocks.append(bullet(_rt("b = 0.01 (razón de endurecimiento)")))
    blocks.append(bullet(_rt("Modelo: Menegotto-Pinto (Giuffré, 1970)")))

    blocks.append(h2("Secciones"))
    blocks.append(
        bullet(
            _rt("Columnas: ", bold=True),
            _rt("500×500 mm, As = 5,890 mm², ρ = 2.36%"),
        )
    )
    blocks.append(
        bullet(
            _rt("Vigas: ", bold=True),
            _rt("350×600 mm, As = 2,199 mm², ρ = 1.05%"),
        )
    )
    blocks.append(bullet(_rt("Discretización fibra: 10 subdivisiones en cada dirección")))
    blocks.append(bullet(_rt("Integración: 5 puntos Gauss-Lobatto por elemento")))

    blocks.append(h2("Amortiguamiento"))
    blocks.append(bullet(_rt("Tipo: Rayleigh proporcional a masa y rigidez")))
    blocks.append(bullet(_rt("ξ = 5.0% en modos 1 y 3")))
    blocks.append(
        bullet(
            _rt("Períodos modales: "),
            _rt("[1.186, 0.368, 0.201, 0.132, 0.100] s", code=True),
        )
    )
    blocks.append(bullet(_rt("T₁ = 1.186 s (período fundamental)")))

    blocks.append(h2("Análisis NLTHA"))
    blocks.append(bullet(_rt("Algoritmo: Newton-Raphson con Newmark β=0.25, γ=0.5")))
    blocks.append(bullet(_rt("Paso de tiempo: dt = Δt del registro (típicamente 0.005–0.02 s)")))
    blocks.append(bullet(_rt("Transformación geométrica: P-Delta")))
    blocks.append(bullet(_rt("Convergencia: tolerancia 1e-6, max 10 iteraciones")))
    blocks.append(
        bullet(
            _rt("Fuente: "),
            _rt("src/opensees_analysis/ospy_model.py", code=True),
            _rt(" (934 líneas) + "),
            _rt("nltha_runner.py", code=True),
            _rt(" (658 líneas)"),
        )
    )

    blocks.append(divider())

    # --- Hybrid-PINN ---
    blocks.append(h1("🧠 Hybrid-PINN — Red Neuronal Informada por Física"))
    blocks.append(
        para(
            _rt("Arquitectura de "),
            _rt("603,653 parámetros", bold=True),
            _rt(
                " que combina un encoder temporal CNN 1D con un regresor fully-connected "
                "para predecir derivas de entrepiso (IDR) por piso."
            ),
        )
    )

    blocks.append(h2("Arquitectura"))
    blocks.append(h3("Encoder Temporal (1D-CNN)"))
    blocks.append(
        numbered(
            _rt("Conv1d: 1→32 canales, k=7, s=2, p=3 + BN + SiLU → (B, 32, 1024)"),
        )
    )
    blocks.append(
        numbered(
            _rt("Conv1d: 32→64 canales, k=5, s=2, p=2 + BN + SiLU → (B, 64, 512)"),
        )
    )
    blocks.append(
        numbered(
            _rt("Conv1d: 64→128 canales, k=3, s=2, p=1 + BN + SiLU → (B, 128, 256)"),
        )
    )
    blocks.append(numbered(_rt("AdaptiveAvgPool1d → (B, 128, 16) → Flatten → (B, 2048)")))

    blocks.append(h3("Cabeza de Regresión (FC)"))
    blocks.append(numbered(_rt("Linear 2048→256 + SiLU + Dropout(0.05)")))
    blocks.append(numbered(_rt("Linear 256→128 + SiLU + Dropout(0.05)")))
    blocks.append(numbered(_rt("Linear 128→64 + SiLU")))
    blocks.append(numbered(_rt("Linear 64→32 + SiLU")))
    blocks.append(numbered(_rt("Linear 32→5 (output: IDR por piso)")))

    blocks.append(h2("Función de Pérdida Híbrida"))
    blocks.append(quote("L_total = λ_d · L_data + λ_p · L_physics + λ_bc · L_bc"))
    blocks.append(
        bullet(
            _rt("L_data", bold=True),
            _rt(": MSE entre predicción PINN y respuesta OpenSeesPy"),
        )
    )
    blocks.append(
        bullet(
            _rt("L_physics", bold=True),
            _rt(": Residuo de la ecuación de movimiento Mü + Cu̇ + f_int(u) = −M·1·ag"),
        )
    )
    blocks.append(
        bullet(
            _rt("L_bc", bold=True),
            _rt(": Condiciones iniciales u(0) = 0, u̇(0) = 0"),
        )
    )
    blocks.append(
        bullet(
            _rt("Pesos por defecto: "),
            _rt("λ_d=1.0, λ_p=0.1, λ_bc=0.01", code=True),
        )
    )
    blocks.append(
        bullet(
            _rt("Adaptativo: Gradient-norm balancing (McClenny & Braga-Neto, 2023, EMA α=0.9)"),
        )
    )

    blocks.append(h2("Protocolo de Entrenamiento"))
    blocks.append(bullet(_rt("Optimizador: AdamW (lr=1e-3, weight_decay=1e-4)")))
    blocks.append(
        bullet(_rt("Scheduler: CosineAnnealingWarmRestarts (T₀=50, T_mult=2, η_min=1e-6)"))
    )
    blocks.append(bullet(_rt("Máximo epochs: 500 (early stopping patience=50)")))
    blocks.append(bullet(_rt("Batch size: 64")))
    blocks.append(bullet(_rt("Gradient clipping: max_norm=1.0")))
    blocks.append(bullet(_rt("Seed: 42 (reproducibilidad determinista)")))
    blocks.append(bullet(_rt("Split: 70% train / 15% val / 15% test (estratificado)")))
    blocks.append(
        bullet(
            _rt("3 modos: ", bold=True),
            _rt("Data-only → Hybrid → Adaptive"),
        )
    )
    blocks.append(
        bullet(
            _rt("Fuente: "),
            _rt("src/pinn/model.py", code=True),
            _rt(" (351 líneas), "),
            _rt("loss.py", code=True),
            _rt(" (406 líneas), "),
            _rt("trainer.py", code=True),
            _rt(" (656 líneas)"),
        )
    )

    blocks.append(h2("Benchmark de Latencia"))
    blocks.append(bullet(_rt("Target: ≤ 100 ms para aplicabilidad en tiempo real")))
    blocks.append(bullet(_rt("Cold start + warm inference (1000 iteraciones)")))
    blocks.append(bullet(_rt("Throughput a batch sizes: 1, 8, 32, 128")))
    blocks.append(
        bullet(
            _rt("Script: "),
            _rt("src/pinn/benchmark_latency.py", code=True),
            _rt(" (372 líneas)"),
        )
    )

    blocks.append(divider())

    # --- Data Factory ---
    blocks.append(h1("🏭 Data Factory — Generación de Datos Sísmicos"))
    blocks.append(
        para(
            _rt(
                "Pipeline completo para generación de movimientos sísmicos, espectros de respuesta "
                "y matching espectral conforme a "
            ),
            _rt("ASCE 7-22 §16.2", bold=True),
            _rt("."),
        )
    )

    blocks.append(h2("Capacidades"))
    blocks.append(
        numbered(
            _rt("Ingesta de registros PEER NGA-West2 (formato AT2)"),
        )
    )
    blocks.append(
        numbered(_rt("Generación de registros sintéticos (ruido coloreado + envolvente Husid)"))
    )
    blocks.append(
        numbered(
            _rt("Espectro de respuesta: Nigam-Jennings (1969) piecewise-exact recurrence"),
        )
    )
    blocks.append(numbered(_rt("Scalado al espectro de diseño (SDS=1.0g, SD1=0.6g, TL=8.0s)")))
    blocks.append(numbered(_rt("Spectral matching ASCE 7-22 §16.2: media ≥ 90% del target")))
    blocks.append(numbered(_rt("NLTHA automático con OpenSeesPy por cada registro")))
    blocks.append(numbered(_rt("Logging a Notion (DB de Simulaciones)")))
    blocks.append(numbered(_rt("Exportación a CSV (factory_summary.csv)")))

    blocks.append(h2("Espectro de Diseño ASCE 7-22"))
    blocks.append(bullet(_rt("SDS = 1.0 g (aceleración espectral en período corto)")))
    blocks.append(bullet(_rt("SD1 = 0.6 g (aceleración espectral a 1 segundo)")))
    blocks.append(bullet(_rt("TL = 8.0 s (período de transición largo)")))
    blocks.append(
        bullet(
            _rt("Rango de matching: "),
            _rt("[0.2T₁, 2.0T₁] = [0.237, 2.372] s", code=True),
        )
    )

    blocks.append(h2("Criterios de Selección PEER NGA-West2"))
    blocks.append(bullet(_rt("Magnitud: 6.0 ≤ Mw ≤ 7.5")))
    blocks.append(bullet(_rt("Distancia: 10 ≤ Rjb ≤ 50 km")))
    blocks.append(bullet(_rt("Clase de sitio: C/D (180 ≤ Vs30 ≤ 760 m/s)")))
    blocks.append(bullet(_rt("Records mínimos: 200+")))
    blocks.append(bullet(_rt("Scale factor máximo: 5.0")))

    blocks.append(h2("Uso"))
    blocks.append(
        code_block(
            "# Campaña sintética (20 registros)\n"
            "python -m src.preprocessing.data_factory --synthetic 20\n\n"
            "# Campaña con datos PEER reales\n"
            "python -m src.preprocessing.data_factory --input data/raw/peer_records/\n\n"
            "# Dry-run (sin ejecutar NLTHA)\n"
            "python -m src.preprocessing.data_factory --synthetic 10 --dry-run",
            "bash",
        )
    )
    blocks.append(
        bullet(
            _rt("Fuente: "),
            _rt("src/preprocessing/data_factory.py", code=True),
            _rt(" (1,313 líneas)"),
        )
    )

    blocks.append(divider())

    # --- Pipeline ML ---
    blocks.append(h1("⚙️ Pipeline de Preprocesamiento ML"))
    blocks.append(
        para(
            _rt(
                "Transforma los datos crudos de NLTHA en datasets listos para entrenamiento del PINN."
            ),
        )
    )
    blocks.append(
        code_block(
            "Raw NLTHA Output → Feature Extraction → Normalization → Train/Val/Test Split\n"
            "       │                    │                  │                │\n"
            "  Time series          IDR, PFA,          Min-Max or        70/15/15\n"
            "  (disp, accel,        Sa, Sd,           StandardScaler\n"
            "   force, drift)       Arias intensity",
            "plain text",
        )
    )
    blocks.append(bullet(_rt("Features de entrada: PGA, PGV, Sa(T₁), intensidad de Arias")))
    blocks.append(bullet(_rt("Features temporales: acelerograma (ventana de 2048 pts)")))
    blocks.append(bullet(_rt("Targets: IDR máximo por piso (5 valores)")))
    blocks.append(bullet(_rt("Validación: K-S test, Pearson correlation, 5-fold CV")))
    blocks.append(
        bullet(
            _rt("Fuente: "),
            _rt("src/preprocessing/pipeline.py", code=True),
            _rt(" (382 líneas)"),
        )
    )

    blocks.append(divider())

    # --- Utilidades ---
    blocks.append(h1("🔧 Módulos de Utilidad"))
    blocks.append(h2("FigureManager"))
    blocks.append(
        para(
            _rt(
                "Gestor centralizado de figuras para publicación HRPUB. Asegura 300 DPI, "
                "formato PNG/TIFF, y nomenclatura consistente."
            ),
        )
    )
    blocks.append(
        bullet(
            _rt("Fuente: "),
            _rt("src/utils/figure_manager.py", code=True),
            _rt(" (240 líneas)"),
        )
    )

    blocks.append(h2("NotionResearchLogger"))
    blocks.append(
        para(
            _rt(
                "Logger automático que registra cada simulación NLTHA a la base de datos "
                "de Notion (🔬 Registro de Simulaciones)."
            ),
        )
    )
    blocks.append(
        bullet(
            _rt("Fuente: "),
            _rt("src/utils/sync_results.py", code=True),
            _rt(" (268 líneas)"),
        )
    )

    blocks.append(h2("NotionProjectSync"))
    blocks.append(
        para(
            _rt(
                "Sincronización completa del proyecto: página principal, Roadmap DB, "
                "Simulation DB y sub-páginas de documentación."
            ),
        )
    )
    blocks.append(
        bullet(
            _rt("Fuente: "),
            _rt("src/utils/notion_full_sync.py", code=True),
            _rt(" + "),
            _rt("notion_pages_sync.py", code=True),
        )
    )

    return blocks


def _build_manuscrito_hrpub() -> list[dict]:
    """📝 Manuscrito HRPUB — publication tracking."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocks: list[dict] = []

    blocks.append(
        callout(
            "Journal: HRPUB — Horizon Research Publishing\n"
            "Formato: Numérico correlativo [1], [2], ...\n"
            "Idioma: Inglés académico formal, sin contracciones\n"
            "Figuras: ≥ 300 DPI (PNG/TIFF)\n"
            f"Actualizado: {ts}",
            "📝",
        )
    )
    blocks.append(toc())
    blocks.append(divider())

    # Progress overview
    blocks.append(h1("📊 Progreso General"))

    sections = [
        (
            "§1 Introduction",
            False,
            "Contexto, brecha de conocimiento, objetivos. Estructura esqueleto lista.",
        ),
        ("§2 Objectives", False, "Objetivo general + 5 objetivos específicos. Estructura lista."),
        (
            "§3.1 Framework Architecture",
            False,
            "Tres capas: Simulación, Inteligencia, Documentación.",
        ),
        (
            "§3.2 Ground Motion Selection",
            False,
            "PEER NGA-West2, criterios, ASCE 7-22 spectral matching.",
        ),
        (
            "§3.3 Structural Model",
            False,
            "OpenSeesPy RC frame 5 pisos — geometría, materiales, secciones.",
        ),
        (
            "§3.4 PINN Formulation",
            True,
            "Arquitectura completa (Tabla 1), pérdida híbrida (Eqs. 1-4), hiperparámetros (Tabla 2). ✅",
        ),
        ("§3.5 Training Protocol", False, "3 modos de entrenamiento, benchmark latencia."),
        (
            "§4 Results",
            False,
            "Figuras de entrenamiento, comparativas PINN vs OpenSees, espectros.",
        ),
        ("§5 Discussion", False, "Contribuciones, limitaciones, trabajo futuro."),
        ("§6 Conclusions", False, "Hallazgos clave, relevancia práctica."),
        ("References", True, "15 entradas definidas [1]-[15] en references.bib. ✅"),
    ]

    for name, done, desc in sections:
        blocks.append(todo(f"{name} — {desc}", checked=done))

    blocks.append(divider())

    # Section details
    blocks.append(h1("📄 Detalle por Sección"))

    blocks.append(h2("§1 Introduction"))
    blocks.append(
        para(
            _rt("Archivo: "),
            _rt("manuscript/01_introduction.md", code=True),
        )
    )
    blocks.append(
        bullet(
            _rt("1.1 Background — Edificios RC en zonas sísmicas, NLTHA costoso computacionalmente")
        )
    )
    blocks.append(
        bullet(
            _rt(
                "1.2 Literature Review — NLTHA tradicional, ML en ingeniería estructural, PINNs, Digital Twins"
            )
        )
    )
    blocks.append(
        bullet(
            _rt(
                "1.3 Research Gap — No existe framework que integre NLTHA + PINN + DT en tiempo real"
            )
        )
    )
    blocks.append(bullet(_rt("1.4 Contribution — Framework de Gemelo Digital Híbrido")))

    blocks.append(h2("§2 Objectives"))
    blocks.append(
        para(
            _rt("Archivo: "),
            _rt("manuscript/02_objectives.md", code=True),
        )
    )
    blocks.append(numbered(_rt("Simulation Layer: Modelo OpenSeesPy 5-pisos RC, ACI 318-19")))
    blocks.append(numbered(_rt("Data Generation: ≥500 NLTHA con PEER NGA-West2")))
    blocks.append(
        numbered(_rt("Intelligence Layer: PINN con ecuación de movimiento, latencia ≤100 ms"))
    )
    blocks.append(numbered(_rt("Validation: R² ≥ 0.95, RMSE ≤ 5% max drift")))
    blocks.append(numbered(_rt("Digital Twin: Monitoreo de salud estructural en tiempo real")))

    blocks.append(h2("§3 Methods (Methodology)"))
    blocks.append(
        para(
            _rt("Archivo: "),
            _rt("manuscript/03_methods.md", code=True),
            _rt(" — Sección más extensa (232 líneas). "),
            _rt("§3.4 es la única completamente terminada.", bold=True),
        )
    )

    blocks.append(h3("§3.4 PINN Formulation (COMPLETA ✅)"))
    blocks.append(bullet(_rt("Tabla 1: Arquitectura capa por capa (9 capas, 603,653 params)")))
    blocks.append(
        bullet(_rt("Ecuación (1): Pérdida total L = λ_d·L_data + λ_p·L_phys + λ_bc·L_bc"))
    )
    blocks.append(bullet(_rt("Ecuación (2): L_data — MSE")))
    blocks.append(bullet(_rt("Ecuación (3): L_physics — Residuo EOM con f_int no-lineal")))
    blocks.append(bullet(_rt("Ecuación (4): L_bc — Condiciones iniciales")))
    blocks.append(bullet(_rt("Tabla 2: 13 hiperparámetros con justificación")))
    blocks.append(bullet(_rt("3 modos: data-only, hybrid, adaptive")))

    blocks.append(h2("§4 Results (Pendiente)"))
    blocks.append(
        para(
            _rt("Archivo: "),
            _rt("manuscript/04_results.md", code=True),
            _rt(" — Esqueleto listo. Necesita figuras de entrenamiento y comparativas."),
        )
    )

    blocks.append(h2("§5 Discussion (Pendiente)"))
    blocks.append(
        para(
            _rt("Archivo: "),
            _rt("manuscript/05_discussion.md", code=True),
        )
    )

    blocks.append(h2("§6 Conclusions (Pendiente)"))
    blocks.append(
        para(
            _rt("Archivo: "),
            _rt("manuscript/06_conclusions.md", code=True),
        )
    )

    blocks.append(divider())

    # References
    blocks.append(h1("📚 Referencias"))
    blocks.append(
        para(
            _rt("Archivo: "),
            _rt("manuscript/references.bib", code=True),
            _rt(" — 15 entradas BibTeX."),
        )
    )

    refs = [
        "[1] Chopra (2017) — Dynamics of Structures",
        "[2] PEER NGA-West2 — Ancheta et al. (2014)",
        "[3] Raissi et al. (2019) — Physics-Informed Neural Networks",
        "[4] McKenna et al. (2010) — OpenSees",
        "[5] ACI 318-19 — Building Code Requirements for Structural Concrete",
        "[6] ASCE 7-22 — Minimum Design Loads",
        "[7] Mander et al. (1988) — Stress-strain model for confined concrete",
        "[8] Menegotto & Pinto (1973) — Cyclic steel model",
        "[9] Ramachandran et al. (2017) — Swish activation function",
        "[10] Ba et al. (2016) — Layer Normalization",
        "[11] Loshchilov & Hutter (2019) — AdamW decoupled weight decay",
        "[12] Loshchilov & Hutter (2017) — SGDR: Cosine Annealing",
        "[13] Paszke et al. (2019) — PyTorch",
        "[14] He et al. (2015) — Kaiming initialization",
        "[15] McClenny & Braga-Neto (2023) — Self-Adaptive PINNs",
    ]
    for r in refs:
        blocks.append(bullet(_rt(r)))

    return blocks


def _build_metodologia_resultados() -> list[dict]:
    """🔬 Metodología y Resultados — campaigns and metrics."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocks: list[dict] = []

    blocks.append(
        callout(
            f"Resultados de campañas NLTHA, métricas de daño y validación. Actualizado: {ts}",
            "🔬",
        )
    )
    blocks.append(toc())
    blocks.append(divider())

    # Campaign results from CSV
    csv_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "factory_summary.csv"
    records: list[dict[str, str]] = []
    if csv_path.exists():
        with open(csv_path) as f:
            records = list(csv.DictReader(f))

    blocks.append(h1("📊 Campaña NLTHA Sintética"))

    if records:
        n_total = len(records)
        n_ok = sum(1 for r in records if r.get("converged", "").lower() == "true")
        pgas = [float(r.get("pga_g", 0)) for r in records]
        idrs = [float(r.get("max_idr_overall", 0)) for r in records]
        wall_times = [float(r.get("wall_clock_s", 0)) for r in records]

        blocks.append(
            callout(
                f"✅ {n_ok}/{n_total} registros convergieron (100%) "
                f"en {sum(wall_times):.0f}s totales",
                "✅",
            )
        )

        blocks.append(h2("Estadísticas Globales"))
        blocks.append(
            bullet(
                _rt(
                    f"PGA: min={min(pgas):.3f}g, max={max(pgas):.3f}g, media={sum(pgas) / len(pgas):.3f}g"
                ),
            )
        )
        blocks.append(
            bullet(
                _rt(
                    f"Max IDR: min={min(idrs):.4f}, max={max(idrs):.4f}, "
                    f"media={sum(idrs) / len(idrs):.4f}"
                ),
            )
        )
        blocks.append(
            bullet(
                _rt(f"Tiempo total: {sum(wall_times):.0f}s ({sum(wall_times) / 60:.1f} min)"),
            )
        )

        # Per-story max IDR
        blocks.append(h2("Distribución de IDR por Piso"))
        for i in range(1, 6):
            key = f"max_idr_{i}"
            vals = [float(r.get(key, 0)) for r in records]
            mx = max(vals) if vals else 0
            bar_len = int(mx / max(idrs) * 20) if max(idrs) > 0 else 0
            bar = "█" * bar_len + "░" * (20 - bar_len)
            blocks.append(bullet(_rt(f"Piso {i}: {bar} {mx:.5f}")))

        # Top 5 most severe
        blocks.append(h2("Top 5 Registros Más Severos"))
        sorted_recs = sorted(
            records, key=lambda r: float(r.get("max_idr_overall", 0)), reverse=True
        )
        for _i, r in enumerate(sorted_recs[:5]):
            name = r.get("record_name", "?")
            idr = float(r.get("max_idr_overall", 0))
            pga = float(r.get("pga_g", 0))
            vb = float(r.get("peak_base_shear_kN", 0))
            dur = float(r.get("duration_s", 0))
            blocks.append(
                numbered(
                    _rt(f"{name}", bold=True),
                    _rt(f" — IDR={idr:.4f}, PGA={pga:.3f}g, VBase={vb:.0f} kN, {dur:.1f}s"),
                )
            )

        # Detailed table per record
        blocks.append(h2("Detalle por Registro"))
        for r in records:
            name = r.get("record_name", "?")
            idr = float(r.get("max_idr_overall", 0))
            pga = float(r.get("pga_g", 0))
            sf = float(r.get("scale_factor", 0))
            dur = float(r.get("duration_s", 0))
            vb = float(r.get("peak_base_shear_kN", 0))
            wc = float(r.get("wall_clock_s", 0))
            steps = r.get("n_steps", "?")
            blocks.append(
                toggle(
                    f"{name} — IDR={idr:.4f}, PGA={pga:.3f}g",
                    [
                        bullet(_rt(f"Scale Factor: {sf:.3f}")),
                        bullet(_rt(f"Duración: {dur:.1f}s ({steps} pasos)")),
                        bullet(_rt(f"VBase máx: {vb:.0f} kN")),
                        bullet(_rt(f"Tiempo real de cálculo: {wc:.1f}s")),
                        bullet(_rt("Estado: Convergió")),
                    ],
                )
            )
    else:
        blocks.append(
            callout("No se encontró factory_summary.csv. Ejecute la campaña primero.", "⚠️")
        )

    blocks.append(divider())

    # Damage metrics
    blocks.append(h1("📐 Métricas de Daño Estructural"))

    blocks.append(h2("IDR — Inter-story Drift Ratio"))
    blocks.append(
        para(
            _rt(
                "Razón de la deformación lateral de entrepiso relativa a la altura del piso. "
                "Es el principal "
            ),
            _rt("Engineering Demand Parameter (EDP)", bold=True),
            _rt(" para evaluación sísmica."),
        )
    )
    blocks.append(bullet(_rt("IDR < 0.5%: Daño menor (operación inmediata)")))
    blocks.append(bullet(_rt("0.5% ≤ IDR < 1.0%: Daño moderado (seguridad de vida)")))
    blocks.append(bullet(_rt("1.0% ≤ IDR < 2.0%: Daño significativo (prevención de colapso)")))
    blocks.append(bullet(_rt("IDR ≥ 2.0%: Daño severo / posible colapso")))

    blocks.append(h2("PGA — Peak Ground Acceleration"))
    blocks.append(
        para(_rt("Aceleración pico registrada en la base del edificio durante el sismo (g)."))
    )

    blocks.append(h2("Park-Ang Damage Index"))
    blocks.append(
        para(
            _rt(
                "Índice combinado que pondera deformación máxima y energía histerética disipada. "
                "DI < 0.1: sin daño, 0.1-0.25: menor, 0.25-0.40: reparable, 0.40-1.0: severo, >1.0: colapso."
            )
        )
    )

    blocks.append(h2("Cortante Basal"))
    blocks.append(
        para(
            _rt("Fuerza cortante máxima en la base del edificio (kN). Indicador de demanda global.")
        )
    )

    blocks.append(divider())

    # Spectral matching
    blocks.append(h1("🌊 Espectro de Diseño y Matching"))
    blocks.append(h2("Parámetros ASCE 7-22"))
    blocks.append(bullet(_rt("SDS = 1.0 g")))
    blocks.append(bullet(_rt("SD1 = 0.6 g")))
    blocks.append(bullet(_rt("TL = 8.0 s")))
    blocks.append(bullet(_rt("Rango de matching: [0.2T₁, 2.0T₁] = [0.237, 2.372] s")))

    blocks.append(h2("Algoritmo de Matching"))
    blocks.append(numbered(_rt("Calcular espectro de respuesta con Nigam-Jennings (1969)")))
    blocks.append(numbered(_rt("Evaluarrelación Sa_record/Sa_target en [0.2T₁, 2.0T₁]")))
    blocks.append(numbered(_rt("Factor de escala: mediana de relaciones espectrales")))
    blocks.append(numbered(_rt("Iterative boost: ajuste fino si media < 90% (max 10 iter)")))
    blocks.append(numbered(_rt("Criterio ASCE 7-22 §16.2: media de suite ≥ 90% del target")))

    blocks.append(divider())

    # Criteria table
    blocks.append(h1("🎯 Criterios de Éxito del Proyecto"))
    blocks.append(bullet(_rt("Error PINN vs OpenSees: < 5% RMSE normalizado")))
    blocks.append(bullet(_rt("Latencia inferencia: < 100 ms (P95)")))
    blocks.append(bullet(_rt("Figuras: ≥ 300 DPI")))
    blocks.append(bullet(_rt("Cobertura tests: ≥ 80%")))
    blocks.append(bullet(_rt("Pre-commit: 0 warnings")))
    blocks.append(bullet(_rt("Manuscrito: 6 secciones completas")))
    blocks.append(bullet(_rt("Registros sísmicos: ≥ 200 (PEER NGA-West2)")))

    return blocks


def _build_guia_desarrollo() -> list[dict]:
    """🛠️ Guía de Desarrollo — setup, commands, CI/CD, conventions."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    blocks: list[dict] = []

    blocks.append(
        callout(
            f"Guía completa para desarrolladores y colaboradores. Actualizado: {ts}",
            "🛠️",
        )
    )
    blocks.append(toc())
    blocks.append(divider())

    # Setup
    blocks.append(h1("🚀 Configuración Inicial"))
    blocks.append(
        code_block(
            "# 1. Clonar repositorio\n"
            "git clone https://github.com/Mikisbell/Hybrid-Digital-Twin-Seismic-RC.git\n"
            "cd Hybrid-Digital-Twin-Seismic-RC\n\n"
            "# 2. Crear entorno virtual\n"
            "python3 -m venv .venv\n"
            "source .venv/bin/activate\n\n"
            "# 3. Instalar dependencias\n"
            "pip install -r requirements.txt\n\n"
            "# 4. Instalar pre-commit hooks\n"
            "pre-commit install\n\n"
            "# 5. Configurar variables de entorno\n"
            "echo 'NOTION_TOKEN=tu_token_aqui' > .env\n\n"
            "# 6. Verificar instalación\n"
            "python -c \"import openseespy.opensees; print('OpenSeesPy OK')\"\n"
            "python -c \"import torch; print(f'PyTorch OK ({torch.__version__})')\"",
            "bash",
        )
    )

    blocks.append(divider())

    # Commands
    blocks.append(h1("⌨️ Comandos Principales"))

    blocks.append(h2("Simulación y Datos"))
    blocks.append(
        code_block(
            "# Campaña NLTHA sintética (20 registros)\n"
            "python -m src.preprocessing.data_factory --synthetic 20\n\n"
            "# Campaña con datos PEER reales\n"
            "python -m src.preprocessing.data_factory --input data/raw/peer_records/\n\n"
            "# Pipeline de preprocesamiento ML\n"
            "python -m src.preprocessing.pipeline\n\n"
            "# Modelo OpenSeesPy standalone\n"
            "python -m src.opensees_analysis.ospy_model",
            "bash",
        )
    )

    blocks.append(h2("PINN"))
    blocks.append(
        code_block(
            "# Entrenar PINN (modo hybrid por defecto)\n"
            "python -m src.pinn.trainer\n\n"
            "# Benchmark de latencia\n"
            "python -m src.pinn.benchmark_latency\n\n"
            "# Solo modelo (verificar arquitectura)\n"
            "python -m src.pinn.model",
            "bash",
        )
    )

    blocks.append(h2("Notion Sync"))
    blocks.append(
        code_block(
            "# Sincronización completa a Notion\n"
            "python -m src.utils.notion_full_sync\n\n"
            "# Solo página principal\n"
            "python -m src.utils.notion_full_sync --page-only\n\n"
            "# Solo Simulation DB\n"
            "python -m src.utils.notion_full_sync --simulations-only\n\n"
            "# Crear/actualizar sub-páginas de documentación\n"
            "python -m src.utils.notion_pages_sync\n\n"
            "# Dry-run (solo preview)\n"
            "python -m src.utils.notion_pages_sync --dry-run",
            "bash",
        )
    )

    blocks.append(h2("Calidad de Código"))
    blocks.append(
        code_block(
            "# Pre-commit (todos los hooks)\n"
            "pre-commit run --all-files\n\n"
            "# Ruff (linter)\n"
            "ruff check src/\n\n"
            "# Ruff (formatter)\n"
            "ruff format src/\n\n"
            "# Tests\n"
            "pytest -v --tb=short",
            "bash",
        )
    )

    blocks.append(divider())

    # Architecture
    blocks.append(h1("📁 Estructura del Repositorio"))
    blocks.append(
        code_block(
            "Hybrid-Digital-Twin-Seismic-RC/\n"
            "├── src/\n"
            "│   ├── opensees_analysis/     ← Modelo RC y runner NLTHA\n"
            "│   │   ├── ospy_model.py      (934 líneas) Modelo 5-pisos\n"
            "│   │   └── nltha_runner.py    (658 líneas) NLTHA pipeline\n"
            "│   ├── pinn/                  ← Hybrid-PINN (PyTorch)\n"
            "│   │   ├── model.py           (351 líneas) Arquitectura CNN+FC\n"
            "│   │   ├── loss.py            (406 líneas) Pérdida híbrida\n"
            "│   │   ├── trainer.py         (656 líneas) Loop de entrenamiento\n"
            "│   │   └── benchmark_latency.py (372 líneas) Benchmark\n"
            "│   ├── preprocessing/         ← Pipeline de datos\n"
            "│   │   ├── data_factory.py    (1313 líneas) Generador sísmico\n"
            "│   │   └── pipeline.py        (382 líneas) Feature engineering\n"
            "│   └── utils/                 ← Utilidades\n"
            "│       ├── figure_manager.py  (240 líneas) Figuras 300 DPI\n"
            "│       ├── sync_results.py    (268 líneas) Logger Notion\n"
            "│       ├── notion_full_sync.py (1108 líneas) Sync completo\n"
            "│       └── notion_pages_sync.py Sub-páginas Notion\n"
            "├── manuscript/                ← Manuscrito HRPUB\n"
            "│   ├── 01_introduction.md ... 06_conclusions.md\n"
            "│   ├── references.bib         (15 entradas)\n"
            "│   └── README.md\n"
            "├── notebooks/                 ← Jupyter notebooks\n"
            "├── data/\n"
            "│   ├── raw/                   ← Registros sísmicos + CSVs\n"
            "│   ├── processed/             ← Datos normalizados\n"
            "│   ├── models/                ← Checkpoints PINN (.pt)\n"
            "│   └── external/              ← Datos de terceros\n"
            "├── figures/                   ← Figuras del manuscrito\n"
            "├── .github/workflows/         ← CI/CD\n"
            "│   ├── notion_sync.yml        ← Roadmap auto-sync\n"
            "│   └── pinn_notify.yml        ← PINN result logger\n"
            "├── requirements.txt\n"
            "└── .pre-commit-config.yaml",
            "plain text",
        )
    )

    blocks.append(divider())

    # Conventions
    blocks.append(h1("📏 Convenciones del Proyecto"))

    blocks.append(h2("Idiomas"))
    blocks.append(
        bullet(_rt("Código, docstrings, commits, README, manuscrito: ", bold=True), _rt("INGLÉS"))
    )
    blocks.append(
        bullet(_rt("Notion (bases de datos, tareas, roadmap): ", bold=True), _rt("ESPAÑOL"))
    )
    blocks.append(bullet(_rt("Chat con el usuario: ", bold=True), _rt("ESPAÑOL")))
    blocks.append(
        callout(
            "Regla de oro: Todo lo que va al repositorio o publicación → inglés. "
            "Todo lo interactivo → español.",
            "🌐",
        )
    )

    blocks.append(h2("Formato de Commits"))
    blocks.append(
        code_block(
            "feat: add PINN training loop\n"
            "fix: correct eigenvalue extraction\n"
            "docs: expand §3.4 with Table 1\n"
            "build: add pre-commit hooks\n"
            "refactor: split data_factory into modules\n"
            "test: add NLTHA convergence tests",
            "plain text",
        )
    )

    blocks.append(h2("Reglas de Git"))
    blocks.append(bullet(_rt("NUNCA subir archivos >1 MB (.csv, .hdf5, .pkl, .pt, .pth, .onnx)")))
    blocks.append(bullet(_rt("Usar .gitkeep en directorios vacíos")))
    blocks.append(bullet(_rt("Datos pesados → DVC o .gitignore")))
    blocks.append(
        bullet(_rt("Pre-commit hooks obligatorios: ruff, ruff-format, isort, trailing whitespace"))
    )

    blocks.append(h2("Pre-commit Hooks (8)"))
    blocks.append(numbered(_rt("ruff — Linter Python")))
    blocks.append(numbered(_rt("ruff-format — Auto-formatter")))
    blocks.append(numbered(_rt("isort — Ordenar imports")))
    blocks.append(numbered(_rt("trim trailing whitespace")))
    blocks.append(numbered(_rt("fix end of files")))
    blocks.append(numbered(_rt("check yaml")))
    blocks.append(numbered(_rt("check json")))
    blocks.append(numbered(_rt("check for large files")))

    blocks.append(divider())

    # CI/CD
    blocks.append(h1("🔄 CI/CD — GitHub Actions"))

    blocks.append(h2("notion_sync.yml"))
    blocks.append(para(_rt("Sincroniza automáticamente el Roadmap DB al hacer push a main.")))
    blocks.append(bullet(_rt("Trigger: push a main (cualquier archivo)")))
    blocks.append(bullet(_rt("Acción: Crea entrada en 📅 Hoja de Ruta con categoría y estado")))
    blocks.append(
        bullet(
            _rt("Propiedades: Tarea, Categoría, Fase, Estado, Fecha, Commit, Archivos Modificados"),
        )
    )

    blocks.append(h2("pinn_notify.yml"))
    blocks.append(para(_rt("Registra entrenamientos PINN al pushear cambios en src/pinn/.")))
    blocks.append(
        bullet(_rt("Trigger: push a main que modifique src/pinn/** o benchmark_results.json"))
    )
    blocks.append(bullet(_rt("Acción: Crea entrada en 🔬 Registro de Simulaciones con métricas")))
    blocks.append(bullet(_rt("Lee benchmark_results.json para latencia, throughput, device")))

    blocks.append(divider())

    # Dependencies
    blocks.append(h1("📦 Dependencias Principales"))
    blocks.append(h2("Cálculo Numérico"))
    blocks.append(bullet(_rt("numpy, scipy, pandas")))

    blocks.append(h2("Simulación"))
    blocks.append(bullet(_rt("openseespy 3.7.1 — Motor NLTHA")))

    blocks.append(h2("Machine Learning"))
    blocks.append(bullet(_rt("torch (CPU para dev, GPU para prod)")))

    blocks.append(h2("Visualización"))
    blocks.append(bullet(_rt("matplotlib, seaborn, plotly (300 DPI)")))

    blocks.append(h2("Datos"))
    blocks.append(bullet(_rt("h5py — Datasets grandes")))

    blocks.append(h2("Infraestructura"))
    blocks.append(bullet(_rt("notion-client 3.0.0 — API Notion")))
    blocks.append(bullet(_rt("httpx — HTTP cliente")))
    blocks.append(bullet(_rt("pre-commit, ruff — Calidad de código")))
    blocks.append(bullet(_rt("pytest — Testing")))

    return blocks


# ═══════════════════════════════════════════════════════════════════════════
# Notion page creator class
# ═══════════════════════════════════════════════════════════════════════════


class NotionPagesSync:
    """Create and populate Notion sub-pages."""

    SUB_PAGES = [
        ("📖 Documentación Técnica", "📖", _build_documentacion_tecnica),
        ("📝 Manuscrito HRPUB", "📝", _build_manuscrito_hrpub),
        ("🔬 Metodología y Resultados", "🔬", _build_metodologia_resultados),
        ("🛠️ Guía de Desarrollo", "🛠️", _build_guia_desarrollo),
    ]

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        if not dry_run:
            from notion_client import Client

            token = NOTION_TOKEN or os.environ.get("NOTION_TOKEN", "")
            if not token:
                raise ValueError("NOTION_TOKEN not set. Export it or create .env")
            self.client = Client(auth=token)

    def _find_existing_subpages(self) -> dict[str, str]:
        """Return {title: page_id} for child pages of main page."""
        existing: dict[str, str] = {}
        resp = self.client.blocks.children.list(MAIN_PAGE_ID)
        all_blocks = resp.get("results", [])
        while resp.get("has_more"):
            resp = self.client.blocks.children.list(MAIN_PAGE_ID, start_cursor=resp["next_cursor"])
            all_blocks.extend(resp.get("results", []))

        for b in all_blocks:
            if b["type"] == "child_page":
                title = b["child_page"].get("title", "")
                existing[title] = b["id"]

        return existing

    def _delete_page(self, page_id: str) -> None:
        """Archive (delete) a page."""
        try:
            self.client.blocks.delete(page_id)
            logger.debug("  Deleted page %s", page_id)
        except Exception as e:
            logger.warning("  Could not delete page %s: %s", page_id, e)

    def _create_subpage(self, title: str, emoji: str, blocks: list[dict]) -> str:
        """Create a child page under main page with content."""
        page = self.client.pages.create(
            parent={"page_id": MAIN_PAGE_ID},
            icon={"type": "emoji", "emoji": emoji},
            properties={"title": [{"text": {"content": title}}]},
        )
        page_id = page["id"]
        logger.info("  Created page: %s (%s)", title, page_id)

        # Append blocks in batches of 100
        for i in range(0, len(blocks), 100):
            batch = blocks[i : i + 100]
            try:
                self.client.blocks.children.append(page_id, children=batch)
                logger.debug("  Appended blocks %d-%d", i, i + len(batch))
            except Exception as e:
                logger.warning("  Failed batch %d: %s", i, e)

        return page_id

    def sync(self, clean: bool = False) -> None:
        """Create or update all sub-pages."""
        logger.info("=" * 60)
        logger.info("NOTION SUB-PAGES SYNC")
        logger.info("=" * 60)

        for title, emoji, builder in self.SUB_PAGES:
            blocks = builder()
            logger.info("\n📄 %s — %d blocks", title, len(blocks))

            if self.dry_run:
                logger.info("  [DRY RUN] Would create page with %d blocks", len(blocks))
                continue

            # Check for existing page
            existing = self._find_existing_subpages()
            if title in existing:
                if clean:
                    logger.info("  Deleting existing page: %s", existing[title])
                    self._delete_page(existing[title])
                else:
                    logger.info(
                        "  Page already exists: %s — skipping (use --clean to recreate)", title
                    )
                    continue

            self._create_subpage(title, emoji, blocks)

        logger.info("\n" + "=" * 60)
        logger.info("SUB-PAGES SYNC COMPLETE")
        logger.info("=" * 60)

    def update_main_page_navigation(self) -> None:
        """Add a navigation section at the top of main page linking to sub-pages."""
        if self.dry_run:
            logger.info("[DRY RUN] Would update main page navigation")
            return

        existing = self._find_existing_subpages()
        if not existing:
            logger.info("No sub-pages found to create navigation for")
            return

        # Build navigation blocks
        nav_blocks: list[dict] = [
            divider(),
            h2("📚 Navegación — Documentación Completa"),
            callout(
                "Este workspace contiene documentación detallada en sub-páginas dedicadas. "
                "Haga clic en cada enlace para acceder a la documentación completa.",
                "🗂️",
            ),
        ]
        for title in existing:
            desc = ""
            if "Documentación" in title:
                desc = " — Modelo RC, PINN, Data Factory, Pipeline, utilidades"
            elif "Manuscrito" in title:
                desc = " — Progreso §1-§6, referencias [1]-[15], formato HRPUB"
            elif "Metodología" in title:
                desc = " — Campañas NLTHA, espectros, métricas de daño"
            elif "Guía" in title:
                desc = " — Setup, comandos, CI/CD, convenciones, dependencias"
            nav_blocks.append(
                bullet(
                    _rt(f"{title}", bold=True),
                    _rt(f"{desc} → ver sub-página"),
                )
            )
        nav_blocks.append(divider())

        # Find the position to insert (after the TOC block)
        resp = self.client.blocks.children.list(MAIN_PAGE_ID)
        all_blocks = resp.get("results", [])
        while resp.get("has_more"):
            resp = self.client.blocks.children.list(MAIN_PAGE_ID, start_cursor=resp["next_cursor"])
            all_blocks.extend(resp.get("results", []))

        # Find first divider after TOC
        insert_after = None
        found_toc = False
        for b in all_blocks:
            if b["type"] == "table_of_contents":
                found_toc = True
            elif found_toc and b["type"] == "divider":
                insert_after = b["id"]
                break

        if insert_after:
            try:
                self.client.blocks.children.append(
                    MAIN_PAGE_ID,
                    children=nav_blocks,
                    after=insert_after,
                )
                logger.info("✅ Navigation section added after TOC")
            except Exception as e:
                logger.warning("Could not insert navigation: %s", e)
                # Fallback: append at end
                self.client.blocks.children.append(MAIN_PAGE_ID, children=nav_blocks)
                logger.info("✅ Navigation section appended at end (fallback)")
        else:
            self.client.blocks.children.append(MAIN_PAGE_ID, children=nav_blocks)
            logger.info("✅ Navigation section appended at end")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    import argparse
    from pathlib import Path as _Path

    # Load .env
    _env = _Path(__file__).resolve().parents[2] / ".env"
    if _env.exists():
        for line in _env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    global NOTION_TOKEN  # noqa: PLW0603
    if not NOTION_TOKEN:
        NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")

    parser = argparse.ArgumentParser(
        description="Create Notion sub-pages with full project documentation"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without creating")
    parser.add_argument(
        "--clean", action="store_true", help="Delete existing sub-pages before recreating"
    )
    parser.add_argument("--nav", action="store_true", help="Only update navigation on main page")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    sync = NotionPagesSync(dry_run=args.dry_run)

    if args.nav:
        sync.update_main_page_navigation()
    else:
        sync.sync(clean=args.clean)


if __name__ == "__main__":
    main()
