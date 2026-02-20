"""
ospy_model.py — 5-Story 3-Bay RC Frame (ACI 318-19)
====================================================

Defines a complete nonlinear finite element model of a reinforced concrete
moment-resisting frame using OpenSeesPy with fiber-section elements.

Design Basis
------------
    - ACI 318-19  : Structural concrete design
    - ASCE 7-22   : Seismic load combinations
    - Mander (1988): Confined concrete model [7]
    - Menegotto-Pinto (1973): Cyclic steel model [8]

Material Models
---------------
    - Concrete02 (Kent-Scott-Park + Mander confinement)
    - Steel02 (Giuffré-Menegotto-Pinto with isotropic hardening)

Element Formulation
-------------------
    - forceBeamColumn with Gauss-Lobatto integration (5 points)
    - Fiber sections for columns and beams
    - P-Delta geometric transformation for columns
    - Linear transformation for beams

References
----------
    [5] ACI 318-19, Building Code Requirements for Structural Concrete.
    [7] Mander, Priestley, Park (1988), J. Struct. Eng., 114(8), 1804-1826.
    [8] Menegotto, Pinto (1973), IABSE Symposium, pp. 15-22.

Unit System
-----------
    Length : m
    Force  : kN
    Mass   : tonne (Mg) = kN·s²/m
    Stress : kPa  (kN/m²)  ← material inputs in MPa are converted ×1e3
    Time   : s

Author: Mikisbell
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import OpenSeesPy (graceful fallback for CI environments)
# ---------------------------------------------------------------------------
try:
    import openseespy.opensees as ops

    OPS_AVAILABLE = True
except ImportError:
    OPS_AVAILABLE = False
    logger.warning("OpenSeesPy not installed. Model build will be unavailable.")


# ═══════════════════════════════════════════════════════════════════════════
# Configuration dataclasses
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ConcreteProperties:
    """Concrete02 material parameters (Mander confined concrete model).

    The confined properties are computed from transverse reinforcement
    using the Mander confinement model per ACI 318-19 Section 18.7.5.
    """

    # Unconfined properties
    fc: float = 28.0  # Compressive strength f'c (MPa)
    ec0: float = 0.002  # Strain at f'c
    ecu_unconf: float = 0.006  # Ultimate strain (unconfined)
    Ec: float = 0.0  # Elastic modulus (auto-computed if 0)

    # Confined properties (Mander model)
    fcc: float = 0.0  # Confined strength (auto-computed if 0)
    ecc: float = 0.0  # Strain at fcc (auto-computed if 0)
    ecu_conf: float = 0.015  # Ultimate confined strain

    # Tensile properties
    ft: float = 0.0  # Tensile strength (auto-computed if 0)
    ets: float = 0.001  # Tensile softening slope strain

    def __post_init__(self) -> None:
        # Auto-compute derived properties using object.__setattr__ (frozen)
        if self.Ec == 0.0:
            object.__setattr__(self, "Ec", 4700.0 * math.sqrt(self.fc))
        if self.ft == 0.0:
            object.__setattr__(self, "ft", 0.62 * math.sqrt(self.fc))
        if self.fcc == 0.0:
            # Default confinement ratio ~1.3 for typical columns
            object.__setattr__(self, "fcc", 1.3 * self.fc)
        if self.ecc == 0.0:
            object.__setattr__(self, "ecc", self.ec0 * (1 + 5 * (self.fcc / self.fc - 1)))


@dataclass(frozen=True)
class SteelProperties:
    """Steel02 material parameters (Giuffré-Menegotto-Pinto model).

    Models cyclic behavior with Bauschinger effect, critical for
    seismic analysis of RC frames.
    """

    fy: float = 420.0  # Yield strength (MPa)
    Es: float = 200_000.0  # Elastic modulus (MPa)
    b: float = 0.01  # Strain hardening ratio
    R0: float = 18.5  # Transition parameter
    cR1: float = 0.925  # Transition coefficient 1  # noqa: N815
    cR2: float = 0.15  # Transition coefficient 2  # noqa: N815
    a1: float = 0.04  # Isotropic hardening: increase in compression yield
    a2: float = 1.0  # a1 * fy / E0
    a3: float = 0.04  # Isotropic hardening: increase in tension yield
    a4: float = 1.0  # a3 * fy / E0


@dataclass(frozen=True)
class SectionGeometry:
    """Cross-section dimensions for columns and beams (mm → m internally)."""

    # Column section (square for simplicity, per ACI 318-19 §18.7)
    col_width: float = 0.50  # m
    col_depth: float = 0.50  # m
    col_cover: float = 0.04  # m (clear cover per ACI 318-19 §20.6.1)
    col_bar_dia: float = 0.025  # m (#8 = 25.4 mm)
    col_n_bars_top: int = 4  # Bars per face
    col_n_bars_bot: int = 4
    col_n_bars_side: int = 2  # Intermediate bars per side

    # Beam section (rectangular)
    beam_width: float = 0.35  # m
    beam_depth: float = 0.60  # m
    beam_cover: float = 0.04  # m
    beam_bar_dia: float = 0.020  # m (#6 = 19.1 mm)
    beam_n_bars_top: int = 4
    beam_n_bars_bot: int = 3

    # Transverse reinforcement (for Mander confinement)
    tie_bar_dia: float = 0.010  # m (#3 = 9.5 mm)
    tie_spacing: float = 0.100  # m (per ACI 318-19 §18.7.5.3)


@dataclass
class FrameGeometry:
    """5-story 3-bay frame geometry."""

    n_stories: int = 5
    n_bays: int = 3
    first_story_height: float = 3.5  # m (taller first story)
    typical_story_height: float = 3.0  # m
    bay_width: float = 5.0  # m

    def story_height(self, story: int) -> float:
        """Height of a given story (1-indexed)."""
        return self.first_story_height if story == 1 else self.typical_story_height

    @property
    def total_height(self) -> float:
        return self.first_story_height + (self.n_stories - 1) * self.typical_story_height

    @property
    def total_width(self) -> float:
        return self.n_bays * self.bay_width

    def node_coords(self) -> list[tuple[int, float, float]]:
        """Generate (node_tag, x, y) for all frame nodes.

        Numbering: node_tag = story * 100 + bay
        story=0 is base, bay=0..n_bays
        """
        coords: list[tuple[int, float, float]] = []
        for story in range(self.n_stories + 1):
            y = sum(self.story_height(s) for s in range(1, story + 1))
            for bay in range(self.n_bays + 1):
                x = bay * self.bay_width
                tag = story * 100 + bay
                coords.append((tag, x, y))
        return coords


@dataclass
class GravityLoads:
    """Gravity loads per ACI 318-19 + ASCE 7-22."""

    dead_load_floor: float = 25.0  # kN/m² (slab + finishes)
    live_load_floor: float = 2.0  # kN/m² (residential)
    dead_load_roof: float = 20.0  # kN/m² (lighter roof)
    live_load_roof: float = 1.0  # kN/m²
    tributary_width: float = 5.0  # m (perpendicular to frame)

    def beam_udl(self, story: int, n_stories: int) -> float:
        """Factored UDL on beam (kN/m) for seismic combo: 1.0D + 0.25L."""
        is_roof = story == n_stories
        d = self.dead_load_roof if is_roof else self.dead_load_floor
        live = self.live_load_roof if is_roof else self.live_load_floor
        return (1.0 * d + 0.25 * live) * self.tributary_width


@dataclass
class ModelConfig:
    """Master configuration for the 5-story RC frame model."""

    concrete: ConcreteProperties = field(default_factory=ConcreteProperties)
    steel: SteelProperties = field(default_factory=SteelProperties)
    sections: SectionGeometry = field(default_factory=SectionGeometry)
    frame: FrameGeometry = field(default_factory=FrameGeometry)
    loads: GravityLoads = field(default_factory=GravityLoads)
    damping_ratio: float = 0.05  # 5% Rayleigh
    damping_modes: tuple[int, int] = (1, 3)  # Modes for Rayleigh
    n_integration_pts: int = 5  # Gauss-Lobatto points per element
    mass_type: str = "lumped"  # "lumped" or "consistent"


# ═══════════════════════════════════════════════════════════════════════════
# Mander Confinement Model Calculator
# ═══════════════════════════════════════════════════════════════════════════


def compute_mander_confinement(
    fc: float,
    col_width: float,
    col_depth: float,
    cover: float,
    tie_dia: float,
    tie_spacing: float,
    n_legs_x: int = 2,
    n_legs_y: int = 2,
) -> tuple[float, float, float]:
    """Compute confined concrete properties per Mander et al. (1988).

    Parameters
    ----------
    fc : float
        Unconfined compressive strength (MPa).
    col_width, col_depth : float
        Column dimensions (m).
    cover : float
        Clear cover (m).
    tie_dia : float
        Tie bar diameter (m).
    tie_spacing : float
        Tie spacing (m).
    n_legs_x, n_legs_y : int
        Number of tie legs in each direction.

    Returns
    -------
    fcc : float
        Confined compressive strength (MPa).
    ecc : float
        Strain at fcc.
    ecu : float
        Ultimate confined strain.
    """
    # Core dimensions
    bc = col_width - 2 * cover  # Core width (center-to-center of ties)
    dc = col_depth - 2 * cover  # Core depth

    # Area of one tie leg
    a_tie = math.pi * (tie_dia**2) / 4

    # Volumetric ratio of transverse reinforcement
    rho_x = (n_legs_x * a_tie) / (dc * tie_spacing)
    rho_y = (n_legs_y * a_tie) / (bc * tie_spacing)

    # Effective lateral confining stress
    fy_tie = 420.0  # MPa (same grade)
    fl_x = rho_x * fy_tie
    fl_y = rho_y * fy_tie

    # Confinement effectiveness coefficient (Mander 1988, Eq. 3)
    # Simplified for rectangular hoops
    ke = 0.75  # Typical for well-detailed columns per ACI 318-19

    fl_eff_x = ke * fl_x
    fl_eff_y = ke * fl_y
    fl_eff = (fl_eff_x + fl_eff_y) / 2

    # Confined strength (Mander 1988, Eq. 6)
    ratio = fl_eff / fc
    fcc = fc * (2.254 * math.sqrt(1 + 7.94 * ratio) - 2 * ratio - 1.254)

    # Confined strain at peak
    ec0 = 0.002
    ecc = ec0 * (1 + 5 * (fcc / fc - 1))

    # Ultimate confined strain (Paulay & Priestley approximation)
    rho_s = rho_x + rho_y
    ecu = 0.004 + 1.4 * rho_s * fy_tie * 0.1 / fcc  # Simplified

    logger.info(
        "Mander confinement: f'cc=%.1f MPa (ratio=%.2f), εcc=%.4f, εcu=%.4f",
        fcc,
        fcc / fc,
        ecc,
        ecu,
    )

    return fcc, ecc, ecu


# ═══════════════════════════════════════════════════════════════════════════
# Model Builder
# ═══════════════════════════════════════════════════════════════════════════


class RCFrameModel:
    """Builds and manages a 5-story 3-bay RC frame in OpenSeesPy.

    This class encapsulates the complete finite element model including:
    - Nodes and constraints
    - Material definitions (Concrete02 + Steel02)
    - Fiber sections (columns and beams)
    - Elements (forceBeamColumn with P-Delta/Linear transformations)
    - Gravity loading
    - Rayleigh damping
    - Mass assignment

    Usage
    -----
        model = RCFrameModel()
        model.build()
        T1 = model.get_fundamental_period()
        model.apply_gravity()
    """

    # Material tags
    MAT_CONCRETE_CONF = 1
    MAT_CONCRETE_UNCONF = 2
    MAT_STEEL = 3

    # Section tags
    SEC_COL = 10
    SEC_BEAM = 20
    # Variable sections for tall buildings (N >= 8)
    SEC_COL_BOT = 11  # Stories 1-4
    SEC_COL_MID = 12  # Stories 5-7
    SEC_COL_TOP = 13  # Stories 8+

    # Transformation tags
    TRANSF_PDELTA = 1  # Columns (P-Delta)
    TRANSF_LINEAR = 2  # Beams (Linear)

    def __init__(self, config: ModelConfig | None = None) -> None:
        if not OPS_AVAILABLE:
            raise RuntimeError("OpenSeesPy is required. Install with: pip install openseespy")
        self.config = config or ModelConfig()
        self._is_built = False
        self._node_tags: list[int] = []
        self._element_tags: list[int] = []
        self._floor_nodes: dict[int, list[int]] = {}  # story -> [node_tags]
        self._periods: list[float] | None = None

    # ── Public API ─────────────────────────────────────────────────────

    def build(self) -> None:
        """Build the complete FE model from scratch."""
        logger.info(
            "Building %d-story %d-bay RC frame model...",
            self.config.frame.n_stories,
            self.config.frame.n_bays,
        )
        self._wipe_and_init()
        self._define_nodes()
        self._define_materials()
        self._define_sections()
        self._define_transformations()
        self._define_elements()
        self._assign_masses()
        self._is_built = True
        logger.info(
            "Model built: %d nodes, %d elements",
            len(self._node_tags),
            len(self._element_tags),
        )

    def apply_gravity(self) -> bool:
        """Apply gravity loads and run static analysis.

        Returns
        -------
        bool
            True if gravity analysis converged.
        """
        self._check_built()
        logger.info("Applying gravity loads...")

        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)

        frame = self.config.frame
        loads = self.config.loads

        # Distributed beam loads (UDL on each beam)
        for story in range(1, frame.n_stories + 1):
            udl = loads.beam_udl(story, frame.n_stories)
            for bay in range(frame.n_bays):
                # Beam element tags start after columns
                beam_tag = 2000 + story * 100 + bay
                # eleLoad for forceBeamColumn: Wy in local coords
                ops.eleLoad("-ele", beam_tag, "-type", "-beamUniform", -udl, 0.0)

        # Static analysis
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("BandGeneral")
        ops.test("NormDispIncr", 1.0e-8, 100)
        ops.algorithm("Newton")
        ops.integrator("LoadControl", 0.1)
        ops.analysis("Static")

        converged = True
        for step in range(10):
            ok = ops.analyze(1)
            if ok != 0:
                logger.warning("Gravity step %d failed, trying ModifiedNewton", step)
                ops.algorithm("ModifiedNewton")
                ok = ops.analyze(1)
                if ok != 0:
                    logger.error("Gravity analysis failed at step %d", step)
                    converged = False
                    break
                ops.algorithm("Newton")

        if converged:
            logger.info("Gravity analysis converged.")
            ops.loadConst("-time", 0.0)

        return converged

    def get_fundamental_period(self, n_modes: int = 5) -> float:
        """Run eigenvalue analysis and return T1.

        Parameters
        ----------
        n_modes : int
            Number of modes to compute.

        Returns
        -------
        float
            Fundamental period T1 (seconds).
        """
        self._check_built()
        eigenvalues = ops.eigen("-genBandArpack", n_modes)
        self._periods = [2.0 * math.pi / math.sqrt(ev) for ev in eigenvalues]
        logger.info("Modal periods: %s", [f"{T:.3f}s" for T in self._periods])
        return self._periods[0]

    def get_periods(self, n_modes: int = 5) -> list[float]:
        """Return all computed periods."""
        if self._periods is None:
            self.get_fundamental_period(n_modes)
        return self._periods  # type: ignore[return-value]

    def setup_rayleigh_damping(self) -> None:
        """Assign Rayleigh damping based on configured modes.

        Damping calibrated on modes 1 & 3 per standard practice for
        nonlinear time-history analysis of RC frames (Chopra, 2017;
        PEER TBI Guidelines, 2017; NIST GCR 10-917-8).  Higher modes
        will be overdamped (ξ_eff grows with ω), which is an accepted
        trade-off to avoid under-damping intermediate modes.
        """
        self._check_built()
        periods = self.get_periods()
        m1, m2 = self.config.damping_modes
        xi = self.config.damping_ratio

        w1 = 2.0 * math.pi / periods[m1 - 1]
        w2 = 2.0 * math.pi / periods[m2 - 1]

        # Rayleigh coefficients: a0 * M + a1 * K
        a0 = 2.0 * xi * w1 * w2 / (w1 + w2)
        a1 = 2.0 * xi / (w1 + w2)

        ops.rayleigh(a0, 0.0, 0.0, a1)
        logger.info(
            "Rayleigh damping: ξ=%.1f%%, modes %d&%d, a0=%.6f, a1=%.6f",
            xi * 100,
            m1,
            m2,
            a0,
            a1,
        )

    def get_floor_node_tags(self) -> dict[int, list[int]]:
        """Return mapping of story -> list of node tags at that level."""
        return self._floor_nodes

    def get_story_drift_recorders(self) -> list[tuple[int, int, int]]:
        """Return (story, top_node, bot_node) for IDR calculation."""
        frame = self.config.frame
        recorders = []
        for story in range(1, frame.n_stories + 1):
            top_node = story * 100  # leftmost column node
            bot_node = (story - 1) * 100
            recorders.append((story, top_node, bot_node))
        return recorders

    def reset(self) -> None:
        """Wipe the model completely."""
        if OPS_AVAILABLE:
            ops.wipe()
        self._is_built = False
        self._node_tags = []
        self._element_tags = []
        self._floor_nodes = {}
        self._periods = None

    # ── Internal construction methods ─────────────────────────────────

    def _check_built(self) -> None:
        if not self._is_built:
            raise RuntimeError("Model not built. Call build() first.")

    def _wipe_and_init(self) -> None:
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)

    def _define_nodes(self) -> None:
        frame = self.config.frame
        coords = frame.node_coords()

        for tag, x, y in coords:
            ops.node(tag, x, y)
            self._node_tags.append(tag)

            story = tag // 100
            if story not in self._floor_nodes:
                self._floor_nodes[story] = []
            self._floor_nodes[story].append(tag)

        # Fix base nodes (story 0)
        for tag in self._floor_nodes[0]:
            ops.fix(tag, 1, 1, 1)

        # Rigid diaphragm: constrain all nodes at each floor to have same
        # horizontal displacement (master = leftmost node)
        for story in range(1, frame.n_stories + 1):
            floor_tags = self._floor_nodes[story]
            master = floor_tags[0]  # Leftmost node
            for slave in floor_tags[1:]:
                ops.equalDOF(master, slave, 1)  # Tie horizontal DOF

        logger.info(
            "Nodes: %d created, %d fixed, floor diaphragms applied",
            len(self._node_tags),
            len(self._floor_nodes[0]),
        )

    def _define_materials(self) -> None:
        c = self.config.concrete
        s = self.config.steel
        sec = self.config.sections

        # ── Unit conversion ─────────────────────────────────────────────
        # Model uses m-kN-s unit system → stresses in kPa (kN/m²).
        # Material dataclasses store values in MPa for readability;
        # multiply by 1e3 to convert MPa → kPa.
        mpa = 1e3  # MPa → kPa conversion factor

        # Compute Mander confinement for columns (returns MPa)
        fcc, ecc, ecu_conf = compute_mander_confinement(
            fc=c.fc,
            col_width=sec.col_width,
            col_depth=sec.col_depth,
            cover=sec.col_cover,
            tie_dia=sec.tie_bar_dia,
            tie_spacing=sec.tie_spacing,
        )

        # ── Concrete02: confined ────────────────────────────────────────
        # Concrete02(matTag, fpc, epsc0, fpcu, epsU, lambda, ft, Ets)
        fpcu_conf = 0.2 * fcc  # Residual strength (20% of peak)
        ops.uniaxialMaterial(
            "Concrete02",
            self.MAT_CONCRETE_CONF,
            -fcc * mpa,  # fpc  (negative = compression, kPa)
            -ecc,  # epsc0 (strain, dimensionless)
            -fpcu_conf * mpa,  # fpcu (kPa)
            -ecu_conf,  # epsU  (strain, dimensionless)
            0.1,  # lambda (tension softening, dimensionless)
            c.ft * mpa,  # ft (tensile strength, kPa)
            c.Ec * mpa * c.ets,  # Ets (tension stiffening slope, kPa)
        )

        # ── Concrete02: unconfined (cover) ──────────────────────────────
        fpcu_unconf = 0.0  # Cover spalls off completely
        ops.uniaxialMaterial(
            "Concrete02",
            self.MAT_CONCRETE_UNCONF,
            -c.fc * mpa,  # fpc (kPa)
            -c.ec0,  # epsc0 (strain)
            -fpcu_unconf,  # fpcu  (0 kPa)
            -c.ecu_unconf,  # epsU  (strain)
            0.1,  # lambda (dimensionless)
            c.ft * mpa,  # ft (kPa)
            c.Ec * mpa * c.ets,  # Ets (kPa)
        )

        # ── Steel02: reinforcing bars ───────────────────────────────────
        # Steel02(matTag, Fy, E0, b, R0, cR1, cR2, a1, a2, a3, a4)
        ops.uniaxialMaterial(
            "Steel02",
            self.MAT_STEEL,
            s.fy * mpa,  # Fy (kPa)
            s.Es * mpa,  # E0 (kPa)
            s.b,  # b (strain hardening ratio, dimensionless)
            s.R0,  # R0  (dimensionless)
            s.cR1,  # cR1 (dimensionless)
            s.cR2,  # cR2 (dimensionless)
            s.a1,  # a1  (dimensionless)
            s.a2,  # a2  (dimensionless)
            s.a3,  # a3  (dimensionless)
            s.a4,  # a4  (dimensionless)
        )

        logger.info(
            "Materials defined: Concrete02 (conf f'cc=%.1f MPa, unconf f'c=%.1f MPa), "
            "Steel02 (fy=%.0f MPa)",
            fcc,
            c.fc,
            s.fy,
        )

    def _define_sections(self) -> None:
        """Define fiber sections for columns and beams."""
        sec = self.config.sections

        # ── Column fiber section ────────────────────────────────────────
        if self.config.frame.n_stories >= 8:
            # Variable sections for tall buildings
            logger.info("Building variable column sections for N=%d", self.config.frame.n_stories)

            # Bot (70x70)
            self._build_rc_fiber_section(
                sec_tag=self.SEC_COL_BOT,
                width=0.70,
                depth=0.70,
                cover=sec.col_cover,
                n_bars_top=5,
                n_bars_bot=5,
                n_bars_side=3,
                bar_dia=sec.col_bar_dia,
                label="Column (Bot 700)",
            )
            # Mid (60x60)
            self._build_rc_fiber_section(
                sec_tag=self.SEC_COL_MID,
                width=0.60,
                depth=0.60,
                cover=sec.col_cover,
                n_bars_top=4,
                n_bars_bot=4,
                n_bars_side=2,
                bar_dia=sec.col_bar_dia,
                label="Column (Mid 600)",
            )
            # Top (50x50 - same as default)
            self._build_rc_fiber_section(
                sec_tag=self.SEC_COL_TOP,
                width=sec.col_width,
                depth=sec.col_depth,
                cover=sec.col_cover,
                n_bars_top=sec.col_n_bars_top,
                n_bars_bot=sec.col_n_bars_bot,
                n_bars_side=sec.col_n_bars_side,
                bar_dia=sec.col_bar_dia,
                label="Column (Top 500)",
            )

            # Map default SEC_COL to Bot for compatibility if needed, though we'll use specific tags
            # We still build SEC_COL as a fallback or for beam integration registration
            self._build_rc_fiber_section(
                sec_tag=self.SEC_COL,
                width=sec.col_width,
                depth=sec.col_depth,
                cover=sec.col_cover,
                n_bars_top=sec.col_n_bars_top,
                n_bars_bot=sec.col_n_bars_bot,
                n_bars_side=sec.col_n_bars_side,
                bar_dia=sec.col_bar_dia,
                label="Column (Default)",
            )

        else:
            # Standard single section
            self._build_rc_fiber_section(
                sec_tag=self.SEC_COL,
                width=sec.col_width,
                depth=sec.col_depth,
                cover=sec.col_cover,
                n_bars_top=sec.col_n_bars_top,
                n_bars_bot=sec.col_n_bars_bot,
                n_bars_side=sec.col_n_bars_side,
                bar_dia=sec.col_bar_dia,
                label="Column",
            )

        # ── Beam fiber section ──────────────────────────────────────────
        self._build_rc_fiber_section(
            sec_tag=self.SEC_BEAM,
            width=sec.beam_width,
            depth=sec.beam_depth,
            cover=sec.beam_cover,
            n_bars_top=sec.beam_n_bars_top,
            n_bars_bot=sec.beam_n_bars_bot,
            n_bars_side=0,
            bar_dia=sec.beam_bar_dia,
            label="Beam",
        )

    def _build_rc_fiber_section(
        self,
        sec_tag: int,
        width: float,
        depth: float,
        cover: float,
        n_bars_top: int,
        n_bars_bot: int,
        n_bars_side: int,
        bar_dia: float,
        label: str = "",
    ) -> None:
        """Build a rectangular RC fiber section.

        Fiber layout:
            +--[unconfined cover]--+
            |  +-[confined core]-+ |
            |  | o   o   o   o  | |  ← top bars
            |  | o           o  | |  ← side bars
            |  | o   o   o      | |  ← bottom bars
            |  +----------------+ |
            +---------------------+
        """
        a_bar = math.pi * (bar_dia**2) / 4

        # Core dimensions
        core_width = width - 2 * cover
        core_depth = depth - 2 * cover

        # Fiber discretization
        n_fib_core_y = 16  # Fibers in depth (core)
        n_fib_core_z = 8  # Fibers in width (core)
        n_fib_cover_y = 4  # Cover fibers
        n_fib_cover_z = 1

        ops.section("Fiber", sec_tag)

        # ── Confined concrete core ──────────────────────────────────
        ops.patch(
            "rect",
            self.MAT_CONCRETE_CONF,
            n_fib_core_y,
            n_fib_core_z,
            -core_depth / 2,
            -core_width / 2,  # y1, z1 (bottom-left)
            core_depth / 2,
            core_width / 2,  # y2, z2 (top-right)
        )

        # ── Unconfined cover: top and bottom ────────────────────────
        # Top cover
        ops.patch(
            "rect",
            self.MAT_CONCRETE_UNCONF,
            n_fib_cover_y,
            n_fib_cover_z,
            core_depth / 2,
            -width / 2,
            depth / 2,
            width / 2,
        )
        # Bottom cover
        ops.patch(
            "rect",
            self.MAT_CONCRETE_UNCONF,
            n_fib_cover_y,
            n_fib_cover_z,
            -depth / 2,
            -width / 2,
            -core_depth / 2,
            width / 2,
        )
        # Left cover
        ops.patch(
            "rect",
            self.MAT_CONCRETE_UNCONF,
            n_fib_core_y,
            n_fib_cover_z,
            -core_depth / 2,
            -width / 2,
            core_depth / 2,
            -core_width / 2,
        )
        # Right cover
        ops.patch(
            "rect",
            self.MAT_CONCRETE_UNCONF,
            n_fib_core_y,
            n_fib_cover_z,
            -core_depth / 2,
            core_width / 2,
            core_depth / 2,
            width / 2,
        )

        # ── Reinforcing steel layers ────────────────────────────────
        # Top bars
        y_top = core_depth / 2 - bar_dia / 2
        ops.layer(
            "straight",
            self.MAT_STEEL,
            n_bars_top,
            a_bar,
            y_top,
            -core_width / 2 + bar_dia / 2,
            y_top,
            core_width / 2 - bar_dia / 2,
        )

        # Bottom bars
        y_bot = -core_depth / 2 + bar_dia / 2
        ops.layer(
            "straight",
            self.MAT_STEEL,
            n_bars_bot,
            a_bar,
            y_bot,
            -core_width / 2 + bar_dia / 2,
            y_bot,
            core_width / 2 - bar_dia / 2,
        )

        # Side bars (intermediate, columns only)
        if n_bars_side > 0:
            dy = (y_top - y_bot) / (n_bars_side + 1)
            for i in range(1, n_bars_side + 1):
                y_side = y_bot + i * dy
                # Left face
                ops.fiber(y_side, -core_width / 2 + bar_dia / 2, a_bar, self.MAT_STEEL)
                # Right face
                ops.fiber(y_side, core_width / 2 - bar_dia / 2, a_bar, self.MAT_STEEL)

        total_as = (n_bars_top + n_bars_bot + 2 * n_bars_side) * a_bar
        rho = total_as / (width * depth) * 100
        logger.info(
            "%s section (tag=%d): %.0fx%.0f mm, As=%.0f mm², ρ=%.2f%%",
            label,
            sec_tag,
            width * 1000,
            depth * 1000,
            total_as * 1e6,
            rho,
        )

    def _define_transformations(self) -> None:
        """P-Delta for columns, Linear for beams."""
        ops.geomTransf("PDelta", self.TRANSF_PDELTA)
        ops.geomTransf("Linear", self.TRANSF_LINEAR)

    def _define_elements(self) -> None:
        """Create forceBeamColumn elements for all columns and beams.

        Uses explicit beamIntegration objects (required in OpenSeesPy ≥3.5).
        Gauss-Lobatto integration provides optimal accuracy for force-based elements.
        """
        frame = self.config.frame
        n_ip = self.config.n_integration_pts

        # Define beam integration rules (Lobatto for accuracy at element ends)
        integ_col = 1
        integ_beam = 2
        # Default
        ops.beamIntegration("Lobatto", integ_col, self.SEC_COL, n_ip)
        ops.beamIntegration("Lobatto", integ_beam, self.SEC_BEAM, n_ip)

        # Variable sections
        integ_col_bot = 11
        integ_col_mid = 12
        integ_col_top = 13
        if frame.n_stories >= 8:
            ops.beamIntegration("Lobatto", integ_col_bot, self.SEC_COL_BOT, n_ip)
            ops.beamIntegration("Lobatto", integ_col_mid, self.SEC_COL_MID, n_ip)
            ops.beamIntegration("Lobatto", integ_col_top, self.SEC_COL_TOP, n_ip)

        # ── Columns ─────────────────────────────────────────────────────
        for story in range(1, frame.n_stories + 1):
            for bay in range(frame.n_bays + 1):
                bot_node = (story - 1) * 100 + bay
                top_node = story * 100 + bay
                tag = 1000 + story * 100 + bay

                # Select section based on story
                if frame.n_stories >= 8:
                    if story <= 4:
                        integ_tag = integ_col_bot
                    elif story <= 7:
                        integ_tag = integ_col_mid
                    else:
                        integ_tag = integ_col_top
                else:
                    integ_tag = integ_col

                ops.element(
                    "forceBeamColumn",
                    tag,
                    bot_node,
                    top_node,
                    self.TRANSF_PDELTA,
                    integ_tag,
                )
                self._element_tags.append(tag)

        # ── Beams ───────────────────────────────────────────────────────
        for story in range(1, frame.n_stories + 1):
            for bay in range(frame.n_bays):
                left_node = story * 100 + bay
                right_node = story * 100 + bay + 1
                tag = 2000 + story * 100 + bay

                ops.element(
                    "forceBeamColumn",
                    tag,
                    left_node,
                    right_node,
                    self.TRANSF_LINEAR,
                    integ_beam,
                )
                self._element_tags.append(tag)

        n_cols = (frame.n_stories) * (frame.n_bays + 1)
        n_beams = frame.n_stories * frame.n_bays
        logger.info("Elements: %d columns + %d beams = %d total", n_cols, n_beams, n_cols + n_beams)

    def _assign_masses(self) -> None:
        """Assign lumped masses at floor nodes from gravity loads."""
        frame = self.config.frame
        loads = self.config.loads
        g = 9.81  # m/s²

        for story in range(1, frame.n_stories + 1):
            nodes = self._floor_nodes[story]
            n_nodes = len(nodes)

            # Total floor weight
            udl = loads.beam_udl(story, frame.n_stories)
            floor_weight = udl * frame.total_width  # kN
            node_mass = floor_weight / g / n_nodes  # tonnes (kN→kg: /9.81)

            for tag in nodes:
                # Mass in X direction only (2D frame, horizontal excitation)
                ops.mass(tag, node_mass, 1.0e-9, 0.0)

        logger.info("Lumped masses assigned at %d floor levels", frame.n_stories)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience factory
# ═══════════════════════════════════════════════════════════════════════════


def build_default_model() -> RCFrameModel:
    """Build the default 5-story RC frame and return the model object.

    This is the primary entry point for other modules.
    """
    model = RCFrameModel()
    model.build()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# CLI verification
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("  5-Story 3-Bay RC Frame — Model Verification")
    print("=" * 60)

    model = RCFrameModel()
    model.build()

    # Eigenvalue analysis
    T1 = model.get_fundamental_period(n_modes=5)
    periods = model.get_periods()

    print(f"\n{'Mode':<8} {'Period (s)':<12} {'Frequency (Hz)':<15}")
    print("-" * 35)
    for i, T in enumerate(periods, 1):
        print(f"{i:<8} {T:<12.4f} {1 / T:<15.4f}")

    # Gravity analysis
    ok = model.apply_gravity()
    print(f"\nGravity analysis: {'CONVERGED ✅' if ok else 'FAILED ❌'}")

    # Summary
    config = model.config
    print(f"\n{'=' * 60}")
    print("  Model Summary")
    print(f"{'=' * 60}")
    print(f"  Stories:          {config.frame.n_stories}")
    print(f"  Bays:             {config.frame.n_bays}")
    print(f"  Total height:     {config.frame.total_height:.1f} m")
    print(f"  Total width:      {config.frame.total_width:.1f} m")
    print(f"  Concrete f'c:     {config.concrete.fc:.0f} MPa")
    print(f"  Steel fy:         {config.steel.fy:.0f} MPa")
    print(
        f"  Column:           {config.sections.col_width * 1000:.0f}x{config.sections.col_depth * 1000:.0f} mm"
    )
    print(
        f"  Beam:             {config.sections.beam_width * 1000:.0f}x{config.sections.beam_depth * 1000:.0f} mm"
    )
    print(f"  T1:               {T1:.4f} s")
    print(f"  Damping:          {config.damping_ratio * 100:.0f}% (modes {config.damping_modes})")
