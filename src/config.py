"""
config.py — GlobalConfig: Single source of truth for the Hybrid Digital Twin
=============================================================================

Centralizes structural and analysis parameters (n_stories, n_bays, seq_len, dt)
so that data_factory.py, pipeline.py, and train.py always operate on consistent
values.  When the factory finishes a simulation campaign, it persists this
config to the output directory.  Downstream scripts load it to validate that
they are operating on matching parameters, preventing silent shape-mismatch
errors.

Usage::

    # In data_factory.py: save after simulation campaign
    from src.config import GlobalConfig
    GlobalConfig(n_stories=10, n_bays=3).save("data/raw")

    # In pipeline.py: load & validate
    cfg = GlobalConfig.load("data/raw")
    assert cfg.n_stories == args.n_stories

    # In train.py: auto-detect from processed data
    cfg = GlobalConfig.from_processed_dir("data/processed")

Author: Mikisbell
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "global_config.json"


@dataclass
class GlobalConfig:
    """Single source of truth for structural and analysis parameters.

    Saved alongside simulation outputs so that all downstream scripts
    (pipeline, train) can validate they are operating on consistent data.
    """

    # ------------------------------------------------------------------ #
    # Structural parameters
    # ------------------------------------------------------------------ #
    n_stories: int = 5
    n_bays: int = 3
    story_height_first: float = 3.5  # m (ground floor)
    story_height_typical: float = 3.0  # m (upper floors)

    # ------------------------------------------------------------------ #
    # Analysis parameters
    # ------------------------------------------------------------------ #
    dt: float = 0.01  # s  (time step, must match NLTHA output)
    seq_len: int = 2048  # samples (~20.5 s at dt=0.01 s)

    # ------------------------------------------------------------------ #
    # Data paths (relative to workspace root)
    # ------------------------------------------------------------------ #
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, directory: Path | str) -> None:
        """Serialize config to *directory*/global_config.json."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        config_file = directory / _CONFIG_FILENAME
        with open(config_file, "w") as f:
            json.dump(asdict(self), f, indent=4)
        logger.info("GlobalConfig saved → %s", config_file)

    @classmethod
    def load(cls, directory: Path | str) -> GlobalConfig:
        """Load config from *directory*/global_config.json.

        Parameters
        ----------
        directory:
            Path to a directory containing ``global_config.json``, or directly
            to the JSON file itself.

        Raises
        ------
        FileNotFoundError
            If the config file does not exist in the given directory.
        """
        path = Path(directory)
        config_file = path if path.suffix == ".json" else path / _CONFIG_FILENAME
        if not config_file.exists():
            raise FileNotFoundError(
                f"global_config.json not found in '{directory}'. "
                "Run data_factory.py first to generate and save the config."
            )
        with open(config_file) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_processed_dir(cls, processed_dir: str | Path) -> GlobalConfig:
        """Auto-detect config from a processed data directory.

        Looks for ``global_config.json`` inside ``processed_dir``.
        """
        return cls.load(processed_dir)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Raise ``ValueError`` if parameters are out of valid range."""
        if self.n_stories < 1:
            raise ValueError(f"n_stories must be ≥ 1, got {self.n_stories}")
        if self.n_bays < 1:
            raise ValueError(f"n_bays must be ≥ 1, got {self.n_bays}")
        if self.seq_len < 128:
            raise ValueError(
                f"seq_len={self.seq_len} is too short for seismic analysis (minimum 128 samples)"
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
