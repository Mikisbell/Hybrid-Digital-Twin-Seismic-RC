"""
Notion Sync Engine for Hybrid Digital Twin Simulations
=======================================================

Automatically logs OpenSeesPy simulation results to a Notion database.
This module is the bridge between the Simulation Layer and the Research
Documentation Layer, ensuring every data point is traceable (HRPUB compliance).

Usage:
    from src.utils.sync_results import NotionResearchLogger

    logger = NotionResearchLogger()
    logger.log_simulation(
        ground_motion="RSN953_Northridge",
        max_drift=0.0234,
        peak_acceleration=0.82,
        convergence_status="Converged",
        num_stories=5,
        notes="ACI 318-19 design, Rayleigh damping 5%"
    )

Environment Variables Required:
    NOTION_TOKEN       - Notion internal integration token
    NOTION_DATABASE_ID - Target database ID (32-char hex)

Author: Mikisbell
Project: Hybrid Digital Twin for Seismic RC Buildings
"""

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard import: notion-client is optional at module level
# ---------------------------------------------------------------------------
try:
    from notion_client import Client as NotionClient

    NOTION_AVAILABLE = True
except ImportError:
    NOTION_AVAILABLE = False
    logger.warning("notion-client not installed. Run: pip install notion-client")


class NotionResearchLogger:
    """
    Logs simulation results and research milestones to a Notion database.

    The database should have the following properties (columns):
        - Movimiento Sísmico  (Title)
        - Deriva Máxima      (Number, format: percent)
        - PGA (g)            (Number, unit: g)
        - Pisos              (Number)
        - Estado             (Select: Convergió / Divergió / En ejecución / En cola)
        - Fase               (Select: Métodos / Resultados / Validación)
        - Date            (Date)
        - Notas              (Rich Text)
        - Referencia         (Rich Text)  ← For HRPUB citation tracing
    """

    def __init__(self, token: str | None = None, database_id: str | None = None):
        if not NOTION_AVAILABLE:
            raise ImportError("notion-client is required. Install with: pip install notion-client")

        self.token = token or os.environ.get("NOTION_TOKEN")
        self.database_id = database_id or os.environ.get("NOTION_DATABASE_ID")

        if not self.token:
            raise ValueError(
                "NOTION_TOKEN not found. Set it as an environment variable or pass it directly."
            )
        if not self.database_id:
            raise ValueError(
                "NOTION_DATABASE_ID not found. Set it as an environment variable or pass it directly."
            )

        self.client = NotionClient(auth=self.token)
        logger.info("NotionResearchLogger initialized successfully.")

    # -----------------------------------------------------------------------
    # Core: Log a single simulation run
    # -----------------------------------------------------------------------
    def log_simulation(
        self,
        ground_motion: str,
        max_drift: float,
        peak_acceleration: float = 0.0,
        convergence_status: str = "Converged",
        num_stories: int = 5,
        phase: str = "Methods",
        notes: str = "",
        source_ref: str = "",
    ) -> dict:
        """
        Log one NLTHA simulation result to the Notion database.

        Parameters
        ----------
        ground_motion : str
            Name/ID of the ground motion record (e.g., "RSN953_Northridge").
        max_drift : float
            Maximum inter-story drift ratio (e.g., 0.023 = 2.3%).
        peak_acceleration : float
            Peak ground acceleration in g.
        convergence_status : str
            "Converged", "Diverged", or "Running".
        num_stories : int
            Number of stories in the RC model.
        phase : str
            HRPUB manuscript phase: "Methods", "Results", "Validation".
        notes : str
            Free-form notes about the simulation.
        source_ref : str
            HRPUB-compatible citation reference (e.g., "[cite: 15]").

        Returns
        -------
        dict
            Notion API response.
        """
        # Map convergence status to Spanish DB values
        status_map = {
            "Converged": "Convergi\u00f3",
            "Diverged": "Divergi\u00f3",
            "Running": "En ejecuci\u00f3n",
            "Queued": "En cola",
        }
        # Map phase to Spanish DB values
        phase_map = {
            "Methods": "M\u00e9todos",
            "Results": "Resultados",
            "Validation": "Validaci\u00f3n",
        }

        properties = {
            "Movimiento S\u00edsmico": {"title": [{"text": {"content": ground_motion}}]},
            "Deriva M\u00e1xima": {"number": round(max_drift, 6)},
            "PGA (g)": {"number": round(peak_acceleration, 4)},
            "Pisos": {"number": num_stories},
            "Estado": {"select": {"name": status_map.get(convergence_status, convergence_status)}},
            "Fase": {"select": {"name": phase_map.get(phase, phase)}},
            "Fecha": {"date": {"start": datetime.now(timezone.utc).isoformat()}},
            "Notas": {
                "rich_text": [{"text": {"content": notes[:2000]}}]  # Notion limit
            },
        }

        # Only add Referencia if provided (keeps the table clean)
        if source_ref:
            properties["Referencia"] = {"rich_text": [{"text": {"content": source_ref}}]}

        try:
            response = self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=properties,
            )
            logger.info(
                f"✅ Logged to Notion: {ground_motion} | "
                f"Drift={max_drift:.4f} | Status={convergence_status}"
            )
            return response
        except Exception as e:
            logger.error(f"❌ Notion sync failed: {e}")
            raise

    # -----------------------------------------------------------------------
    # Batch: Log multiple simulation results at once
    # -----------------------------------------------------------------------
    def log_batch(self, results: list[dict]) -> list[dict]:
        """
        Log a batch of simulation results.

        Parameters
        ----------
        results : list[dict]
            Each dict should have keys matching log_simulation() params.

        Returns
        -------
        list[dict]
            List of Notion API responses.
        """
        responses = []
        for i, result in enumerate(results):
            try:
                resp = self.log_simulation(**result)
                responses.append(resp)
            except Exception as e:
                logger.error(f"Failed on result {i}: {e}")
                responses.append({"error": str(e), "index": i})
        logger.info(f"Batch complete: {len(responses)}/{len(results)} logged.")
        return responses

    # -----------------------------------------------------------------------
    # Utility: Update status of an existing entry
    # -----------------------------------------------------------------------
    def update_status(self, page_id: str, new_status: str) -> dict:
        """Update the Estado property of an existing Notion page."""
        return self.client.pages.update(
            page_id=page_id,
            properties={"Estado": {"select": {"name": new_status}}},
        )

    # -----------------------------------------------------------------------
    # Utility: Query the database for existing records
    # -----------------------------------------------------------------------
    def query_simulations(self, status_filter: str | None = None) -> list:
        """
        Query simulation records, optionally filtering by status.

        Uses ``client.search()`` (notion-client v3 compatible) since
        ``databases.query()`` was removed in v3.

        Parameters
        ----------
        status_filter : str, optional
            Filter by "Convergi\u00f3", "Divergi\u00f3", etc.

        Returns
        -------
        list
            List of Notion page objects.
        """
        pages: list = []
        search_results = self.client.search(
            query="", filter={"value": "page", "property": "object"}
        )
        for p in search_results.get("results", []):
            parent = p.get("parent", {})
            db_id = parent.get("database_id", "")
            if self.database_id not in db_id:
                continue
            if status_filter:
                props = p.get("properties", {})
                estado = props.get("Estado", {})
                sel = estado.get("select", {}) or {}
                if sel.get("name") != status_filter:
                    continue
            pages.append(p)
        return pages


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Log a simulation result to Notion")
    parser.add_argument("--gm", required=True, help="Ground motion name")
    parser.add_argument("--drift", type=float, required=True, help="Max drift ratio")
    parser.add_argument("--pga", type=float, default=0.0, help="PGA in g")
    parser.add_argument("--status", default="Converged", help="Convergence status")
    parser.add_argument("--notes", default="", help="Additional notes")
    args = parser.parse_args()

    log = NotionResearchLogger()
    log.log_simulation(
        ground_motion=args.gm,
        max_drift=args.drift,
        peak_acceleration=args.pga,
        convergence_status=args.status,
        notes=args.notes,
    )
    print("Done.")
