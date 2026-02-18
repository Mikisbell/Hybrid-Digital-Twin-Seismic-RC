"""
Utility package for the Hybrid Digital Twin Framework.

Modules:
    sync_results   - Notion API integration for automatic research logging
    figure_manager - HRPUB-compliant figure saving and caption management
"""

from src.utils.figure_manager import FigureManager
from src.utils.sync_results import NotionResearchLogger

__all__ = ["NotionResearchLogger", "FigureManager"]
