from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from ..backtest.engine import BacktestParams


@dataclass(frozen=True)
class HysteresisCooldownPreset:
    """Hysteresis and Cooldown preset values."""
    
    name: str
    tp_hysteresis: float  # As decimal (e.g., 0.05 for 5%)
    sl_hysteresis: float  # As decimal (e.g., 0.05 for 5%)
    tp_cooldown_days: int
    sl_cooldown_days: int


# Preset definitions
CONSERVATIVE_PRESET = HysteresisCooldownPreset(
    name="Conservative",
    tp_hysteresis=0.05,  # 5%
    sl_hysteresis=0.05,   # 5%
    tp_cooldown_days=5,
    sl_cooldown_days=3,
)

MODERATE_PRESET = HysteresisCooldownPreset(
    name="Moderate",
    tp_hysteresis=0.03,  # 3%
    sl_hysteresis=0.03,   # 3%
    tp_cooldown_days=3,
    sl_cooldown_days=2,
)

AGGRESSIVE_PRESET = HysteresisCooldownPreset(
    name="Aggressive",
    tp_hysteresis=0.01,  # 1%
    sl_hysteresis=0.01,   # 1%
    tp_cooldown_days=1,
    sl_cooldown_days=1,
)

# All presets
ALL_PRESETS = {
    "Conservative": CONSERVATIVE_PRESET,
    "Moderate": MODERATE_PRESET,
    "Aggressive": AGGRESSIVE_PRESET,
}


class PresetManager:
    """Manage saved backtest parameter presets."""
    
    def __init__(self, presets_dir: Optional[Path] = None) -> None:
        """Initialize preset manager.
        
        Args:
            presets_dir: Directory for preset files. Defaults to .presets/ in project root.
        """
        if presets_dir is None:
            # Default to .presets/ in project root
            project_root = Path(__file__).parent.parent.parent
            presets_dir = project_root / ".presets"
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def _preset_path(self, name: str) -> Path:
        """Get preset file path."""
        # Sanitize name for filename
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        return self.presets_dir / f"{safe_name}.json"

    def save(self, name: str, params: BacktestParams) -> None:
        """Save a preset.
        
        Args:
            name: Preset name.
            params: BacktestParams to save.
        """
        preset_dict = asdict(params)
        preset_path = self._preset_path(name)
        
        with open(preset_path, "w", encoding="utf-8") as f:
            json.dump(preset_dict, f, indent=2, default=str)

    def load(self, name: str) -> Optional[BacktestParams]:
        """Load a preset.
        
        Args:
            name: Preset name.
            
        Returns:
            BacktestParams if found, None otherwise.
        """
        preset_path = self._preset_path(name)
        
        if not preset_path.exists():
            return None
            
        try:
            with open(preset_path, "r", encoding="utf-8") as f:
                preset_dict = json.load(f)
            
            # Convert string dates back to date objects if needed
            # BacktestParams doesn't have dates, so this is mainly for future compatibility
            
            return BacktestParams(**preset_dict)
        except Exception:
            return None

    def list_presets(self) -> List[str]:
        """List all saved preset names.
        
        Returns:
            List of preset names (without .json extension).
        """
        if not self.presets_dir.exists():
            return []
            
        presets = []
        for preset_file in self.presets_dir.glob("*.json"):
            # Extract name from filename (remove .json extension)
            name = preset_file.stem
            presets.append(name)
            
        return sorted(presets)

    def delete(self, name: str) -> bool:
        """Delete a preset.
        
        Args:
            name: Preset name.
            
        Returns:
            True if deleted, False if not found.
        """
        preset_path = self._preset_path(name)
        
        if not preset_path.exists():
            return False
            
        try:
            preset_path.unlink()
            return True
        except Exception:
            return False


# Global preset manager instance
_preset_manager: Optional[PresetManager] = None


def get_preset_manager() -> PresetManager:
    """Get global preset manager instance."""
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = PresetManager()
    return _preset_manager


