from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

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
    tp_hysteresis=0.025,  # 2.5%
    sl_hysteresis=0.015,   # 1.5%
    tp_cooldown_days=3,
    sl_cooldown_days=5,
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
    
    def __init__(self, presets_dir: Path | None = None) -> None:
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

    def save(self, name: str, params: BacktestParams, start_date: date | None = None, end_date: date | None = None) -> None:
        """Save a preset.
        
        Args:
            name: Preset name.
            params: BacktestParams to save.
            start_date: Optional start date for backtest.
            end_date: Optional end date for backtest.
        """
        preset_dict = asdict(params)
        # Add start/end dates if provided
        if start_date:
            preset_dict['start_date'] = start_date.isoformat()
        if end_date:
            preset_dict['end_date'] = end_date.isoformat()
        preset_path = self._preset_path(name)
        
        with open(preset_path, "w", encoding="utf-8") as f:
            json.dump(preset_dict, f, indent=2, default=str)

    def load(self, name: str) -> tuple[BacktestParams | None, date | None, date | None]:
        """Load a preset.
        
        Args:
            name: Preset name.
            
        Returns:
            Tuple of (BacktestParams, start_date, end_date). Returns (None, None, None) if not found.
        """
        preset_path = self._preset_path(name)
        
        if not preset_path.exists():
            return None, None, None
            
        try:
            with open(preset_path, encoding="utf-8") as f:
                preset_dict = json.load(f)
            
            # Extract start/end dates if present
            start_date = None
            end_date = None
            if 'start_date' in preset_dict:
                start_date = date.fromisoformat(preset_dict['start_date'])
                del preset_dict['start_date']  # Remove from dict before creating BacktestParams
            if 'end_date' in preset_dict:
                end_date = date.fromisoformat(preset_dict['end_date'])
                del preset_dict['end_date']  # Remove from dict before creating BacktestParams
            
            # Validate required fields before creating BacktestParams
            if 'threshold' not in preset_dict:
                return None, None, None
            
            # Ensure shares_per_signal is set (budget-based mode no longer supported)
            # If preset has weekly_budget but no shares_per_signal, skip it
            if 'shares_per_signal' not in preset_dict or preset_dict['shares_per_signal'] is None:
                # Try to convert from budget-based preset (if exists)
                # For now, skip budget-based presets
                if 'weekly_budget' in preset_dict and preset_dict['weekly_budget'] is not None:
                    # Budget-based preset - skip it (no longer supported)
                    return None, None, None
                # No shares_per_signal and no weekly_budget - invalid preset
                return None, None, None
            
            # Ensure enable_tp_sl is boolean
            if 'enable_tp_sl' in preset_dict:
                preset_dict['enable_tp_sl'] = bool(preset_dict['enable_tp_sl'])
            
            # Remove budget-related fields if present (for backward compatibility)
            preset_dict.pop('weekly_budget', None)
            preset_dict.pop('mode', None)
            preset_dict.pop('carryover', None)
            
            params = BacktestParams(**preset_dict)
            return params, start_date, end_date
        except (ValueError, TypeError, KeyError):
            # Return None on validation errors
            return None, None, None

    def list_presets(self) -> list[str]:
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
_preset_manager: PresetManager | None = None


def get_preset_manager() -> PresetManager:
    """Get global preset manager instance."""
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = PresetManager()
    return _preset_manager


