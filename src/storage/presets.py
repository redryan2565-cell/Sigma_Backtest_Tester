from __future__ import annotations

import json
import os
import platform
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


@dataclass(frozen=True)
class UniversalPreset:
    """Universal preset for quick setup with ticker and parameters.
    
    These presets are available across all sessions and devices.
    """
    name: str  # "TQQQ", "SOXL", "QLD"
    ticker: str
    params: BacktestParams
    start_date: date  # Fixed start date
    end_date: date | None  # None means use current date


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

# Universal presets (available across all sessions)
UNIVERSAL_PRESETS: dict[str, UniversalPreset] = {}

# Initialize universal presets
def _init_universal_presets() -> None:
    """Initialize universal presets with predefined values."""
    global UNIVERSAL_PRESETS
    
    # Common settings
    common_start_date = date(2020, 1, 1)
    common_shares = 1.0
    common_tp_sl_enabled = False
    
    # TQQQ preset
    tqqq_params = BacktestParams(
        threshold=-0.041,  # -4.1%
        shares_per_signal=common_shares,
        fee_rate=0.008,  # 0.8%
        slippage_rate=0.0013,  # 0.13%
        enable_tp_sl=common_tp_sl_enabled,
        tp_threshold=None,
        sl_threshold=None,
        tp_sell_percentage=1.0,
        sl_sell_percentage=1.0,
        reset_baseline_after_tp_sl=True,
        tp_hysteresis=0.0,
        sl_hysteresis=0.0,
        tp_cooldown_days=0,
        sl_cooldown_days=0,
    )
    UNIVERSAL_PRESETS["TQQQ"] = UniversalPreset(
        name="TQQQ",
        ticker="TQQQ",
        params=tqqq_params,
        start_date=common_start_date,
        end_date=None,  # Use current date
    )
    
    # SOXL preset
    soxl_params = BacktestParams(
        threshold=-0.072,  # -7.2%
        shares_per_signal=common_shares,
        fee_rate=0.0075,  # 0.75%
        slippage_rate=-0.042,  # -4.2% (as specified by user)
        enable_tp_sl=common_tp_sl_enabled,
        tp_threshold=None,
        sl_threshold=None,
        tp_sell_percentage=1.0,
        sl_sell_percentage=1.0,
        reset_baseline_after_tp_sl=True,
        tp_hysteresis=0.0,
        sl_hysteresis=0.0,
        tp_cooldown_days=0,
        sl_cooldown_days=0,
    )
    UNIVERSAL_PRESETS["SOXL"] = UniversalPreset(
        name="SOXL",
        ticker="SOXL",
        params=soxl_params,
        start_date=common_start_date,
        end_date=None,  # Use current date
    )
    
    # QLD preset
    qld_params = BacktestParams(
        threshold=-0.025,  # -2.5%
        shares_per_signal=common_shares,
        fee_rate=0.0095,  # 0.95%
        slippage_rate=-0.0089,  # -0.89% (as specified by user)
        enable_tp_sl=common_tp_sl_enabled,
        tp_threshold=None,
        sl_threshold=None,
        tp_sell_percentage=1.0,
        sl_sell_percentage=1.0,
        reset_baseline_after_tp_sl=True,
        tp_hysteresis=0.0,
        sl_hysteresis=0.0,
        tp_cooldown_days=0,
        sl_cooldown_days=0,
    )
    UNIVERSAL_PRESETS["QLD"] = UniversalPreset(
        name="QLD",
        ticker="QLD",
        params=qld_params,
        start_date=common_start_date,
        end_date=None,  # Use current date
    )

# Initialize universal presets on module load
_init_universal_presets()


def get_universal_preset(name: str) -> UniversalPreset | None:
    """Get a universal preset by name.
    
    Args:
        name: Preset name ("TQQQ", "SOXL", or "QLD")
        
    Returns:
        UniversalPreset if found, None otherwise.
    """
    return UNIVERSAL_PRESETS.get(name.upper())


def _is_streamlit_cloud() -> bool:
    """Check if running on Streamlit Cloud.

    Returns:
        True if running on Streamlit Cloud, False otherwise.
    """
    # Check for Streamlit Cloud environment variables
    if os.getenv('STREAMLIT_SERVER_ENV') == 'cloud':
        return True
    # Check for Streamlit Cloud mount path
    if os.path.exists('/mount/src'):
        return True
    # Check for Streamlit Cloud specific paths
    if os.path.exists('/home/appuser') or os.path.exists('/home/adminuser'):
        return True
    return False


def _get_default_presets_dir() -> Path:
    """Get default presets directory based on OS.

    Returns:
        Path to user-specific presets directory:
        - Windows: %APPDATA%\\normal-dip-bt\\presets\\
        - Linux: ~/.config/normal-dip-bt/presets/
        - Mac: ~/Library/Application Support/normal-dip-bt/presets/
    """
    system = platform.system()
    home = Path.home()

    if system == "Windows":
        # Use APPDATA if available, otherwise use USERPROFILE
        appdata = os.getenv("APPDATA")
        if appdata:
            base_dir = Path(appdata)
        else:
            base_dir = home
        presets_dir = base_dir / "normal-dip-bt" / "presets"
    elif system == "Darwin":  # macOS
        presets_dir = home / "Library" / "Application Support" / "normal-dip-bt" / "presets"
    else:  # Linux and other Unix-like systems
        presets_dir = home / ".config" / "normal-dip-bt" / "presets"

    return presets_dir


class SessionPresetManager:
    """Preset manager using Streamlit session_state (user-specific).

    This manager stores presets in the browser session, ensuring each user
    has their own isolated preset storage. Suitable for Streamlit Cloud deployment.
    """

    def __init__(self) -> None:
        """Initialize session-based preset manager."""
        try:
            import streamlit as st
            if 'user_presets' not in st.session_state:
                st.session_state['user_presets'] = {}
        except ImportError:
            # If streamlit is not available, use a fallback dict
            # This should not happen in normal usage
            self._fallback_storage: dict[str, dict] = {}
            if not hasattr(self, '_fallback_storage'):
                self._fallback_storage = {}

    def _get_storage(self) -> dict:
        """Get the storage dictionary (session_state or fallback)."""
        try:
            import streamlit as st
            if 'user_presets' not in st.session_state:
                st.session_state['user_presets'] = {}
            return st.session_state['user_presets']
        except ImportError:
            if not hasattr(self, '_fallback_storage'):
                self._fallback_storage = {}
            return self._fallback_storage

    def save(self, name: str, params: BacktestParams, start_date: date | None = None, end_date: date | None = None) -> None:
        """Save a preset.

        Args:
            name: Preset name.
            params: BacktestParams to save.
            start_date: Optional start date for backtest.
            end_date: Optional end date for backtest.
        """
        storage = self._get_storage()
        storage[name] = {
            'params': asdict(params),
            'start_date': start_date.isoformat() if start_date else None,
            'end_date': end_date.isoformat() if end_date else None
        }

    def load(self, name: str) -> tuple[BacktestParams | None, date | None, date | None]:
        """Load a preset.

        Args:
            name: Preset name.

        Returns:
            Tuple of (BacktestParams, start_date, end_date). Returns (None, None, None) if not found.
        """
        storage = self._get_storage()
        preset_data = storage.get(name)

        if not preset_data:
            return None, None, None

        try:
            # Extract start/end dates
            start_date = None
            end_date = None
            if preset_data.get('start_date'):
                start_date = date.fromisoformat(preset_data['start_date'])
            if preset_data.get('end_date'):
                end_date = date.fromisoformat(preset_data['end_date'])

            # Get params dict
            params_dict = preset_data.get('params', {})
            if not params_dict or 'threshold' not in params_dict:
                return None, None, None

            # Ensure shares_per_signal is set
            if 'shares_per_signal' not in params_dict or params_dict['shares_per_signal'] is None:
                if 'weekly_budget' in params_dict and params_dict['weekly_budget'] is not None:
                    # Budget-based preset - skip it (no longer supported)
                    return None, None, None
                # No shares_per_signal and no weekly_budget - invalid preset
                return None, None, None

            # Ensure enable_tp_sl is boolean
            if 'enable_tp_sl' in params_dict:
                params_dict['enable_tp_sl'] = bool(params_dict['enable_tp_sl'])

            # Remove budget-related fields if present (for backward compatibility)
            params_dict.pop('weekly_budget', None)
            params_dict.pop('mode', None)
            params_dict.pop('carryover', None)

            params = BacktestParams(**params_dict)
            return params, start_date, end_date
        except (ValueError, TypeError, KeyError):
            # Return None on validation errors
            return None, None, None

    def list_presets(self) -> list[str]:
        """List all saved preset names.

        Returns:
            List of preset names.
        """
        storage = self._get_storage()
        return sorted(list(storage.keys()))

    def delete(self, name: str) -> bool:
        """Delete a preset.

        Args:
            name: Preset name.

        Returns:
            True if deleted, False if not found.
        """
        storage = self._get_storage()
        if name in storage:
            del storage[name]
            return True
        return False

    def export_all(self) -> dict:
        """Export all presets as dictionary for download.

        Returns:
            Dictionary containing all presets.
        """
        storage = self._get_storage()
        return dict(storage)

    def import_all(self, presets_dict: dict) -> None:
        """Import presets from dictionary (from upload).

        Args:
            presets_dict: Dictionary containing presets to import.
        """
        storage = self._get_storage()
        for name, data in presets_dict.items():
            storage[name] = data


class PresetManager:
    """Manage saved backtest parameter presets (file-based).

    This manager stores presets in the local file system. Suitable for local development.
    """

    def __init__(self, presets_dir: Path | None = None) -> None:
        """Initialize preset manager.

        Args:
            presets_dir: Directory for preset files. Defaults to user-specific local directory.
        """
        if presets_dir is None:
            presets_dir = _get_default_presets_dir()
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

    def export_all(self) -> dict:
        """Export all presets as dictionary for download.

        Returns:
            Dictionary containing all presets.
        """
        all_presets = {}
        for preset_name in self.list_presets():
            params, start_date, end_date = self.load(preset_name)
            if params:
                all_presets[preset_name] = {
                    'params': asdict(params),
                    'start_date': start_date.isoformat() if start_date else None,
                    'end_date': end_date.isoformat() if end_date else None
                }
        return all_presets

    def import_all(self, presets_dict: dict) -> None:
        """Import presets from dictionary (from upload).

        Args:
            presets_dict: Dictionary containing presets to import.
        """
        for name, data in presets_dict.items():
            try:
                # Extract params and dates
                params_dict = data.get('params', {})
                start_date_str = data.get('start_date')
                end_date_str = data.get('end_date')

                # Convert dates
                start_date = date.fromisoformat(start_date_str) if start_date_str else None
                end_date = date.fromisoformat(end_date_str) if end_date_str else None

                # Create BacktestParams
                if 'threshold' in params_dict:
                    # Remove budget-related fields
                    params_dict.pop('weekly_budget', None)
                    params_dict.pop('mode', None)
                    params_dict.pop('carryover', None)
                    params = BacktestParams(**params_dict)
                    self.save(name, params, start_date, end_date)
            except (ValueError, TypeError, KeyError):
                # Skip invalid presets
                continue


# Global preset manager instance
_preset_manager: PresetManager | SessionPresetManager | None = None


def get_preset_manager() -> PresetManager | SessionPresetManager:
    """Get global preset manager instance.

    Always returns SessionPresetManager to ensure session-based storage.
    This ensures presets are isolated per browser session and not shared across devices.

    Returns:
        SessionPresetManager instance.
    """
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = SessionPresetManager()
    return _preset_manager


def reset_preset_manager() -> None:
    """Reset global preset manager instance (for cache invalidation).

    Note: SessionPresetManager doesn't need reset as it uses session_state directly.
    This function is kept for backward compatibility but has no effect with SessionPresetManager.
    """
    global _preset_manager
    # SessionPresetManager uses session_state directly, so no reset needed
    # This function is kept for backward compatibility
    pass

