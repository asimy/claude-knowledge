"""Configuration management for the knowledge base."""

import json
from pathlib import Path
from typing import Any


class ConfigManager:
    """Manages configuration for the knowledge base.

    This service handles loading and saving configuration files,
    including sync path and sync state management.
    """

    def __init__(self, base_path: Path) -> None:
        """Initialize the configuration manager.

        Args:
            base_path: Base directory for knowledge base storage.
        """
        self.base_path = base_path
        self.config_path = base_path / "config.json"

    def load_config(self) -> dict[str, Any]:
        """Load configuration from config.json.

        Returns:
            Configuration dictionary.
        """
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def save_config(self, config: dict[str, Any]) -> None:
        """Save configuration to config.json.

        Args:
            config: Configuration dictionary to save.
        """
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def get_sync_path(self) -> Path | None:
        """Get the saved sync path from config.

        Returns:
            Path to sync directory, or None if not configured.
        """
        config = self.load_config()
        sync_path = config.get("sync_path")
        if sync_path:
            return Path(sync_path).expanduser()
        return None

    def set_sync_path(self, path: Path) -> None:
        """Save the sync path to config.

        Args:
            path: Path to sync directory.
        """
        config = self.load_config()
        config["sync_path"] = str(path)
        self.save_config(config)

    def get_local_sync_state(self) -> dict[str, dict[str, Any]]:
        """Get the local record of what was last synced.

        Returns:
            Dictionary mapping entry ID to {content_hash, updated_at}.
        """
        config = self.load_config()
        return config.get("sync_state", {})

    def set_local_sync_state(self, state: dict[str, dict[str, Any]]) -> None:
        """Save the local record of what was last synced.

        Args:
            state: Dictionary mapping entry ID to sync state.
        """
        config = self.load_config()
        config["sync_state"] = state
        self.save_config(config)
