"""Configuration management for collective encoder."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import warnings


class Config:
    """Configuration manager for collective encoder."""

    def __init__(self, config_path: Optional[str] = None, debug: bool = False):
        """Initialize configuration.

        Args:
            config_path: Path to YAML configuration file
            debug: Whether to enable debug mode
        """
        self._config = {}

        if config_path:
            self.load_from_file(config_path)

        if debug:
            self.enable_debug_mode()

    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r') as f:
            self._config = yaml.safe_load(f)

    def enable_debug_mode(self) -> None:
        """Enable debug mode with reduced parameters."""
        debug_overrides = {
            'debug': True,
            'nepochs': 2,
            'nexp': 0,
            'outpath': "train_runs",
            'outfolder': "debug",
            'overwrite': True,
            'wandb': False,
            'output_to_file': True,
            'nogpu': True,
            'export_latent': False,
        }

        debug_data_overrides = {
            'dataset_size': 50,
            'sequential': True,
            'train_size': 40,
            'batch_size': 4,
            'validation_size': 10,
            'val_batch_size': 4,
            'test_full_dataset': True,
            'num_workers': 1,
        }

        # Apply main overrides
        self._config.update(debug_overrides)

        # Apply data args overrides
        if 'data_args' not in self._config:
            self._config['data_args'] = {}
        self._config['data_args'].update(debug_data_overrides)

        print("Debug mode enabled")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()

    def validate(self) -> None:
        """Validate configuration parameters."""
        required_keys = ['network_name', 'data_name']

        for key in required_keys:
            if key not in self._config:
                raise ValueError(f"Required configuration key missing: {key}")

        # Validate network type
        valid_networks = [
            'AE', 'VAE', 'DVAE', 'EDVAE', 'EDVAEGAN',
            'GMVAE', 'VAEGAN', 'VAEGAN_mse', 'VAECGAN',
            'VAECGAN_mse', 'VAEC_mse', 'VAEC', 'BGE'
        ]

        if self._config['network_name'] not in valid_networks:
            raise ValueError(f"Unknown network type: {self._config['network_name']}")

        # Validate data type
        valid_data_types = ['KMC', 'COLVAR', 'MD17', 'XTC', 'XYZ']

        if self._config['data_name'] not in valid_data_types:
            raise ValueError(f"Unknown data type: {self._config['data_name']}")

    def __getitem__(self, key: str) -> Any:
        """Dict-like access to configuration."""
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like setting of configuration."""
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        """Dict-like containment check."""
        return key in self._config