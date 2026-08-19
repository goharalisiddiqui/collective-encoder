"""Robust configuration and dictionary update resolver for hyperparameter optimization."""

import copy
import re
from typing import Any, Dict, List, Optional, Tuple, Union


class ConfigResolver:
    """Resolves nested dictionary overrides, dot-notation paths, list indexing,
    and domain-specific architectural transformations for collective_encoder.
    """

    INDEX_REGEX = re.compile(r"^(.+)\[(-?\d+)\]$")
    DOMAIN_ALIASES = {
        "latent_dim",
        "bottleneck_dim",
        "hidden_layers",
        "beta",
        "kld_max_value",
        "lr",
        "learning_rate",
        "lrate",
        "weight_decay",
        "batch_size",
        "val_batch_size",
    }

    @classmethod
    def apply_overrides(
        cls,
        base_config: Dict[str, Any],
        overrides: Dict[str, Any],
        strict: bool = False,
    ) -> Dict[str, Any]:
        """Applies overrides onto a deep copy of base_config.

        Args:
            base_config: The baseline configuration dictionary.
            overrides: Dictionary of overrides (supports dot-notation, list indices, domain aliases).
            strict: If True, raises an error if an intermediate key does not exist.

        Returns:
            A new dictionary containing the resolved configuration.
        """
        config = copy.deepcopy(base_config)

        # 1. Apply high-level domain transformations first
        cls._apply_domain_adapters(config, overrides)

        # 2. Apply dot-notation and indexed overrides
        for key_path, value in overrides.items():
            if key_path in cls.DOMAIN_ALIASES:
                continue
            cls._set_nested_value(config, key_path, value, strict=strict)

        return config

    @classmethod
    def _set_nested_value(
        cls,
        root: Dict[str, Any],
        key_path: str,
        value: Any,
        strict: bool = False,
    ) -> None:
        """Sets a value at a nested path, creating intermediate dicts if needed."""
        parts = key_path.split(".")
        current = root
        for part in parts[:-1]:
            match = cls.INDEX_REGEX.match(part)
            if match:
                name, idx_str = match.group(1), match.group(2)
                idx = int(idx_str)
                if name not in current:
                    if strict:
                        raise KeyError("Key '" + name + "' not found in config.")
                    current[name] = []
                current = current[name][idx]
            else:
                if part not in current or not isinstance(current[part], dict):
                    if strict and part not in current:
                        raise KeyError("Key '" + part + "' not found in config.")
                    current[part] = {}
                current = current[part]

        # Terminal key
        terminal_part = parts[-1]
        match = cls.INDEX_REGEX.match(terminal_part)
        if match:
            name, idx_str = match.group(1), match.group(2)
            idx = int(idx_str)
            if name not in current:
                if strict:
                    raise KeyError("List key '" + name + "' not found in config.")
                current[name] = []
            target_list = current[name]
            coerced_val = cls._coerce_value(value)
            if idx < 0:
                idx = len(target_list) + idx
            if 0 <= idx < len(target_list):
                target_list[idx] = coerced_val
            elif idx == len(target_list):
                target_list.append(coerced_val)
            else:
                raise IndexError(f"Index {idx} out of range for list '{name}' of length {len(target_list)}")
        else:
            current[terminal_part] = cls._coerce_value(value)

    @classmethod
    def _apply_domain_adapters(
        cls, config: Dict[str, Any], overrides: Dict[str, Any]
    ) -> None:
        """Applies domain-specific high-level hyperparameter adaptations."""
        # Latent dimension adaptation
        if "latent_dim" in overrides or "bottleneck_dim" in overrides:
            ld = int(overrides.get("latent_dim", overrides.get("bottleneck_dim")))
            net_args = config.setdefault("network_args", {})
            if "network" in net_args and isinstance(net_args["network"], list) and len(net_args["network"]) > 0:
                net_args["network"] = list(net_args["network"][:-1]) + [ld]
            if "encoder_network" in net_args and isinstance(net_args["encoder_network"], list) and len(net_args["encoder_network"]) > 0:
                net_args["encoder_network"] = list(net_args["encoder_network"][:-1]) + [ld]
            if "latent_dim" in net_args:
                net_args["latent_dim"] = ld

        # Hidden layers adaptation
        if "hidden_layers" in overrides:
            hidden = overrides["hidden_layers"]
            if isinstance(hidden, str):
                hidden = [int(x.strip("[] ")) for x in hidden.split(",") if x.strip("[] ")]
            elif isinstance(hidden, (tuple, list)):
                hidden = [int(x) for x in hidden]
            net_args = config.setdefault("network_args", {})
            if "network" in net_args and isinstance(net_args["network"], list) and len(net_args["network"]) > 0:
                current_ld = net_args["network"][-1]
                net_args["network"] = hidden + [current_ld]
            if "encoder_network" in net_args and isinstance(net_args["encoder_network"], list) and len(net_args["encoder_network"]) > 0:
                current_ld = net_args["encoder_network"][-1]
                net_args["encoder_network"] = hidden + [current_ld]

        # Beta for VAE / sDVAE
        if "beta" in overrides:
            config.setdefault("network_args", {})["beta"] = float(overrides["beta"])

        # KLD capacity schedule value
        if "kld_max_value" in overrides:
            kld_args = config.setdefault("network_args", {}).setdefault("kld_args", {})
            kld_sched = kld_args.setdefault("kld_max_scheduler_args", {})
            kld_sched["value"] = float(overrides["kld_max_value"])

        # Learning rate
        if "lr" in overrides:
            config["lrate"] = float(overrides["lr"])
        elif "learning_rate" in overrides:
            config["lrate"] = float(overrides["learning_rate"])
        elif "lrate" in overrides:
            config["lrate"] = float(overrides["lrate"])

        # Weight decay
        if "weight_decay" in overrides:
            config["weight_decay"] = float(overrides["weight_decay"])

        # Batch sizes
        if "batch_size" in overrides:
            config.setdefault("datamodule_args", {})["batch_size"] = int(overrides["batch_size"])
        if "val_batch_size" in overrides:
            config.setdefault("datamodule_args", {})["val_batch_size"] = int(overrides["val_batch_size"])

    @staticmethod
    def _coerce_value(value: Any) -> Any:
        """Attempts to convert strings to appropriate python types (bool, int, float, list)."""
        if isinstance(value, str):
            val_strip = value.strip()
            if val_strip.lower() in ("true", "yes"):
                return True
            if val_strip.lower() in ("false", "no"):
                return False
            if val_strip.lower() in ("none", "null"):
                return None
            try:
                if "." in val_strip or "e" in val_strip.lower():
                    return float(val_strip)
                return int(val_strip)
            except ValueError:
                pass
            if val_strip.startswith("[") and val_strip.endswith("]"):
                inner = val_strip[1:-1].strip()
                if not inner:
                    return []
                return [ConfigResolver._coerce_value(x.strip()) for x in inner.split(",")]
        return value

    @classmethod
    def diff(cls, base: Dict[str, Any], resolved: Dict[str, Any], prefix: str = "") -> Dict[str, Tuple[Any, Any]]:
        """Computes differences between base and resolved dictionaries.
        
        Returns:
            Dictionary mapping key_path to (base_value, resolved_value).
        """
        changes = {}
        all_keys = set(base.keys()).union(set(resolved.keys()))
        for key in sorted(all_keys):
            curr_path = f"{prefix}.{key}" if prefix else str(key)
            if key not in base:
                changes[curr_path] = (None, resolved[key])
            elif key not in resolved:
                changes[curr_path] = (base[key], None)
            else:
                val_b = base[key]
                val_r = resolved[key]
                if isinstance(val_b, dict) and isinstance(val_r, dict):
                    changes.update(cls.diff(val_b, val_r, prefix=curr_path))
                elif val_b != val_r:
                    changes[curr_path] = (val_b, val_r)
        return changes
