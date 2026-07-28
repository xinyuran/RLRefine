"""Fail fast on ms-swift releases that freeze the DPO policy adapter."""
import importlib.metadata
import json
from typing import Any

from packaging.version import InvalidVersion, Version


MIN_VERSION = Version("3.12.2")
MAX_VERSION = Version("4.0.0")


def evaluate_version(raw_version: str) -> dict[str, Any]:
    try:
        installed = Version(raw_version)
    except InvalidVersion:
        return {
            "status": "FAIL",
            "reason": "invalid_ms_swift_version",
            "installed_version": raw_version,
        }

    compatible = MIN_VERSION <= installed < MAX_VERSION
    return {
        "status": "PASS" if compatible else "FAIL",
        "reason": (
            "compatible_peft_reference_adapter_loading"
            if compatible
            else "unsupported_peft_reference_adapter_loading"
        ),
        "installed_version": raw_version,
        "required_version": f">={MIN_VERSION},<{MAX_VERSION}",
    }


def main() -> int:
    try:
        raw_version = importlib.metadata.version("ms-swift")
        result = evaluate_version(raw_version)
    except importlib.metadata.PackageNotFoundError:
        result = {
            "status": "FAIL",
            "reason": "ms_swift_not_installed",
            "required_version": f">={MIN_VERSION},<{MAX_VERSION}",
        }
    print(
        json.dumps(
            {"event": "bp4_dpo_ms_swift_compatibility", **result},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
