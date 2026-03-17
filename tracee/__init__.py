"""Local-dev shim for importing tracee from the workspace root."""

from pathlib import Path
import sys

package_root = Path(__file__).resolve().parent
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from backbone.sdk.instrument import init, trace

__all__ = ["init", "trace"]
