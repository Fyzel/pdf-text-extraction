"""Ollama instance health probing."""
import urllib.error
import urllib.request

from pdf_extractor.config import OllamaInstance

_TIMEOUT: int = 5


def probe_instances(instances: list[OllamaInstance]) -> list[OllamaInstance]:
    """Return only instances that respond with HTTP 200 on GET /api/tags.

    Args:
        instances: Candidate Ollama instances to probe.

    Returns:
        Subset of ``instances`` that responded successfully within the timeout.
        Empty list if all instances are unreachable.
    """
    live: list[OllamaInstance] = []
    for instance in instances:
        try:
            with urllib.request.urlopen(  # nosec B310
                f"{instance.url}/api/tags", timeout=_TIMEOUT
            ) as resp:
                if resp.status == 200:
                    live.append(instance)
        except (urllib.error.URLError, OSError):
            pass
    return live
