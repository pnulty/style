from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class NormalizedExample:
    text: str
    label: str
    source: str
    domain: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
