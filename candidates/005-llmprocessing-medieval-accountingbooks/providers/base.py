# Author: Maximilian Vogeltanz, University of Graz, 2026

# defining classes for LLM Processing.
# Script gets called in dhinfra_client.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

@dataclass
class GenResult:
    text: str
    usage: Usage

class ProviderClient(Protocol):
    def generate(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        temperature: float,
        thinking: Optional[dict] = None,
    ) -> GenResult:
        ...
