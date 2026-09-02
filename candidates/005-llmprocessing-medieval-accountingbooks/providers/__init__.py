# Author: Maximilian Vogeltanz, University of Graz, 2026

# Provider factory for LLM API processing in project "Aldersbach Digital".
# This standalone repository ships one provider: the DH-Infra cluster of the
# University of Graz. Other providers of the full pipeline are not included here.


from .dhinfra_client import DhinfraClient

def make_client(provider: str):
    p = provider.lower()
    if p == "dhinfra":
        return DhinfraClient()
    raise ValueError(
        f"Unknown provider: {provider}. This repository only supports 'dhinfra'."
    )
