from __future__ import annotations

from runtime.config import RuntimeConfig
from runtime.engines.base import DecisionEngine
from runtime.engines.fsm_engine import FSMDecisionEngine
from runtime.engines.legacy_engine import LegacyRuleEngine
from runtime.engines.ml_engine import MLDecisionEngine


ENGINE_REGISTRY: dict[str, type[DecisionEngine]] = {
    "fsm": FSMDecisionEngine,
    "legacy": LegacyRuleEngine,
    "ml": MLDecisionEngine,
}


def available_engines() -> list[str]:
    return sorted(ENGINE_REGISTRY.keys())


def create_engine(name: str, config: RuntimeConfig) -> DecisionEngine:
    if name not in ENGINE_REGISTRY:
        available = ", ".join(available_engines())
        raise ValueError(f"Unsupported decision engine '{name}'. Available: {available}")
    return ENGINE_REGISTRY[name](config)
