from __future__ import annotations

from abc import ABC, abstractmethod

from fsm import DrowsinessSignals
from runtime.contracts import DecisionResult, EngineContext


class DecisionEngine(ABC):
    name: str

    @abstractmethod
    def initialize(self, context: EngineContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, signals: DrowsinessSignals) -> DecisionResult:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
