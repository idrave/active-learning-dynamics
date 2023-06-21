from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class Agent(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        pass

    def reset(self):
        pass

class AgentReset(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def act(self, obs: np.ndarray) -> Optional[np.ndarray]:
        pass

    def reset(self):
        pass