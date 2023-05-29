from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def act(self, obs: np.ndarray) -> np.ndarray:
        pass

    def reset(self):
        pass