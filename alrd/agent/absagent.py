from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def act(obs: np.ndarray) -> np.ndarray:
        pass