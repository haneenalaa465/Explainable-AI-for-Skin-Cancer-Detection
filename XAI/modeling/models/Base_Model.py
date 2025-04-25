import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from XAI.config import MODEL_INPUT_SIZE


class BaseModel(nn.Module, ABC):
    @staticmethod
    @abstractmethod
    def name():
        pass

    @staticmethod
    def inputSize():
        return MODEL_INPUT_SIZE
