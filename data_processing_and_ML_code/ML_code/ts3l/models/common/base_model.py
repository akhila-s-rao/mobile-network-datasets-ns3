from abc import ABC, abstractmethod
import torch
from torch import nn
from typing import Any, Union, Tuple

class TS3LModule(ABC, nn.Module):
    def __init__(self) -> None:
        super(TS3LModule, self).__init__()
        
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        def init_decorator(cls_init):
            def new_init(self, *args, **kwargs):
                cls_init(self, *args, **kwargs)
                cls.set_first_phase(self)
            return new_init
        cls.__init__ = init_decorator(cls.__init__) # type: ignore

    # Akhila
    pred_head_size = None 
    
    @property
    @abstractmethod
    def encoder(self):
        raise NotImplementedError
    
    def set_first_phase(self):
        """Set first phase step as the forward pass
        """
        self.forward = self._first_phase_step
        self.encoder.requires_grad_(True)

    # Akhila
    #def get_first_phase_output(self, x: torch.Tensor) -> torch.Tensor:
    #    # Get the output of the enmedding layers 
    #    print('Inside get_first_phase_output function')
    #    print(self.encoder)
    #    print(x)
    #    return self.encoder(x)
        
    
    # Akhila
    def set_second_phase(self, freeze_encoder: bool = True, pred_head_size: int = 1):
        """Set second phase step as the forward pass
        """
        # Akhila 
        self.pred_head_size = pred_head_size
        
        self.forward = self._second_phase_step
        self.encoder.requires_grad_(not freeze_encoder)
    
    @abstractmethod
    def _first_phase_step(self, *args: Any, **kwargs: Any) -> Any:
        pass
    
    @abstractmethod
    def _second_phase_step(self, *args: Any, **kwargs: Any) -> Any:
        pass