from torch import nn
from pydantic import BaseModel, model_validator
import torch


class CommonModule(nn.Module):
    def __init__(self,para:BaseModel,namespace=None,para_cls=None):
        if para_cls is None:
            para_cls = namespace.__dict__[f'{self.__class__.__name__}']
        if type(para) is dict: para = para_cls(**para)
        self.para = para.model_copy(deep=True)
        super().__init__()
        
    @staticmethod
    @torch.no_grad()            
    def convert_to_ternary(module: nn.Module,mods=None) -> nn.Module:
        """
        Recursively replace Bit.Conv2d/Bit.Linear with Ternary*Infer modules.
        Returns a new nn.Module (original left untouched if you deepcopy before).
        """
        for name, child in list(module.named_children()):
            if mods is not None:
                if name not in mods:continue
            if hasattr(child, 'to_ternary'):
                setattr(module, name, child.to_ternary())
            else:
                CommonModule.convert_to_ternary(child)
                
                
class CommonModel(BaseModel):
    bit: bool = True
    scale_op: str = "median"

    @model_validator(mode='after')
    def valid_model(self): return self

    def update(self,**kwargs):
        self.__dict__.update(**kwargs)
    
    @staticmethod
    def _build(obj,namespace):
        obj = obj.valid_model()
        return namespace.__dict__[f'{obj.__class__.__name__}'](obj)