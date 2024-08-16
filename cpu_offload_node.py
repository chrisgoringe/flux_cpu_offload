import torch
from comfy.ldm.flux.model import Flux
from comfy.ldm.flux.layers import SingleStreamBlock, DoubleStreamBlock
from comfy.model_patcher import ModelPatcher
from nodes import UNETLoader
import folder_paths

class CastMode:
    PRE = 1
    AUTO = 2
    MANUAL = 3

    @classmethod
    def get(cls, s:str):
        if s.lower() == 'pre': return CastMode.PRE
        if s.lower() == 'auto': return CastMode.AUTO
        return CastMode.MANUAL

class OnDemandLinear(torch.nn.Module):
    def __init__(self, linear:torch.nn.Linear, mode:int):
        super().__init__()
        self.mode = mode
        if mode==CastMode.PRE: linear.to(torch.bfloat16)
        self.wrapped = linear

    def _apply(self, fn, recurse=True): pass

    def forward(self, x:torch.Tensor):
        if self.mode == CastMode.MANUAL:
            weight = self.wrapped.weight.to(x)
            bias   = self.wrapped.bias.to(x) if self.wrapped.bias is not None else None
        else:
            weight = self.wrapped.weight.cuda()
            bias   = self.wrapped.bias.cuda() if self.wrapped.bias is not None else None
        with torch.autocast("cuda", enabled=(self.mode==CastMode.AUTO)):
            return torch.nn.functional.linear(x, weight, bias)

def lock_SingleStreamBlock_to_cpu(module:SingleStreamBlock, mode:int):
    module.linear1 = OnDemandLinear(module.linear1, mode)
    module.linear2 = OnDemandLinear(module.linear2, mode)
    # attention to add

def lock_DoubleStreamBlock_to_cpu(module:DoubleStreamBlock, mode:int):
    module.txt_mlp = torch.nn.Sequential(OnDemandLinear(module.txt_mlp[0], mode), module.txt_mlp[1], 
                                         OnDemandLinear(module.txt_mlp[2], mode))
    module.img_mlp = torch.nn.Sequential(OnDemandLinear(module.img_mlp[0], mode), module.img_mlp[1], 
                                         OnDemandLinear(module.img_mlp[2], mode))
    # attention to add

def split_model(model:ModelPatcher, number_of_single_blocks:int, number_of_double_blocks:int, mode:int):
    '''
    Run the SingleStreamBlock stack on the CPU
    '''
    diffusion_model:Flux = model.model.diffusion_model
    for i in range(min(number_of_single_blocks, len(diffusion_model.single_blocks))):
        lock_SingleStreamBlock_to_cpu(diffusion_model.single_blocks[i], mode)
    for i in range(min(number_of_double_blocks, len(diffusion_model.double_blocks))):
        lock_DoubleStreamBlock_to_cpu(diffusion_model.double_blocks[i], mode)
    return model
    
class CPUOffLoad:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "func"
    CATEGORY = "flux_tools"

    @classmethod
    def INPUT_TYPES(s): return { "required": { 
        "model": ("MODEL",{}),
        "double_blocks_on_cpu": ("INT",{"default":0, "min":0, "max":19}),
        "single_blocks_on_cpu": ("INT",{"default":0, "min":0, "max":38}),
        "cast_mode":            (["pre","auto","manual"],{})
    } }

    def func(self, model, double_blocks_on_cpu, single_blocks_on_cpu, cast_mode): 
        m = split_model(model.clone(), number_of_single_blocks=single_blocks_on_cpu, 
                        number_of_double_blocks=double_blocks_on_cpu, mode=CastMode.get(cast_mode))
        
        fraction_pinned   = ( 2*double_blocks_on_cpu + single_blocks_on_cpu ) / 57 
        fraction_possible = 0.5 # what fraction of the whole thing could we possibly have pinned? Work out better!

        m.model.memory_usage_factor *= (1 - fraction_possible*fraction_pinned)
        return (m,)

class UNETLoaderForce(UNETLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("unet"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                              "force_reload": (["no","yes"], {})                            }}
    FUNCTION = "func"
    CATEGORY = "flux_tools"

    def func(self, unet_name, weight_dtype, force_reload):
        return self.load_unet(unet_name, weight_dtype)
    
    @classmethod
    def IS_CHANGED(self, unet_name, weight_dtype, force_reload):
        if force_reload=='yes': return float('NaN')
        return unet_name+