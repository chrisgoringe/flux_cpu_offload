
from .cpu_offload_node import CPUOffLoad, UNETLoaderForce

version = "0.1"

NODE_CLASS_MAPPINGS = { 
    "CPU Offload Layers"            : CPUOffLoad,
    "Load Diffusion Model Force"    : UNETLoaderForce,
                      }

__all__ = ["NODE_CLASS_MAPPINGS",]