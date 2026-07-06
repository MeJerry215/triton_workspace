import pytest
import torch
import triton
import importlib
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
# __all__ = ["async_copy", "fence_async_shared", "mbarrier", "mma_v2", "tma", "warpgroup_mma", "warpgroup_mma_wait"]
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
# __all__ = ["async_copy", "mbarrier", "mma_v2"]
# from triton.experimental.gluon.language.nvidia.ampere import 


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9

if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")