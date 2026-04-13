import torch 
import time 

def get_torch_device_backend(force_cpu=False, deviceidx=0):
    if torch.cuda.is_available() and not force_cpu:
        device = torch.device("cuda:%d"%deviceidx)
        torch_backend = torch.cuda 
    elif torch.backends.mps.is_available() and not force_cpu:
        torch_backend = torch.mps
        device = torch.device("mps:%d"%deviceidx)
        assert torch.get_default_dtype()==torch.float32
    else: # Neither CUDA nor MPS available
        device = torch.device("cpu")
        torch_backend = torch.cpu
    return device,torch_backend

class Timer():
    def __init__(self, torch_backend):
        assert torch_backend in [torch.cpu,torch.cuda,torch.mps]
        self.torch_backend = torch_backend
    def tic(self):
        if self.torch_backend==torch.cpu:
            self.t0 = time.perf_counter()
        else:
            self.torch_backend.empty_cache()
            self.t0 = self.torch_backend.Event(enable_timing=True)
            self.tend = self.torch_backend.Event(enable_timing=True)
            self.t0.record()
    def toc(self):
        if self.torch_backend==torch.cpu:
            tdelta = time.perf_counter()-self.t0
        else:
            self.tend.record()
            self.torch_backend.synchronize()
            tdelta = self.t0.elapsed_time(self.tend)/1000
        return tdelta
