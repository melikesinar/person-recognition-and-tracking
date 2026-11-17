"""PyTorch runtime helpers."""
import torch


def configure_cpu_inference(num_threads: int):
    """
    CPU settings for PyTorch inference, applied once per process.

    flush_denormal: this network produces many "denormal" intermediate floats (values
    extremely close to zero). x86 CPUs process those up to ~100x slower than normal
    floats. Flushing them to zero made OSNet ~4x faster here with bit-identical
    output embeddings (measured max difference: 0.0).
    """
    import warnings
    # numpy warns once that the smallest subnormal is now zero; that is expected.
    warnings.filterwarnings("ignore", message="The value of the smallest subnormal")
    torch.set_flush_denormal(True)
    torch.set_num_threads(max(1, int(num_threads)))
