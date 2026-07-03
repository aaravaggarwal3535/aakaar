from ._C import generate_random

def rand(size: int, seed: int = 42):
    """
    Generates a 1D NumPy array filled with uniform random numbers 
    sampled between 0.0 and 1.0, accelerated by CUDA cuRAND.
    """
    return generate_random(size, seed)