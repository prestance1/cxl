from math import ceil, floor


def get_max_depth(no_variables: int) -> None:
    return no_variables - 2


# block and thread calculations inspired from: https://github.com/hpi-epic/gpucsl


def calculate_blocks_and_threads_compact(variable_count: int):
    """
    Calculate the number of blocks and threads per block for a compact kernel.

    Args:
        variable_count (int): Number of variables.

    Returns:
        tuple: Blocks per grid and threads per block.
    """
    blocks_per_grid = (int(ceil(variable_count / 512)),)
    threads_per_block = (min(512, variable_count),)
    return (blocks_per_grid, threads_per_block)


def calculate_blocks_and_threads_kernel_level_n(variable_count: int):
    """
    Calculate the number of blocks and threads per block for a kernel at level N.

    Args:
        variable_count (int): Number of variables.

    Returns:
        tuple: Blocks per grid and threads per block.
    """

    blocks_per_grid = (variable_count, variable_count)
    threads_per_block = (32,)
    return (blocks_per_grid, threads_per_block)


def calculate_blocks_and_threads_kernel_level_0(variable_count: int):
    """
    Calculate the number of blocks and threads per block for a kernel at level 0.

    Args:
        variable_count (int): Number of variables.

    Returns:
        tuple: Blocks per grid and threads per block.
    """
    needed_threads = floor((variable_count * (variable_count + 1)) / 2)
    blocks_per_grid = (ceil(needed_threads / 1024),)
    threads_per_block = (min(max(32, needed_threads), 1024),)

    return (blocks_per_grid, threads_per_block)
