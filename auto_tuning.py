import tempfile

import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.space_generator import ScheduleFn

from evaluate import test_numerical_correctness
from gemm_relu_add import gemm_relu_add
from tvm.tir.schedule import BlockRV
from tvm.script import tir as T


def auto_tuning_schedule(sch: tir.Schedule) -> tir.Schedule:
    """The function that defines the schedule space for automatic tuning.

    Parameters
    ----------
    sch : tir.Schedule
        An empty schedule of the GeMM + ReLU + add workload.

    Returns
    -------
    sch : tir.Schedule
        The updated schedule of the GeMM + ReLU + add workload.
    """

    sch = tir.Schedule(gemm_relu_add)
    # Define the shared memory tile sizes and register tile sizes.
    
    # thread_tile_x, thread_tile_y, thread_tile_k = 4, 4, 1

    # Step 1. Shared memory tiling.
    block_gemm = sch.get_block("gemm")
    # Fetch the loops outside the "gemm" block.
    i, j, k = sch.get_loops(block_gemm)
    _, tile_x = sch.sample_perfect_tile(i, n = 2)
    _, tile_y = sch.sample_perfect_tile(j, n = 2)
    _, tile_k = sch.sample_perfect_tile(k, n = 2)

    i_outer, i_inner = sch.split(i, factors=[None, tile_x])
    j_outer, j_inner = sch.split(j, factors=[None, tile_y])
    k_outer, k_inner = sch.split(k, factors=[None, tile_k])
    sch.reorder(i_outer, j_outer, k_outer, i_inner, j_inner, k_inner)
    sch.bind(i_outer, "blockIdx.x")
    sch.bind(j_outer, "blockIdx.y")
    A_shared = sch.cache_read(block_gemm, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(A_shared, sch.get_loops(block_gemm)[2], preserve_unit_loops = True)
    B_shared = sch.cache_read(block_gemm, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(B_shared, sch.get_loops(block_gemm)[2], preserve_unit_loops = True)

    # Step 2. Register tiling.
    block_gemm = sch.get_block("gemm")
    i, j, k = sch.get_loops(block_gemm)[-3:]
    thread_extent_x, thread_tile_x = sch.sample_perfect_tile(i, n = 2)
    thread_extent_y, thread_tile_y = sch.sample_perfect_tile(j, n = 2)
    _, thread_tile_k = sch.sample_perfect_tile(k, n = 2)

    i_outer, i_inner = sch.split(i, factors=[None, thread_tile_x])
    j_outer, j_inner = sch.split(j, factors=[None, thread_tile_y])
    k_outer, k_inner = sch.split(k, factors=[None, thread_tile_k])
    sch.reorder(i_outer, j_outer, k_outer, i_inner, j_inner, k_inner)
    sch.bind(i_outer, "threadIdx.x")
    sch.bind(j_outer, "threadIdx.y")
    A_local = sch.cache_read(block_gemm, read_buffer_index=0, storage_scope="local")
    sch.compute_at(A_local, sch.get_loops(block_gemm)[5], preserve_unit_loops = True)
    B_local = sch.cache_read(block_gemm, read_buffer_index=1, storage_scope="local")
    sch.compute_at(B_local, sch.get_loops(block_gemm)[5], preserve_unit_loops = True)

    # Step 3. Cooperative fetching.
    def _cooperative_fetching_impl(block: BlockRV):
        read_x, read_y = sch.get_loops(block)[-2: ]
        combined_loop = sch.fuse(read_x, read_y)
        _, th_x, th_y= sch.split(
            combined_loop, 
            factors = [
                None,
                thread_extent_x, 
                thread_extent_y
            ]
        )
        sch.bind(th_x, "threadIdx.x")
        sch.bind(th_y, "threadIdx.y")

    _cooperative_fetching_impl(A_shared)
    _cooperative_fetching_impl(B_shared)

    # Step 4. Write cache.
    block_gemm = sch.get_block("gemm")
    loop_index = 5
    write_cache_loc = sch.get_loops(block_gemm)[loop_index]
    wirte_local = sch.cache_write(block_gemm, write_buffer_index = 0 , storage_scope = "local")
    sch.reverse_compute_at(wirte_local, write_cache_loc, preserve_unit_loops = True)

    # Step 5. Epilogue fusion.
    block_relu = sch.get_block("relu")
    block_add = sch.get_block("add")
    sch.reverse_compute_inline(block_relu)
    sch.reverse_compute_inline(block_add)

    return sch


def auto_tune():
    with tempfile.TemporaryDirectory() as work_dir:
        target = tvm.target.Target(
            {
                "kind": "cuda",
                "max_shared_memory_per_block": 49152,
                "max_threads_per_block": 1024,
                "thread_warp_size": 32,
            }
        )
        # Tune the workload and record the evaluated schedules into the database.
        database = ms.tir_integration.tune_tir(
            mod=gemm_relu_add,
            target=target,
            work_dir=work_dir,
            max_trials_global=64,  # We try 64 schedules in the search space.
            num_trials_per_iter=32,
            space=ScheduleFn(sch_fn=auto_tuning_schedule),
        )
        # Retrieve the best performant schedule from the database.
        sch = ms.tir_integration.compile_tir(database, gemm_relu_add, target)
        assert sch is not None, "No valid schedule found!"
        # Print out the optimized function and the schedule.
        sch.mod.show()
        sch.trace.show()
        # Test the numerical correctness.
        test_numerical_correctness(sch)


if __name__ == "__main__":
    auto_tune()
