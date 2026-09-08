# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def main_kernel(Mean: T.handle("float32", "global"), Rstd: T.handle("float32", "global"), X: T.handle("bfloat16", "global"), Y: T.handle("bfloat16", "global"), beta: T.handle("bfloat16", "global"), gamma: T.handle("bfloat16", "global")):
        T.func_attr({"target": T.target({"arch": "sm_80", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tirx.is_global_func": True, "tirx.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [2, 4, 5]})
        Y_1 = T.decl_buffer((33554432,), "bfloat16", data=Y)
        Rstd_1 = T.decl_buffer((4096,), data=Rstd)
        Mean_1 = T.decl_buffer((4096,), data=Mean)
        beta_1 = T.decl_buffer((8192,), "bfloat16", data=beta)
        gamma_1 = T.decl_buffer((8192,), "bfloat16", data=gamma)
        X_1 = T.decl_buffer((33554432,), "bfloat16", data=X)
        bx = T.launch_thread("blockIdx.x", 4096)
        buf_dyn_shmem = T.alloc_buffer((50176,), "uint8", scope="shared.dyn")
        X_smem: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 0)
        G_smem: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 16384)
        B_smem: T.handle("bfloat16", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 32768)
        workspace: T.handle("float32", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 49152)
        workspace_1: T.handle("float32", "shared.dyn") = T.handle_add_byte_offset(buf_dyn_shmem.data, 49152)
        X_local = T.alloc_buffer((32,), scope="local")
        X_smem_local_cast = T.alloc_buffer((8,), "bfloat16", scope="local")
        X_sq_local = T.alloc_buffer((32,), scope="local")
        sum_row = T.alloc_buffer((1,), scope="local")
        sumsq_row = T.alloc_buffer((1,), scope="local")
        mean_row = T.alloc_buffer((1,), scope="local")
        rstd_row = T.alloc_buffer((1,), scope="local")
        G_smem_local_cast_2 = T.alloc_buffer((8,), "bfloat16", scope="local")
        B_smem_local_cast_3 = T.alloc_buffer((8,), "bfloat16", scope="local")
        X_smem_local_cast_1 = T.alloc_buffer((8,), "bfloat16", scope="local")
        tx = T.launch_thread("threadIdx.x", 256)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        X_smem_1 = T.decl_buffer((8192,), "bfloat16", data=X_smem, scope="shared.dyn")
        G_smem_1 = T.decl_buffer((8192,), "bfloat16", data=G_smem, scope="shared.dyn")
        B_smem_1 = T.decl_buffer((8192,), "bfloat16", data=B_smem, scope="shared.dyn")
        X_local_1 = T.decl_buffer((32,), data=X_local.data, scope="local")
        X_sq_local_1 = T.decl_buffer((32,), data=X_sq_local.data, scope="local")
        sum_row_1 = T.decl_buffer((1,), data=sum_row.data, scope="local")
        sumsq_row_1 = T.decl_buffer((1,), data=sumsq_row.data, scope="local")
        mean_row_1 = T.decl_buffer((1,), data=mean_row.data, scope="local")
        rstd_row_1 = T.decl_buffer((1,), data=rstd_row.data, scope="local")
        for i in T.unroll(4):
            X_smem_1[i * 2048 + tx * 8:i * 2048 + tx * 8 + 8] = X_1[bx * 8192 + i * 2048 + tx * 8:bx * 8192 + i * 2048 + tx * 8 + 8]
        for i in T.unroll(4):
            G_smem_1[i * 2048 + tx * 8:i * 2048 + tx * 8 + 8] = gamma_1[i * 2048 + tx * 8:i * 2048 + tx * 8 + 8]
        for i in T.unroll(4):
            B_smem_1[i * 2048 + tx * 8:i * 2048 + tx * 8 + 8] = beta_1[i * 2048 + tx * 8:i * 2048 + tx * 8 + 8]
        for i in T.unroll(4):
            X_smem_local_cast[0:8] = X_smem_1[i * 2048 + tx * 8:i * 2048 + tx * 8 + 8]
            for vec in range(2):
                X_local_1[i * 8 + vec * 4:i * 8 + vec * 4 + 4] = T.Cast("float32x4", X_smem_local_cast[vec * 4:vec * 4 + 4])
        for i in T.unroll(32):
            X_sq_local_1[i] = X_local_1[i] * X_local_1[i]
        sum_row_1[0] = T.float32(0.0)
        for rv in T.unroll(32):
            sum_row_1[0] = sum_row_1[0] + X_local_1[rv % 4 * 8 + rv // 4]
        sum_row_1[0] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 256, 1, 0>::run", sum_row_1[0], T.tvm_access_ptr(T.type_annotation("float32"), workspace_1, 0, 256, 2))
        sumsq_row_1[0] = T.float32(0.0)
        for rv in T.unroll(32):
            sumsq_row_1[0] = sumsq_row_1[0] + X_sq_local_1[rv % 4 * 8 + rv // 4]
        sumsq_row_1[0] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 256, 1, 0>::run", sumsq_row_1[0], T.tvm_access_ptr(T.type_annotation("float32"), workspace, 0, 256, 2))
        mean_row_1[0] = sum_row_1[0] * T.float32(0.0001220703125)
        rstd_row_1[0] = T.rsqrt(sumsq_row_1[0] * T.float32(0.0001220703125) - mean_row_1[0] * mean_row_1[0] + T.float32(1.0000000000000001e-05))
        Mean_1[bx] = mean_row_1[0]
        Rstd_1[bx] = rstd_row_1[0]
        for i in T.unroll(4):
            G_smem_local_cast_2[0:8] = G_smem_1[i * 2048 + tx * 8:i * 2048 + tx * 8 + 8]
            B_smem_local_cast_3[0:8] = B_smem_1[i * 2048 + tx * 8:i * 2048 + tx * 8 + 8]
            for vec in range(2):
                X_smem_local_cast_1[vec * 4:vec * 4 + 4] = T.Cast("bfloat16x4", (X_local_1[i * 8 + vec * 4:i * 8 + vec * 4 + 4] - T.Broadcast(mean_row_1[0], 4)) * T.Broadcast(rstd_row_1[0], 4) * T.Cast("float32x4", G_smem_local_cast_2[vec * 4:vec * 4 + 4]) + T.Cast("float32x4", B_smem_local_cast_3[vec * 4:vec * 4 + 4]))
            X_smem_1[i * 2048 + tx * 8:i * 2048 + tx * 8 + 8] = X_smem_local_cast_1[0:8]
        for i in T.unroll(4):
            Y_1[bx * 8192 + i * 2048 + tx * 8:bx * 8192 + i * 2048 + tx * 8 + 8] = X_smem_1[i * 2048 + tx * 8:i * 2048 + tx * 8 + 8]

    @T.prim_func
    def main(X_handle: T.handle, gamma_handle: T.handle, beta_handle: T.handle, Y_handle: T.handle, Mean_handle: T.handle, Rstd_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_80", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tirx.is_entry_func": True, "tl.has_tma": T.bool(False), "tl.readonly_param_indices": [0, 1, 2, 3, 4, 5], "tma_descriptor_args": {}})
        X = T.match_buffer(X_handle, (4096, 8192), "bfloat16", strides=(8192, 1))
        gamma = T.match_buffer(gamma_handle, (8192,), "bfloat16", strides=(1,))
        beta = T.match_buffer(beta_handle, (8192,), "bfloat16", strides=(1,))
        Y = T.match_buffer(Y_handle, (4096, 8192), "bfloat16", strides=(8192, 1))
        Mean = T.match_buffer(Mean_handle, (4096,), strides=(1,))
        Rstd = T.match_buffer(Rstd_handle, (4096,), strides=(1,))
        Module.main_kernel(Mean.data, Rstd.data, X.data, Y.data, beta.data, gamma.data)