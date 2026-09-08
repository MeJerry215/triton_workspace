# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(X_handle: T.handle, gamma_handle: T.handle, beta_handle: T.handle, Y_handle: T.handle, Mean_handle: T.handle, Rstd_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_80", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tl.has_tma": T.bool(False)})
        X = T.match_buffer(X_handle, (4096, 8192), "bfloat16", strides=(8192, 1))
        gamma = T.match_buffer(gamma_handle, (8192,), "bfloat16", strides=(1,))
        beta = T.match_buffer(beta_handle, (8192,), "bfloat16", strides=(1,))
        Y = T.match_buffer(Y_handle, (4096, 8192), "bfloat16", strides=(8192, 1))
        Mean = T.match_buffer(Mean_handle, (4096,), strides=(1,))
        Rstd = T.match_buffer(Rstd_handle, (4096,), strides=(1,))
        bx = T.launch_thread("blockIdx.x", 4096)
        X_smem = T.alloc_buffer((8192,), "bfloat16", scope="shared.dyn")
        G_smem = T.alloc_buffer((8192,), "bfloat16", scope="shared.dyn")
        B_smem = T.alloc_buffer((8192,), "bfloat16", scope="shared.dyn")
        X_local = T.alloc_buffer((32,), scope="local")
        X_smem_local_cast = T.alloc_buffer((8,), "bfloat16", scope="local")
        X_sq_local = T.alloc_buffer((32,), scope="local")
        sum_row = T.alloc_buffer((1,), scope="local")
        workspace = T.alloc_buffer((256,), scope="shared.dyn")
        sumsq_row = T.alloc_buffer((1,), scope="local")
        workspace_1 = T.alloc_buffer((256,), scope="shared.dyn")
        mean_row = T.alloc_buffer((1,), scope="local")
        rstd_row = T.alloc_buffer((1,), scope="local")
        G_smem_local_cast_2 = T.alloc_buffer((8,), "bfloat16", scope="local")
        B_smem_local_cast_3 = T.alloc_buffer((8,), "bfloat16", scope="local")
        X_smem_local_cast_1 = T.alloc_buffer((8,), "bfloat16", scope="local")
        tx = T.launch_thread("threadIdx.x", 256)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        X_smem_1 = T.decl_buffer((8192,), "bfloat16", data=X_smem.data, scope="shared.dyn")
        G_smem_1 = T.decl_buffer((8192,), "bfloat16", data=G_smem.data, scope="shared.dyn")
        B_smem_1 = T.decl_buffer((8192,), "bfloat16", data=B_smem.data, scope="shared.dyn")
        X_local_1 = T.decl_buffer((32,), data=X_local.data, scope="local")
        X_sq_local_1 = T.decl_buffer((32,), data=X_sq_local.data, scope="local")
        sum_row_1 = T.decl_buffer((1,), data=sum_row.data, scope="local")
        sumsq_row_1 = T.decl_buffer((1,), data=sumsq_row.data, scope="local")
        mean_row_1 = T.decl_buffer((1,), data=mean_row.data, scope="local")
        rstd_row_1 = T.decl_buffer((1,), data=rstd_row.data, scope="local")
        i = T.int32()
        with T.attr(i, "pragma_unroll_explicit", T.bool(False)):
            for i in T.unroll(4):
                X_1 = T.Buffer((33554432,), "bfloat16", data=X.data)
                X_smem_1[i * 2048 + tx * 8:i * 2048 + tx * 8 + 8] = X_1[bx * 8192 + i * 2048 + tx * 8:bx * 8192 + i * 2048 + tx * 8 + 8]
        i_1 = T.int32()
        with T.attr(i_1, "pragma_unroll_explicit", T.bool(False)):
            for i_1 in T.unroll(4):
                gamma_1 = T.Buffer((8192,), "bfloat16", data=gamma.data)
                G_smem_1[i_1 * 2048 + tx * 8:i_1 * 2048 + tx * 8 + 8] = gamma_1[i_1 * 2048 + tx * 8:i_1 * 2048 + tx * 8 + 8]
        i_2 = T.int32()
        with T.attr(i_2, "pragma_unroll_explicit", T.bool(False)):
            for i_2 in T.unroll(4):
                beta_1 = T.Buffer((8192,), "bfloat16", data=beta.data)
                B_smem_1[i_2 * 2048 + tx * 8:i_2 * 2048 + tx * 8 + 8] = beta_1[i_2 * 2048 + tx * 8:i_2 * 2048 + tx * 8 + 8]
        i_3 = T.int32()
        with T.attr(i_3, "pragma_unroll_explicit", T.bool(False)):
            for i_3 in T.unroll(4):
                X_smem_local_cast[0:8] = X_smem_1[i_3 * 2048 + tx * 8:i_3 * 2048 + tx * 8 + 8]
                for vec in range(2):
                    X_local_1[i_3 * 8 + vec * 4:i_3 * 8 + vec * 4 + 4] = T.Cast("float32x4", X_smem_local_cast[vec * 4:vec * 4 + 4])
        i_4 = T.int32()
        with T.attr(i_4, "pragma_unroll_explicit", T.bool(False)):
            for i_4 in T.unroll(32):
                X_sq_local_1[i_4] = X_local_1[i_4] * X_local_1[i_4]
        sum_row_1[0] = T.float32(0.0)
        rv = T.int32()
        with T.attr(rv, "pragma_unroll_explicit", T.bool(False)):
            for rv in T.unroll(32):
                sum_row_1[0] = sum_row_1[0] + X_local_1[rv % 4 * 8 + rv // 4]
        sum_row_1[0] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 256, 1, 0>::run", sum_row_1[0], T.tvm_access_ptr(T.type_annotation("float32"), workspace.data, 0, 256, 2))
        sumsq_row_1[0] = T.float32(0.0)
        rv_1 = T.int32()
        with T.attr(rv_1, "pragma_unroll_explicit", T.bool(False)):
            for rv_1 in T.unroll(32):
                sumsq_row_1[0] = sumsq_row_1[0] + X_sq_local_1[rv_1 % 4 * 8 + rv_1 // 4]
        sumsq_row_1[0] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 256, 1, 0>::run", sumsq_row_1[0], T.tvm_access_ptr(T.type_annotation("float32"), workspace_1.data, 0, 256, 2))
        mean_row_1[0] = sum_row_1[0] * T.float32(0.0001220703125)
        rstd_row_1[0] = T.rsqrt(sumsq_row_1[0] * T.float32(0.0001220703125) - mean_row_1[0] * mean_row_1[0] + T.float32(1.0000000000000001e-05))
        Mean_1 = T.Buffer((4096,), data=Mean.data)
        Mean_1[bx] = mean_row_1[0]
        Rstd_1 = T.Buffer((4096,), data=Rstd.data)
        Rstd_1[bx] = rstd_row_1[0]
        i_5 = T.int32()
        with T.attr(i_5, "pragma_unroll_explicit", T.bool(False)):
            for i_5 in T.unroll(4):
                G_smem_local_cast_2[0:8] = G_smem_1[i_5 * 2048 + tx * 8:i_5 * 2048 + tx * 8 + 8]
                B_smem_local_cast_3[0:8] = B_smem_1[i_5 * 2048 + tx * 8:i_5 * 2048 + tx * 8 + 8]
                for vec in range(2):
                    X_smem_local_cast_1[vec * 4:vec * 4 + 4] = T.Cast("bfloat16x4", (X_local_1[i_5 * 8 + vec * 4:i_5 * 8 + vec * 4 + 4] - T.Broadcast(mean_row_1[0], 4)) * T.Broadcast(rstd_row_1[0], 4) * T.Cast("float32x4", G_smem_local_cast_2[vec * 4:vec * 4 + 4]) + T.Cast("float32x4", B_smem_local_cast_3[vec * 4:vec * 4 + 4]))
                X_smem_1[i_5 * 2048 + tx * 8:i_5 * 2048 + tx * 8 + 8] = X_smem_local_cast_1[0:8]
        i_6 = T.int32()
        T.attr(i_6, "pragma_unroll_explicit", T.bool(False))
        for i_6 in T.unroll(4):
            Y_1 = T.Buffer((33554432,), "bfloat16", data=Y.data)
            Y_1[bx * 8192 + i_6 * 2048 + tx * 8:bx * 8192 + i_6 * 2048 + tx * 8 + 8] = X_smem_1[i_6 * 2048 + tx * 8:i_6 * 2048 + tx * 8 + 8]