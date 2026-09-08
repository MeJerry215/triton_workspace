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
        with T.sblock("root"):
            T.reads()
            T.writes()
            X_sq_local = T.Buffer((32,), scope="local")
            sum_row = T.Buffer((1,), scope="local")
            sumsq_row = T.Buffer((1,), scope="local")
            X_local = T.Buffer((32,), scope="local")
            mean_row = T.Buffer((1,), scope="local")
            rstd_row = T.Buffer((1,), scope="local")
            T.sblock_attr({"layout_map": {X_sq_local: metadata["tl.Fragment"][0], sum_row: metadata["tl.Fragment"][1], sumsq_row: metadata["tl.Fragment"][2], X_local: metadata["tl.Fragment"][3], mean_row: metadata["tl.Fragment"][4], rstd_row: metadata["tl.Fragment"][5]}})
            bx = T.launch_thread("blockIdx.x", 4096)
            tx = T.launch_thread("threadIdx.x", 256)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.sblock("tilelang_root"):
                T.reads()
                T.writes()
                T.sblock_attr({"layout_map": {X_sq_local: metadata["tl.Fragment"][0], sum_row: metadata["tl.Fragment"][1], sumsq_row: metadata["tl.Fragment"][2], X_local: metadata["tl.Fragment"][3], mean_row: metadata["tl.Fragment"][4], rstd_row: metadata["tl.Fragment"][5]}})
                X_smem = T.sblock_alloc_buffer((1, 8192), "bfloat16", scope="shared.dyn")
                G_smem = T.sblock_alloc_buffer((8192,), "bfloat16", scope="shared.dyn")
                B_smem = T.sblock_alloc_buffer((8192,), "bfloat16", scope="shared.dyn")
                X_local = T.sblock_alloc_buffer((32,), data=X_local.data, scope="local")
                X_sq_local = T.sblock_alloc_buffer((32,), data=X_sq_local.data, scope="local")
                sum_row = T.sblock_alloc_buffer((1,), data=sum_row.data, scope="local")
                sumsq_row = T.sblock_alloc_buffer((1,), data=sumsq_row.data, scope="local")
                mean_row = T.sblock_alloc_buffer((1,), data=mean_row.data, scope="local")
                rstd_row = T.sblock_alloc_buffer((1,), data=rstd_row.data, scope="local")
                workspace = T.sblock_alloc_buffer((256,), scope="shared.dyn")
                workspace_1 = T.sblock_alloc_buffer((256,), scope="shared.dyn")
                if tx < 256 and tx >= 0:
                    for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        for vec in T.vectorized(8):
                            X_smem[0, (i * 8 + vec) // 8 * 2048 + tx * 8 + (i * 8 + vec) % 8] = X[bx, (i * 8 + vec) // 8 * 2048 + tx * 8 + (i * 8 + vec) % 8]
                if tx < 256 and tx >= 0:
                    for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        for vec in T.vectorized(8):
                            G_smem[(i * 8 + vec) // 8 * 2048 + tx * 8 + (i * 8 + vec) % 8] = gamma[(i * 8 + vec) // 8 * 2048 + tx * 8 + (i * 8 + vec) % 8]
                if tx < 256 and tx >= 0:
                    for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        for vec in T.vectorized(8):
                            B_smem[(i * 8 + vec) // 8 * 2048 + tx * 8 + (i * 8 + vec) % 8] = beta[(i * 8 + vec) // 8 * 2048 + tx * 8 + (i * 8 + vec) % 8]
                for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    for vec in T.vectorized(8):
                        X_local[i * 8 + vec] = T.Cast("float32", X_smem[0, (i * 8 + vec) // 8 * 2048 + tx * 8 + (i * 8 + vec) % 8])
                for i in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    X_sq_local[i] = X_local[i] * X_local[i]
                for i in T.unroll(1, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    sum_row[0] = T.float32(0.0)
                    for rv in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        sum_row[0] = sum_row[0] + X_local[rv % 4 * 8 + rv // 4]
                    sum_row[0] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 256, 1, 0>::run", sum_row[0], T.tvm_access_ptr(T.type_annotation("float32"), workspace.data, 0, 256, 2))
                for i in T.unroll(1, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    sumsq_row[0] = T.float32(0.0)
                    for rv in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        sumsq_row[0] = sumsq_row[0] + X_sq_local[rv % 4 * 8 + rv // 4]
                    sumsq_row[0] = T.call_extern("float32", "tl::AllReduce<tl::SumOp, 256, 1, 0>::run", sumsq_row[0], T.tvm_access_ptr(T.type_annotation("float32"), workspace_1.data, 0, 256, 2))
                for i in T.unroll(1, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    mean_row[0] = sum_row[0] * T.float32(0.0001220703125)
                    rstd_row[0] = T.rsqrt(sumsq_row[0] * T.float32(0.0001220703125) - mean_row[0] * mean_row[0] + T.float32(1.0000000000000001e-05))
                    Mean[bx] = mean_row[0]
                    Rstd[bx] = rstd_row[0]
                for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    for vec in T.vectorized(8):
                        norm: T.float32 = (X_local[i * 8 + vec] - mean_row[0]) * rstd_row[0]
                        X_smem[0, (i * 8 + vec) // 8 * 2048 + tx * 8 + (i * 8 + vec) % 8] = T.Cast("bfloat16", norm * T.Cast("float32", G_smem[(i * 8 + vec) // 8 * 2048 + tx * 8 + (i * 8 + vec) % 8]) + T.Cast("float32", B_smem[(i * 8 + vec) // 8 * 2048 + tx * 8 + (i * 8 + vec) % 8]))
                if tx < 256 and tx >= 0:
                    for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                        for vec in T.vectorized(8):
                            Y[bx, (i * 8 + vec) // 8 * 2048 + tx * 8 + (i * 8 + vec) % 8] = X_smem[0, (i * 8 + vec) // 8 * 2048 + tx * 8 + (i * 8 + vec) % 8]

# Metadata omitted. Use show_meta=True in script() method to show it.