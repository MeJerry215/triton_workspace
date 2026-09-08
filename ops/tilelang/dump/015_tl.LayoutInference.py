# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(X_handle: T.handle, gamma_handle: T.handle, beta_handle: T.handle, Y_handle: T.handle, Mean_handle: T.handle, Rstd_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_80", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32})})
        X = T.match_buffer(X_handle, (4096, 8192), "bfloat16", strides=(8192, 1))
        gamma = T.match_buffer(gamma_handle, (8192,), "bfloat16", strides=(1,))
        beta = T.match_buffer(beta_handle, (8192,), "bfloat16", strides=(1,))
        Y = T.match_buffer(Y_handle, (4096, 8192), "bfloat16", strides=(8192, 1))
        Mean = T.match_buffer(Mean_handle, (4096,), strides=(1,))
        Rstd = T.match_buffer(Rstd_handle, (4096,), strides=(1,))
        with T.sblock("root"):
            T.reads()
            T.writes()
            mean_row = T.Buffer((1,), scope="local.fragment")
            sum_row = T.Buffer((1,), scope="local.fragment")
            rstd_row = T.Buffer((1,), scope="local.fragment")
            sumsq_row = T.Buffer((1,), scope="local.fragment")
            X_local = T.Buffer((1, 8192), scope="local.fragment")
            X_sq_local = T.Buffer((1, 8192), scope="local.fragment")
            T.sblock_attr({"layout_map": {mean_row: metadata["tl.Fragment"][0], sum_row: metadata["tl.Fragment"][1], rstd_row: metadata["tl.Fragment"][2], sumsq_row: metadata["tl.Fragment"][3], X_local: metadata["tl.Fragment"][4], X_sq_local: metadata["tl.Fragment"][5]}})
            bx = T.launch_thread("blockIdx.x", 4096)
            tx = T.launch_thread("threadIdx.x", 256)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.sblock("tilelang_root"):
                T.reads()
                T.writes()
                T.sblock_attr({"layout_map": {mean_row: metadata["tl.Fragment"][0], sum_row: metadata["tl.Fragment"][1], rstd_row: metadata["tl.Fragment"][2], sumsq_row: metadata["tl.Fragment"][3], X_local: metadata["tl.Fragment"][4], X_sq_local: metadata["tl.Fragment"][5]}})
                X_smem = T.sblock_alloc_buffer((1, 8192), "bfloat16", scope="shared.dyn")
                G_smem = T.sblock_alloc_buffer((8192,), "bfloat16", scope="shared.dyn")
                B_smem = T.sblock_alloc_buffer((8192,), "bfloat16", scope="shared.dyn")
                X_local = T.sblock_alloc_buffer((1, 8192), data=X_local.data, scope="local.fragment")
                X_sq_local = T.sblock_alloc_buffer((1, 8192), data=X_sq_local.data, scope="local.fragment")
                sum_row = T.sblock_alloc_buffer((1,), data=sum_row.data, scope="local.fragment")
                sumsq_row = T.sblock_alloc_buffer((1,), data=sumsq_row.data, scope="local.fragment")
                mean_row = T.sblock_alloc_buffer((1,), data=mean_row.data, scope="local.fragment")
                rstd_row = T.sblock_alloc_buffer((1,), data=rstd_row.data, scope="local.fragment")
                T.copy(T.region(X[bx, 0], 1, 1, 8192), T.region(X_smem[0, 0], 2, 1, 8192))
                T.copy(T.region(gamma[0], 1, 8192), T.region(G_smem[0], 2, 8192))
                T.copy(T.region(beta[0], 1, 8192), T.region(B_smem[0], 2, 8192))
                for i in T.parallel(1, annotations={"parallel_loop_layout": metadata["tl.Fragment"][6]}):
                    for j in T.parallel(8192):
                        X_local[0, j] = T.Cast("float32", X_smem[0, j])
                for i in T.parallel(1, annotations={"parallel_loop_layout": metadata["tl.Fragment"][7]}):
                    for j in T.parallel(8192):
                        X_sq_local[0, j] = X_local[0, j] * X_local[0, j]
                T.reduce(T.region(X_local[0, 0], 1, 1, 8192), T.region(sum_row[0], 2, 1), "sum", 1, T.bool(True))
                T.reduce(T.region(X_sq_local[0, 0], 1, 1, 8192), T.region(sumsq_row[0], 2, 1), "sum", 1, T.bool(True))
                for i in T.parallel(1, annotations={"parallel_loop_layout": metadata["tl.Fragment"][8]}):
                    mean_row[0] = sum_row[0] * T.float32(0.0001220703125)
                    rstd_row[0] = T.rsqrt(sumsq_row[0] * T.float32(0.0001220703125) - mean_row[0] * mean_row[0] + T.float32(1.0000000000000001e-05))
                    Mean[bx] = mean_row[0]
                    Rstd[bx] = rstd_row[0]
                for i in T.parallel(1, annotations={"parallel_loop_layout": metadata["tl.Fragment"][9]}):
                    for j in T.parallel(8192):
                        norm: T.float32 = (X_local[0, j] - mean_row[0]) * rstd_row[0]
                        X_smem[0, j] = T.Cast("bfloat16", norm * T.Cast("float32", G_smem[j]) + T.Cast("float32", B_smem[j]))
                T.copy(T.region(X_smem[0, 0], 1, 1, 8192), T.region(Y[bx, 0], 2, 1, 8192))

# Metadata omitted. Use show_meta=True in script() method to show it.