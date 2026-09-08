# from tvm.script import ir as I
# from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func
    def main(X_handle: T.handle, gamma_handle: T.handle, beta_handle: T.handle, Y_handle: T.handle, Mean_handle: T.handle, Rstd_handle: T.handle):
        X = T.match_buffer(X_handle, (4096, 8192), "bfloat16", strides=(8192, 1))
        gamma = T.match_buffer(gamma_handle, (8192,), "bfloat16", strides=(1,))
        beta = T.match_buffer(beta_handle, (8192,), "bfloat16", strides=(1,))
        Y = T.match_buffer(Y_handle, (4096, 8192), "bfloat16", strides=(8192, 1))
        Mean = T.match_buffer(Mean_handle, (4096,), strides=(1,))
        Rstd = T.match_buffer(Rstd_handle, (4096,), strides=(1,))
        # with T.sblock("root"):
        for bx in T.thread_binding(4096, thread="blockIdx.x"):
            for tx in T.thread_binding(256, thread="threadIdx.x"):
                for ty in T.thread_binding(1, thread="threadIdx.y"):
                    for tz in T.thread_binding(1, thread="threadIdx.z"):
                        with T.sblock("tilelang_root"):
                            T.reads()
                            T.writes()
                            X_smem = T.sblock_alloc_buffer((1, 8192), "bfloat16", scope="shared.dyn")
                            G_smem = T.sblock_alloc_buffer((8192,), "bfloat16", scope="shared.dyn")
                            B_smem = T.sblock_alloc_buffer((8192,), "bfloat16", scope="shared.dyn")
                            X_local = T.sblock_alloc_buffer((1, 8192), scope="local.fragment")
                            X_sq_local = T.sblock_alloc_buffer((1, 8192), scope="local.fragment")
                            sum_row = T.sblock_alloc_buffer((1,), scope="local.fragment")
                            sumsq_row = T.sblock_alloc_buffer((1,), scope="local.fragment")
                            mean_row = T.sblock_alloc_buffer((1,), scope="local.fragment")
                            rstd_row = T.sblock_alloc_buffer((1,), scope="local.fragment")
                            T.copy(T.region(X[bx, 0], 1, 1, 8192), T.region(X_smem[0, 0], 2, 1, 8192))
                            T.copy(T.region(gamma[0], 1, 8192), T.region(G_smem[0], 2, 8192))
                            T.copy(T.region(beta[0], 1, 8192), T.region(B_smem[0], 2, 8192))
                            for i in T.parallel(1):
                                for j in T.parallel(8192):
                                    X_local[i, j] = T.Cast("float32", X_smem[i, j])
                            for i in T.parallel(1):
                                for j in T.parallel(8192):
                                    X_sq_local[i, j] = X_local[i, j] * X_local[i, j]
                            T.reduce(T.region(X_local[0, 0], 1, 1, 8192), T.region(sum_row[0], 2, 1), "sum", 1, T.bool(True))
                            T.reduce(T.region(X_sq_local[0, 0], 1, 1, 8192), T.region(sumsq_row[0], 2, 1), "sum", 1, T.bool(True))
                            inv_D: T.float32 = T.float32(1.0) / T.Cast("float32", 8192)
                            for i in T.parallel(1):
                                mean_row[i] = sum_row[i] * inv_D
                                rstd_row[i] = T.rsqrt(sumsq_row[i] * inv_D - mean_row[i] * mean_row[i] + T.Cast("float32", T.float32(1.0000000000000001e-05)))
                                Mean[bx + i] = mean_row[i]
                                Rstd[bx + i] = rstd_row[i]
                            for i in T.parallel(1):
                                for j in T.parallel(8192):
                                    norm: T.float32 = (X_local[i, j] - mean_row[i]) * rstd_row[i]
                                    X_smem[i, j] = T.Cast("bfloat16", norm * T.Cast("float32", G_smem[j]) + T.Cast("float32", B_smem[j]))
                            T.copy(T.region(X_smem[0, 0], 1, 1, 8192), T.region(Y[bx, 0], 2, 1, 8192))