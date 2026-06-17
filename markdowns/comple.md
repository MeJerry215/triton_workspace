# triton 编译


```shell
$ cd $HOME/llvm-project  # your clone of LLVM.
$ mkdir build
$ cd build
$ cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON ../llvm -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU"
$ ninja

# Modify as appropriate to point to your LLVM build.
$ export LLVM_BUILD_DIR=$HOME/llvm-project/build

$ cd <triton install>
$ LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include \
  LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib \
  LLVM_SYSPATH=$LLVM_BUILD_DIR \
  pip install -e .

```