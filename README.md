# libthom

`libthom` is a collection of CUDA implementations for common linear algebra operations. It includes hand-written kernels and reference implementations such as cuBLAS and Thrust, with command-line tools to check results and compare performance.

![libthom performance plots: SGEMV on A100, SDOT and Reduction on RTX 2060](README-banner.png)

## Routines

| Routine | Description | Details |
| --- | --- | --- |
| `reduction` | Sum a vector with custom CUDA and Thrust implementations. | [Routine notes and benchmark](routines/reduction/README.md) |
| `sdot` | Single-precision dot product. | [Routine notes](routines/sdot/README.md) |
| `sgemv` | Single-precision general matrix-vector multiplication. | [Routine notes](routines/sgemv/README.md) |
| `dgemm` | Double-precision general matrix multiplication. | [Source and implementations](routines/dgemm/) |
| `hgemm` | Half-precision general matrix multiplication. | [Source and implementations](routines/hgemm/) |

The plots above show selected benchmark results: SGEMV on an NVIDIA A100, and SDOT and Reduction on an NVIDIA RTX 2060. Performance depends on the GPU, CUDA toolkit, and selected implementation.

## Requirements

- NVIDIA GPU and a compatible NVIDIA driver
- CUDA Toolkit with `nvcc`, cuBLAS, and cuRAND
- CMake 3.28 or newer
- A C++ compiler supported by the installed CUDA Toolkit

The build compiles for all CUDA architectures by default. To target a specific compute capability, set `ARCH` (for example, `80` for A100). Choose an architecture supported by your GPU and CUDA Toolkit.

## Build

Build all routines:

```sh
cmake -S . -B build -DARCH=80
cmake --build build
```

Build a single routine:

```sh
cmake -S . -B build -DARCH=80 -DROUTINE=sgemv
cmake --build build
```

Available `ROUTINE` values are `reduction`, `sdot`, `sgemv`, `dgemm`, and `hgemm`. Use `-DROUTINE=all` (the default) to build them all.

## Run

The executable is `build/libthom`. Each routine has its own implementations and arguments. The general command format is:

```sh
./build/libthom <test|bench> <routine> <implementation> [arguments...]
```

For example, list the implementation names in the selected routine's source, then run a correctness check or a benchmark:

```sh
./build/libthom test reduction warp 1000000 10
./build/libthom bench reduction warp
```

Exact argument counts vary by routine. The `test` mode accepts a size, iteration count, and optional result-check flag; `bench` runs the routine's configured set of sizes. Some implementations are only compiled for certain GPU architectures. See the routine source and notes for implementation-specific details.
