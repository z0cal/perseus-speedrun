// Host shim: lets the real kernel sources (crates/pc-kernels/kernels/*.cuh) compile with g++ and run on the CPU.
// Only the CUDA vocabulary is defined here; no kernel logic lives in this file.
#pragma once
#include <cstddef>
#include <cstdint>
typedef void* cudaStream_t;
typedef int cudaError_t;
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(...)
#define __shared__ static thread_local
struct HostIdx { int x, y, z; };
extern thread_local HostIdx blockIdx, threadIdx;
extern thread_local int pc_host_phase;
// A block runs every thread twice: phase 0 stops at the barrier (shared-memory prologue), phase 1 runs through. The
// prologue only writes constants, so running it again is harmless.
#define __syncthreads() do { if (pc_host_phase == 0) return; } while (0)
inline uint32_t __umulhi(uint32_t a, uint32_t b) { return (uint32_t)(((uint64_t)a * b) >> 32); }
inline uint64_t __umul64hi(uint64_t a, uint64_t b) { return (uint64_t)(((unsigned __int128)a * b) >> 64); }
inline uint32_t __brev(uint32_t v) { uint32_t r = 0; for (int i = 0; i < 32; ++i) { r = (r << 1) | (v & 1); v >>= 1; } return r; }
struct uint4 { uint32_t x, y, z, w; };
struct ushort4 { uint16_t x, y, z, w; };
struct ulonglong2 { unsigned long long x, y; };
inline uint4 make_uint4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { return {a, b, c, d}; }
inline ushort4 make_ushort4(uint16_t a, uint16_t b, uint16_t c, uint16_t d) { return {a, b, c, d}; }
inline ulonglong2 make_ulonglong2(unsigned long long a, unsigned long long b) { return {a, b}; }
template <class T> inline T __ldg(const T* p) { return *p; }
template <class T> inline T __shfl_xor_sync(unsigned, T v, int = 0) { return v; }
