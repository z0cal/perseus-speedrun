// Launch geometry copied from the launchers at the end of bts.cuh (mod_raise_l, small_int_l), W = u32, MM = Barrett64
// (the instantiation pc-py's session uses).
#include <cuda_runtime.h>
#include "bts_host.cuh"
thread_local HostIdx blockIdx, threadIdx;
thread_local int pc_host_phase;
using W = uint32_t;
using MM = pc::Barrett64<W>;
extern "C" int host_mod_raise(const ModRaiseArgs* a) {
    if (a->d < 1 || a->d > 4 || a->r0 < a->d || a->r1 > 64 || a->r1 <= a->r0 || a->nparts < 1 || a->nparts > 2) return -1;
    for (int z = 0; z < a->K * a->nparts; ++z)
        for (int y = 0; y < a->r1 - a->r0; ++y)
            for (int x = 0; x < pc::N / pc::bts::BX; ++x) {
                blockIdx = {x, y, z};
                for (int t = 0; t < pc::bts::BX; ++t) {
                    threadIdx = {t, 0, 0};
                    pc::bts::mod_raise<W, MM>(*a);
                }
            }
    return 0;
}
extern "C" int host_small_int(const SmallIntArgs* a) {
    if (a->nparts < 1 || a->nparts > 2 || a->K < 1) return -1;
    for (int z = 0; z < a->K; ++z)
        for (int y = 0; y < a->nparts; ++y)
            for (int x = 0; x < pc::N / pc::bts::BX; ++x) {
                blockIdx = {x, y, z};
                for (int ph = 0; ph < 2; ++ph) {
                    pc_host_phase = ph;
                    for (int t = 0; t < pc::bts::BX; ++t) {
                        threadIdx = {t, 0, 0};
                        pc::bts::small_int<W, MM>(*a);
                    }
                }
            }
    return 0;
}
