// HIP virtual memory management (VMM) probe for gfx1151: reserve/create/
// map/access/unmap/release, kernel and copy access across physical chunks,
// captured-graph replay after more chunks are mapped behind the same VA,
// map/unmap cost, GTT/VRAM accounting, and gather bandwidth vs hipMalloc.
//
//   tools/bench/build.sh tools/bench/vmm_probe.hip && /tmp/vmm_probe
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

#define CK(x)                                                               \
  do {                                                                      \
    const hipError_t e_ = (x);                                              \
    if (e_ != hipSuccess) {                                                 \
      std::printf("FAIL line %d: %s -> %s\n", __LINE__, #x,                 \
                  hipGetErrorString(e_));                                   \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

namespace {

constexpr std::size_t kMiB = std::size_t{1} << 20;

long long Sys(const char* name) {
  std::ifstream f(std::string("/sys/class/drm/renderD128/device/") + name);
  long long v = -1;
  f >> v;
  return v;
}

struct Mem {
  long long gtt, vram;
  std::size_t free;
};
Mem Snap() {
  std::size_t free = 0, total = 0;
  (void)hipMemGetInfo(&free, &total);
  return {Sys("mem_info_gtt_used"), Sys("mem_info_vram_used"), free};
}
void Delta(const char* what, const Mem& a, const Mem& b) {
  std::printf("  %-34s gtt %+8.1f MiB  vram %+8.1f MiB  hipfree %+8.1f MiB\n",
              what, double(b.gtt - a.gtt) / kMiB, double(b.vram - a.vram) / kMiB,
              (double(b.free) - double(a.free)) / kMiB);
}

double Us(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - t0)
      .count();
}

hipMemAllocationProp Prop() {
  hipMemAllocationProp p{};
  p.type = hipMemAllocationTypePinned;
  p.location.type = hipMemLocationTypeDevice;
  p.location.id = 0;
  return p;
}

void Access(void* va, std::size_t bytes) {
  hipMemAccessDesc d{};
  d.location.type = hipMemLocationTypeDevice;
  d.location.id = 0;
  d.flags = hipMemAccessFlagsProtReadWrite;
  CK(hipMemSetAccess(va, bytes, &d, 1));
}

__global__ void Fill(std::uint32_t* p, std::size_t n, std::uint32_t salt) {
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x)
    p[i] = std::uint32_t(i) * 2654435761u ^ salt;
}

__global__ void Verify(const std::uint32_t* p, std::size_t n,
                       std::uint32_t salt, unsigned long long* bad) {
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x)
    if (p[i] != (std::uint32_t(i) * 2654435761u ^ salt))
      atomicAdd(bad, 1ull);
}

// Graph-replayable row write/read: the row comes from device memory, the
// base pointer is captured, like the decode graphs' KV stores.
__global__ void WriteRow(std::uint32_t* base, const std::uint32_t* row,
                         std::uint32_t words, std::uint32_t value) {
  base[std::size_t{*row} * words + threadIdx.x] = value + threadIdx.x;
}
__global__ void SumRow(const std::uint32_t* base, const std::uint32_t* row,
                       std::uint32_t words, unsigned long long* out) {
  atomicAdd(out, base[std::size_t{*row} * words + threadIdx.x]);
}

// One 1 KiB K-cache-like row per 64-thread block, rows picked at random.
__global__ void Gather(const uint4* base, const std::uint32_t* rows,
                       std::uint32_t n_rows, uint4* sink) {
  uint4 acc{0, 0, 0, 0};
  for (std::uint32_t r = blockIdx.x; r < n_rows; r += gridDim.x) {
    const uint4 v = base[std::size_t{rows[r]} * 64 + threadIdx.x];
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  }
  if (acc.x == 0x12345678u)
    sink[threadIdx.x] = acc;
}
__global__ void Stream(const uint4* base, std::size_t n, uint4* sink) {
  uint4 acc{0, 0, 0, 0};
  for (std::size_t i = blockIdx.x * std::size_t{blockDim.x} + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    const uint4 v = base[i];
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  }
  if (acc.x == 0x12345678u)
    sink[threadIdx.x] = acc;
}

// A VA range backed by `chunks` physical allocations of `chunk` bytes.
struct Vmm {
  void* va{nullptr};
  std::size_t reserved{0}, chunk{0};
  std::vector<hipMemGenericAllocationHandle_t> handles;
  Vmm(std::size_t reserve, std::size_t chunk_bytes)
      : reserved(reserve), chunk(chunk_bytes) {
    CK(hipMemAddressReserve(&va, reserved, 0, nullptr, 0));
  }
  void Grow(std::size_t chunks) {
    const auto prop = Prop();
    for (std::size_t i = handles.size(); i < chunks; ++i) {
      hipMemGenericAllocationHandle_t h{};
      CK(hipMemCreate(&h, chunk, &prop, 0));
      CK(hipMemMap(static_cast<char*>(va) + i * chunk, chunk, 0, h, 0));
      Access(static_cast<char*>(va) + i * chunk, chunk);
      handles.push_back(h);
    }
  }
  void Shrink(std::size_t chunks) {
    while (handles.size() > chunks) {
      CK(hipMemUnmap(static_cast<char*>(va) + (handles.size() - 1) * chunk,
                     chunk));
      CK(hipMemRelease(handles.back()));
      handles.pop_back();
    }
  }
  ~Vmm() {
    Shrink(0);
    CK(hipMemAddressFree(va, reserved));
  }
};

void Timing(std::size_t chunk, std::size_t n) {
  const auto prop = Prop();
  void* va = nullptr;
  CK(hipMemAddressReserve(&va, chunk * n, 0, nullptr, 0));
  std::vector<hipMemGenericAllocationHandle_t> h(n);
  double create = 0, map = 0, access = 0, unmap = 0, release = 0;
  const Mem m0 = Snap();
  for (std::size_t i = 0; i < n; ++i) {
    auto t = std::chrono::steady_clock::now();
    CK(hipMemCreate(&h[i], chunk, &prop, 0));
    create += Us(t);
    t = std::chrono::steady_clock::now();
    CK(hipMemMap(static_cast<char*>(va) + i * chunk, chunk, 0, h[i], 0));
    map += Us(t);
    t = std::chrono::steady_clock::now();
    Access(static_cast<char*>(va) + i * chunk, chunk);
    access += Us(t);
  }
  const Mem m1 = Snap();
  Fill<<<256, 256>>>(static_cast<std::uint32_t*>(va), chunk * n / 4, 7);
  CK(hipDeviceSynchronize());
  for (std::size_t i = 0; i < n; ++i) {
    auto t = std::chrono::steady_clock::now();
    CK(hipMemUnmap(static_cast<char*>(va) + i * chunk, chunk));
    unmap += Us(t);
    t = std::chrono::steady_clock::now();
    CK(hipMemRelease(h[i]));
    release += Us(t);
  }
  const Mem m2 = Snap();
  CK(hipMemAddressFree(va, chunk * n));
  std::printf(
      "chunk %4zu MiB x%3zu: create %7.1f us  map %7.1f us  access %7.1f us  "
      "unmap %7.1f us  release %7.1f us (per chunk)\n",
      chunk / kMiB, n, create / n, map / n, access / n, unmap / n,
      release / n);
  Delta("mapped (vs before create)", m0, m1);
  Delta("released (vs before create)", m0, m2);
}

double GatherGBs(const void* base, std::size_t bytes, const std::uint32_t* rows,
                 std::uint32_t n_rows, uint4* sink) {
  hipEvent_t a, b;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  Gather<<<4096, 64>>>(static_cast<const uint4*>(base), rows, n_rows, sink);
  CK(hipEventRecord(a));
  for (int i = 0; i < 5; ++i)
    Gather<<<4096, 64>>>(static_cast<const uint4*>(base), rows, n_rows, sink);
  CK(hipEventRecord(b));
  CK(hipEventSynchronize(b));
  float ms = 0;
  CK(hipEventElapsedTime(&ms, a, b));
  (void)bytes;
  return 5.0 * n_rows * 1024.0 / (ms * 1e6);
}
double StreamGBs(const void* base, std::size_t bytes, uint4* sink) {
  hipEvent_t a, b;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  Stream<<<1024, 256>>>(static_cast<const uint4*>(base), bytes / 16, sink);
  CK(hipEventRecord(a));
  for (int i = 0; i < 5; ++i)
    Stream<<<1024, 256>>>(static_cast<const uint4*>(base), bytes / 16, sink);
  CK(hipEventRecord(b));
  CK(hipEventSynchronize(b));
  float ms = 0;
  CK(hipEventElapsedTime(&ms, a, b));
  return 5.0 * bytes / (ms * 1e6);
}


// When does physical memory go back: unmap, release, address free, later?
// Is a released chunk reused by the next create? Can a handle move to
// another VA (a recycle cache)?
void ReleaseTest() {
  const auto prop = Prop();
  const std::size_t chunk = 2 * kMiB, n = 64;
  void* va = nullptr;
  void* vb = nullptr;
  CK(hipMemAddressReserve(&va, chunk * n, 0, nullptr, 0));
  CK(hipMemAddressReserve(&vb, chunk * n, 0, nullptr, 0));
  std::vector<hipMemGenericAllocationHandle_t> h(n);
  auto map_all = [&](void* base) {
    for (std::size_t i = 0; i < n; ++i) {
      CK(hipMemMap(static_cast<char*>(base) + i * chunk, chunk, 0, h[i], 0));
      Access(static_cast<char*>(base) + i * chunk, chunk);
    }
  };
  auto unmap_all = [&](void* base) {
    for (std::size_t i = 0; i < n; ++i)
      CK(hipMemUnmap(static_cast<char*>(base) + i * chunk, chunk));
  };
  const Mem m0 = Snap();
  for (auto& x : h)
    CK(hipMemCreate(&x, chunk, &prop, 0));
  Mem m = Snap();
  Delta("create 64 x 2 MiB (unmapped)", m0, m);
  map_all(va);
  Fill<<<256, 256>>>(static_cast<std::uint32_t*>(va), chunk * n / 4, 5);
  CK(hipDeviceSynchronize());
  Delta("map + touch", m0, m = Snap());
  unmap_all(va);
  Delta("unmap all", m0, m = Snap());
  // Recycle: same handles mapped into another reservation keep their data.
  map_all(vb);
  unsigned long long* bad = nullptr;
  CK(hipMalloc(&bad, 8));
  CK(hipMemset(bad, 0, 8));
  Verify<<<256, 256>>>(static_cast<std::uint32_t*>(vb), chunk * n / 4, 5, bad);
  unsigned long long h_bad = 1;
  CK(hipMemcpy(&h_bad, bad, 8, hipMemcpyDeviceToHost));
  CK(hipFree(bad));
  Delta("remap handles into 2nd VA", m0, m = Snap());
  std::printf("  recycled-handle data mismatches: %llu\n", h_bad);
  unmap_all(vb);
  for (auto x : h)
    CK(hipMemRelease(x));
  Delta("unmap + release all", m0, m = Snap());
  usleep(300000);
  Delta("+300 ms", m0, m = Snap());
  for (auto& x : h)
    CK(hipMemCreate(&x, chunk, &prop, 0));
  map_all(va);
  Delta("create + map again (reuse?)", m0, m = Snap());
  // Release while mapped, then unmap.
  for (auto x : h)
    CK(hipMemRelease(x));
  Delta("release while mapped", m0, m = Snap());
  unmap_all(va);
  Delta("then unmap", m0, m = Snap());
  CK(hipMemAddressFree(va, chunk * n));
  CK(hipMemAddressFree(vb, chunk * n));
  Delta("address free", m0, m = Snap());
  usleep(300000);
  Delta("+300 ms", m0, m = Snap());
  void* p = nullptr;
  CK(hipMalloc(&p, 128 * kMiB));
  Fill<<<256, 256>>>(static_cast<std::uint32_t*>(p), 32 * kMiB, 1);
  CK(hipDeviceSynchronize());
  Delta("control: hipMalloc 128 MiB", m0, m = Snap());
  CK(hipFree(p));
  Delta("control: hipFree", m0, m = Snap());
  // Churn: 20 grow/release cycles of 128 MiB must not accumulate.
  for (int cycle = 0; cycle < 20; ++cycle) {
    Vmm v(chunk * n, chunk);
    v.Grow(n);
  }
  Delta("after 20 x 128 MiB VMM churn", m0, m = Snap());
}

// Physical pages return only when their reservation is freed. Can pieces
// get their own reservations at fixed addresses (hinted), be freed one by
// one and re-reserved at the same address? Is a sub-range free legal? Does
// freeing a reservation a recycled handle was once mapped into keep it alive?
void PieceTest() {
  const auto prop = Prop();
  const std::size_t chunk = 2 * kMiB, n = 16;
  void* base = nullptr;
  CK(hipMemAddressReserve(&base, chunk * n, 2 * kMiB, nullptr, 0));
  CK(hipMemAddressFree(base, chunk * n));
  std::vector<void*> va(n);
  int honored = 0;
  for (std::size_t i = 0; i < n; ++i) {
    void* want = static_cast<char*>(base) + i * chunk;
    const hipError_t e = hipMemAddressReserve(&va[i], chunk, 0, want, 0);
    honored += e == hipSuccess && va[i] == want;
    if (e != hipSuccess) {
      std::printf("hinted reserve %zu failed: %s\n", i, hipGetErrorString(e));
      return;
    }
  }
  std::printf("hinted per-piece reservations honored: %d of %zu\n", honored, n);
  std::vector<hipMemGenericAllocationHandle_t> h(n);
  const Mem m0 = Snap();
  Mem m;
  for (std::size_t i = 0; i < n; ++i) {
    CK(hipMemCreate(&h[i], chunk, &prop, 0));
    CK(hipMemMap(va[i], chunk, 0, h[i], 0));
    Access(va[i], chunk);
  }
  // One kernel across the adjacent per-piece reservations.
  Fill<<<256, 256>>>(static_cast<std::uint32_t*>(va[0]), chunk * n / 4, 3);
  unsigned long long* bad = nullptr;
  CK(hipMalloc(&bad, 8));
  CK(hipMemset(bad, 0, 8));
  Verify<<<256, 256>>>(static_cast<std::uint32_t*>(va[0]), chunk * n / 4, 3,
                       bad);
  unsigned long long h_bad = 1;
  CK(hipMemcpy(&h_bad, bad, 8, hipMemcpyDeviceToHost));
  std::printf("  kernel across adjacent reservations: mismatches %llu\n",
              h_bad);
  Delta("map 16 pieces", m0, m = Snap());
  double t_free = 0, t_again = 0;
  for (std::size_t i = n / 2; i < n; ++i) {
    auto t = std::chrono::steady_clock::now();
    CK(hipMemUnmap(va[i], chunk));
    CK(hipMemRelease(h[i]));
    CK(hipMemAddressFree(va[i], chunk));
    t_free += Us(t);
  }
  Delta("free upper 8 (unmap+release+addrfree)", m0, m = Snap());
  int same = 0;
  for (std::size_t i = n / 2; i < n; ++i) {
    void* want = static_cast<char*>(base) + i * chunk;
    auto t = std::chrono::steady_clock::now();
    CK(hipMemAddressReserve(&va[i], chunk, 0, want, 0));
    CK(hipMemCreate(&h[i], chunk, &prop, 0));
    CK(hipMemMap(va[i], chunk, 0, h[i], 0));
    Access(va[i], chunk);
    t_again += Us(t);
    same += va[i] == want;
  }
  std::printf("  re-reserved at same address: %d of %zu; free %.1f us, "
              "re-commit %.1f us per piece\n",
              same, n / 2, t_free / (n / 2), t_again / (n / 2));
  Fill<<<256, 256>>>(static_cast<std::uint32_t*>(va[0]), chunk * n / 4, 4);
  CK(hipMemset(bad, 0, 8));
  Verify<<<256, 256>>>(static_cast<std::uint32_t*>(va[0]), chunk * n / 4, 4,
                       bad);
  CK(hipMemcpy(&h_bad, bad, 8, hipMemcpyDeviceToHost));
  std::printf("  after re-commit mismatches %llu\n", h_bad);
  Delta("re-committed", m0, m = Snap());
  // Recycled handle: mapped in A, moved to B, then A freed.
  void* a = nullptr;
  void* b = nullptr;
  CK(hipMemAddressReserve(&a, chunk, 0, nullptr, 0));
  CK(hipMemAddressReserve(&b, chunk, 0, nullptr, 0));
  hipMemGenericAllocationHandle_t x{};
  CK(hipMemCreate(&x, chunk, &prop, 0));
  CK(hipMemMap(a, chunk, 0, x, 0));
  Access(a, chunk);
  Fill<<<256, 256>>>(static_cast<std::uint32_t*>(a), chunk / 4, 8);
  CK(hipDeviceSynchronize());
  CK(hipMemUnmap(a, chunk));
  CK(hipMemMap(b, chunk, 0, x, 0));
  Access(b, chunk);
  const Mem mb = Snap();
  CK(hipMemAddressFree(a, chunk));
  CK(hipMemset(bad, 0, 8));
  Verify<<<256, 256>>>(static_cast<std::uint32_t*>(b), chunk / 4, 8, bad);
  CK(hipMemcpy(&h_bad, bad, 8, hipMemcpyDeviceToHost));
  std::printf("  moved handle after freeing first VA: mismatches %llu\n",
              h_bad);
  Delta("free first VA of moved handle", mb, Snap());
  CK(hipMemUnmap(b, chunk));
  CK(hipMemRelease(x));
  CK(hipMemAddressFree(b, chunk));
  // Sub-range free of one big reservation.
  void* big = nullptr;
  CK(hipMemAddressReserve(&big, chunk * 4, 0, nullptr, 0));
  CK(hipMemCreate(&x, chunk, &prop, 0));
  CK(hipMemMap(static_cast<char*>(big) + chunk, chunk, 0, x, 0));
  CK(hipMemUnmap(static_cast<char*>(big) + chunk, chunk));
  CK(hipMemRelease(x));
  const Mem ms = Snap();
  const hipError_t sub =
      hipMemAddressFree(static_cast<char*>(big) + chunk, chunk);
  std::printf("  sub-range hipMemAddressFree: %s\n", hipGetErrorString(sub));
  Delta("after sub-range free", ms, Snap());
  // Freeing the whole range after a sub-range free crashes the runtime.
  for (std::size_t i = 0; i < n; ++i) {
    CK(hipMemUnmap(va[i], chunk));
    CK(hipMemRelease(h[i]));
    CK(hipMemAddressFree(va[i], chunk));
  }
  CK(hipFree(bad));
  Delta("all freed", m0, Snap());
}

long RssKiB() {
  std::ifstream f("/proc/self/status");
  std::string key;
  long v = -1;
  while (f >> key) {
    if (key == "VmRSS:") {
      f >> v;
      break;
    }
  }
  return v;
}

// Cost of many small reservations (one per committed piece).
void ReservationScale(std::size_t count) {
  void* dev = nullptr;
  CK(hipMalloc(&dev, 64));
  auto copy_us = [&] {
    std::uint32_t x = 0;
    auto t = std::chrono::steady_clock::now();
    for (int i = 0; i < 200; ++i)
      CK(hipMemcpy(&x, dev, 4, hipMemcpyDeviceToHost));
    return Us(t) / 200;
  };
  const double before = copy_us();
  const long rss0 = RssKiB();
  std::vector<void*> va(count);
  auto t = std::chrono::steady_clock::now();
  for (auto& v : va)
    CK(hipMemAddressReserve(&v, 2 * kMiB, 0, nullptr, 0));
  const double reserve = Us(t) / count;
  const long rss1 = RssKiB();
  const double after = copy_us();
  t = std::chrono::steady_clock::now();
  for (auto* v : va)
    CK(hipMemAddressFree(v, 2 * kMiB));
  const double release = Us(t) / count;
  std::printf("%zu reservations: reserve %.2f us, free %.2f us each, host RSS "
              "%+ld KiB, 4 B D2H copy %.1f -> %.1f us\n",
              count, reserve, release, rss1 - rss0, before, after);
  CK(hipFree(dev));
}

// Copy latency with many mapped handles (one reservation), and a copy
// spanning two reservations.
void HandleScale(std::size_t count, std::size_t piece) {
  void* dev = nullptr;
  CK(hipMalloc(&dev, 64));
  auto copy_us = [&] {
    std::uint32_t x = 0;
    auto t = std::chrono::steady_clock::now();
    for (int i = 0; i < 200; ++i)
      CK(hipMemcpy(&x, dev, 4, hipMemcpyDeviceToHost));
    return Us(t) / 200;
  };
  const double before = copy_us();
  Vmm v(count * piece, piece);
  auto t = std::chrono::steady_clock::now();
  v.Grow(count);
  const double grow = Us(t) / count;
  const double after = copy_us();
  std::vector<std::uint8_t> host(4 * piece);
  t = std::chrono::steady_clock::now();
  CK(hipMemcpy(host.data(), static_cast<char*>(v.va) + piece / 2, host.size(),
               hipMemcpyDeviceToHost));
  std::printf("%zu x %zu KiB handles: grow %.1f us/piece, 4 B D2H %.1f -> "
              "%.1f us, 4-piece D2H %.1f us\n",
              count, piece >> 10, grow, before, after, Us(t));
  void* a = nullptr;
  CK(hipMemAddressReserve(&a, 2 * piece, 0, nullptr, 0));
  CK(hipMemAddressFree(a, 2 * piece));
  void* r0 = nullptr;
  void* r1 = nullptr;
  CK(hipMemAddressReserve(&r0, piece, 0, a, 0));
  CK(hipMemAddressReserve(&r1, piece, 0, static_cast<char*>(a) + piece, 0));
  const auto prop = Prop();
  hipMemGenericAllocationHandle_t h0{}, h1{};
  CK(hipMemCreate(&h0, piece, &prop, 0));
  CK(hipMemCreate(&h1, piece, &prop, 0));
  CK(hipMemMap(r0, piece, 0, h0, 0));
  CK(hipMemMap(r1, piece, 0, h1, 0));
  Access(r0, piece);
  Access(r1, piece);
  const hipError_t cross = hipMemcpy(host.data(), static_cast<char*>(r0) + 64,
                                     piece, hipMemcpyDeviceToHost);
  std::printf("copy spanning two reservations: %s\n", hipGetErrorString(cross));
  CK(hipMemUnmap(r0, piece));
  CK(hipMemUnmap(r1, piece));
  CK(hipMemRelease(h0));
  CK(hipMemRelease(h1));
  CK(hipMemAddressFree(r0, piece));
  CK(hipMemAddressFree(r1, piece));
  CK(hipFree(dev));
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  int vmm = -1;
  CK(hipDeviceGetAttribute(
      &vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0));
  hipDeviceProp_t props{};
  CK(hipGetDeviceProperties(&props, 0));
  int rt = 0, drv = 0;
  CK(hipRuntimeGetVersion(&rt));
  CK(hipDriverGetVersion(&drv));
  std::printf("device %s (%s) runtime %d driver %d VMM attribute %d\n",
              props.name, props.gcnArchName, rt, drv, vmm);
  const auto prop = Prop();
  std::size_t gmin = 0, grec = 0;
  CK(hipMemGetAllocationGranularity(&gmin, &prop,
                                    hipMemAllocationGranularityMinimum));
  CK(hipMemGetAllocationGranularity(&grec, &prop,
                                    hipMemAllocationGranularityRecommended));
  std::printf("granularity: minimum %zu B, recommended %zu B\n", gmin, grec);
  {
    // Runtime warm-up so later deltas exclude one-time allocations.
    void* w = nullptr;
    CK(hipMalloc(&w, kMiB));
    Fill<<<1, 64>>>(static_cast<std::uint32_t*>(w), 64, 0);
    CK(hipDeviceSynchronize());
    CK(hipFree(w));
  }
  if (argc > 1 && std::string(argv[1]) == "handles") {
    HandleScale(2048, std::size_t{1} << 20);
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "scale") {
    ReservationScale(1000);
    ReservationScale(60000);
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "pieces") {
    PieceTest();
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "release") {
    ReleaseTest();
    return 0;
  }

  // 1. Reservations cost no memory; many large ones fit.
  {
    const Mem m0 = Snap();
    std::vector<void*> vas(2560);
    for (auto& va : vas)
      CK(hipMemAddressReserve(&va, 128 * kMiB, 0, nullptr, 0));
    const Mem m1 = Snap();
    Delta("2560 x 128 MiB reservations", m0, m1);
    for (auto* va : vas)
      CK(hipMemAddressFree(va, 128 * kMiB));
  }

  // 2. Kernel + copy access across chunks; memory only when mapped.
  {
    const std::size_t chunk = std::max<std::size_t>(gmin, 2 * kMiB);
    if (chunk > 8 * kMiB) {
      std::printf("minimum granularity too large for this probe\n");
      return 1;
    }
    const Mem m0 = Snap();
    Vmm v(64 * chunk, chunk);
    std::printf("VA %p (2 MiB aligned: %d)\n", v.va,
                int(reinterpret_cast<std::uintptr_t>(v.va) % (2 * kMiB) == 0));
    const Mem m1 = Snap();
    Delta("reserve 4 GiB VA", m0, m1);
    v.Grow(8);
    const Mem m2 = Snap();
    Delta("map 8 chunks", m1, m2);
    const std::size_t n = 8 * chunk / 4;
    auto* p = static_cast<std::uint32_t*>(v.va);
    Fill<<<256, 256>>>(p, n, 11);
    CK(hipDeviceSynchronize());
    const Mem m3 = Snap();
    Delta("after kernel touch", m2, m3);
    unsigned long long* bad = nullptr;
    CK(hipMalloc(&bad, sizeof(*bad)));
    CK(hipMemset(bad, 0, sizeof(*bad)));
    Verify<<<256, 256>>>(p, n, 11, bad);
    unsigned long long h_bad = 1;
    CK(hipMemcpy(&h_bad, bad, sizeof(h_bad), hipMemcpyDeviceToHost));
    // Copies that straddle a chunk boundary, pageable and pinned host.
    std::vector<std::uint32_t> host(chunk / 4);
    const std::size_t off = chunk / 4 * 3 / 4;  // words, 3/4 into chunk 0
    CK(hipMemcpy(host.data(), p + off, host.size() * 4,
                 hipMemcpyDeviceToHost));
    std::size_t copy_bad = 0;
    for (std::size_t i = 0; i < host.size(); ++i)
      copy_bad += host[i] != (std::uint32_t(off + i) * 2654435761u ^ 11);
    std::uint32_t* pinned = nullptr;
    CK(hipHostMalloc(&pinned, chunk));
    for (std::size_t i = 0; i < chunk / 4; ++i)
      pinned[i] = std::uint32_t(i) * 2654435761u ^ 99;
    hipStream_t s;
    CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    CK(hipMemcpyAsync(p + chunk / 8, pinned, chunk, hipMemcpyHostToDevice, s));
    CK(hipMemsetAsync(p + 3 * chunk / 4 - 256, 0, 2048, s));  // straddles
    CK(hipStreamSynchronize(s));
    CK(hipMemcpy(host.data(), p + chunk / 8, host.size() * 4,
                 hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < host.size(); ++i)
      copy_bad += host[i] != (std::uint32_t(i) * 2654435761u ^ 99);
    std::vector<std::uint32_t> zeros(512);
    CK(hipMemcpy(zeros.data(), p + 3 * chunk / 4 - 256, 2048,
                 hipMemcpyDeviceToHost));
    for (auto z : zeros)
      copy_bad += z != 0;
    // D2D between two VMM chunks and from hipMalloc.
    CK(hipMemcpyAsync(p + 11 * chunk / 8, p + chunk / 8, chunk,
                      hipMemcpyDeviceToDevice, s));
    CK(hipStreamSynchronize(s));
    CK(hipMemcpy(host.data(), p + 11 * chunk / 8, host.size() * 4,
                 hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < host.size(); ++i)
      copy_bad += host[i] != (std::uint32_t(i) * 2654435761u ^ 99);
    std::printf("kernel verify mismatches %llu, copy mismatches %zu\n", h_bad,
                copy_bad);

    // 3. Graph captured with only the first chunks mapped, replayed at rows
    // in chunks mapped later (and remapped to new physical memory).
    std::uint32_t* row = nullptr;
    unsigned long long* sum = nullptr;
    CK(hipMalloc(&row, 4));
    CK(hipMalloc(&sum, 8));
    const std::uint32_t words = 64;
    CK(hipMemsetAsync(row, 0, 4, s));
    hipGraph_t g;
    hipGraphExec_t ge;
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    CK(hipMemsetAsync(sum, 0, 8, s));
    WriteRow<<<1, words, 0, s>>>(p, row, words, 1000);
    SumRow<<<1, words, 0, s>>>(p, row, words, sum);
    CK(hipStreamEndCapture(s, &g));
    CK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    const unsigned long long expect = 64ull * 1000 + 64 * 63 / 2;
    auto replay = [&](std::uint32_t r) {
      CK(hipMemcpyAsync(row, &r, 4, hipMemcpyHostToDevice, s));
      CK(hipGraphLaunch(ge, s));
      unsigned long long got = 0;
      CK(hipMemcpyAsync(&got, sum, 8, hipMemcpyDeviceToHost, s));
      CK(hipStreamSynchronize(s));
      std::uint32_t direct = 0;
      CK(hipMemcpy(&direct, p + std::size_t{r} * words + 5, 4,
                   hipMemcpyDeviceToHost));
      return got == expect && direct == 1005;
    };
    const auto rows_per_chunk = std::uint32_t(chunk / 4 / words);
    const bool g0 = replay(3);
    v.Grow(64);  // mapped after instantiation
    const bool g1 = replay(rows_per_chunk * 40 + 7);
    const bool g2 = replay(rows_per_chunk * 63 + 1);
    v.Shrink(40);
    v.Grow(64);  // same VA, new physical chunks
    const bool g3 = replay(rows_per_chunk * 63 + 2);
    std::printf("graph replay: pre-map %d, later-mapped %d %d, remapped %d\n",
                g0, g1, g2, g3);
    CK(hipGraphExecDestroy(ge));
    CK(hipGraphDestroy(g));
    const Mem m4 = Snap();
    v.Shrink(1);
    const Mem m5 = Snap();
    Delta("shrink 64 -> 1 chunk", m4, m5);
    CK(hipFree(row));
    CK(hipFree(sum));
    CK(hipFree(bad));
    CK(hipHostFree(pinned));
    CK(hipStreamDestroy(s));
  }

  // 4. Map/unmap cost.
  Timing(2 * kMiB, 128);
  Timing(64 * kMiB, 8);

  // 5. Stream and random 1 KiB-row gather bandwidth (TLB/fragment effects).
  {
    const std::size_t bytes = 1024 * kMiB;
    const std::uint32_t n_rows = 1u << 20;
    std::vector<std::uint32_t> h_rows(n_rows);
    std::mt19937 rng(1);
    for (auto& r : h_rows)
      r = rng() % std::uint32_t(bytes / 1024);
    std::uint32_t* rows = nullptr;
    uint4* sink = nullptr;
    CK(hipMalloc(&rows, n_rows * 4));
    CK(hipMalloc(&sink, 64 * 16));
    CK(hipMemcpy(rows, h_rows.data(), n_rows * 4, hipMemcpyHostToDevice));
    {
      void* p = nullptr;
      CK(hipMalloc(&p, bytes));
      Fill<<<256, 256>>>(static_cast<std::uint32_t*>(p), bytes / 4, 1);
      std::printf("hipMalloc   : stream %6.1f GB/s  gather %6.1f GB/s\n",
                  StreamGBs(p, bytes, sink),
                  GatherGBs(p, bytes, rows, n_rows, sink));
      CK(hipFree(p));
    }
    for (std::size_t chunk : {2 * kMiB, 64 * kMiB}) {
      Vmm v(bytes, chunk);
      v.Grow(bytes / chunk);
      Fill<<<256, 256>>>(static_cast<std::uint32_t*>(v.va), bytes / 4, 1);
      std::printf("VMM %3zu MiB : stream %6.1f GB/s  gather %6.1f GB/s\n",
                  chunk / kMiB, StreamGBs(v.va, bytes, sink),
                  GatherGBs(v.va, bytes, rows, n_rows, sink));
    }
    CK(hipFree(rows));
    CK(hipFree(sink));
  }
  std::printf("done\n");
  return 0;
}
