#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/core/hip/vmm.hpp"

using gufo::hip::GrowableBuffer;
using gufo::hip::PiecePool;

namespace {

void Require(bool ok, const std::string& message) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
    std::exit(1);
  }
}

std::size_t FreeBytes() {
  std::size_t free = 0, total = 0;
  Require(hipMemGetInfo(&free, &total) == hipSuccess, "hipMemGetInfo");
  return free;
}

std::vector<std::uint8_t> Read(const void* device, std::size_t bytes) {
  std::vector<std::uint8_t> host(bytes);
  Require(hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost) ==
              hipSuccess,
          "read back");
  return host;
}

// Graph-replayable write like the decode KV stores: the byte offset comes
// from device memory, the base pointer is captured.
__global__ void Fill(std::uint8_t* base, const std::size_t* offset,
                     std::uint8_t value) {
  base[*offset + blockIdx.x * blockDim.x + threadIdx.x] = value;
}

// Tags the first word of each piece through the kernel's page tables.
__global__ void Stamp(std::uint32_t* base, std::size_t piece_words,
                      std::uint32_t tag) {
  base[blockIdx.x * piece_words] = tag + blockIdx.x;
}

bool All(const std::vector<std::uint8_t>& bytes, std::uint8_t value) {
  for (auto b : bytes)
    if (b != value)
      return false;
  return true;
}

}  // namespace

int main() {
  constexpr std::size_t kStep = std::size_t{2} << 20;
  hipStream_t stream = nullptr;
  Require(hipStreamCreate(&stream) == hipSuccess, "stream");
  std::string error;
  {
    PiecePool pool(kStep);
    pool.SetLimit(3 * kStep);
    GrowableBuffer buffer;
    Require(buffer.Reserve(64 * kStep, pool, &error), error);
    auto* base = static_cast<std::uint8_t*>(buffer.data());
    Require(buffer.committed() == 0 && pool.used() == 0,
            "reservation committed memory");

    // A graph captured before the pages exist writes them once committed.
    std::size_t* offset = nullptr;
    Require(hipMalloc(&offset, sizeof(*offset)) == hipSuccess, "offset");
    const std::size_t at = 2 * kStep + 4096;
    Require(hipMemcpy(offset, &at, sizeof(at), hipMemcpyHostToDevice) ==
                hipSuccess,
            "offset upload");
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    Require(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal) ==
                hipSuccess,
            "begin capture");
    Fill<<<16, 256, 0, stream>>>(base, offset, 0x5A);
    Require(hipStreamEndCapture(stream, &graph) == hipSuccess &&
                hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0) ==
                    hipSuccess,
            "capture");

    const auto free_start = FreeBytes();
    Require(buffer.Commit(kStep + 1, stream, &error) &&
                buffer.committed() == 2 * kStep && pool.used() == 2 * kStep,
            "commit rounds up to whole pieces: " + error);
    Require(FreeBytes() + 2 * kStep <= free_start + (std::size_t{1} << 20),
            "commit did not use device memory");
    Require(hipStreamSynchronize(stream) == hipSuccess &&
                All(Read(base, 2 * kStep), 0),
            "new pages are zero");
    Require(hipMemset(base + kStep - 512, 0x11, 1024) == hipSuccess &&
                All(Read(base + kStep - 512, 1024), 0x11),
            "write and copy across a piece boundary");

    // Exhaustion: the prefix stays committed and intact.
    Require(!buffer.Commit(6 * kStep, stream, &error) &&
                error.find("pool exhausted") != std::string::npos,
            "exhausted pool must fail clearly");
    Require(buffer.committed() == 3 * kStep && pool.used() == 3 * kStep &&
                All(Read(base + kStep - 512, 1024), 0x11),
            "failed commit disturbed the committed prefix");
    Require(hipGraphLaunch(exec, stream) == hipSuccess &&
                hipStreamSynchronize(stream) == hipSuccess &&
                All(Read(base + 2 * kStep + 4096, 4096), 0x5A),
            "graph replay into pages committed after capture");

    // Trimmed pieces return to the pool and serve another buffer, zeroed.
    buffer.Trim(kStep);
    Require(buffer.committed() == kStep && pool.used() == kStep,
            "trim bookkeeping");
    GrowableBuffer other;
    const auto free_reuse = FreeBytes();
    Require(other.Reserve(8 * kStep, pool, &error) &&
                other.Commit(2 * kStep, stream, &error) &&
                hipStreamSynchronize(stream) == hipSuccess &&
                All(Read(other.data(), 2 * kStep), 0),
            "recycled pieces were not zeroed: " + error);
    Require(FreeBytes() + (std::size_t{1} << 20) >= free_reuse,
            "recycling allocated new device memory");
    Require(!buffer.Commit(2 * kStep, stream, &error),
            "pool limit ignored while recycled pieces are in use");
    other.Trim(0);
    Require(buffer.Commit(3 * kStep, stream, &error), error);
    Require(hipStreamSynchronize(stream) == hipSuccess &&
                All(Read(base + 2 * kStep + 4096, 4096), 0) &&
                All(Read(base + kStep - 512, 512), 0x11),
            "recommit kept stale pages or lost the kept piece");
    Require(hipGraphLaunch(exec, stream) == hipSuccess &&
                hipStreamSynchronize(stream) == hipSuccess &&
                All(Read(base + 2 * kStep + 4096, 4096), 0x5A),
            "graph replay into recommitted pages");

    // Moving keeps the address; churn does not grow device memory.
    GrowableBuffer moved = std::move(buffer);
    Require(moved.data() == base && moved.committed() == 3 * kStep,
            "move changed the buffer");
    const auto free_churn = FreeBytes();
    for (int i = 0; i < 20; ++i) {
      moved.Trim(0);
      Require(moved.Commit(3 * kStep, stream, &error), error);
    }
    Require(FreeBytes() + (std::size_t{1} << 20) >= free_churn,
            "grow/trim churn leaked device memory");
    Require(hipGraphExecDestroy(exec) == hipSuccess &&
                hipGraphDestroy(graph) == hipSuccess &&
                hipFree(offset) == hipSuccess,
            "graph cleanup");
    moved.Trim(0);
    Require(pool.used() == 0, "trim did not return pieces to the pool");
  }
  {
    // Shared prefix pages: mapped by two buffers, counted once, kept alive
    // by either holder, invalid once both let go.
    PiecePool pool(kStep);
    GrowableBuffer holder, sharer;
    Require(holder.Reserve(8 * kStep, pool, &error) &&
                sharer.Reserve(8 * kStep, pool, &error) &&
                holder.Commit(3 * kStep, stream, &error) &&
                hipMemsetAsync(holder.data(), 0x33, 3 * kStep, stream) ==
                    hipSuccess &&
                hipStreamSynchronize(stream) == hipSuccess,
            error);
    std::vector<gufo::hip::PieceRef> refs;
    holder.Share(2 * kStep + 7, &refs);
    Require(refs.size() == 2, "share must cover whole pieces only");
    std::vector<std::uint32_t> retained;
    for (const auto ref : refs)
      Require(pool.Retain(ref), "live piece could not be retained");
    for (const auto ref : refs)
      retained.push_back(ref.index);
    Require(sharer.Attach(retained, &error) == 2 && pool.used() == 3 * kStep,
            "shared pieces must map once and count once: " + error);
    Require(sharer.Commit(3 * kStep, stream, &error) &&
                hipStreamSynchronize(stream) == hipSuccess &&
                All(Read(sharer.data(), 2 * kStep), 0x33) &&
                All(Read(static_cast<std::uint8_t*>(sharer.data()) + 2 * kStep,
                         kStep),
                    0),
            "sharer must see the prefix and a private zeroed tail");
    Require(hipMemset(static_cast<std::uint8_t*>(sharer.data()) + 2 * kStep,
                      0x44, kStep) == hipSuccess &&
                All(Read(static_cast<std::uint8_t*>(holder.data()) + 2 * kStep,
                         kStep),
                    0x33),
            "a private tail write reached the holder");
    holder.Trim(0);
    Require(All(Read(sharer.data(), 2 * kStep), 0x33) &&
                pool.used() == 3 * kStep,
            "the sharer lost pages when the holder released them");
    Require(holder.Commit(2 * kStep, stream, &error) &&
                hipStreamSynchronize(stream) == hipSuccess &&
                All(Read(holder.data(), 2 * kStep), 0) &&
                All(Read(sharer.data(), 2 * kStep), 0x33),
            "a recommitting holder reused a shared piece");
    sharer.Trim(0);
    Require(!pool.Retain(refs.front()),
            "a recycled piece still accepted a stale reference");
    holder.Trim(0);
    Require(pool.used() == 0, "shared pieces leaked");

    // A full pool asks its reclaim hook (the executor pages an idle
    // session out) before failing.
    pool.SetLimit(2 * kStep);
    Require(holder.Commit(2 * kStep, stream, &error), error);
    int reclaims = 0;
    pool.SetReclaim([&] {
      ++reclaims;
      if (holder.committed() == 0)
        return false;
      holder.Trim(0);
      return true;
    });
    Require(sharer.Commit(2 * kStep, stream, &error) && reclaims == 1 &&
                holder.committed() == 0,
            "the reclaim hook did not make room: " + error);
    Require(!holder.Commit(kStep, stream, &error) && reclaims == 2 &&
                error.find("pool exhausted") != std::string::npos,
            "nothing reclaimable must fail clearly");
    pool.SetReclaim({});
    sharer.Trim(0);
  }
  {
    // A slot re-mapped to another piece while its old piece now backs a
    // different buffer: its zero fill and kernel writes must reach the new
    // piece. ROCm 7.2 keeps a stale GPU TLB entry after hipMemMap; without
    // the flush between map and fill, a's fill (a blit kernel) and writes
    // land in b's piece (1 MiB pieces as in the Qwen pool; every round
    // aliased without the flush).
    constexpr std::size_t kPiece = std::size_t{1} << 20;
    constexpr std::size_t kWords = kPiece / sizeof(std::uint32_t);
    PiecePool pool(kPiece);
    const auto word = [](const void* device) {
      std::uint32_t w = 0;
      std::memcpy(&w, Read(device, sizeof(w)).data(), sizeof(w));
      return w;
    };
    for (std::uint32_t round = 0; round < 8; ++round) {
      GrowableBuffer a, b;
      Require(a.Reserve(kPiece, pool, &error) &&
                  b.Reserve(kPiece, pool, &error) &&
                  a.Commit(kPiece, stream, &error),
              error);
      Stamp<<<1, 1, 0, stream>>>(static_cast<std::uint32_t*>(a.data()), kWords,
                                 7);
      Require(hipStreamSynchronize(stream) == hipSuccess, "stamp");
      a.Trim(0);
      Require(b.Commit(kPiece, stream, &error), error);
      Stamp<<<1, 1, 0, stream>>>(static_cast<std::uint32_t*>(b.data()), kWords,
                                 2000 + round);
      Require(hipStreamSynchronize(stream) == hipSuccess, "stamp");
      // One flush for both buffers' new pages, fills after it.
      const auto flushes = pool.flushes();
      GrowableBuffer c;
      {
        const gufo::hip::TlbBatch batch;
        Require(c.Reserve(kPiece, pool, &error) &&
                    a.Commit(kPiece, stream, &error) &&
                    c.Commit(kPiece, stream, &error),
                error);
      }
      Require(pool.Settle(&error) && pool.flushes() == flushes + 1, error);
      Require(hipStreamSynchronize(stream) == hipSuccess, "fill");
      Require(word(b.data()) == 2000 + round && word(a.data()) == 0 &&
                  word(c.data()) == 0,
              "round " + std::to_string(round) +
                  ": a zero fill missed the re-mapped piece (a=" +
                  std::to_string(word(a.data())) +
                  " b=" + std::to_string(word(b.data())) + ")");
      Stamp<<<1, 1, 0, stream>>>(static_cast<std::uint32_t*>(a.data()), kWords,
                                 1000 + round);
      Require(hipStreamSynchronize(stream) == hipSuccess, "stamp");
      Require(word(a.data()) == 1000 + round && word(b.data()) == 2000 + round,
              "round " + std::to_string(round) +
                  ": a kernel write missed the re-mapped piece (a=" +
                  std::to_string(word(a.data())) +
                  " b=" + std::to_string(word(b.data())) + ")");
    }
  }
  Require(hipStreamDestroy(stream) == hipSuccess, "stream cleanup");
  std::printf("growable buffer: lazy commit, pool limit, recycling, sharing, graph "
              "replay, churn and re-mapped slots ok\n");
  return 0;
}
