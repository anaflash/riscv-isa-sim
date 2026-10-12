// KUCIF eFlash-PIM custom extension for Spike.
//
// Implements the 3 KUCIF commands needed for the anaflash-pim-compiler Week 1
// demo scope (see docs/kucif_isa.md and
// docs/kucif_custom_dispatch.md in the anaflash-pim-compiler repo):
//   pim_parallel (funct3=5), pim_rbr (funct3=6, unused stub), pim_load (funct3=7).
// pim_erase/pim_program/pim_read/pim_loadraw are out of scope: weight state is
// preloaded at startup/reset from an optional host-side image instead of being
// programmed by the guest binary.
//
// Semantics (accumulate-then-drain) and the rs1 ABI
// (rs1 = (bit_slice_index << 24) | weight_tile_group) are taken from the
// vendor reference runtime shipped in
// sim/eFlash_RISCV_Codegen_Demo/c/ecg_gate/riscv_custom/pim_runtime_chip.h,
// which is the ground truth for the real-chip funct3 5/6/7 encoding.
//
// rs2 for pim_parallel points to a 2-byte guest buffer packing 8 activation
// lanes at 2 bits each (lane il at byte il/4, bits (il%4)*2 .. +1). rs2 for
// pim_load points to an 8-entry int32 guest output buffer.
#include "extension.h"
#include "mmu.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define DECODE_MACRO_USAGE_LOGGED 1
#include "decode_macros.h"

namespace {

constexpr int kOpcode = 0x0b;
constexpr int kFunct3Parallel = 5;
constexpr int kFunct3Rbr = 6;
constexpr int kFunct3Load = 7;
constexpr int kNumIL = 8;    // input lines (parallel MAC reduction width)
constexpr int kNumCO = 8;    // output channels / BL-groups per WL tile
// Spike-only virtual address space for tiled GEMM tests. Physical WL/bank
// mapping is architecture-specific; a real macro currently documents 16 WL.
constexpr int kMaxGroups = 128;

class kucif_t : public extension_t {
 public:
  kucif_t() { init_state(); }

  const char* name() const override { return "kucif"; }

  // Note: processor_t::reset() only walks custom_extensions registered
  // *before* it runs, which is not guaranteed for --extlib/--extension in
  // this htif-less bring-up flow, so state must also be valid straight out
  // of the constructor; init_state() is shared by both paths.
  void reset(processor_t&) override { init_state(); }

  std::vector<insn_desc_t> get_instructions(const processor_t&) override;
  std::vector<disasm_insn_t*> get_disasms(const processor_t*) override { return {}; }

  void pim_parallel(processor_t* p, reg_t rs1, reg_t buf_addr) {
    int bs = (int)((rs1 >> 24) & 0xff);
    int group = (int)(rs1 & 0x00ffffffULL);
    if (group >= loaded_groups_ || bs >= 4)
      illegal_instruction(*p);
    int scale = 1 << (2 * (3 - bs));
    uint8_t buf[2];
    buf[0] = p->get_mmu()->load<uint8_t>(buf_addr + 0);
    buf[1] = p->get_mmu()->load<uint8_t>(buf_addr + 1);
    for (int il = 0; il < kNumIL; il++) {
      int a2 = (buf[il / 4] >> ((il % 4) * 2)) & 0x3;
      if (!a2) continue;
      for (int co = 0; co < kNumCO; co++)
        acc_[co] += (int64_t)wtile_[group][il][co] * a2 * scale;
    }
#ifdef KUCIF_DEBUG
    fprintf(stderr, "[kucif] parallel bs=%d group=%d buf=%02x%02x acc=%ld,%ld,...\n",
            bs, group, buf[0], buf[1], (long)acc_[0], (long)acc_[1]);
#endif
    par_calls_++;
  }

  void pim_load(processor_t* p, reg_t out_addr) {
#ifdef KUCIF_DEBUG
    fprintf(stderr, "[kucif] load out_addr=%lx acc=%ld,%ld,...\n",
            (unsigned long)out_addr, (long)acc_[0], (long)acc_[1]);
#endif
    for (int co = 0; co < kNumCO; co++) {
      p->get_mmu()->store<int32_t>(out_addr + 4 * co, (int32_t)acc_[co]);
      acc_[co] = 0;
    }
    load_calls_++;
  }

  void pim_rbr() { rbr_calls_++; }

 private:
  void init_state() {
    memset(acc_, 0, sizeof(acc_));
    par_calls_ = rbr_calls_ = load_calls_ = 0;
    loaded_groups_ = kMaxGroups;
    // Keep the identity tile as the no-configuration default for smoke tests.
    for (int g = 0; g < kMaxGroups; g++)
      for (int il = 0; il < kNumIL; il++)
        for (int co = 0; co < kNumCO; co++)
          wtile_[g][il][co] = (il == co) ? 129 : 128;

    const char* image_path = std::getenv("KUCIF_WEIGHT_IMAGE");
    if (!image_path || !*image_path) return;

    FILE* image = std::fopen(image_path, "rb");
    if (!image) {
      std::fprintf(stderr, "[kucif] cannot open weight image '%s'\n", image_path);
      std::abort();
    }

    bool invalid_image = std::fseek(image, 0, SEEK_END) != 0;
    long image_size = invalid_image ? -1 : std::ftell(image);
    invalid_image = invalid_image || image_size <= 0 ||
                    image_size > (long)sizeof(wtile_) ||
                    image_size % sizeof(wtile_[0]) != 0 ||
                    std::fseek(image, 0, SEEK_SET) != 0;
    size_t bytes_read = invalid_image
                            ? 0
                            : std::fread(wtile_, 1, (size_t)image_size, image);
    invalid_image = invalid_image || bytes_read != (size_t)image_size ||
                    std::ferror(image);
    std::fclose(image);
    if (invalid_image) {
      std::fprintf(stderr,
                   "[kucif] weight image '%s' must contain 1..%d whole tiles (%zu bytes max)\n",
                   image_path, kMaxGroups, sizeof(wtile_));
      std::abort();
    }
    loaded_groups_ = (int)(image_size / sizeof(wtile_[0]));
#ifdef KUCIF_DEBUG
    std::fprintf(stderr, "[kucif] loaded %d weight groups (%ld bytes) from '%s'\n",
                 loaded_groups_, image_size, image_path);
#endif
  }

  int64_t acc_[kNumCO];
  uint8_t wtile_[kMaxGroups][kNumIL][kNumCO];
  int loaded_groups_;
  long par_calls_, rbr_calls_, load_calls_;
};

reg_t kucif_parallel_insn(processor_t* p, insn_t insn, reg_t pc) {
  auto* ext = static_cast<kucif_t*>(p->get_extension());
  ext->pim_parallel(p, RS1, RS2);
  return pc + 4;
}

reg_t kucif_load_insn(processor_t* p, insn_t insn, reg_t pc) {
  auto* ext = static_cast<kucif_t*>(p->get_extension());
  ext->pim_load(p, RS2);
  return pc + 4;
}

reg_t kucif_rbr_insn(processor_t* p, insn_t UNUSED insn, reg_t pc) {
  auto* ext = static_cast<kucif_t*>(p->get_extension());
  ext->pim_rbr();
  return pc + 4;
}

std::vector<insn_desc_t> kucif_t::get_instructions(const processor_t&) {
  const insn_bits_t mask = 0x707f;  // opcode[6:0] | funct3[14:12]
  auto match = [](int funct3) -> insn_bits_t {
    return (insn_bits_t)((funct3 << 12) | kOpcode);
  };
  std::vector<insn_desc_t> insns;
  insns.push_back({match(kFunct3Parallel), mask,
                    kucif_parallel_insn, kucif_parallel_insn,
                    kucif_parallel_insn, kucif_parallel_insn,
                    kucif_parallel_insn, kucif_parallel_insn,
                    kucif_parallel_insn, kucif_parallel_insn});
  insns.push_back({match(kFunct3Rbr), mask,
                    kucif_rbr_insn, kucif_rbr_insn,
                    kucif_rbr_insn, kucif_rbr_insn,
                    kucif_rbr_insn, kucif_rbr_insn,
                    kucif_rbr_insn, kucif_rbr_insn});
  insns.push_back({match(kFunct3Load), mask,
                    kucif_load_insn, kucif_load_insn,
                    kucif_load_insn, kucif_load_insn,
                    kucif_load_insn, kucif_load_insn,
                    kucif_load_insn, kucif_load_insn});
  return insns;
}

}  // namespace

REGISTER_EXTENSION(kucif, []() { return new kucif_t(); })
