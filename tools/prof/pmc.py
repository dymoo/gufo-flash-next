#!/usr/bin/env python3
"""Hardware-counter (PMC) passes over one command, summarized per kernel.

`tools/prof/prof.py` answers where the time goes; this answers why one kernel
is off its ceiling: achieved clock, occupancy, instruction counts per wave,
LDS bank conflicts, L2 hit rate and fabric read traffic. gfx1151 exposes a
small set of counters (see docs/PERFORMANCE.md) and each rocprofv3 pass fits
only a few of them per hardware block, so the counters are split into groups
and the command runs once per group. Dispatches are serialized and counted
individually, so profile a command that launches the kernel a handful of times
(the first dispatches are cold; --skip drops them).

  python3 tools/prof/pmc.py run --kernel DenseF16GEMMKernel --skip 3 \\
      --out /tmp/pmc -- ./bench --case ssm_in
  python3 tools/prof/pmc.py show /tmp/pmc

rocprofv3 needs libhsa-amd-aqlprofile64.so to collect counters (apt package
hsa-amd-aqlprofile); point --aqlprofile-dir at a directory holding it when it
is not installed system-wide.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import re
import signal
import statistics
import subprocess
import sys

# Each group must fit one pass. GL2C counters are limited to a few per pass;
# a group that exceeds the hardware makes rocprofv3 abort, and this tool kills
# the stuck process (its crash handler can spin for minutes).
GROUPS = {
    "waves": "GRBM_GUI_ACTIVE SQ_WAVES SQ_WAVE_CYCLES SQ_BUSY_CYCLES "
    "SQ_INSTS_VALU SQ_INSTS_SALU SQ_INSTS_LDS",
    "lds": "SQC_LDS_BANK_CONFLICT SQC_LDS_IDX_ACTIVE SQ_BUSY_CYCLES GRBM_GUI_ACTIVE",
    "l2": "GL2C_HIT GL2C_MISS",
    "fabric": "GL2C_EA_RDREQ_128B GL2C_MC_RDREQ",
    "clock": "TA_TA_BUSY GL2C_MC_WRREQ GL2C_MC_WRREQ_STALL GRBM_GUI_ACTIVE GRBM_COUNT",
}


def run_group(name, counters, args, command):
    out = os.path.join(args.out, name)
    env = dict(os.environ)
    if args.aqlprofile_dir:
        env["LD_LIBRARY_PATH"] = args.aqlprofile_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    cmd = ["rocprofv3", "--pmc", *counters.split(), "--output-format", "csv",
           "-d", out, "-o", "run"]
    if args.kernel:
        cmd += ["--kernel-include-regex", args.kernel]
    cmd += ["--", *command]
    print(f"== {name}: {counters}", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        _, err = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        _, err = proc.communicate()
        err = "timed out (a counter group that exceeds the hardware can hang the profiler)\n" + (err or "")
    if proc.returncode != 0:
        lines = [l for l in (err or "").splitlines() if re.search(r"exceeds|error|fail|timed out|cannot", l, re.I)]
        print(f"   FAILED rc={proc.returncode}: " + " | ".join(lines[:2]), flush=True)
        return False
    return True


def kernel_name(raw):
    name = re.sub(r"^void\s+", "", raw)
    name = re.sub(r"^(?:\(anonymous namespace\)::|[A-Za-z_0-9]+::)+", "", name)
    name = re.sub(r"\)\(.*|\(void const.*|\([A-Za-z_:0-9 ]+[*&].*", "", name)
    return name[:80]


def load(out, skip):
    """-> {kernel: {counter: [per-dispatch values]}}, {kernel: [durations us]}, cu_count"""
    counters = collections.defaultdict(lambda: collections.defaultdict(list))
    durations = collections.defaultdict(list)
    cus = 40
    for agent in glob.glob(os.path.join(out, "*", "run_agent_info.csv")):
        for row in csv.DictReader(open(agent)):
            if row.get("Agent_Type") == "GPU":
                cus = int(row["Simd_Count"]) // 2
    first = True
    for path in sorted(glob.glob(os.path.join(out, "*", "run_counter_collection.csv"))):
        per = collections.defaultdict(lambda: collections.defaultdict(float))
        span = {}
        for row in csv.DictReader(open(path)):
            key = (kernel_name(row["Kernel_Name"]), int(row["Dispatch_Id"]))
            per[key][row["Counter_Name"]] += float(row["Counter_Value"])
            span[key] = (int(row["End_Timestamp"]) - int(row["Start_Timestamp"])) / 1000.0
        by_kernel = collections.defaultdict(list)
        for key in sorted(per, key=lambda k: k[1]):
            by_kernel[key[0]].append(key)
        for kernel, keys in by_kernel.items():
            for key in keys[skip:]:
                for counter, value in per[key].items():
                    counters[kernel][counter].append(value)
                if first:
                    durations[kernel].append(span[key])
        first = False
    return counters, durations, cus


def show(out, skip):
    counters, durations, cus = load(out, skip)
    if not counters:
        print("no counter data found (did every group fail?)")
        return 1

    def med(kernel, counter):
        values = counters[kernel].get(counter)
        return statistics.median(values) if values else float("nan")

    def ratio(a, b):
        return a / b if b else float("nan")

    print(f"{'kernel':<58}{'n':>3}{'us':>8}{'MHz':>6}{'VALU/wv':>9}{'LDS/wv':>8}"
          f"{'ldsConf%':>9}{'L2hit%':>7}{'fetchMB':>8}{'occ/CU':>7}{'busy%':>6}")
    for kernel in sorted(counters, key=lambda k: -sum(durations[k])):
        us = statistics.median(durations[kernel]) if durations[kernel] else float("nan")
        waves = med(kernel, "SQ_WAVES")
        active = med(kernel, "GRBM_GUI_ACTIVE")
        hit, miss = med(kernel, "GL2C_HIT"), med(kernel, "GL2C_MISS")
        clock = ratio(med(kernel, "GRBM_COUNT") or active, us)
        print(f"{kernel:<58}{len(durations[kernel]):>3}{us:>8.0f}{clock:>6.0f}"
              f"{ratio(med(kernel, 'SQ_INSTS_VALU'), waves):>9.0f}"
              f"{ratio(med(kernel, 'SQ_INSTS_LDS'), waves):>8.0f}"
              f"{100 * ratio(med(kernel, 'SQC_LDS_BANK_CONFLICT'), med(kernel, 'SQC_LDS_IDX_ACTIVE')):>9.1f}"
              f"{100 * ratio(hit, hit + miss):>7.1f}"
              f"{med(kernel, 'GL2C_EA_RDREQ_128B') * 128 / 1e6:>8.0f}"
              f"{ratio(med(kernel, 'SQ_WAVE_CYCLES'), active) / cus:>7.2f}"
              f"{100 * ratio(med(kernel, 'SQ_BUSY_CYCLES'), active) / (cus // 2):>6.1f}")
    print("\nMHz = GRBM_COUNT / dispatch time; occ/CU = mean resident waves per CU "
          "(SQ_WAVE_CYCLES / GRBM_GUI_ACTIVE / CUs); busy% = share of GRBM-active "
          "cycles with a wave resident per WGP; ldsConf% = SQC_LDS_BANK_CONFLICT / "
          "SQC_LDS_IDX_ACTIVE; fetchMB = 128-byte L2 read requests to the fabric.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run", help="run the command once per counter group, then summarize")
    run.add_argument("--out", required=True, help="output directory (one subdirectory per group)")
    run.add_argument("--kernel", help="regex of kernel names to profile (others run unprofiled)")
    run.add_argument("--skip", type=int, default=0, help="dispatches per kernel to drop (cold start)")
    run.add_argument("--groups", default=",".join(GROUPS), help=f"comma list of {sorted(GROUPS)}")
    run.add_argument("--timeout", type=int, default=180, help="seconds per group")
    run.add_argument("--aqlprofile-dir", help="directory holding libhsa-amd-aqlprofile64.so")
    run.add_argument("command", nargs=argparse.REMAINDER)
    shw = sub.add_parser("show", help="summarize an existing --out directory")
    shw.add_argument("out")
    shw.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "show":
        return show(args.out, args.skip)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the command after --")
    os.makedirs(args.out, exist_ok=True)
    for name in args.groups.split(","):
        run_group(name, GROUPS[name], args, command)
    return show(args.out, args.skip)


if __name__ == "__main__":
    sys.exit(main())
