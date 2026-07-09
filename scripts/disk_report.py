"""One-off HPC disk report (run via `srun python`).

The AIT restricted shell blocks df/du/quota and srun only permits
python, so this uses os.statvfs (filesystem free/used) and os.walk
(per-directory sizes) to answer: total quota, used, and how much
the pre-v5 checkpoint dirs would free.
"""

import os

ROOT = "/media/D1/ait_users/staff18"
CKPT = os.path.join(ROOT, "Search-R1", "verl_checkpoints")

DELETE = {
    "treatment_lora_v1", "treatment_lora_v2", "treatment_lora_v3",
    "treatment_grounding_v1", "treatment_grounding_v2",
    "treatment_grounding_v3", "treatment_grounding_v4",
    "treatment_forced_exec_v1", "treatment_de_floor_v1",
    "treatment_f1_v1", "control_lora_v2",
}


def dir_size(path):
    total = 0
    for dp, _dn, fn in os.walk(path):
        for name in fn:
            fp = os.path.join(dp, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def gb(n):
    return n / (1024 ** 3)


st = os.statvfs(ROOT)
total = st.f_blocks * st.f_frsize
free = st.f_bavail * st.f_frsize
used = total - st.f_bfree * st.f_frsize
print(f"FILESYSTEM at {ROOT}")
print(f"  total={gb(total):.1f}G used={gb(used):.1f}G "
      f"free={gb(free):.1f}G ({100*used/total:.0f}% used)")

def top_level_report(base, label):
    print(f"\n{label} ({base}):")
    rows = []
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p) and not os.path.islink(p):
            rows.append((dir_size(p), name))
        elif os.path.isfile(p):
            rows.append((os.path.getsize(p), name))
    for sz, name in sorted(rows, reverse=True):
        if gb(sz) >= 0.05:
            print(f"  {gb(sz):7.2f}G  {name}")


top_level_report(ROOT, "HOME top-level")
SR1 = os.path.join(ROOT, "Search-R1")
top_level_report(SR1, "Search-R1 top-level")

print("\nCHECKPOINT DIRS (verl_checkpoints):")
del_sum = 0
keep_sum = 0
for name in sorted(os.listdir(CKPT)):
    p = os.path.join(CKPT, name)
    if not os.path.isdir(p):
        continue
    sz = dir_size(p)
    tag = "DELETE" if name in DELETE else "keep"
    if name in DELETE:
        del_sum += sz
    else:
        keep_sum += sz
    print(f"  [{tag:6}] {name:28} {gb(sz):6.2f}G")

print(f"\nDELETE candidates free: {gb(del_sum):.2f}G")
print(f"KEEP total:             {gb(keep_sum):.2f}G")
