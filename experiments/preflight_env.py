"""Check the tomato-ml env before a job spends its GPU allocation.

    python -m experiments.preflight_env            # imports + CUDA (default)
    python -m experiments.preflight_env --no-cuda  # imports only (login node)

Exits 0 if the env is usable, 1 with a plain-language diagnosis otherwise.

Why this exists: SLURM job 3930 queued for a GPU, started, and died on
`import torch.nn` with `No module named 'torch._prims_common'` — a torch that pip
had half-written. The whole allocation was spent discovering a broken import.
Checking first costs about a second.

The two failures this catches, both of which really happened on 2026-07-16 after
an unpinned `pip install` re-resolved the environment:

  * torch <-> numpy. torch 2.3.0 needs numpy<2. Anything demanding numpy>=2
    (a current rembg, say) makes pip move numpy, collide with torch, and leave
    a half-written torch plus a `~orch` directory in site-packages.
  * albumentations <-> albucore. albumentations 1.4.10 declares only
    `albucore>=0.0.11`, so pip is free to upgrade albucore to a version that has
    moved `preserve_channel_dim` out of `albucore.utils` — and the import breaks.
    albucore must therefore be pinned by US; albumentations will not do it.

See experiments/requirements.txt for the pins and the reasoning.
"""
import argparse
import sys

# (import name, pinned version in experiments/requirements.txt or None if unpinned)
EXPECTED = [
    ("torch", "2.3.0"),
    ("torchvision", "0.18.0"),
    ("timm", "1.0.27"),
    ("albumentations", "1.4.10"),
    ("numpy", "1.26.4"),
    ("sklearn", None),
    ("matplotlib", None),
    ("yaml", None),
]

HINT = (
    "The tomato-ml env is broken or has drifted. This is almost always an\n"
    "unpinned `pip install` re-resolving dependencies. Fix it ON THE LOGIN NODE:\n"
    "    pip install -r experiments/requirements.txt\n"
    "and never install packages from inside a SLURM job.\n"
    "If site-packages contains a directory starting with '~' (e.g. ~orch), that is\n"
    "pip's leftover from an interrupted install - delete it and reinstall torch."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cuda", action="store_true",
                        help="Skip the CUDA check (for login-node use).")
    args = parser.parse_args()

    failed = False
    versions = {}
    for name, pin in EXPECTED:
        try:
            mod = __import__(name)
        except Exception as e:
            print(f"FAIL  import {name}: {type(e).__name__}: {e}", file=sys.stderr)
            failed = True
            continue
        got = getattr(mod, "__version__", "?")
        versions[name] = got
        # A drifted-but-importable version is a warning, not an error: it may be
        # deliberate. It is still worth printing, because it makes results
        # non-comparable to the rows already in the thesis.
        #
        # Compare only the public version: torch reports "2.3.0+cu121", where
        # "+cu121" is a PEP 440 local version identifier naming the CUDA build,
        # not a different release. Matching the raw string made this warn on a
        # perfectly correct env every single run — and a warning that always
        # fires is a warning nobody reads.
        if pin and got.split("+")[0] != pin:
            print(f"WARN  {name} {got} != pinned {pin} - results may not be "
                  f"comparable to existing rows.", file=sys.stderr)

    if failed:
        print("\n" + HINT, file=sys.stderr)
        raise SystemExit(1)

    print("  " + " | ".join(f"{n} {v}" for n, v in versions.items() if v != "?"))

    if not args.no_cuda:
        import torch
        if not torch.cuda.is_available():
            print("FAIL  CUDA not available - this job needs a GPU.", file=sys.stderr)
            raise SystemExit(1)
        print(f"  CUDA OK: {torch.cuda.get_device_name(0)}")

    print("Preflight OK.")


if __name__ == "__main__":
    main()
