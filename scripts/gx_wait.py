"""Block until the named datasets reach a terminal state, then print them.

Used to pace the driver: the pipeline runs for hours, so waiting is done by a
backgrounded process that exits on the event rather than by repeated polling
from the foreground.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gxhic  # noqa: E402

TERMINAL = {"ok", "error", "empty", "discarded", "failed_metadata", "paused"}


def main():
    ids = sys.argv[1:]
    if not ids:
        raise SystemExit("usage: gx_wait.py <hda_id> [...]")
    while True:
        states = {}
        for i in ids:
            try:
                d = gxhic.dataset(i)
                states[i] = (d.get("state"), d.get("file_size") or 0,
                             d.get("name", ""))
            except Exception as e:  # noqa
                states[i] = ("unreachable:" + str(e)[:60], 0, "")
        if all(s[0] in TERMINAL for s in states.values()):
            for i, (s, sz, n) in states.items():
                print(f"{i} {s} {sz/1e9:.2f}GB {n[:60]}", flush=True)
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
