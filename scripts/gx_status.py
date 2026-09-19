"""One-screen status of the Hi-C histories on galaxy-heidel."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gxhic  # noqa: E402

EXTRA = {}
_p = Path(__file__).parent.parent / "runs" / "gx_ref_hist.json"
if _p.exists():
    import json
    EXTRA["hg19_reference"] = json.load(open(_p))["ref_hist"]


def main():
    hists = dict(gxhic.HISTS)
    hists.update(EXTRA)
    for name, hid in hists.items():
        h = gxhic.history(hid)
        sd = {k: v for k, v in (h.get("state_details") or {}).items() if v}
        print(f"== {name:22s} {hid}  {sd}")
        for c in gxhic.contents(hid, v="dev",
                                keys="id,hid,name,state,file_size,extension"):
            if c["state"] == "ok" and "--all" not in sys.argv:
                sz = c.get("file_size") or 0
                print(f"   {c['hid']:>3} ok      {sz/1e9:8.2f} GB  {c['name'][:62]}")
            else:
                sz = c.get("file_size") or 0
                print(f"   {c['hid']:>3} {c['state']:<8}{sz/1e9:8.2f} GB  "
                      f"{c['name'][:62]}")


if __name__ == "__main__":
    main()
