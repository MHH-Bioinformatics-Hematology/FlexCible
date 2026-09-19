"""Minimal Galaxy client for the Hi-C fetch/map/build pipeline on galaxy-heidel.

The server sits behind a corporate proxy with a self-signed certificate, so
every process that talks to it must drop the proxy variables and use an
unverified SSL context. Importing this module does that once.
"""
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://galaxy-heidel.mh-hannover.local"
KEY = os.environ.get("GKEY", "acc267e0c1b4ea221e22154bda6c65d5")

for _v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(_v, None)

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE


def _url(path, **q):
    q["key"] = KEY
    return f"{BASE}{path}?{urllib.parse.urlencode(q)}"


def _req(method, path, body=None, tries=4, timeout=180, **q):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    last = None
    for i in range(tries):
        try:
            r = urllib.request.Request(_url(path, **q), data=data,
                                       headers=headers, method=method)
            with urllib.request.urlopen(r, context=CTX, timeout=timeout) as resp:
                raw = resp.read()
            return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:2000]
            raise RuntimeError(f"{method} {path} -> {e.code}: {detail}") from None
        except Exception as e:  # noqa
            last = e
            time.sleep(2 * (i + 1))
    raise last


def get(path, **q):
    return _req("GET", path, **q)


def post(path, body, **q):
    return _req("POST", path, body=body, **q)


def put(path, body, **q):
    return _req("PUT", path, body=body, **q)


def history(hid):
    return get(f"/api/histories/{hid}")


def contents(hid, **q):
    return get(f"/api/histories/{hid}/contents", **q)


def dataset(did):
    return get(f"/api/datasets/{did}")


def job(jid, full=False):
    q = {"full": "true"} if full else {}
    return get(f"/api/jobs/{jid}", **q)


def tool(tid):
    return get(f"/api/tools/{urllib.parse.quote(tid, safe='')}", io_details="true")


def run_tool(hid, tool_id, inputs):
    return post("/api/tools", {"history_id": hid, "tool_id": tool_id,
                               "inputs": inputs})


def download(did, dest, tries=4, timeout=1800):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return os.path.getsize(dest)
    tmp = dest + ".part"
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(_url(f"/api/datasets/{did}/display",
                                             to_ext="data"),
                                        context=CTX, timeout=timeout) as r, \
                    open(tmp, "wb") as o:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    o.write(chunk)
            os.replace(tmp, dest)
            return os.path.getsize(dest)
        except Exception as e:  # noqa
            last = e
            time.sleep(3 * (i + 1))
    raise last


HISTS = {
    "HCT116_HiC_rep1": "2810f54e3c952a81",
    "HCT116_HiC_rep2": "03694320c2305563",
    "leukemia_ZFP36L2": "01ce4583c5019a62",
    "GM13977_depth": "b510be579c1fa8fe",
}
