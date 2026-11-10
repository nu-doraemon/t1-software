"""Static HTML dashboard of the nightly build ledger.

render() writes index.html and one page per build into an output
directory. The pages use no JavaScript and only relative links, so they
open straight from the filesystem or from any static web server. Build
logs are copied next to the pages for the same reason.

This module reads nothing from the ledger itself: nightly.py gathers the
inputs under the ledger lock and passes them in.
"""
import html
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

LOG_TAIL = 200  # lines of build.log shown on a build page

STYLE = """
body { font-family: sans-serif; margin: 1.5em; color: #222; }
table { border-collapse: collapse; margin: 0.5em 0 1.5em; }
th, td { border: 1px solid #ccc; padding: 0.2em 0.6em; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
code, pre { font-family: monospace; }
pre { background: #f7f7f7; border: 1px solid #ddd; padding: 0.6em; overflow-x: auto; }
.banner { padding: 0.6em; margin: 0.6em 0; border: 1px solid; font-weight: bold; }
.stale { background: #fff3cd; border-color: #c9a300; }
.error { background: #f8d7da; border-color: #b02a37; }
.success { color: #146c43; }
.failed, .cancelled, .interrupted { color: #b02a37; }
.queued, .running { color: #0a58ca; }
"""


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def short(sha):
    return esc(sha[:8]) if sha else "-"


def parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def write_atomic(path, text):
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def page(title, body):
    return (f"<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">"
            f"<title>{esc(title)}</title><style>{STYLE}</style></head>\n"
            f"<body>\n{body}\n</body></html>\n")


def table(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>\n" for row in rows)
    return f"<table><tr>{head}</tr>\n{body}</table>"


def log_names(src_dir):
    """build.log first, then attempt-N.log in attempt order."""
    attempts = sorted(src_dir.glob("attempt-*.log"),
                      key=lambda p: int(p.stem.split("-", 1)[1]) if p.stem.split("-", 1)[1].isdigit() else 0)
    names = [p.name for p in attempts]
    if (src_dir / "build.log").exists():
        names.insert(0, "build.log")
    return names


def copy_logs(src_dir, dst_dir):
    """Copy logs that are missing or older than the source. Returns their names."""
    names = log_names(src_dir)
    if names:
        dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src, dst = src_dir / name, dst_dir / name
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dst)
    return names


def images_state(record):
    d = Path(record["images_dir"])
    stem = f"t1-lartpc-{record['variant']}-{record['tag']}"
    if (d / f"{stem}.sif").exists() or (d / f"{stem}.oci.tar").exists():
        return "yes"
    return "pruned" if record["status"] == "success" else "-"


def banners(site, last_poll, now):
    limit = 2 * float(site["POLL_INTERVAL_HOURS"])
    out = []
    if last_poll is None:
        out.append(("stale", "Poller stale: no poll yet"))
        return out
    age = (now - parse_time(last_poll["time"])).total_seconds() / 3600
    if age > limit:
        out.append(("stale", f"Poller stale: last poll {age:.0f} h ago (limit {limit:g} h)"))
    if not last_poll.get("ok"):
        out.append(("error", f"Last poll failed: {last_poll.get('error')}"))
    return out


def render_index(out, site, records, review_states, last_poll, now):
    parts = ["<h1>T1 nightly builds</h1>"]
    for cls, text in banners(site, last_poll, now):
        parts.append(f"<div class=\"banner {cls}\">{esc(text)}</div>")
    if last_poll is not None:
        state = "ok" if last_poll.get("ok") else f"error: {last_poll.get('error')}"
        parts.append(f"<p>Last poll: {esc(last_poll['time'])} ({esc(state)}). "
                     f"Rendered: {esc(now.strftime('%Y-%m-%dT%H:%M:%SZ'))}.</p>")
    rows = []
    for rec in records:
        key = rec["key"]
        rows.append([
            esc((rec.get("queued_at") or "")[:10]),
            f"<a href=\"builds/{esc(key)}.html\"><code>{esc(key)}</code></a>",
            f"<span class=\"{esc(rec['status'])}\">{esc(rec['status'])}</span>",
            esc(review_states.get(key, "")),
            esc(rec.get("failed_stage") or ""),
            esc(rec["attempts"]),
            f"<code>{short(rec['t1_software_commit'])}</code>",
            esc(", ".join(rec.get("changed") or {})),
            esc(images_state(rec)),
        ])
    parts.append(table(["date", "key", "status", "review", "failed stage", "attempts",
                        "t1-software", "changed", "images"], rows))
    write_atomic(out / "index.html", page("T1 nightly builds", "\n".join(parts)))


def render_build(out, record, review, review_state, src_dir):
    key = record["key"]
    logs = copy_logs(src_dir, out / "builds" / key)
    parts = ["<p><a href=\"../index.html\">all builds</a></p>",
             f"<h1>Build <code>{esc(key)}</code></h1>"]
    fields = []
    for name in sorted(record):
        if name in ("pins", "changed"):
            continue
        value = record[name]
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        fields.append([esc(name), esc(value)])
    fields.append([esc("review state"), esc(review_state)])
    parts.append(table(["field", "value"], fields))
    parts.append("<h2>Pins</h2>")
    parts.append(table(["key", "commit"], [[esc(k), f"<code>{esc(v)}</code>"]
                                           for k, v in sorted(record["pins"].items())]))
    parts.append("<h2>Changed since the previous poll</h2>")
    parts.append(table(["name", "old", "new"], [
        [esc(n), f"<code>{short(old)}</code>", f"<code>{short(new)}</code>"]
        for n, (old, new) in sorted((record.get("changed") or {}).items())]))
    parts.append("<h2>Review</h2>")
    if review is None:
        parts.append("<p>No review.</p>")
    else:
        parts.append(table(["time", "user", "attempt", "note"], [[
            esc(review.get("time")), esc(review.get("user")), esc(review.get("attempt")),
            esc(review.get("note"))]]))
    parts.append("<h2>Logs</h2>")
    if logs:
        parts.append("<ul>" + "".join(
            f"<li><a href=\"{esc(key)}/{esc(n)}\">{esc(n)}</a></li>" for n in logs) + "</ul>")
        if "build.log" in logs:
            lines = (src_dir / "build.log").read_text(errors="replace").splitlines()
            parts.append(f"<p>Last {LOG_TAIL} lines of build.log:</p>")
            parts.append("<pre>" + esc("\n".join(lines[-LOG_TAIL:])) + "</pre>")
    else:
        parts.append("<p>No log yet.</p>")
    write_atomic(out / "builds" / f"{key}.html", page(f"T1 nightly build {key}", "\n".join(parts)))


def render(out, *, site, records, reviews, review_states, last_poll, build_dirs, now):
    out = Path(out)
    (out / "builds").mkdir(parents=True, exist_ok=True)
    render_index(out, site, records, review_states, last_poll, now)
    for rec in records:
        key = rec["key"]
        render_build(out, rec, reviews.get(key), review_states.get(key, ""), build_dirs[key])
