"""Static HTML dashboard of the nightly build ledger.

render() writes index.html and one page per build into an output
directory. The pages work without JavaScript. One small inline script
adds a light/dark toggle and remembers the choice; without it the pages
follow the system theme. All links are relative except the commit links
to github.com. Build logs are not published: a build page shows the log
file location and the last lines of build.log.

This module reads nothing from the ledger itself: nightly.py gathers the
inputs under the ledger lock and passes them in.
"""
import html
import os
import re
from datetime import datetime, timezone
from pathlib import Path

LOG_TAIL = 200  # lines of build.log shown on a build page
INDEX_TITLE = "T1-LArTPC nightly builds"
T1 = "T1_SOFTWARE_COMMIT"
REVIEW_SECTION = ("failed", "interrupted", "cancelled")

_LIGHT = ("--bg:#fafaf7;--fg:#222;--muted:#777;--line:#dcdcd4;--head:#efefe8;--link:#005cc5;"
          "--ok:#22863a;--bad:#d73a49;--run:#005cc5;--warn:#b08800;--warnbg:#fff5b1;"
          "--code:#efefe8;--row:#f3f3ec")
_DARK = ("--bg:#14161a;--fg:#d4d7dd;--muted:#80868f;--line:#2a2e35;--head:#1c1f25;--link:#79b8ff;"
         "--ok:#56d364;--bad:#ff7b72;--run:#79b8ff;--warn:#e3b341;--warnbg:#3a2f0b;"
         "--code:#1c1f25;--row:#1a1d22")

STYLE = (
    f":root{{color-scheme:light;{_LIGHT}}}"
    f"@media(prefers-color-scheme:dark){{:root:not([data-theme=light]){{color-scheme:dark;{_DARK}}}}}"
    f":root[data-theme=dark]{{color-scheme:dark;{_DARK}}}"
    """
*{box-sizing:border-box}
[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--fg);font:13.5px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--link);text-decoration:none}a:hover{text-decoration:underline}
code,pre,table{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
code{background:var(--code);padding:.05em .35em;border-radius:4px;font-size:.92em}
pre{background:var(--code);border:1px solid var(--line);border-radius:6px;padding:.8em;overflow-x:auto;font-size:.82em;line-height:1.45}
.wrap{max-width:1240px;margin:0 auto;padding:0 20px 40px}
header.top{border-bottom:1px solid var(--line)}
header.top .wrap{display:flex;align-items:center;gap:16px;padding-top:14px;padding-bottom:14px}
header.top h1{margin:0;font-size:1.05rem;flex:1;font-family:ui-monospace,Menlo,Consolas,monospace}
.toggle{cursor:pointer;border:1px solid var(--line);border-radius:999px;padding:3px 12px;background:var(--bg);color:var(--fg);font:inherit;font-size:.85rem}
.toggle:hover{border-color:var(--muted)}
.ok{color:var(--ok)}.bad{color:var(--bad)}.run{color:var(--run)}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th{text-align:left;font-weight:600;color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.03em}
th,td{padding:3px 8px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:hover td{background:var(--row)}
tr.pruned td{color:var(--muted)}
time,th,.nowrap{white-space:nowrap}
.scroll{overflow-x:auto}
.banner{border:1px solid var(--warn);background:var(--warnbg);color:var(--warn);padding:8px 12px;border-radius:6px;margin:12px 0;font-weight:600}
.banner.error{border-color:var(--bad);color:var(--bad)}
.meta{color:var(--muted);margin:14px 0}
.dot{display:inline-block;width:.7em;height:.7em;border-radius:50%;background:currentColor;margin-right:.45em}
.crumb{margin:16px 0 4px}
h2{font-size:1.05rem;margin:28px 0 8px}
dl.kv{display:grid;grid-template-columns:max-content 1fr;gap:6px 22px;margin:8px 0}
dl.kv dt{color:var(--muted)}dl.kv dd{margin:0;overflow-wrap:anywhere}
"""
)

THEME_SCRIPT = (
    '<script>(function(){var r=document.documentElement,k="t1-theme";'
    'try{var s=localStorage.getItem(k);if(s==="dark"||s==="light")r.dataset.theme=s}catch(e){}'
    'addEventListener("DOMContentLoaded",function(){var b=document.getElementById("theme-toggle");'
    'b.hidden=false;b.onclick=function(){var d=r.dataset.theme||'
    '(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");'
    'd=d==="dark"?"light":"dark";r.dataset.theme=d;'
    'try{localStorage.setItem(k,d)}catch(e){}}})})();</script>'
)

GITHUB = re.compile(r"https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?")


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def short(sha):
    return esc(sha[:8]) if sha else "-"


def parse_time(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def stamp(text):
    """A UTC time to the minute; the full ISO time is in the tooltip."""
    if not text:
        return "-"
    return f"<time datetime=\"{esc(text)}\" title=\"{esc(text)}\">{esc(text[:10])} {esc(text[11:16])}Z</time>"


def write_atomic(path, text):
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def page(title, heading, body):
    return (f"<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{esc(title)}</title><style>{STYLE}</style>{THEME_SCRIPT}</head>\n"
            f"<body>\n<header class=\"top\"><div class=\"wrap\"><h1>{heading}</h1>"
            f"<button id=\"theme-toggle\" class=\"toggle\" type=\"button\" hidden "
            f"title=\"Toggle light/dark\">&#9680; theme</button></div></header>\n"
            f"<div class=\"wrap\">\n{body}\n</div>\n</body></html>\n")


def table(headers, rows):
    """rows: (css class, [cell html, ...]); one table row per output line."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        ("<tr class=\"" + cls + "\">" if cls else "<tr>")
        + "".join(f"<td>{c}</td>" for c in cells) + "</tr>\n"
        for cls, cells in rows)
    return f"<div class=\"scroll\"><table><thead><tr>{head}</tr></thead>\n<tbody>\n{body}</tbody></table></div>"


def commit_link(record, key, sha):
    """The short commit, linked to its GitHub page when the clone URL is on github.com."""
    if not sha:
        return "-"
    text = f"<code>{short(sha)}</code>"
    m = GITHUB.fullmatch((record.get("repos") or {}).get(key) or "")
    if not m:
        return text
    return f"<a href=\"https://github.com/{esc(m.group(1))}/{esc(m.group(2))}/commit/{esc(sha)}\">{text}</a>"


def pkg_name(key):
    return key.removesuffix("_COMMIT").lower().replace("_", "-")


def log_names(src_dir):
    """build.log first, then attempt-N.log in attempt order."""
    attempts = sorted(src_dir.glob("attempt-*.log"),
                      key=lambda p: int(p.stem.split("-", 1)[1]) if p.stem.split("-", 1)[1].isdigit() else 0)
    names = [p.name for p in attempts]
    if (src_dir / "build.log").exists():
        names.insert(0, "build.log")
    return names


def image_file(record):
    d = Path(record["images_dir"])
    return d / f"t1-lartpc-{record['variant']}-{record['tag']}.sif"


def is_pruned(record):
    """A successful build whose image files retention has deleted."""
    if record["status"] != "success":
        return False
    sif = image_file(record)
    return not (sif.exists() or sif.with_suffix(".oci.tar").exists())


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


def is_initial(record):
    """The first poll of a ledger: every package is new."""
    changed = record.get("changed") or {}
    return bool(changed) and all(old is None for old, _ in changed.values())


def changed_cell(record):
    changed = record.get("changed") or {}
    if not changed:
        return "-"
    if is_initial(record):
        return "<span class=\"muted\">initial</span>"
    names = [pkg_name(k) for k in sorted(changed)]
    text = ", ".join(names[:2]) + (f" +{len(names) - 2}" if len(names) > 2 else "")
    return f"<span title=\"{esc(', '.join(names))}\">{esc(text)}</span>"


def status_html(record):
    cls = {"success": "ok", "running": "run", "queued": "run"}.get(record["status"], "bad")
    return f"<span class=\"{cls}\"><span class=\"dot\"></span>{esc(record['status'])}</span>"


def review_html(review_state):
    cls = "bad" if review_state == "needs review" else "muted"
    return f"<span class=\"{cls}\">{esc(review_state)}</span>"


def nightly_build(link, records):
    """The record whose .sif the nightly symlink points at, or None."""
    if link is None or not os.path.islink(link):
        return None
    name = os.path.basename(os.readlink(link))
    return next((r for r in records if name == image_file(r).name), None)


def nightly_line(link, records):
    """The nightly symlink and the build it points at; "" when there is no link."""
    if link is None or not os.path.islink(link):
        return ""
    target = os.readlink(link)
    build = nightly_build(link, records)
    of = (f" (build <a href=\"builds/{esc(build['key'])}.html\"><code>{esc(build['key'])}</code></a>)"
          if build else "")
    return (f"Nightly image: <code>{esc(link)}</code> &rarr; <code>{esc(target)}</code>{of}<br>")


KEY_COMPONENTS = ("PyTorch", "JAX", "ROOT", "Geant4", "Python")


def key_versions(info):
    """One line with the main versions of an image."""
    vals = {r["name"]: r["installed"].split(" ")[0] for r in info["core"]}
    return " &middot; ".join(f"{esc(n)} <code>{esc(vals[n])}</code>"
                             for n in KEY_COMPONENTS if n in vals)


def render_image_page(out, record, info):
    """builds/<key>-image.html: facts, core stack (pinned vs installed), package list."""
    key = record["key"]
    facts = [("Image file", f"<code>{esc(info.get('sif'))}</code>"),
             ("Size", f"{info['size'] / 1e9:.1f} GB" if info.get("size") else "-"),
             ("Built", stamp(info.get("built_at"))),
             ("Python packages", esc(len(info["pip"]))),
             ("Conda packages", esc(info["conda_count"]))]
    rows = []
    for r in info["core"]:
        pinned = esc(r["pinned"]) if r["pinned"] else "<span class=\"muted\">-</span>"
        flag = " <span class=\"bad\">differs</span>" if r["differs"] else ""
        note = f"<span class=\"muted\">{esc(r['note'])}</span>"
        rows.append(("", [esc(r["name"]), pinned, esc(r["installed"]) + flag, note]))
    pip_rows = "".join(f"<tr><td>{esc(n)}</td><td>{esc(v)}</td></tr>" for n, v in info["pip"])
    body = (f"<p class=\"crumb\"><a href=\"../index.html\">&larr; all builds</a> / "
            f"<a href=\"{esc(key)}.html\">build <code>{esc(key)}</code></a></p>"
            "<dl class=\"kv\">" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in facts) + "</dl>"
            "<h2>Core stack</h2>" + table(["component", "pinned", "installed", "note"], rows)
            + f"<details><summary>All {len(info['pip'])} Python packages</summary>"
            "<div class=\"scroll\" style=\"max-height:420px;overflow:auto\"><table><thead>"
            "<tr><th>package</th><th>version</th></tr></thead>"
            f"<tbody>{pip_rows}</tbody></table></div></details>")
    write_atomic(out / "builds" / f"{key}-image.html",
                 page(f"{INDEX_TITLE}: image of build {key}",
                      f"Image of build <code>{esc(key)}</code>", body))


def render_index(out, site, records, review_states, last_poll, nightly_link, nightly_info, now):
    parts = []
    for cls, text in banners(site, last_poll, now):
        parts.append(f"<div class=\"banner {cls}\">{esc(text)}</div>")
    n_ok = sum(r["status"] == "success" for r in records)
    n_bad = sum(r["status"] == "failed" for r in records)
    n_rev = sum(review_states.get(r["key"]) == "needs review" for r in records)
    meta = [f"{len(records)} build{'' if len(records) == 1 else 's'}: "
            f"{n_ok} succeeded, {n_bad} failed, {n_rev} need review"]
    if last_poll is not None:
        state = "ok" if last_poll.get("ok") else f"error: {last_poll.get('error')}"
        meta.append(f"Last poll {esc(last_poll['time'])} ({esc(state)}). "
                    f"Rendered {esc(now.strftime('%Y-%m-%dT%H:%M:%SZ'))}.")
    parts.append(f"<p class=\"meta\">{nightly_line(nightly_link, records)}{'<br>'.join(meta)}</p>")
    if nightly_info:
        parts.append(f"<p class=\"meta\">Base image: {key_versions(nightly_info)} &middot; "
                     f"<a href=\"builds/{esc(nightly_build(nightly_link, records)['key'])}-image.html\">details</a></p>")
    rows = []
    for rec in records:
        key = rec["key"]
        pruned = is_pruned(rec)
        status = status_html(rec) + (
            " <span class=\"muted\" title=\"Image files removed by retention (NIGHTLY_KEEP)\">pruned</span>"
            if pruned else "")
        rows.append(("pruned" if pruned else "", [
            stamp(rec.get("queued_at")),
            f"<a href=\"builds/{esc(key)}.html\"><code>{esc(key)}</code></a>",
            status,
            review_html(review_states.get(key, "")),
            esc(rec.get("failed_stage") or ""),
            esc(rec["attempts"]),
            commit_link(rec, T1, rec["t1_software_commit"]),
            changed_cell(rec),
        ]))
    parts.append(table(["queued (UTC)", "build", "status", "review", "failed stage", "tries",
                        "t1-software", "changed"], rows))
    write_atomic(out / "index.html", page(INDEX_TITLE, esc(INDEX_TITLE), "\n".join(parts)))


def duration(record):
    if not (record.get("started_at") and record.get("finished_at")):
        return "-"
    minutes = int((parse_time(record["finished_at"]) - parse_time(record["started_at"])).total_seconds() // 60)
    return f"{minutes // 60} h {minutes % 60:02d} min"


def render_build(out, record, review, review_state, src_dir, info=None):
    key = record["key"]
    logs = log_names(src_dir)
    changed = record.get("changed") or {}
    t1_old = (changed.get(T1) or [None])[0]
    t1_cell = commit_link(record, T1, record["t1_software_commit"])
    if t1_old:
        t1_cell += f" <span class=\"muted\">was {commit_link(record, T1, t1_old)}</span>"
    if record["status"] != "success":
        image = "-"
    elif is_pruned(record):
        image = "<span class=\"muted\">pruned (removed by retention)</span>"
    else:
        image = f"<code>{esc(image_file(record))}</code>"
    fields = [("Status", status_html(record)), ("Review", esc(review_state) or "-"),
              ("Queued", stamp(record.get("queued_at"))), ("Started", stamp(record.get("started_at"))),
              ("Finished", stamp(record.get("finished_at"))), ("Duration", duration(record)),
              ("Attempts", esc(record["attempts"])),
              ("Slurm jobs", esc(", ".join(record.get("slurm_job_ids") or [])) or "-"),
              ("Variant", esc(record["variant"])), ("Tag", f"<code>{esc(record['tag'])}</code>"),
              ("t1-software", t1_cell), ("Image", image)]
    if record.get("failed_stage"):
        fields.insert(1, ("Failed stage", esc(record["failed_stage"])))
    parts = ["<p class=\"crumb\"><a href=\"../index.html\">&larr; all builds</a></p>",
             "<dl class=\"kv\">" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in fields) + "</dl>",
             "<h2>Packages</h2>"]
    rows = []
    for name, sha in sorted(record["pins"].items()):
        if name not in changed:
            note = "<span class=\"muted\">unchanged</span>"
        elif changed[name][0] is None:
            note = "<span class=\"muted\">first build</span>"
        else:
            note = f"was {commit_link(record, name, changed[name][0])}"
        rows.append(("", [esc(pkg_name(name)), commit_link(record, name, sha), note]))
    parts.append(table(["package", "pinned commit", "since previous poll"], rows))
    if review is not None or record["status"] in REVIEW_SECTION:
        parts.append("<h2>Review</h2>")
        if review is None:
            parts.append("<p class=\"muted\">No review.</p>")
        else:
            parts.append(
                f"<dl class=\"kv\"><dt>Reviewed</dt><dd>{stamp(review.get('time'))} by "
                f"{esc(review.get('user'))} (attempt {esc(review.get('attempt'))})</dd>"
                f"<dt>Note</dt><dd>{esc(review.get('note'))}</dd></dl>")
    if info:
        parts.append(f"<h2>Base image</h2><p>{key_versions(info)} &middot; "
                     f"<a href=\"{esc(key)}-image.html\">details</a></p>")
    parts.append("<h2>Logs</h2>")
    if logs:
        parts.append("<ul>" + "".join(
            f"<li><code>{esc(src_dir / n)}</code></li>" for n in logs) + "</ul>")
        if "build.log" in logs:
            lines = (src_dir / "build.log").read_text(errors="replace").splitlines()
            parts.append(f"<p class=\"meta\">Last {LOG_TAIL} lines of build.log</p>")
            parts.append("<pre>" + esc("\n".join(lines[-LOG_TAIL:])) + "</pre>")
    else:
        parts.append("<p class=\"muted\">No log yet.</p>")
    write_atomic(out / "builds" / f"{key}.html",
                 page(f"{INDEX_TITLE}: build {key}", f"Build <code>{esc(key)}</code>", "\n".join(parts)))
    if info:
        render_image_page(out, record, info)


def render(out, *, site, records, reviews, review_states, last_poll, build_dirs,
           image_infos=None, nightly_link=None, now):
    out = Path(out)
    image_infos = image_infos or {}
    (out / "builds").mkdir(parents=True, exist_ok=True)
    nightly = nightly_build(nightly_link, records)
    render_index(out, site, records, review_states, last_poll, nightly_link,
                 image_infos.get(nightly["key"]) if nightly else None, now)
    for rec in records:
        key = rec["key"]
        render_build(out, rec, reviews.get(key), review_states.get(key, ""), build_dirs[key],
                     image_infos.get(key))
