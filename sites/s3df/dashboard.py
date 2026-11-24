"""Static HTML dashboard of the release and nightly build ledger.

render() writes into an output directory:

    index.html                 releases, one section per subgroup
    nightly.html               nightly builds, one section and table per subgroup
    builds/<key>.html          one page per build
    builds/<key>-image.html    the installed software of a build's image

The pages work without JavaScript. One small inline script adds a
light/dark toggle and remembers the choice; without it the pages follow
the system theme. All links are relative except the links to github.com.
Build logs are not published: a build page shows the log file location
and the last lines of build.log.

This module reads nothing from the ledger itself: nightly.py gathers the
inputs under the ledger lock and passes them in.
"""
import html
import os
import re
from datetime import datetime, timezone
from pathlib import Path

LOG_TAIL = 200  # lines of build.log shown on a build page
INDEX_TITLE = "T1 software"
NIGHTLY_TITLE = "T1 software nightly builds"
T1 = "T1_SOFTWARE_COMMIT"
REVIEW_SECTION = ("failed", "interrupted", "cancelled")

TITLES = {"lartpc": "T1-LArTPC", "wc": "T1-WC"}
# Subgroups whose nightly is a test build, not an official one: the note
# shown on the nightly page.
EXPERIMENTAL = {
    "wc": ("For testing only; not for datasets (use the <a href=\"index.html\">LUCiD release</a>). "
           "LUCiD and PhotonSim HEAD on the shared T1 base. PhotonSim uses Geant4 <code>11.3.2</code>, "
           "the version LUCiD requires; the base stays on <code>10.7.4</code> until T1-LArTPC moves "
           "to 11.3.2. A failure here does not block the T1-LArTPC nightly."),
}
# Display names of the /opt/VERSIONS sources.
SOURCE_NAMES = {"lucid": "LUCiD", "photonsim": "PhotonSim"}

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
h3{font-size:.95rem;margin:20px 0 6px}
dl.kv{display:grid;grid-template-columns:max-content 1fr;gap:6px 22px;margin:8px 0}
dl.kv dt{color:var(--muted)}dl.kv dd{margin:0;overflow-wrap:anywhere}
.badge{display:inline-block;font-size:.75rem;font-weight:600;padding:0 8px;border-radius:999px;border:1px solid currentColor;margin-left:8px;vertical-align:1px}
.badge.official{color:var(--ok)}.badge.exp{color:var(--warn)}.badge.none{color:var(--muted)}
.note{border-left:3px solid var(--warn);padding:4px 12px;margin:8px 0;color:var(--muted)}
section.card{border:1px solid var(--line);border-radius:8px;padding:2px 16px 10px;margin:16px 0}
nav.tabs{display:flex;gap:18px;margin:14px 0 0;border-bottom:1px solid var(--line)}
nav.tabs a{padding:4px 2px 8px;color:var(--muted)}
nav.tabs a.on{color:var(--fg);border-bottom:2px solid var(--link);font-weight:600}
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


def nav(current):
    tabs = [("index.html", "Releases"), ("nightly.html", "Nightly builds")]
    on = ' class="on"'
    return "<nav class=\"tabs\">" + "".join(
        f"<a href=\"{href}\"{on if href == current else ''}>{name}</a>" for href, name in tabs) + "</nav>"


def table(headers, rows):
    """rows: (css class, [cell html, ...]); one table row per output line."""
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        ("<tr class=\"" + cls + "\">" if cls else "<tr>")
        + "".join(f"<td>{c}</td>" for c in cells) + "</tr>\n"
        for cls, cells in rows)
    return f"<div class=\"scroll\"><table><thead><tr>{head}</tr></thead>\n<tbody>\n{body}</tbody></table></div>"


def kv(pairs):
    return "<dl class=\"kv\">" + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in pairs) + "</dl>"


def url_commit(url, sha):
    """The short commit, linked to its GitHub page when url is on github.com."""
    if not sha:
        return "-"
    text = f"<code>{short(sha)}</code>"
    m = GITHUB.fullmatch(url or "")
    if not m:
        return text
    return f"<a href=\"https://github.com/{esc(m.group(1))}/{esc(m.group(2))}/commit/{esc(sha)}\">{text}</a>"


def commit_link(record, key, sha):
    return url_commit((record.get("repos") or {}).get(key), sha)


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
    return d / f"t1-{record.get('subgroup', 'lartpc')}-{record['variant']}-{record['tag']}.sif"


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


def banner_html(cls, text):
    return f"<div class=\"banner {cls}\">{esc(text)}</div>"


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


def link_target(link):
    """(target text, target file name) of a symlink, or None when it is not one."""
    if link is None or not os.path.islink(link):
        return None
    target = os.readlink(link)
    return target, os.path.basename(target)


def nightly_build(link, records):
    """The record whose .sif the nightly symlink points at, or None."""
    t = link_target(link)
    if t is None:
        return None
    return next((r for r in records if t[1] == image_file(r).name), None)


def nightly_line(link, records):
    """The nightly symlink and the build it points at; "" when there is no link."""
    t = link_target(link)
    if t is None:
        return ""
    build = nightly_build(link, records)
    of = (f" (build <a href=\"builds/{esc(build['key'])}.html\"><code>{esc(build['key'])}</code></a>)"
          if build else "")
    return f"<code>{esc(link)}</code> &rarr; <code>{esc(t[0])}</code>{of}"


KEY_COMPONENTS = ("PyTorch", "JAX", "ROOT", "Geant4", "Python")


def key_versions(info):
    """The main versions of an image's /opt/conda, one line."""
    vals = {r["name"]: r["installed"].split(" ")[0] for r in info["core"] if r["installed"] != "-"}
    sidecar = "geant4" in (info.get("versions") or {})
    parts = []
    for name in KEY_COMPONENTS:
        if name not in vals:
            continue
        label = "Geant4 (base)" if name == "Geant4" and sidecar else name
        extra = " (CUDA)" if name == "JAX" and info.get("jax_gpu") else ""
        parts.append(f"{esc(label)} <code>{esc(vals[name])}</code>{extra}")
    return " &middot; ".join(parts)


def sources(versions):
    """The sources /opt/VERSIONS names: [(name, ref, commit link)], in file order."""
    out = []
    for key, sha in versions.items():
        if key.endswith("_sha"):
            name = key.removesuffix("_sha")
            ref = versions.get(f"{name}_ref", "")
            out.append((SOURCE_NAMES.get(name, name), "" if ref in ("", "none", "unknown") else ref,
                        url_commit(versions.get(f"{name}_source"), sha)))
    return out


def stack_line(info):
    """Provenance (sources, PhotonSim's Geant4) and the main versions of an image."""
    versions = info.get("versions") or {}
    parts = [f"{esc(n)}{f' <code>{esc(ref)}</code>' if ref else ''} {link}"
             for n, ref, link in sources(versions)]
    if versions.get("geant4"):
        side = " (PhotonSim)" if "photonsim_sha" in versions else ""
        parts.append(f"Geant4 <code>{esc(versions['geant4'])}</code>{side}")
    main = key_versions(info)
    return " &middot; ".join([*parts, *([main] if main else [])])


def gpu_line(info, image, hint=""):
    if not info:
        return "-"
    if info.get("jax_gpu"):
        return f"yes: <code>apptainer exec --nv {esc(image)} ...</code>"
    return f"<span class=\"muted\">no: the image's jaxlib is CPU-only.{hint}</span>"


def counts(records, review_states):
    n_ok = sum(r["status"] == "success" for r in records)
    n_bad = sum(r["status"] == "failed" for r in records)
    n_rev = sum(review_states.get(r["key"]) == "needs review" for r in records)
    return (f"{len(records)}: {n_ok} succeeded, {n_bad} failed, "
            f"{n_rev} need{'s' if n_rev == 1 else ''} review")


def size_text(size):
    return f"{size / 1e9:.1f} GB" if size else "-"


def render_image_page(out, record, info):
    """builds/<key>-image.html: facts, core stack (pinned vs installed), package list."""
    key = record["key"]
    facts = [("Image file", f"<code>{esc(info.get('sif'))}</code>"),
             ("Size", size_text(info.get("size"))),
             ("Built", stamp(info.get("built_at"))),
             ("Python packages", esc(len(info["pip"]))),
             ("Conda packages", esc(info["conda_count"]))]
    if info.get("versions"):
        facts.append(("Provenance", stack_line({"core": [], "versions": info["versions"]})))
    rows = []
    for r in info["core"]:
        pinned = esc(r["pinned"]) if r["pinned"] else "<span class=\"muted\">-</span>"
        flag = " <span class=\"bad\">differs</span>" if r["differs"] else ""
        note = f"<span class=\"muted\">{esc(r['note'])}</span>"
        rows.append(("", [esc(r["name"]), pinned, esc(r["installed"]) + flag, note]))
    pip_rows = "".join(f"<tr><td>{esc(n)}</td><td>{esc(v)}</td></tr>" for n, v in info["pip"])
    body = (f"<p class=\"crumb\"><a href=\"../nightly.html\">&larr; nightly builds</a> / "
            f"<a href=\"{esc(key)}.html\">build <code>{esc(key)}</code></a></p>"
            + kv(facts)
            + "<h2>Core stack</h2>" + table(["component", "pinned", "installed", "note"], rows)
            + f"<details><summary>All {len(info['pip'])} Python packages</summary>"
            "<div class=\"scroll\" style=\"max-height:420px;overflow:auto\"><table><thead>"
            "<tr><th>package</th><th>version</th></tr></thead>"
            f"<tbody>{pip_rows}</tbody></table></div></details>")
    write_atomic(out / "builds" / f"{key}-image.html",
                 page(f"{NIGHTLY_TITLE}: image of build {key}",
                      f"Image of build <code>{esc(key)}</code>", body))


# -- index.html: releases ----------------------------------------------------

def release_card(sub, rel, site):
    title = TITLES.get(sub, sub)
    cfg, recs, infos = rel["config"], rel["records"], rel["infos"]
    done = [r for r in recs if r["status"] == "success"]
    image, repo = cfg["image"], cfg["repo"]
    name = image.rsplit("/", 1)[-1]
    images = Path(site["IMAGES_DIR"])
    parts = []
    if done:
        rec = done[0]
        tag, info = rec["tag"], infos.get(rec["tag"]) or {}
        versions = info.get("versions") or {}
        docker = [f"<code>{esc(image)}:{esc(tag)}</code>"]
        if image.startswith("ghcr.io/"):
            docker.append(f"<a href=\"{esc(repo)}/pkgs/container/{esc(name)}\">ghcr.io package</a>")
        if cfg.get("recipe"):
            docker.append(f"<a href=\"{esc(repo)}/tree/{esc(tag)}/{esc(cfg['recipe'])}\">recipe</a>")
        t = link_target(rel["link"])
        apptainer = (f"<code>{esc(rel['link'])}</code> &rarr; <code>{esc(t[0])}</code>" if t
                     else f"<code>{esc(rec['sif'])}</code>")
        hint = (f" For GPU JAX, use the <a href=\"nightly.html#{esc(sub)}\">{esc(title)} nightly</a>"
                f"{' (experimental)' if sub in EXPERIMENTAL else ''}.")
        checks = ["image inspected"] + ([f"<code>{esc(Path(cfg['test']).name)}</code> passed"]
                                        if cfg.get("test") else [])
        parts.append(kv([
            ("Release", f"<a href=\"{esc(repo)}/releases/tag/{esc(tag)}\"><code>{esc(tag)}</code></a>"
                        + (f" &middot; built {esc(versions['build_date'][:10])}"
                           if versions.get("build_date", "unknown") != "unknown" else "")),
            ("Docker / Podman", " &middot; ".join(docker)),
            ("Apptainer (S3DF)", apptainer),
            ("Verify a copy", f"sha256 <code>{esc(rec['sha256'])}</code>, also in "
                              f"<code>{esc(Path(rec['sif']).name)}.sha256</code>"),
            ("Contents", stack_line(info) if info else "-"),
            ("GPU", gpu_line(info, rel["link"], hint)),
            ("Conversion checks", "<span class=\"ok\"><span class=\"dot\"></span>passed</span>"
                                  + f" <span class=\"muted\">({', '.join(checks)})</span>"),
        ]))
    else:
        parts.append(f"<p class=\"meta\">No converted release yet. Until the first one, the newest "
                     f"tested image is the <a href=\"nightly.html#{esc(sub)}\">{esc(title)} nightly</a>.</p>")
    rows = []
    for rec in recs:
        info = infos.get(rec["tag"]) or {}
        versions = info.get("versions") or {}
        src = sources(versions)
        sif = Path(rec["sif"]) if rec.get("sif") else None
        pruned = rec["status"] == "success" and not (sif and sif.exists())
        status = status_html(rec)
        if rec.get("failed_stage"):
            status += f" <span class=\"muted\">{esc(rec['failed_stage'])}</span>"
        if pruned:
            status += " <span class=\"muted\" title=\"Removed by retention (RELEASE_KEEP)\">pruned</span>"
        jax = next((r["installed"].split(" ")[0] for r in info.get("core", []) if r["name"] == "JAX"), "-")
        rows.append(("pruned" if pruned else "", [
            f"<a href=\"{esc(repo)}/releases/tag/{esc(rec['tag'])}\"><code>{esc(rec['tag'])}</code></a>",
            esc(versions.get("build_date", "-")[:10]),
            f"<code title=\"{esc(rec['digest'])}\">{esc(rec['digest'].removeprefix('sha256:')[:12])}</code>",
            src[0][2] if src else "-",
            f"<code>{esc(versions['geant4'])}</code>" if versions.get("geant4") else "-",
            f"<code>{esc(jax)}</code>" if jax != "-" else "-",
            stamp(rec.get("finished_at")),
            status,
            size_text(rec.get("size")),
            f"<code>{esc(os.path.relpath(sif, images))}</code>" if sif and not pruned else "-",
        ]))
    if rows:
        parts.append("<h3>Converted releases</h3>" + table(
            ["release", "built", "digest", "commit", "Geant4", "JAX", "converted (UTC)", "status",
             "size", "sif"], rows))
    badge = ("<span class=\"badge official\">release</span>" if done
             else "<span class=\"badge none\">no release yet</span>")
    keep = site["RELEASE_KEEP"]
    intro = (f"<p class=\"meta\">The {esc(title)} release is the upstream image "
             f"<code>{esc(image)}</code>, built and published by its own CI on each release tag. "
             f"S3DF converts each release to Apptainer for HPC sites; the newest {esc(keep)} "
             f"conversions are kept.</p>")
    return (f"<section class=\"card\" id=\"{esc(sub)}\"><h2>{esc(title)} {badge}</h2>"
            + intro + "".join(parts) + "</section>")


def render_index(out, site, subgroups, releases, release_check, now):
    parts = [nav("index.html")]
    if release_check is not None and not release_check.get("ok"):
        parts.append(banner_html("error", f"Last release check failed: {release_check.get('error')}"))
    meta = ["Release images of the T1 software, one per detector subgroup. Use a release for "
            "datasets. Nightly builds are on the <a href=\"nightly.html\">nightly page</a>."]
    if release_check is not None:
        st = "ok" if release_check.get("ok") else "error"
        meta.append(f"Last release check {esc(release_check['time'])} ({st}).")
    meta.append(f"Rendered {esc(now.strftime('%Y-%m-%dT%H:%M:%SZ'))}.")
    parts.append(f"<p class=\"meta\">{'<br>'.join(meta)}</p>")
    for sub in subgroups:
        if sub in releases:
            parts.append(release_card(sub, releases[sub], site))
        else:
            title = TITLES.get(sub, sub)
            parts.append(
                f"<section class=\"card\" id=\"{esc(sub)}\"><h2>{esc(title)} "
                "<span class=\"badge none\">no release yet</span></h2>"
                f"<p class=\"meta\">No {esc(title)} release exists yet. Until the first one, the newest "
                f"tested image is the <a href=\"nightly.html#{esc(sub)}\">{esc(title)} nightly</a>.</p>"
                "</section>")
    write_atomic(out / "index.html", page(INDEX_TITLE, esc(INDEX_TITLE), "\n".join(parts)))


# -- nightly.html ------------------------------------------------------------

def build_rows(records, review_states):
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
            esc(rec.get("failed_stage") or ""),
            review_html(review_states.get(key, "")),
            esc(rec["attempts"]),
            commit_link(rec, T1, rec["t1_software_commit"]),
            changed_cell(rec),
        ]))
    return rows


def render_nightly(out, site, subgroups, records, review_states, last_poll, nightly_links,
                   image_infos, now):
    parts = [nav("nightly.html")]
    parts += [banner_html(cls, text) for cls, text in banners(site, last_poll, now)]
    meta = ["Daily builds of the HEAD of t1-software and of every tracked package. For testing; "
            "use a <a href=\"index.html\">release</a> for datasets."]
    if last_poll is not None:
        st = "ok" if last_poll.get("ok") else f"error: {last_poll.get('error')}"
        meta.append(f"Last poll {esc(last_poll['time'])} ({esc(st)}). "
                    f"Rendered {esc(now.strftime('%Y-%m-%dT%H:%M:%SZ'))}.")
    parts.append(f"<p class=\"meta\">{'<br>'.join(meta)}</p>")
    for sub in subgroups:
        title = TITLES.get(sub, sub)
        recs = [r for r in records if r.get("subgroup", "lartpc") == sub]
        link = nightly_links.get(sub)
        linked = nightly_build(link, recs)
        info = image_infos.get(linked["key"]) if linked else None
        badge = " <span class=\"badge exp\">experimental &middot; unofficial</span>" if sub in EXPERIMENTAL else ""
        card = [f"<section class=\"card\" id=\"{esc(sub)}\"><h2>{esc(title)} nightly{badge}</h2>"]
        if sub in EXPERIMENTAL:
            card.append(f"<div class=\"note\">{EXPERIMENTAL[sub]}</div>")
        facts = [("Image", nightly_line(link, recs) or "<span class=\"muted\">no nightly image yet</span>")]
        if info:
            facts.append(("Stack", stack_line(info)
                          + f" &middot; <a href=\"builds/{esc(linked['key'])}-image.html\">details</a>"))
            facts.append(("GPU", gpu_line(info, link)))
        facts.append(("Builds", counts(recs, review_states)))
        card.append(kv(facts))
        card.append("<h3>Builds</h3>")
        if recs:
            card.append(table(["queued (UTC)", "build", "status", "failed stage", "review", "tries",
                               "t1-software", "changed"], build_rows(recs, review_states)))
        else:
            card.append("<p class=\"muted\">No builds yet.</p>")
        card.append("</section>")
        parts.append("".join(card))
    write_atomic(out / "nightly.html", page(NIGHTLY_TITLE, esc(INDEX_TITLE), "\n".join(parts)))


# -- builds/<key>.html -------------------------------------------------------

def duration(record):
    if not (record.get("started_at") and record.get("finished_at")):
        return "-"
    minutes = int((parse_time(record["finished_at"]) - parse_time(record["started_at"])).total_seconds() // 60)
    return f"{minutes // 60} h {minutes % 60:02d} min"


def render_build(out, record, review, review_state, src_dir, info=None):
    key = record["key"]
    sub = record.get("subgroup", "lartpc")
    logs = log_names(src_dir)
    job_logs = [p for p in (src_dir.parent.parent / "jobs" / f"{j}.log"
                            for j in record.get("slurm_job_ids") or []) if p.exists()]
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
    fields = [("Subgroup", esc(TITLES.get(sub, sub))),
              ("Status", status_html(record)), ("Review", esc(review_state) or "-"),
              ("Queued", stamp(record.get("queued_at"))), ("Started", stamp(record.get("started_at"))),
              ("Finished", stamp(record.get("finished_at"))), ("Duration", duration(record)),
              ("Attempts", esc(record["attempts"])),
              ("Slurm jobs", esc(", ".join(record.get("slurm_job_ids") or [])) or "-"),
              ("Variant", esc(record["variant"])), ("Tag", f"<code>{esc(record['tag'])}</code>"),
              ("t1-software", t1_cell), ("Image", image)]
    if record.get("failed_stage"):
        fields.insert(2, ("Failed stage", esc(record["failed_stage"])))
    parts = [f"<p class=\"crumb\"><a href=\"../nightly.html#{esc(sub)}\">&larr; nightly builds</a></p>",
             kv(fields), "<h2>Packages</h2>"]
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
        parts.append(f"<h2>Image</h2><p>{stack_line(info)} &middot; "
                     f"<a href=\"{esc(key)}-image.html\">details</a></p>")
    parts.append("<h2>Logs</h2>")
    if logs or job_logs:
        parts.append("<ul>" + "".join(f"<li><code>{esc(src_dir / n)}</code></li>" for n in logs)
                     + "".join(f"<li><code>{esc(p)}</code> <span class=\"muted\">(Slurm job)</span></li>"
                               for p in job_logs) + "</ul>")
        if "build.log" in logs:
            lines = (src_dir / "build.log").read_text(errors="replace").splitlines()
            parts.append(f"<p class=\"meta\">Last {LOG_TAIL} lines of build.log</p>")
            parts.append("<pre>" + esc("\n".join(lines[-LOG_TAIL:])) + "</pre>")
    else:
        parts.append("<p class=\"muted\">No log yet.</p>")
    write_atomic(out / "builds" / f"{key}.html",
                 page(f"{NIGHTLY_TITLE}: build {key}", f"Build <code>{esc(key)}</code>", "\n".join(parts)))
    if info:
        render_image_page(out, record, info)


def render(out, *, site, records, reviews, review_states, last_poll, build_dirs,
           image_infos=None, nightly_links=None, subgroups=("lartpc", "wc"), releases=None,
           release_check=None, now):
    """nightly_links: {subgroup: path of its nightly .sif symlink}.
    releases: {subgroup: {"config", "link", "records", "infos": {tag: image info}}},
    only for the subgroups whose releases are converted.
    """
    out = Path(out)
    image_infos = image_infos or {}
    nightly_links = nightly_links or {}
    (out / "builds").mkdir(parents=True, exist_ok=True)
    render_index(out, site, subgroups, releases or {}, release_check, now)
    render_nightly(out, site, subgroups, records, review_states, last_poll, nightly_links,
                   image_infos, now)
    for rec in records:
        key = rec["key"]
        render_build(out, rec, reviews.get(key), review_states.get(key, ""), build_dirs[key],
                     image_infos.get(key))
