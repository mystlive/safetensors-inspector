# -*- coding: utf-8 -*-
"""Single-file HTML report for stinspect.

Takes a page dict whose strings are already resolved for the chosen language
and writes one self-contained file: no external CSS, no external JS, no network
access. It deliberately imports neither rules.py nor i18n.py - everything it
needs is in the dict, so display work never reaches back into the detection
logic.

Escaping is the one failure this output format has that the others do not.
Model metadata routinely carries raw JSON from ComfyUI workflows, angle
brackets and all, and a `</script>` inside it would end the payload element. So
the embedded JSON has `<`, `>` and `&` replaced by their \\uXXXX escapes (in
JSON those three characters only ever occur inside string literals, so the
result still parses), and every value reaches the DOM through textContent.
"""
from __future__ import annotations

import json

CSS = """
:root {
  --bg: #16181d; --panel: #1e2128; --line: #30343d; --fg: #dfe3ea;
  --dim: #949cab; --accent: #7fb4ff; --warn: #ffb454; --bad: #ff7b72;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.6 "Segoe UI", "Yu Gothic UI", "Hiragino Sans", system-ui, sans-serif;
}
header { padding: 18px 20px 12px; border-bottom: 1px solid var(--line); }
h1 { margin: 0 0 10px; font-size: 17px; font-weight: 600; letter-spacing: .02em; }
.controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
input, select {
  background: var(--panel); color: var(--fg); border: 1px solid var(--line);
  border-radius: 4px; padding: 6px 9px; font: inherit;
}
input { min-width: min(340px, 60vw); }
input:focus, select:focus { outline: 1px solid var(--accent); }
.count { color: var(--dim); font-size: 13px; margin-left: auto; }
.wrap { padding: 0 20px 40px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th {
  position: sticky; top: 0; background: var(--bg); cursor: pointer;
  font-weight: 600; color: var(--dim); white-space: nowrap; user-select: none;
}
th:hover { color: var(--fg); }
th .arrow { color: var(--accent); }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
tr.item { cursor: pointer; }
tr.item:hover > td { background: #232732; }
tr.item.open > td { background: #262b36; }
td.name { font-weight: 600; overflow-wrap: anywhere; min-width: 18ch; }
td.dim { color: var(--dim); }
/* Type and base are full sentences; one line each keeps rows scannable, and
   the untruncated text is a hover away and always in the detail panel. */
span.clip {
  display: block; max-width: 30ch; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap;
}
.bad { color: var(--bad); }
.warn { color: var(--warn); }
tr.detail > td { background: var(--panel); padding: 0; }
.detail-inner { padding: 14px 18px 18px; border-left: 3px solid var(--accent); }
.path {
  color: var(--dim); font-family: ui-monospace, Consolas, monospace;
  font-size: 12px; word-break: break-all; margin-bottom: 4px;
}
.stats {
  font-family: ui-monospace, Consolas, monospace; font-size: 12px;
  color: var(--dim); margin-bottom: 10px;
}
dl { display: grid; grid-template-columns: max-content 1fr; gap: 3px 16px; margin: 0; }
dt { color: var(--dim); white-space: nowrap; }
dd { margin: 0; min-width: 0; }
dd div { word-break: break-word; }
dd div.sub { color: var(--dim); }
dd .prefix { color: var(--dim); }
.empty { padding: 40px 0; color: var(--dim); text-align: center; }
"""

JS = """
(function () {
  var page = JSON.parse(document.getElementById('stinspect-data').textContent);
  var ui = page.ui, cols = page.columns, rows = page.rows;
  var tbody = document.getElementById('rows');
  var q = document.getElementById('q');
  var kindSel = document.getElementById('kind');
  var count = document.getElementById('count');
  var sortKey = null, sortDir = 1;
  var open = {};

  var kinds = [];
  rows.forEach(function (r) {
    if (r.kind && kinds.indexOf(r.kind) < 0) kinds.push(r.kind);
  });
  kinds.sort();
  kinds.forEach(function (k) {
    var o = document.createElement('option');
    o.value = k; o.textContent = k;
    kindSel.appendChild(o);
  });

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }

  function matches(r) {
    if (kindSel.value && r.kind !== kindSel.value) return false;
    var s = q.value.trim().toLowerCase();
    if (!s) return true;
    return (r.haystack || '').indexOf(s) >= 0;
  }

  function sorted(list) {
    if (!sortKey) return list;
    var col = null;
    cols.forEach(function (c) { if (c.key === sortKey) col = c; });
    var numeric = col && col.numeric;
    return list.slice().sort(function (a, b) {
      var x = numeric ? (a.sort[sortKey] || 0) : (a[sortKey] || '');
      var y = numeric ? (b.sort[sortKey] || 0) : (b[sortKey] || '');
      if (x < y) return -sortDir;
      if (x > y) return sortDir;
      return a.name.localeCompare(b.name);
    });
  }

  function detailNode(r) {
    var box = el('div', 'detail-inner');
    box.appendChild(el('div', 'path', r.path));
    if (r.error) {
      box.appendChild(el('div', 'stats bad', r.error));
      return box;
    }
    box.appendChild(el('div', 'stats', r.detail.stats));
    if (r.detail.size_warning) {
      box.appendChild(el('div', 'stats warn', r.detail.size_warning));
    }
    var dl = el('dl');
    r.detail.rows.forEach(function (entry) {
      dl.appendChild(el('dt', null, entry.label));
      var dd = el('dd');
      entry.lines.forEach(function (line, i) {
        var d = el('div', i ? 'sub' : null);
        if (line.prefix) d.appendChild(el('span', 'prefix', line.prefix));
        d.appendChild(document.createTextNode(line.text));
        dd.appendChild(d);
      });
      dl.appendChild(dd);
    });
    box.appendChild(dl);
    return box;
  }

  function render() {
    var list = sorted(rows.filter(matches));
    tbody.textContent = '';
    count.textContent = ui.showing
      .replace('{n}', list.length).replace('{total}', rows.length);

    if (!list.length) {
      var tr = el('tr');
      var td = el('td', 'empty', ui.no_match);
      td.colSpan = cols.length;
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }

    list.forEach(function (r) {
      var tr = el('tr', 'item' + (open[r.id] ? ' open' : ''));
      cols.forEach(function (c, i) {
        var cls = c.numeric ? 'num' : (i === 0 ? 'name' : 'dim');
        var text = r[c.key] || '';
        var td = el('td', r.error && i === 0 ? 'name bad' : cls);
        if (c.clip && text) {
          var span = el('span', 'clip', text);
          span.title = text;
          td.appendChild(span);
        } else {
          td.textContent = text;
        }
        tr.appendChild(td);
      });
      tr.addEventListener('click', function () {
        open[r.id] = !open[r.id];
        render();
      });
      tbody.appendChild(tr);

      if (open[r.id]) {
        var dtr = el('tr', 'detail');
        var dtd = el('td');
        dtd.colSpan = cols.length;
        dtd.appendChild(detailNode(r));
        dtr.appendChild(dtd);
        tbody.appendChild(dtr);
      }
    });
  }

  var head = document.getElementById('head');
  cols.forEach(function (c) {
    var th = el('th', null, c.label);
    th.addEventListener('click', function () {
      if (sortKey === c.key) { sortDir = -sortDir; } else { sortKey = c.key; sortDir = 1; }
      head.querySelectorAll('.arrow').forEach(function (a) { a.remove(); });
      th.appendChild(el('span', 'arrow', sortDir > 0 ? ' \\u25b2' : ' \\u25bc'));
      render();
    });
    head.appendChild(th);
  });

  q.addEventListener('input', render);
  kindSel.addEventListener('change', render);
  render();
})();
"""


def _attr(s):
    """Escape a value going into an HTML attribute or text node."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _payload(page):
    """Serialise the page for a <script type="application/json"> element.

    In JSON, `<`, `>` and `&` can only appear inside string literals, so
    replacing them with \\uXXXX escapes keeps the document parseable while
    making a `</script>` inside model metadata impossible.
    """
    s = json.dumps(page, ensure_ascii=False)
    return (s.replace("&", "\\u0026")
             .replace("<", "\\u003c")
             .replace(">", "\\u003e"))


def render(page):
    """Return the complete HTML document as a string."""
    return f"""<!DOCTYPE html>
<html lang="{_attr(page.get('lang', 'en'))}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_attr(page['title'])}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>{_attr(page['title'])}</h1>
  <div class="controls">
    <input id="q" type="search" placeholder="{_attr(page['ui']['search'])}">
    <select id="kind"><option value="">{_attr(page['ui']['all_kinds'])}</option></select>
    <span class="count" id="count"></span>
  </div>
</header>
<div class="wrap">
  <table><thead><tr id="head"></tr></thead><tbody id="rows"></tbody></table>
</div>
<script type="application/json" id="stinspect-data">{_payload(page)}</script>
<script>{JS}</script>
</body>
</html>
"""


def write_html(page, path):
    """Write the report. UTF-8 without BOM: browsers read the charset meta."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(render(page))
