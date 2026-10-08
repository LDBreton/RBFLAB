"""Check expanded snippets and math in the generated site."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
for relative in (
    "tutorials/one-stencil/index.html",
    "tutorials/heat-equation/index.html",
    "tutorials/stokes/index.html",
    "theory/rbf-fd/index.html",
):
    html = (root / "site" / relative).read_text(encoding="utf-8")
    assert "--8<--" not in html, relative
    assert 'class="arithmatex"' in html, relative
    assert '<em>i)' not in html, relative

heat = (root / "site/tutorials/heat-equation/index.html").read_text(encoding="utf-8")
for name in ("five_point_laplacian", "rbf_fd_laplacian", "symbolic_solution"):
    assert name in heat, name
print("Rendered documentation includes math and runnable source.")

# Every page must carry the same versioned manual assets, including nested pages.
from hashlib import sha256
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
        self.stylesheets = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "link" and attrs.get("rel") == "stylesheet":
            self.stylesheets.append(attrs["href"])
        for key in ("href", "src"):
            if attrs.get(key):
                self.urls.append(attrs[key])

site = root / "site"
pages = {}
for path in site.rglob("*.html"):
    parser = PageLinks()
    parser.feed(path.read_text(encoding="utf-8"))
    pages[path.resolve()] = parser

asset = "stylesheets/manual.css"
digest = sha256((root / "docs" / asset).read_bytes()).hexdigest()[:12]
for path, parser in pages.items():
    assert any(url.endswith(asset + "?v=" + digest) for url in parser.stylesheets), path
    for url in parser.urls:
        parts = urlsplit(url)
        if parts.scheme or parts.netloc or parts.path.startswith("/"):
            continue
        target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
        if target.is_dir():
            target /= "index.html"
        assert target.exists(), (path, url)
        if parts.fragment and target in pages:
            assert unquote(parts.fragment) in pages[target].ids, (path, url)
print(f"Shared versioned styling and local links verified on {len(pages)} pages.")
