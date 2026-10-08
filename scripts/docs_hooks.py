"""Keep custom documentation assets fresh across GitHub Pages deployments."""
from hashlib import sha256
from pathlib import Path
import re

def version_assets(output, config):
    """Version custom asset URLs while preserving each relative prefix."""
    for relative in ("stylesheets/manual.css", "javascripts/mathjax.js"):
        digest = sha256((Path(config["docs_dir"]) / relative).read_bytes()).hexdigest()[:12]
        pattern = re.escape(relative) + r'(?:\?v=[a-f0-9]+)?(?=["\x27])'
        output = re.sub(pattern, relative + "?v=" + digest, output)
    return output


def on_post_build(config):
    """Cover every HTML output, including the separately rendered 404 page."""
    for path in Path(config["site_dir"]).rglob("*.html"):
        output = path.read_text(encoding="utf-8")
        path.write_text(version_assets(output, config), encoding="utf-8")
