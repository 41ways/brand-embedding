"""embed/out/preview.json 을 viewer.template.html 에 넣어 embed/out/brand-space.html 을 만든다."""
from pathlib import Path

HERE = Path(__file__).parent
data = (HERE / "out" / "preview.json").read_text().replace("</", "<\\/")
html = (HERE / "viewer.template.html").read_text().replace("__DATA__", data)
(HERE / "out" / "brand-space.html").write_text(html)
print("ok", len(html) // 1024, "KB")
