"""Design tokens: color, type, spacing, shape, and the stylesheet built from them.

Every size in the UI comes from here. Text is always black; grey is used only
for lines, tracks and empty placeholders. Teal marks interactive elements, red
marks only the distance warning.
"""
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase

# ---- color
WHITE = QColor("#FFFFFF")
BLACK = QColor("#000000")
TEAL = QColor("#2B4F60")        # interactive elements only
HAIRLINE = QColor("#BDC7C9")    # borders and dividers only
TRACK = QColor("#E6EBEC")       # empty part of bars and sliders
PLACEHOLDER = QColor("#F2F4F4") # stream area before the first frame
RED = QColor("#C62828")         # distance warning only

# ---- type scale (px). Three sizes, nothing in between.
TEXT = 14        # buttons, captions, footnotes, tags, header meta, frame counter
TITLE = 18       # header, card titles, card messages, units (mm, %)
DISPLAY = 56     # the two readouts

# ---- spacing (8 px grid) and shape
S1, S2, S3, S4 = 8, 16, 24, 32
RADIUS = 8                  # every control, card and tag
CONTROL_H = 36              # every button and the view selector
HEADER_H = 56
PANEL_W = 320
CAPTION_H = S4              # caption line under a stream, including its gap

FAMILIES = ["Inter", "Segoe UI Variable", "Segoe UI", "SF Pro Text", "Helvetica Neue",
            "Ubuntu", "Noto Sans", "DejaVu Sans", "Arial"]

STYLESHEET = f"""
#root {{ background: {WHITE.name()}; }}
QLabel {{ color: black; background: transparent; }}
QLabel#header {{ font-size: {TITLE}px; font-weight: 600; }}
#divider {{ background: {HAIRLINE.name()}; }}

QPushButton#primary {{
    background: {TEAL.name()}; color: white; border: none; border-radius: {RADIUS}px;
    min-height: {CONTROL_H}px; max-height: {CONTROL_H}px; padding: 0 {S2}px;
    font-size: {TEXT}px; font-weight: 600;
}}
QSlider {{ min-height: {CONTROL_H}px; }}
QSlider::groove:horizontal {{ height: 4px; background: {TRACK.name()}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: black; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: black; width: {S2}px; height: {S2}px; margin: -6px 0; border-radius: {RADIUS}px; }}
"""


def font(px: int, weight: QFont.Weight = QFont.Weight.Normal, tabular: bool = False) -> QFont:
    f = QFont()
    f.setFamilies(FAMILIES)
    f.setPixelSize(px)
    f.setWeight(weight)
    if tabular:                       # equal-width digits, so numbers do not jitter as they change
        try:
            f.setFeature(QFont.Tag("tnum"), 1)
        except (AttributeError, TypeError):
            pass
    return f


def load_fonts(app):
    """Use the bundled Inter (fonts/), else the first installed family in FAMILIES."""
    for ttf in sorted((Path(__file__).parent / "fonts").glob("*.ttf")):
        QFontDatabase.addApplicationFont(str(ttf))
    app.setFont(font(TEXT))
