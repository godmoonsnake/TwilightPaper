from io import BytesIO

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.util import Inches, Pt

from .agents import SlideDeck

NAVY = RGBColor(19, 38, 46)
TEAL = RGBColor(0, 101, 94)
PAPER = RGBColor(248, 249, 245)
WHITE = RGBColor(255, 255, 255)
MINT = RGBColor(182, 234, 218)


def add_text(slide, text, x, y, width, height, size, color, *, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    frame.margin_left = frame.margin_right = Inches(0)
    for index, line in enumerate(text.splitlines()):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.space_after = Pt(14)
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color


def build_powerpoint(deck: SlideDeck, source_filename: str) -> bytes:
    presentation = Presentation()
    presentation.slide_width = Inches(13.33)
    presentation.slide_height = Inches(7.5)
    presentation.core_properties.title = deck.title
    blank = presentation.slide_layouts[6]

    cover = presentation.slides.add_slide(blank)
    cover.background.fill.solid()
    cover.background.fill.fore_color.rgb = NAVY
    add_text(cover, deck.title, 0.85, 1.6, 11.45, 3.75, 46, WHITE, bold=True)
    add_text(cover, f"Source: {source_filename}", 0.85, 6.35, 11.45, 0.55, 16, MINT)

    for index, content in enumerate(deck.slides, start=1):
        slide = presentation.slides.add_slide(blank)
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = PAPER
        add_text(slide, content.title, 0.85, 0.62, 11.45, 1.25, 36, NAVY, bold=True)
        bullets = "\n".join(f"•  {bullet}" for bullet in content.bullets)
        add_text(slide, bullets, 0.9, 2.0, 11.35, 4.25, 26, NAVY)
        pages = ", ".join(map(str, sorted(set(content.pages))))
        slide.notes_slide.notes_text_frame.text = (
            f"{content.notes or 'Discuss the supporting evidence on the listed pages.'}\n"
            f"Source: {source_filename}, PDF pages {pages}"
        )
        add_text(slide, f"PDF pages {pages}", 0.9, 6.72, 10.6, 0.45, 14, TEAL)
        add_text(slide, str(index + 1), 11.8, 6.72, 0.65, 0.45, 14, TEAL)

    output = BytesIO()
    presentation.save(output)
    return output.getvalue()
