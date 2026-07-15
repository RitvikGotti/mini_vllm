from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, Preformatted, HRFlowable,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "transformer_revision_notes.md"
OUTDIR = ROOT / "output" / "pdf"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTDIR / "transformer_revision_notes.pdf"

PAGE_W, PAGE_H = A4
NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1769AA")
TEAL = colors.HexColor("#007C83")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
PALE = colors.HexColor("#EAF3F8")
LINE = colors.HexColor("#C9D8E5")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleX", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=26,
    leading=31, textColor=NAVY, alignment=TA_CENTER, spaceAfter=7))
styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=10.5,
    leading=15, textColor=MUTED, alignment=TA_CENTER, spaceAfter=18))
styles.add(ParagraphStyle(name="H1X", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16,
    leading=20, textColor=NAVY, spaceBefore=16, spaceAfter=7, keepWithNext=True))
styles.add(ParagraphStyle(name="H2X", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
    leading=14, textColor=TEAL, spaceBefore=11, spaceAfter=4, keepWithNext=True))
styles.add(ParagraphStyle(name="BodyX", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.8,
    leading=12.3, textColor=INK, spaceAfter=4.5))
styles.add(ParagraphStyle(name="BulletX", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.8,
    leading=12.1, leftIndent=13, firstLineIndent=-9, textColor=INK, spaceAfter=2.5))
styles.add(ParagraphStyle(name="FormulaX", parent=styles["BodyText"], fontName="Courier", fontSize=8.0,
    leading=11, leftIndent=7, rightIndent=7, textColor=NAVY, backColor=colors.HexColor("#F5F9FC"),
    borderColor=LINE, borderWidth=.35, borderPadding=5, spaceBefore=3, spaceAfter=5))
styles.add(ParagraphStyle(name="SmallX", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.4,
    leading=9.4, textColor=INK))

def inline(s):
    s = escape(s.strip())
    s = re.sub(r"`([^`]+)`", r'<font name="Courier" color="#0B5C82">\1</font>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s

def add_table(rows, story):
    cells = [[inline(c) for c in r] for r in rows]
    n = max(len(r) for r in cells)
    cells = [r + [""] * (n-len(r)) for r in cells]
    width = (PAGE_W - 34*mm) / n
    data = [[Paragraph(c, styles["SmallX"]) for c in row] for row in cells]
    tbl = Table(data, colWidths=[width]*n, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), .3, LINE), ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BACKGROUND", (0,1), (-1,-1), colors.white),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F5F9FC")]),
        ("LEFTPADDING", (0,0), (-1,-1), 5), ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4), ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.extend([Spacer(1, 3), tbl, Spacer(1, 7)])

def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE); canvas.setLineWidth(.4)
    canvas.line(17*mm, 13*mm, PAGE_W-17*mm, 13*mm)
    canvas.setFont("Helvetica", 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(17*mm, 8.5*mm, "GPT-Style Transformers — Engineering Revision Notes")
    canvas.drawRightString(PAGE_W-17*mm, 8.5*mm, f"{doc.page}")
    canvas.restoreState()

def build():
    story = []
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story += [Spacer(1, 34*mm), Paragraph("GPT-Style Transformers", styles["TitleX"]),
              Paragraph("Engineering revision notes for reconstructing a decoder-only Transformer forward pass", styles["Subtitle"]),
              HRFlowable(width="45%", thickness=2, color=TEAL, spaceBefore=6, spaceAfter=16),
              Paragraph("Designed as a dense, notebook-ready reference: what each component is, why it exists, how it works, and what it feeds next.", styles["BodyX"]),
              Spacer(1, 14*mm), Paragraph("Reading convention", styles["H2X"]),
              Paragraph("Learned parameters are persistent tensors updated during training and frozen during inference. Activations are temporary, input-dependent tensors. Shapes use B=batch size, T=sequence length, V=vocabulary size, d=hidden size.", styles["BodyX"]), PageBreak()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("# "):
            i += 1; continue
        if line.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=.55, color=LINE, spaceBefore=8, spaceAfter=5)); i += 1; continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["H1X"])); i += 1; continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["H2X"])); i += 1; continue
        if line.startswith("```"):
            block=[]; i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i]); i += 1
            # Diagrams are intentionally preformatted: vertical arrows occupy their own lines.
            story.extend([Spacer(1,3), Preformatted("\n".join(block), ParagraphStyle("Diagram", fontName="Courier", fontSize=8.3, leading=11.2, textColor=NAVY, backColor=colors.HexColor("#F2F7FA"), borderColor=LINE, borderWidth=.45, borderPadding=8, leftIndent=3, rightIndent=3)), Spacer(1,7)])
            i += 1; continue
        if line.startswith("|") and i+1 < len(lines) and lines[i+1].startswith("|"):
            rows=[]
            while i < len(lines) and lines[i].startswith("|"):
                raw=[c.strip() for c in lines[i].strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c.replace(" ","")) for c in raw): rows.append(raw)
                i += 1
            add_table(rows, story); continue
        if line.startswith("- "):
            story.append(Paragraph("• " + inline(line[2:]), styles["BulletX"])); i += 1; continue
        if not line.strip():
            story.append(Spacer(1, 2)); i += 1; continue
        # Isolate display-style equations / calculations to make scanning easier.
        if ("=" in line and (line.startswith("`") or line.startswith("$") or line.startswith("For "))):
            story.append(Paragraph(inline(line), styles["FormulaX"])); i += 1; continue
        story.append(Paragraph(inline(line), styles["BodyX"])); i += 1
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=17*mm, rightMargin=17*mm,
        topMargin=17*mm, bottomMargin=19*mm, title="GPT-Style Transformers — Engineering Revision Notes", author="Codex")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)

if __name__ == "__main__":
    build()
