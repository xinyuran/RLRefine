from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "social-preview.png"
W, H = 1280, 640
BG = (247, 244, 236)
NAVY = (18, 40, 61)
TEAL = (79, 179, 163)
MINT = (217, 238, 233)
TEXT = (36, 67, 84)
MUTED = (101, 121, 133)
CREAM = (251, 248, 241)


def font(path: str, size: int):
    return ImageFont.truetype(path, size)


SC = r"C:\Windows\Fonts\Noto Sans SC (TrueType).otf"
SC_BOLD = r"C:\Windows\Fonts\Noto Sans SC Bold (TrueType).otf"
MONO = r"C:\Windows\Fonts\DejaVuSansMono.ttf"


def main() -> None:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    draw.ellipse((950, -230, 1370, 190), fill=(217, 238, 233))
    draw.ellipse((-100, 500, 200, 800), fill=(230, 220, 203))

    # Mark and title.
    draw.rounded_rectangle((78, 78, 156, 156), radius=22, fill=NAVY)
    draw.arc((101, 101, 143, 143), 35, 325, fill=(139, 211, 199), width=10)
    draw.text((180, 88), "RLRefine", font=font(SC_BOLD, 54), fill=NAVY)
    draw.text((180, 151), "结构化抽取 · 可验证输出", font=font(SC, 21), fill=(75, 97, 112))

    # Left message panel.
    draw.rounded_rectangle((82, 232, 564, 456), radius=26, fill=NAVY)
    draw.text((116, 267), "SCHEMA-DRIVEN EXTRACTION", font=font(SC_BOLD, 16), fill=(139, 211, 199))
    draw.text((116, 316), "Schema-driven extraction.", font=font(SC_BOLD, 34), fill=CREAM)
    draw.text((116, 359), "Source-grounded validation.", font=font(SC_BOLD, 30), fill=CREAM)
    draw.rounded_rectangle((116, 408, 358, 434), radius=13, fill=(32, 68, 90))
    draw.text((130, 412), "SFT → DPO → GRPO  ·  contracts", font=font(SC, 14), fill=MINT)

    # Right card.
    card = (620, 70, 1198, 570)
    draw.rounded_rectangle(card, radius=30, fill=(255, 255, 255))
    draw.rounded_rectangle((620, 70, 1198, 80), radius=5, fill=TEAL)
    draw.text((658, 108), "ILLUSTRATIVE EXAMPLE · NO LIVE INFERENCE", font=font(SC_BOLD, 15), fill=MUTED)
    draw.text((658, 148), "输入评论", font=font(SC_BOLD, 22), fill=NAVY)
    draw.rounded_rectangle((658, 190, 1160, 266), radius=16, fill=(241, 246, 244))
    draw.text((680, 212), "包装很好，屏幕清晰。", font=font(SC, 27), fill=TEXT)
    draw.line((906, 280, 906, 306), fill=TEAL, width=3)
    draw.polygon([(896, 300), (906, 310), (916, 300)], fill=TEAL)
    draw.text((658, 330), "结构化 JSON", font=font(SC_BOLD, 22), fill=NAVY)
    draw.rounded_rectangle((658, 374, 1160, 510), radius=16, fill=NAVY)
    draw.text((680, 403), '{ "keywords": [', font=font(MONO, 18), fill=(166, 225, 214))
    draw.text((699, 436), '["包装", 0.96], ["屏幕", 0.94],', font=font(MONO, 17), fill=CREAM)
    draw.text((699, 465), '["清晰", 0.91] ] }', font=font(MONO, 17), fill=CREAM)
    draw.rounded_rectangle((658, 530, 802, 556), radius=13, fill=MINT)
    draw.text((674, 534), "✓ source grounded", font=font(SC_BOLD, 14), fill=(23, 102, 93))
    draw.rounded_rectangle((814, 530, 940, 556), radius=13, fill=(237, 231, 218))
    draw.text((830, 534), "✓ schema checked", font=font(SC_BOLD, 14), fill=(114, 94, 62))

    image.save(OUT, format="PNG", optimize=True)
    assert image.size == (1280, 640)
    assert OUT.stat().st_size < 1_000_000


if __name__ == "__main__":
    main()
