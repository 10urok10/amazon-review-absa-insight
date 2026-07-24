"""
Generates the extension's toolbar icons programmatically (a simple magnifying
glass on a solid background) so no external image-editing tool is needed.
Re-run only if you want to change the icon design.

Usage:
    python generate_icons.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

OUT_DIR = Path(__file__).resolve().parent / "icons"
BG_COLOR = (42, 120, 214, 255)  # brand blue, matches the project's categorical palette
GLASS_COLOR = (255, 255, 255, 255)
SIZES = [16, 48, 128]


def draw_icon(size: int) -> Image.Image:
    # Draw at 4x then downscale for clean anti-aliased edges at small sizes.
    scale = 4
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    corner_radius = s * 0.22
    draw.rounded_rectangle([0, 0, s - 1, s - 1], radius=corner_radius, fill=BG_COLOR)

    # Magnifying glass: circle + handle, roughly centered, sized proportionally.
    lens_r = s * 0.22
    cx, cy = s * 0.42, s * 0.42
    stroke = max(2 * scale, int(s * 0.07))
    draw.ellipse(
        [cx - lens_r, cy - lens_r, cx + lens_r, cy + lens_r],
        outline=GLASS_COLOR,
        width=stroke,
    )
    handle_start = (cx + lens_r * 0.72, cy + lens_r * 0.72)
    handle_end = (s * 0.78, s * 0.78)
    draw.line([handle_start, handle_end], fill=GLASS_COLOR, width=stroke)

    return img.resize((size, size), Image.LANCZOS)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    for size in SIZES:
        icon = draw_icon(size)
        path = OUT_DIR / f"icon{size}.png"
        icon.save(path)
        print(f"[icons] wrote {path.name} ({size}x{size})")


if __name__ == "__main__":
    main()
