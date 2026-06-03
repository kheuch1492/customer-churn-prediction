"""
Génère des images de fond professionnelles pour le dashboard Power BI
(format 1920x1080, ratio 16:9 = page Power BI par défaut), aux couleurs
du projet. Deux versions :
  - bg_branded_1920x1080.png : bandeau d'en-tête avec titre + pied de page contact
  - bg_subtle_1920x1080.png  : fond clair sobre avec accents discrets (sans texte)
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from . import config
except ImportError:
    import config

W, H = 1920, 1080
FONT = "C:/Windows/Fonts/arial.ttf"
FONTB = "C:/Windows/Fonts/arialbd.ttf"


def f(size, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, size)


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def vgradient(w, h, top, bottom):
    """Dégradé vertical (top -> bottom)."""
    t, b = np.array(hex2rgb(top)), np.array(hex2rgb(bottom))
    ramp = np.linspace(0, 1, h)[:, None]
    arr = (t[None, :] * (1 - ramp) + b[None, :] * ramp).astype(np.uint8)
    return Image.fromarray(np.repeat(arr[:, None, :], w, axis=1), "RGB")


def hgradient(w, h, left, right):
    """Dégradé horizontal (left -> right)."""
    l, r = np.array(hex2rgb(left)), np.array(hex2rgb(right))
    ramp = np.linspace(0, 1, w)[:, None]
    arr = (l[None, :] * (1 - ramp) + r[None, :] * ramp).astype(np.uint8)
    return Image.fromarray(np.repeat(arr[None, :, :], h, axis=0), "RGB")


def soft_circle(img, cx, cy, radius, color, alpha):
    """Dessine un cercle translucide (effet décoratif)."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
              fill=hex2rgb(color) + (alpha,))
    img.alpha_composite(overlay)


def build_branded():
    base = vgradient(W, H, "#f5f8fb", "#e8eef4").convert("RGBA")

    # Bandeau d'en-tête (dégradé bleu nuit -> bleu)
    header = hgradient(W, 210, "#14304a", "#2E86AB").convert("RGBA")
    base.paste(header, (0, 0))

    # Cercles décoratifs translucides dans l'en-tête
    soft_circle(base, 1750, 40, 200, "#ffffff", 22)
    soft_circle(base, 1600, 200, 120, "#ffffff", 16)

    # Ligne d'accent rouge sous l'en-tête
    d = ImageDraw.Draw(base)
    d.rectangle([0, 210, W, 218], fill=hex2rgb("#E15759"))

    # Titre + sous-titre
    d.text((60, 58), "Prédiction du Churn Client", font=f(56, True), fill="#ffffff")
    d.text((62, 138),
           "Tableau de bord décisionnel  ·  Telco · 7 043 clients  ·  modèle Gradient Boosting (ROC-AUC 0,844)",
           font=f(24), fill=(255, 255, 255, 220))

    # Pied de page : bande claire + contact
    d.rectangle([0, H - 56, W, H], fill=hex2rgb("#dfe7ee"))
    d.text((60, H - 42), "Source de données : Telco Customer Churn (IBM)",
           font=f(20), fill=hex2rgb("#5a6b7a"))
    contact = "Data Analyste / Analyse BI — Cheikh Sall  ·  sall1969@outlook.fr  ·  Tél. 77 245 62 22"
    tw = d.textlength(contact, font=f(20, True))
    d.text((W - tw - 60, H - 42), contact, font=f(20, True), fill=hex2rgb("#1B3A4B"))

    out = config.POWERBI_DIR / "bg_branded_1920x1080.png"
    base.convert("RGB").save(out, "PNG")
    print(f"[bg] {out.name}")


def build_subtle():
    base = vgradient(W, H, "#f7fafc", "#e9eff5").convert("RGBA")

    # Accents géométriques très discrets (coins)
    soft_circle(base, 1850, -60, 380, "#2E86AB", 16)
    soft_circle(base, 1700, 120, 220, "#4C9F70", 12)
    soft_circle(base, -80, 1140, 360, "#1B3A4B", 12)

    d = ImageDraw.Draw(base)
    # Fine ligne d'accent en haut
    d.rectangle([0, 0, W, 8], fill=hex2rgb("#2E86AB"))
    # Pied de page discret (droite)
    contact = "Data Analyste / Analyse BI — Cheikh Sall  ·  sall1969@outlook.fr  ·  Tél. 77 245 62 22"
    tw = d.textlength(contact, font=f(19))
    d.text((W - tw - 50, H - 40), contact, font=f(19), fill=hex2rgb("#94a3b8"))

    out = config.POWERBI_DIR / "bg_subtle_1920x1080.png"
    base.convert("RGB").save(out, "PNG")
    print(f"[bg] {out.name}")


if __name__ == "__main__":
    build_branded()
    build_subtle()
    print("Fonds Power BI générés dans powerbi/.")
