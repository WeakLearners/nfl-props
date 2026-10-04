"""Team colours for the report pages. Display only.

Each team has a primary colour and an alt colour. Depth badges always use the
primary. Curves and swatches use the alt in dark mode when the primary is too
dark to see on the dark background. The away team also uses its alt when the
two primaries are too close to tell apart on one chart.
"""
TEAM_COLORS = {
    "ARI": ("#97233F", "#FFB612"), "ATL": ("#A71930", "#E8495C"),
    "BAL": ("#241773", "#9E7C0C"), "BUF": ("#00338D", "#C60C30"),
    "CAR": ("#0085CA", "#5FC3F0"), "CHI": ("#0B162A", "#C83803"),
    "CIN": ("#FB4F14", "#FB4F14"), "CLE": ("#FF3C00", "#FF3C00"),
    "DAL": ("#003594", "#869397"), "DEN": ("#FB4F14", "#FB4F14"),
    "DET": ("#0076B6", "#4FB0E8"), "GB": ("#203731", "#FFB612"),
    "HOU": ("#03202F", "#E0303F"), "IND": ("#002C5F", "#A2AAAD"),
    "JAX": ("#006778", "#D7A22A"), "KC": ("#E31837", "#FFB81C"),
    "LV": ("#000000", "#A5ACAF"), "LAC": ("#0080C6", "#FFC20E"),
    "LA": ("#003594", "#FFA300"), "MIA": ("#008E97", "#FC4C02"),
    "MIN": ("#4F2683", "#FFC62F"), "NE": ("#002244", "#C60C30"),
    "NO": ("#101820", "#D3BC8D"), "NYG": ("#0B2265", "#E0303F"),
    "NYJ": ("#125740", "#4CAF7D"), "PHI": ("#004C54", "#A5ACAF"),
    "PIT": ("#101820", "#FFB612"), "SF": ("#AA0000", "#B3995D"),
    "SEA": ("#002244", "#69BE28"), "TB": ("#D50A0A", "#FF7A6B"),
    "TEN": ("#0C2340", "#4B92DB"), "WAS": ("#5A1414", "#FFB612"),
}


def _rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def _lum(h):
    """Relative luminance, 0 (black) to 1 (white)."""
    c = [v / 255 for v in _rgb(h)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _near(a, b):
    return sum((x - y) ** 2 for x, y in zip(_rgb(a), _rgb(b))) ** 0.5 < 110


def _fill(h, a):
    r, g, b = _rgb(h)
    return f"rgba({r},{g},{b},{a})"


def team_css(away, home):
    """<style> that sets the two team colours on a report page, or "" for an unknown team.

    --away/--home: depth badges. --den/--kc: curves and swatches (away/home).
    The selectors repeat :root to win over the :root rules baked into each report file.
    """
    if away not in TEAM_COLORS or home not in TEAM_COLORS:
        return ""
    (ap, aa), (hp, ha) = TEAM_COLORS[away], TEAM_COLORS[home]
    a_line = aa if _near(ap, hp) else ap

    def dark(p, alt):
        return alt if _lum(p) < 0.06 else p

    a_dark, h_dark = dark(a_line, aa), dark(hp, ha)
    if _near(a_dark, h_dark):
        a_dark = aa

    def tokens(a, h, alpha):
        return (f"--den:{a}; --kc:{h}; --den-fill:{_fill(a, alpha)}; --kc-fill:{_fill(h, alpha)};")

    return ("<style>"
            f':root:root:root{{--away:{ap}; --home:{hp}; {tokens(a_line, hp, .16)}}}'
            '@media (prefers-color-scheme: dark){'
            f':root:root:root:not([data-theme="light"]){{{tokens(a_dark, h_dark, .22)}}}}}'
            f':root:root:root[data-theme="dark"]{{{tokens(a_dark, h_dark, .22)}}}'
            "</style>")
