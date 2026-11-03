#!/bin/sh
# One-off: download the 32 NFL logos from the ESPN CDN into static/logos/{ABBR}.png, 96px.
# Site abbreviations differ from ESPN for LA (lar) and WAS (wsh). Run: sh scripts/fetch_logos.sh
cd "$(dirname "$0")/.." && mkdir -p static/logos
for pair in ARI:ari ATL:atl BAL:bal BUF:buf CAR:car CHI:chi CIN:cin CLE:cle DAL:dal DEN:den DET:det GB:gb HOU:hou IND:ind JAX:jax KC:kc LA:lar LAC:lac LV:lv MIA:mia MIN:min NE:ne NO:no NYG:nyg NYJ:nyj PHI:phi PIT:pit SEA:sea SF:sf TB:tb TEN:ten WAS:wsh; do
  a=${pair%%:*}; e=${pair##*:}
  curl -sf -o "static/logos/$a.png" "https://a.espncdn.com/i/teamlogos/nfl/500/$e.png" && sips -Z 96 "static/logos/$a.png" >/dev/null || echo "missing $a"
done
# Optional: trim the transparent border so a logo fills its box (needs Pillow).
python3 - <<'PY' 2>/dev/null || echo "Pillow not installed: logos keep their border"
from PIL import Image
import glob
for f in glob.glob('static/logos/*.png'):
    im=Image.open(f).convert('RGBA'); bb=im.getchannel('A').point(lambda a:255 if a>8 else 0).getbbox()
    if bb: im=im.crop(bb)
    w,h=im.size; n=max(w,h); sq=Image.new('RGBA',(n,n),(0,0,0,0)); sq.paste(im,((n-w)//2,(n-h)//2)); sq.resize((96,96),Image.LANCZOS).save(f)
PY
