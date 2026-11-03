#!/bin/sh
# One-off: download the 32 NFL logos from the ESPN CDN into static/logos/{ABBR}.png, 96px.
# Site abbreviations differ from ESPN for LA (lar) and WAS (wsh). Run: sh scripts/fetch_logos.sh
cd "$(dirname "$0")/.." && mkdir -p static/logos
for pair in ARI:ari ATL:atl BAL:bal BUF:buf CAR:car CHI:chi CIN:cin CLE:cle DAL:dal DEN:den DET:det GB:gb HOU:hou IND:ind JAX:jax KC:kc LA:lar LAC:lac LV:lv MIA:mia MIN:min NE:ne NO:no NYG:nyg NYJ:nyj PHI:phi PIT:pit SEA:sea SF:sf TB:tb TEN:ten WAS:wsh; do
  a=${pair%%:*}; e=${pair##*:}
  curl -sf -o "static/logos/$a.png" "https://a.espncdn.com/i/teamlogos/nfl/500/$e.png" && sips -Z 96 "static/logos/$a.png" >/dev/null || echo "missing $a"
done
