"""Player tray on the report pages. A click on a player card opens it from the right.

It shows the player's games this season, a chart of the card's stat, and one
paragraph about the game on the report. The paragraph is built by fixed rules
from numbers already on the page and in the database, so it states no new facts.
The server adds TRAY_HTML to every report per request (old reports included),
and the tray reads /api/player for the game log. Read-only on the database.
"""
import sqlite3

from .odds import name_key

COLS = ("completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions",
        "carries", "rushing_yards", "rushing_tds", "receptions", "targets",
        "receiving_yards", "receiving_tds", "target_share")


def player_season(db_path, name, team, season):
    """Regular-season games for one player, oldest first: week, opp, away, team, COLS."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "SELECT p.player_id, p.player_display_name, p.week, p.team, p.opponent_team, "
            "s.away_team, " + ", ".join(f"p.{c}" for c in COLS) + " "
            "FROM player_games p LEFT JOIN schedules s ON s.game_id = p.game_id "
            "WHERE p.season = ? AND p.season_type = 'REG' ORDER BY p.week", (season,)).fetchall()
    finally:
        db.close()
    key = name_key(name)
    rows = [r for r in rows if name_key(r[1]) == key]
    if not rows:
        return []
    # Two players can share a name. Keep the one who played for this team.
    ids = {}
    for r in rows:
        ids.setdefault(r[0], 0)
        ids[r[0]] += r[3] == team
    pid = max(ids, key=ids.get)
    return [{"week": r[2], "team": r[3], "opp": r[4], "away": r[5] == r[3],
             **{c: r[6 + i] for i, c in enumerate(COLS)}} for r in rows if r[0] == pid]


TRAY_CSS = """
  .ptray-bg{position:fixed; inset:0; background:rgba(0,0,0,.45); opacity:0; pointer-events:none;
    transition:opacity .2s; z-index:900}
  .ptray{position:fixed; top:0; right:0; bottom:0; width:min(560px,100vw); z-index:901;
    background:var(--bg); color:var(--ink); border-left:2px solid var(--ink);
    box-shadow:-8px 0 32px rgba(0,0,0,.35); transform:translateX(102%); transition:transform .22s ease-out;
    overflow-y:auto; padding:24px 32px 48px; font-family:"Public Sans",ui-sans-serif,system-ui,sans-serif}
  .ptray-open .ptray{transform:none}
  .ptray-open .ptray-bg{opacity:1; pointer-events:auto}
  @media (prefers-reduced-motion: reduce){.ptray,.ptray-bg{transition:none}}
  .ptray-x{position:absolute; top:16px; right:16px; width:32px; height:32px; border:1px solid var(--line);
    background:var(--surface); color:var(--ink-2); border-radius:3px; font-size:18px; line-height:1; cursor:pointer}
  .ptray-x:hover{color:var(--ink); border-color:var(--ink-3)}
  .ptray-eb{font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.14em; text-transform:uppercase;
    color:var(--ink-3); margin:0 0 8px}
  .ptray h2{font-family:"Barlow Condensed",sans-serif; font-weight:700; font-size:36px; line-height:1;
    text-transform:uppercase; margin:0 40px 8px 0; display:flex; align-items:center; gap:8px;
    border:0; padding:0}
  .ptray-mk{display:flex; gap:4px; align-items:center; margin:0 0 24px}
  .ptray h3{font-family:"IBM Plex Mono",monospace; font-size:11px; font-weight:500; letter-spacing:.14em;
    text-transform:uppercase; color:var(--ink-3); margin:24px 0 8px; padding-bottom:4px; border-bottom:1px solid var(--line)}
  .ptray-an{font-size:15px; line-height:1.6; color:var(--ink); margin:0}
  .ptray-chart svg{display:block; width:100%; height:auto}
  .ptray-chart .bar{fill:var(--tcol)}
  .ptray-chart .bar.over{fill:var(--good)} .ptray-chart .bar.under{fill:var(--bad)}
  .ptray-chart .bar.now{stroke:var(--ink); stroke-width:2}
  .ptray-chart text{font-family:"IBM Plex Mono",monospace; font-size:10px; fill:var(--ink-3)}
  .ptray-chart text.v{fill:var(--ink); font-weight:600; paint-order:stroke; stroke:var(--bg); stroke-width:4px; stroke-linejoin:round}
  .ptray-chart .ln{stroke:var(--ink); stroke-width:1.5}
  .ptray-chart .mu{stroke:var(--tcol); stroke-width:1.5; stroke-dasharray:4 3}
  .ptray-key{display:flex; gap:16px; font-size:12px; color:var(--ink-2); margin-top:8px; flex-wrap:wrap}
  .ptray-key i{display:inline-block; width:14px; vertical-align:middle; margin-right:6px; border-top:1.5px solid var(--ink)}
  .ptray-key i.mu{border-top:1.5px dashed var(--tcol)}
  .ptray table{border-collapse:collapse; width:100%; font-size:13px}
  .ptray th,.ptray td{padding:5px 6px; text-align:right; border-bottom:1px solid var(--line-soft); white-space:nowrap;
    font-family:"IBM Plex Mono",monospace}
  .ptray th{font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3); font-weight:500;
    border-bottom:1px solid var(--ink)}
  .ptray th:nth-child(-n+2),.ptray td:nth-child(-n+2){text-align:left}
  .ptray td.hl,.ptray th.hl{color:var(--ink); font-weight:600}
  .ptray tr.now td{background:var(--line-soft)}
  .ptray-none{color:var(--ink-3); font-size:13px}
  .wrap[data-page="report"] .card{cursor:pointer}
  .wrap[data-page="report"] .card:hover{border-color:var(--ink-3)}
"""

# Plain string, not an f-string: the JS braces stay as they are.
TRAY_JS = r"""
(function(){
  if(typeof DATA==="undefined"||typeof STATS==="undefined") return;
  var m=location.pathname.match(/report_(\d{4})_w(\d+)_([A-Z]+)-([A-Z]+)\.html$/);
  if(!m) return;
  var SEASON=+m[1], WEEK=+m[2];
  var COLS={
    QB:[["Cmp","completions"],["Att","attempts"],["Pass yds","passing_yards"],["TD","passing_tds"],["Int","passing_interceptions"],["Rush yds","rushing_yards"]],
    RB:[["Car","carries"],["Rush yds","rushing_yards"],["Tgt","targets"],["Rec","receptions"],["Rec yds","receiving_yards"],["TD","_tds"]],
    WR:[["Tgt","targets"],["Rec","receptions"],["Rec yds","receiving_yards"],["TD","receiving_tds"],["Tgt %","_share"]]
  };
  COLS.TE=COLS.WR; COLS.FB=COLS.RB;
  var esc=function(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/"/g,"&quot;");};
  var f1=function(v){return Math.abs(v)>=20?String(Math.round(v)):String(Math.round(v*10)/10);};
  var ordn=function(n){var t=n%100;return n+(t>=11&&t<=13?"th":({1:"st",2:"nd",3:"rd"})[n%10]||"th");};
  var cell=function(g,k){
    if(k==="_tds") return (g.rushing_tds||0)+(g.receiving_tds||0);
    if(k==="_share") return g.target_share==null?"—":Math.round(g.target_share*100)+"%";
    return g[k]==null?0:g[k];
  };
  var SUFFIX=/^(jr|sr|ii|iii|iv|v)\.?$/i;
  var lastName=function(p){var w=p.split(" ").filter(function(x){return !SUFFIX.test(x);});return w[w.length-1]||p;};

  var bg=document.createElement("div"); bg.className="ptray-bg";
  var tray=document.createElement("aside"); tray.className="ptray";
  tray.setAttribute("role","dialog"); tray.setAttribute("aria-modal","true"); tray.setAttribute("aria-label","Player history");
  tray.dataset.inspectId="player-tray";
  document.body.appendChild(bg); document.body.appendChild(tray);
  var last=null;
  function close(){document.documentElement.classList.remove("ptray-open"); if(last) last.focus&&last.focus();}
  bg.addEventListener("click",close);
  document.addEventListener("keydown",function(e){if(e.key==="Escape") close();});

  function analysis(d,pre,unit,label){
    var nm=lastName(d.p), out=[], v=pre.map(function(g){return cell(g,d.s);}), n=v.length;
    var avg=function(a){return a.reduce(function(s,x){return s+x;},0)/a.length;};
    if(!n) out.push(d.p+" has no "+SEASON+" games before week "+WEEK+".");
    else if(n>=4) out.push(d.p+" averages "+f1(avg(v.slice(-3)))+" "+unit+" over the last 3 games, against "+f1(avg(v))+" over "+n+" games this season.");
    else out.push(d.p+" averages "+f1(avg(v))+" "+unit+" over "+n+" game"+(n>1?"s":"")+" this season.");
    if(n&&d.line!=null){
      var k=v.filter(function(x){return x>d.line;}).length;
      out.push(nm+" went over "+d.line+" in "+k+" of "+n+" game"+(n>1?"s":"")+".");
    }
    (d.bd||[]).forEach(function(b){
      if(/\bbd-up\b/.test(b[0])) out.push("The trend marker is up. Production per game over the last 3 games is well above the games before, with opponents removed.");
      if(/\bbd-down\b/.test(b[0])) out.push("The trend marker is down. Production per game over the last 3 games is well below the games before, with opponents removed.");
    });
    var pos=d.role.replace(/[0-9]+$/,"");
    if(d.dr){
      var word=d.dr.t==="tough"?"tougher than most":d.dr.t==="soft"?"softer than most":"about average";
      out.push(d.vs+" ranks "+ordn(d.dr.rk)+" of 32 against "+pos+"s, "+word+", at "+d.dr.y.toFixed(1)+" yards a game allowed (adjusted).");
    } else if(d.def!=null){
      out.push(d.vs+" is "+(d.def>1.03?"softer than average":d.def<0.97?"tougher than average":"about average")+" against this stat.");
    }
    (d.bd||[]).forEach(function(b){
      if(/\bbd-big\b/.test(b[0])) out.push("This is a Big day matchup: a top-15% "+pos+" against one of the 5 defenses that allow the most to the position.");
      if(/\bbd-caution\b/.test(b[0])) out.push("This is a Caution matchup: a top-15% "+pos+" against one of the 5 toughest defenses to the position.");
    });
    out.push("The model puts the middle of the range at "+f1(d.mu)+" "+unit+", with most games between "+f1(d.lo)+" and "+f1(d.hi)+".");
    if(d.line!=null&&d.mp!=null&&d.bp!=null){
      var mp=Math.round(d.mp*100), bp=Math.round(d.bp*100);
      out.push("It clears "+d.line+" in "+mp+"% of outcomes, against "+bp+"% implied by the "+(d.price>0?"+":"")+d.price+" price.");
      out.push(mp-bp>=5?"The model is more confident than the book.":bp-mp>=5?"The book is more confident than the model.":"The model and the book agree within 5 points.");
    } else out.push("FanDuel offered no number near −300 for "+label.toLowerCase()+".");
    var a=(window.__ACTUALS__||{})[d.p+"|"+d.s];
    if(a!=null) out.push("Final: "+f1(a)+" "+unit+(d.line!=null?(a>d.line?", over the "+d.line+" line.":", under the "+d.line+" line."):"."));
    return out.join(" ");
  }

  function chart(d,games){
    if(!games.length) return '<p class="ptray-none">No games to chart.</p>';
    var W=496,H=170,T=18,B=22,vals=games.map(function(g){return cell(g,d.s);});
    var top=Math.max.apply(null,vals.concat([d.line||0,d.mu||0]))*1.12||1;
    var bw=Math.min(36,(W-8)/games.length-6), step=(W-8)/games.length;
    var Y=function(v){return H-B-(v/top)*(H-B-T);};
    // Rules first, then bars, then labels: a label always sits on top, with a halo.
    var s='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+esc(d.p)+' by week">', lb='';
    if(d.line!=null) s+='<line class="ln" x1="0" x2="'+W+'" y1="'+Y(d.line).toFixed(1)+'" y2="'+Y(d.line).toFixed(1)+'"></line>';
    s+='<line class="mu" x1="0" x2="'+W+'" y1="'+Y(d.mu).toFixed(1)+'" y2="'+Y(d.mu).toFixed(1)+'"></line>';
    games.forEach(function(g,i){
      var v=vals[i], x=4+i*step+(step-bw)/2, cls="bar";
      if(d.line!=null) cls+=v>d.line?" over":" under";
      if(g.week===WEEK) cls+=" now";
      s+='<rect class="'+cls+'" x="'+x.toFixed(1)+'" y="'+Y(v).toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+Math.max(1,(H-B)-Y(v)).toFixed(1)+'">'
        +'<title>Week '+g.week+(g.away?" at ":" vs ")+g.opp+": "+f1(v)+'</title></rect>';
      lb+='<text class="v" x="'+(x+bw/2).toFixed(1)+'" y="'+(Y(v)-4).toFixed(1)+'" text-anchor="middle">'+f1(v)+'</text>'
        +'<text x="'+(x+bw/2).toFixed(1)+'" y="'+(H-6)+'" text-anchor="middle">W'+g.week+'</text>';
    });
    s+=lb+'</svg>';
    return s+'<div class="ptray-key">'+(d.line!=null?'<span><i></i>FanDuel line '+d.line+'</span>':'')
      +'<span><i class="mu"></i>model expected '+f1(d.mu)+'</span>'
      +(d.line!=null?'<span>bars: green over, red under</span>':'')+'</div>';
  }

  function table(d,games){
    if(!games.length) return '<p class="ptray-none">No '+SEASON+' games yet.</p>';
    var cols=COLS[d.role.replace(/[0-9]+$/,"")]||COLS.WR;
    var h='<table><thead><tr><th>Wk</th><th>Opp</th>'+cols.map(function(c){return '<th'+(c[1]===d.s?' class="hl"':'')+'>'+c[0]+'</th>';}).join("")+'</tr></thead><tbody>';
    games.forEach(function(g){
      h+='<tr'+(g.week===WEEK?' class="now"':'')+'><td>'+g.week+'</td><td>'+(g.away?"@ ":"vs ")+g.opp+'</td>'
        +cols.map(function(c){return '<td'+(c[1]===d.s?' class="hl"':'')+'>'+cell(g,c[1])+'</td>';}).join("")+'</tr>';
    });
    return h+'</tbody></table>';
  }

  function open(d,from){
    last=from;
    var st=STATS.filter(function(x){return x[0]===d.s;})[0]||[d.s,d.s,""];
    var a=d.tm===META.teamA;
    tray.style.setProperty("--tcol",a?"var(--den)":"var(--kc)");
    tray.innerHTML='<button class="ptray-x" type="button" aria-label="Close" data-inspect-id="player-tray-close">×</button>'
      +'<p class="ptray-eb">'+SEASON+' season · '+esc(st[1])+' · week '+WEEK+' vs '+esc(d.vs)+'</p>'
      +'<h2>'+esc(d.p)+'</h2>'
      +'<div class="ptray-mk"><span class="role '+(a?"a":"b")+'">'+esc(d.role)+'</span>'
      +(d.rk?'<span class="rk" title="'+esc(d.rkt||"")+'">'+esc(d.rk)+'</span>':'')
      +'<span class="dp" style="font-family:IBM Plex Mono,monospace;font-size:12px;color:var(--ink-3)">'+esc(d.tm)+'</span></div>'
      +'<h3>This game</h3><p class="ptray-an" data-inspect-id="player-tray-analysis">Loading games…</p>'
      +'<h3>'+esc(st[1])+' by week</h3><div class="ptray-chart" data-inspect-id="player-tray-chart"></div>'
      +'<h3>Game log</h3><div data-inspect-id="player-tray-log"></div>';
    tray.querySelector(".ptray-x").addEventListener("click",close);
    document.documentElement.classList.add("ptray-open");
    tray.querySelector(".ptray-x").focus();
    fetch("/api/player?name="+encodeURIComponent(d.p)+"&team="+encodeURIComponent(d.tm)+"&season="+SEASON)
      .then(function(r){return r.ok?r.json():[];}).catch(function(){return [];})
      .then(function(all){
        var games=all.filter(function(g){return g.week<=WEEK;}), pre=games.filter(function(g){return g.week<WEEK;});
        tray.querySelector(".ptray-an").textContent=analysis(d,pre,st[2],st[1]);
        tray.querySelector(".ptray-chart").innerHTML=chart(d,games);
        tray.querySelector("[data-inspect-id=player-tray-log]").innerHTML=table(d,games);
      });
  }

  document.addEventListener("click",function(e){
    var card=e.target.closest&&e.target.closest(".card");
    if(!card||e.target.closest("a,button,summary")) return;
    var grid=card.closest(".grid"), h=grid&&grid.previousElementSibling;
    var nameEl=card.querySelector(".name");
    if(!h||!nameEl) return;
    var label=(h.firstChild&&h.firstChild.textContent||"").trim();
    var st=STATS.filter(function(x){return x[1]===label;})[0];
    var p=nameEl.textContent.replace(/[▲▼]/g,"").trim();
    var d=st&&DATA.filter(function(x){return x.p===p&&x.s===st[0];})[0];
    if(d) open(d,card);
  });
  // Link form: #p=Player Name|stat_key opens the tray on load.
  var hp=decodeURIComponent(location.hash).match(/^#p=(.+)\|(\w+)$/);
  var hd=hp&&DATA.filter(function(x){return x.p===hp[1]&&x.s===hp[2];})[0];
  if(hd) open(hd,null);
})();
"""

TRAY_HTML = f"<style>{TRAY_CSS}</style>\n<script>{TRAY_JS}</script>"
