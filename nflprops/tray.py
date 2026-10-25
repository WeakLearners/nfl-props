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


_POOLS = {}


def _pool(season, week):
    """star_pool for one week, kept in memory: the report's own depth-role source."""
    if (season, week) not in _POOLS:
        from .reliability import star_pool
        _POOLS[(season, week)] = star_pool(season, week)
    return _POOLS[(season, week)]


_RANKS = {}


def _ranks(season, week):
    """{(name_key, position): "#11 QB"} as of the games BEFORE `week`.

    Same ranking and same chip text as the shortlist tag (rating.chip_lookup),
    but chip_lookup only does the latest completed week. This calls the same
    rating.current_ratings(season, week) it is built on, with week - 1, so a
    game in week 5 uses games through week 4. Players below the ranking floor
    are absent, and so is every player in week 1 (nothing played yet).
    """
    if (season, week) not in _RANKS:
        from .rating import current_ratings
        df = current_ratings(season=season, week=week - 1)[2]
        _RANKS[(season, week)] = {(name_key(r["name"]), r.position): f"#{r['rank']} {r.position}"
                                  for _, r in df.iterrows()}
    return _RANKS[(season, week)]


def opp_role_log(db_path, opp, role, stat, season, week):
    """What one team's defense faced at one depth role, game by game, before `week`.

    Role holder = the same rule as the report's role tag: reliability.star_pool
    for that game's week (usage rank on the roster). If that player has no stat
    line in the game, the nth-highest carries/targets/attempts at the
    position that day stands in, and the row says so (sub=True).
    Returns [{week, team, away, player, value, sub, rank}], oldest first. [] when
    opp, role or stat is unknown. Read-only; no odds call.
    """
    import re
    m = re.fullmatch(r"(QB|RB|WR|TE)([0-9]+)", role or "")
    if not m or stat not in COLS or not opp:
        return []
    pos, n = m.group(1), int(m.group(2))
    use = {"QB": "attempts", "RB": "carries", "WR": "targets", "TE": "targets"}[pos]
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        games = db.execute(
            "SELECT DISTINCT p.week, p.team, s.away_team FROM player_games p "
            "LEFT JOIN schedules s ON s.game_id = p.game_id "
            "WHERE p.season=? AND p.season_type='REG' AND p.opponent_team=? AND p.week<? "
            "ORDER BY p.week", (season, opp, week)).fetchall()
        out = []
        for wk, vs, away in games:
            rows = db.execute(
                "SELECT player_display_name, " + stat + ", " + use + " FROM player_games "
                "WHERE season=? AND season_type='REG' AND week=? AND team=? AND position=?",
                (season, wk, vs, pos)).fetchall()
            pool = _pool(season, wk)
            pool = pool[(pool.team == vs) & (pool.position == pos) & (pool["rank"] == n)]
            holder = pool.iloc[0].player_key if len(pool) else None
            hit = [r for r in rows if holder and name_key(r[0]) == holder]
            sub = False
            if not hit:
                rows = sorted((r for r in rows if r[2] is not None), key=lambda r: -r[2])
                if len(rows) < n:
                    continue
                hit, sub = [rows[n - 1]], True
            out.append({"week": wk, "team": vs, "away": away == opp, "player": hit[0][0],
                        "value": hit[0][1] or 0, "sub": sub,
                        "rank": _ranks(season, wk).get((name_key(hit[0][0]), pos))})
        return out
    finally:
        db.close()


TRAY_CSS = """
  /* Shortlist names open the tray; only here, because the tray exists only on served pages. */
  #picklist .pk-nm[data-p]{cursor:pointer}
  #picklist .pk-nm[data-p]:hover{text-decoration:underline; text-decoration-color:currentColor; text-underline-offset:3px}
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
  .ptray-chart .bar{fill:var(--ink-3)}
  .ptray-chart .bar.over{fill:var(--good)} .ptray-chart .bar.under{fill:var(--bad)}
  .ptray-chart .bar.now{stroke:var(--ink); stroke-width:2}
  .ptray-chart text{font-family:"IBM Plex Mono",monospace; font-size:10px; fill:var(--ink-3)}
  .ptray-chart text.v{fill:var(--ink); font-weight:600; paint-order:stroke; stroke:var(--bg); stroke-width:4px; stroke-linejoin:round}
  .ptray-chart .ln{stroke:var(--ink); stroke-width:2}
  .ptray .ptray-dim{color:var(--ink-3); font-weight:400}
  .ptray b.rs{font-weight:600} .ptray b.rs.hit{color:var(--good)} .ptray b.rs.miss{color:var(--bad)}
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
  // This game's own line and result for the leg's stat: [line, hit] or null. hit is null until final.
  var gl=function(g,stat){var l=g.lines&&g.lines[stat];return l?{line:l[0],hit:l[1]}:null;};
  var SUFFIX=/^(jr|sr|ii|iii|iv|v)\.?$/i;
  var lastName=function(p){var w=p.split(" ").filter(function(x){return !SUFFIX.test(x);});return w[w.length-1]||p;};

  var bg=document.createElement("div"); bg.className="ptray-bg";
  var tray=document.createElement("aside"); tray.className="ptray";
  tray.setAttribute("role","dialog"); tray.setAttribute("aria-modal","true"); tray.setAttribute("aria-label","Player history");
  tray.dataset.inspectId="player-tray";
  document.body.appendChild(bg); document.body.appendChild(tray);
  var last=null, seq=0;
  function close(){document.documentElement.classList.remove("ptray-open"); if(last) last.focus&&last.focus();}
  bg.addEventListener("click",close);
  document.addEventListener("keydown",function(e){if(e.key==="Escape") close();});

  function analysis(d,pre,unit,label){
    var nm=lastName(d.p), out=[], v=pre.map(function(g){return cell(g,d.s);}), n=v.length;
    var avg=function(a){return a.reduce(function(s,x){return s+x;},0)/a.length;};
    if(!n) out.push(d.p+" has no "+SEASON+" games before week "+WEEK+".");
    else if(n>=4) out.push(d.p+" averages "+f1(avg(v.slice(-3)))+" "+unit+" over the last 3 games, against "+f1(avg(v))+" over "+n+" games this season.");
    else out.push(d.p+" averages "+f1(avg(v))+" "+unit+" over "+n+" game"+(n>1?"s":"")+" this season.");
    // Each past week is judged against its own line. A week with no line of its own
    // is judged against this week's line, as before.
    var wl=pre.filter(function(g){var x=gl(g,d.s);return x&&x.hit!=null;});
    if(wl.length){
      var kh=wl.filter(function(g){return gl(g,d.s).hit;}).length;
      out.push(nm+" cleared his line in "+kh+" of "+wl.length+" game"+(wl.length>1?"s":"")+" with a line.");
    }
    var nl=pre.filter(function(g){var x=gl(g,d.s);return !x||x.hit==null;});
    if(nl.length&&d.line!=null){
      var k=nl.filter(function(g){return cell(g,d.s)>d.line;}).length;
      out.push(nm+" went over "+d.line+" in "+k+" of "+nl.length+" game"+(nl.length>1?"s":"")+" with no line of his own.");
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
    var eng=(d.pe==="new"?"new model":"old engine");
    out.push("The "+eng+" puts the middle of the range at "+f1(d.mu)+" "+unit+", with most games between "+f1(d.lo)+" and "+f1(d.hi)+".");
    if(d.line!=null&&d.mp!=null&&d.bp!=null){
      var mp=Math.round(d.mp*100), bp=Math.round(d.bp*100);
      out.push("It clears "+d.line+" in "+mp+"% of outcomes, against "+bp+"% implied by the "+(d.price>0?"+":"")+d.price+" price.");
      out.push(mp-bp>=5?"The "+eng+" is more confident than the book.":bp-mp>=5?"The book is more confident than the "+eng+".":"The "+eng+" and the book agree within 5 points.");
    } else out.push("FanDuel offered no number near −300 for "+label.toLowerCase()+".");
    var a=(window.__ACTUALS__||{})[d.p+"|"+d.s];
    if(a!=null) out.push("Final: "+f1(a)+" "+unit+(d.line!=null?(a>d.line?", over the "+d.line+" line.":", under the "+d.line+" line."):"."));
    return out.join(" ");
  }

  function chart(d,games){
    if(!games.length) return '<p class="ptray-none">No games to chart.</p>';
    var W=496,H=170,T=18,B=22,vals=games.map(function(g){return cell(g,d.s);});
    var lns=games.map(function(g){var x=gl(g,d.s);return x?x.line:0;});
    var top=Math.max.apply(null,vals.concat(lns,[d.mu||0]))*1.12||1;
    var bw=Math.min(36,(W-8)/games.length-6), step=(W-8)/games.length;
    var Y=function(v){return H-B-(v/top)*(H-B-T);};
    // Rules first, then bars, then labels: a label always sits on top, with a halo.
    var s='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+esc(d.p)+' by week">', lb='';
    s+='<line class="mu" x1="0" x2="'+W+'" y1="'+Y(d.mu).toFixed(1)+'" y2="'+Y(d.mu).toFixed(1)+'"></line>';
    games.forEach(function(g,i){
      var v=vals[i], x=4+i*step+(step-bw)/2, cls="bar", w=gl(g,d.s);
      if(w&&w.hit!=null) cls+=w.hit?" over":" under";
      if(g.week===WEEK) cls+=" now";
      s+='<rect class="'+cls+'" x="'+x.toFixed(1)+'" y="'+Y(v).toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+Math.max(1,(H-B)-Y(v)).toFixed(1)+'">'
        +'<title>Week '+g.week+(g.away?" at ":" vs ")+g.opp+": "+f1(v)+(w?", line "+w.line+(w.hit==null?"":w.hit?" HIT":" MISS"):", no line")+'</title></rect>'
        +(w?'<line class="ln" data-inspect-id="player-tray-tick" x1="'+(x-3).toFixed(1)+'" x2="'+(x+bw+3).toFixed(1)+'" y1="'+Y(w.line).toFixed(1)+'" y2="'+Y(w.line).toFixed(1)+'"></line>':'');
      lb+='<text class="v" x="'+(x+bw/2).toFixed(1)+'" y="'+(Y(v)-4).toFixed(1)+'" text-anchor="middle">'+f1(v)+'</text>'
        +'<text x="'+(x+bw/2).toFixed(1)+'" y="'+(H-6)+'" text-anchor="middle">W'+g.week+'</text>';
    });
    s+=lb+'</svg>';
    return s+'<div class="ptray-key">'+(games.some(function(g){return gl(g,d.s);})?'<span><i></i>tick: that week\'s FanDuel line</span>':'')
      +'<span><i class="mu"></i>'+(d.pe==="new"?"new model":"old engine")+' expected '+f1(d.mu)+'</span>'
      +(games.some(function(g){return gl(g,d.s);})?'<span>bars: green hit, red miss, grey no line</span>':'')+'</div>';
  }

  function table(d,games){
    if(!games.length) return '<p class="ptray-none">No '+SEASON+' games yet.</p>';
    var cols=COLS[d.role.replace(/[0-9]+$/,"")]||COLS.WR;
    // Line and Result go right after the leg's stat column (at the end if the stat is not a column).
    var at=cols.map(function(c){return c[1];}).indexOf(d.s);
    var extraH='<th>Line</th><th>Result</th>';
    var extra=function(g){
      var x=gl(g,d.s), dash='<span class="ptray-dim">—</span>';
      return '<td>'+(x?x.line:dash)+'</td><td>'+(!x||x.hit==null?dash:x.hit?'<b class="rs hit">HIT</b>':'<b class="rs miss">MISS</b>')+'</td>';
    };
    var h='<table><thead><tr><th>Wk</th><th>Opp</th>'+cols.map(function(c,i){return '<th'+(c[1]===d.s?' class="hl"':'')+'>'+c[0]+'</th>'+(i===at?extraH:'');}).join("")+(at<0?extraH:'')+'</tr></thead><tbody>';
    games.forEach(function(g){
      h+='<tr'+(g.week===WEEK?' class="now"':'')+'><td>'+g.week+'</td><td>'+(g.away?"@ ":"vs ")+g.opp+'</td>'
        +cols.map(function(c,i){return '<td'+(c[1]===d.s?' class="hl"':'')+'>'+cell(g,c[1])+'</td>'+(i===at?extra(g):'');}).join("")+(at<0?extra(g):'')+'</tr>';
    });
    return h+'</tbody></table>';
  }


  // Opponent against the same depth role: bars coloured against THIS report's line, mark at that line.
  function oppChart(d,rows){
    var W=496,H=170,T=18,B=22,vals=rows.map(function(r){return r.value;});
    var top=Math.max.apply(null,vals.concat([d.line||0]))*1.12||1;
    var bw=Math.min(36,(W-8)/rows.length-6), step=(W-8)/rows.length;
    var Y=function(v){return H-B-(v/top)*(H-B-T);};
    var s='<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+esc(d.vs)+' against '+esc(d.role)+' by week">', lb='';
    rows.forEach(function(r,i){
      var x=4+i*step+(step-bw)/2, cls="bar"+(d.line==null?"":r.value>d.line?" over":" under");
      s+='<rect class="'+cls+'" x="'+x.toFixed(1)+'" y="'+Y(r.value).toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+Math.max(1,(H-B)-Y(r.value)).toFixed(1)+'">'
        +'<title>Week '+r.week+(r.away?" at ":" vs ")+r.team+": "+esc(r.player)+" "+f1(r.value)+'</title></rect>'
        +(d.line==null?'':'<line class="ln" x1="'+(x-3).toFixed(1)+'" x2="'+(x+bw+3).toFixed(1)+'" y1="'+Y(d.line).toFixed(1)+'" y2="'+Y(d.line).toFixed(1)+'"></line>');
      lb+='<text class="v" x="'+(x+bw/2).toFixed(1)+'" y="'+(Y(r.value)-4).toFixed(1)+'" text-anchor="middle">'+f1(r.value)+'</text>'
        +'<text x="'+(x+bw/2).toFixed(1)+'" y="'+(H-6)+'" text-anchor="middle">W'+r.week+'</text>';
    });
    return s+lb+'</svg>';
  }
  function oppTable(d,rows){
    var h='<table><thead><tr><th>Wk</th><th>Opp</th><th>Player</th><th class="hl">'+esc((STATS.filter(function(x){return x[0]===d.s;})[0]||[0,d.s])[1])+'</th></tr></thead><tbody>';
    rows.forEach(function(r){
      h+='<tr><td>'+r.week+'</td><td>'+(r.away?"@ ":"vs ")+esc(r.team)+'</td><td>'+esc(r.player)+(r.rank?' <span class="rk" style="margin-left:6px" title="Position rank by recent production before this game (not the depth chart).">'+esc(r.rank)+'</span>':'')+(r.sub?' <span class="ptray-dim" title="The usual role holder had no stat line this game. The player with the same usage rank at the position that day stands in.">(sub)</span>':'')+'</td><td class="hl">'+f1(r.value)+'</td></tr>';
    });
    return h+'</tbody></table>';
  }
  function oppSection(d,rows){
    var sum="";
    if(d.line!=null){
      var k=rows.filter(function(r){return r.value>d.line;}).length;
      sum='<p class="ptray-an" data-inspect-id="player-tray-opp-summary">'+esc(d.role)+'s cleared '+d.line+' in '+k+' of '+rows.length+' game'+(rows.length>1?'s':'')+' against '+esc(d.vs)+'.</p>';
    }
    return '<h3>'+esc(d.vs)+' vs '+esc(d.role)+'</h3>'+sum
      +'<div class="ptray-chart">'+oppChart(d,rows)+'</div>'+oppTable(d,rows);
  }

  function open(d,from){
    last=from; var tok=++seq;
    var st=STATS.filter(function(x){return x[0]===d.s;})[0]||[d.s,d.s,""];
    var a=d.tm===META.teamA;
    tray.style.setProperty("--tcol",a?"var(--den)":"var(--kc)");
    tray.innerHTML='<button class="ptray-x" type="button" aria-label="Close" data-inspect-id="player-tray-close">×</button>'
      +'<p class="ptray-eb">'+SEASON+' season · '+esc(st[1])+' · week '+WEEK+' vs '+esc(d.vs)+'</p>'
      +'<h2>'+esc(d.p)+(window.markStar?window.markStar(window.markKey(SEASON,WEEK,d.p,d.s),"mark-star-tray"):"")+'</h2>'
      +'<div class="ptray-mk"><span class="role '+(a?"a":"b")+'">'+esc(d.role)+'</span>'
      +(d.rk?'<span class="rk" title="'+esc(d.rkt||"")+'">'+esc(d.rk)+'</span>':'')
      +'<span class="dp" style="font-family:IBM Plex Mono,monospace;font-size:12px;color:var(--ink-3)">'+esc(d.tm)+'</span></div>'
      +'<h3>This game</h3><p class="ptray-an" data-inspect-id="player-tray-analysis">Loading games…</p>'
      +'<h3>'+esc(st[1])+' by week</h3><div class="ptray-chart" data-inspect-id="player-tray-chart"></div>'
      +'<h3>Game log</h3><div data-inspect-id="player-tray-log"></div>'
      +'<div data-inspect-id="player-tray-opp" hidden></div>';
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
    // Opponent against the same role. Hidden unless it has rows: no empty box.
    if(d.vs&&d.role) fetch("/api/opp-role?opp="+encodeURIComponent(d.vs)+"&role="+encodeURIComponent(d.role)
        +"&stat="+encodeURIComponent(d.s)+"&season="+SEASON+"&week="+WEEK)
      .then(function(r){return r.ok?r.json():[];}).catch(function(){return [];})
      .then(function(rows){
        var box=tray.querySelector("[data-inspect-id=player-tray-opp]");
        if(!box||tok!==seq||!rows.length) return;
        box.innerHTML=oppSection(d,rows); box.hidden=false;
      });
  }

  // Shortlist names: the tray shows the stat of that shortlist leg.
  var PK="#picklist .pk-nm[data-p]";
  function pickOpen(el){
    var p=el.getAttribute("data-p"), st=el.getAttribute("data-s");
    var d=DATA.filter(function(x){return x.p===p&&x.s===st;})[0];
    if(d) open(d,el);
  }
  Array.prototype.forEach.call(document.querySelectorAll(PK),function(el){
    el.setAttribute("role","button"); el.setAttribute("tabindex","0");
  });
  document.addEventListener("keydown",function(e){
    var el=e.target.matches&&e.target.matches(PK)?e.target:null;
    if(el&&(e.key==="Enter"||e.key===" ")){e.preventDefault(); pickOpen(el);}
  });
  document.addEventListener("click",function(e){
    var nm=e.target.closest&&e.target.closest(PK);
    if(nm){pickOpen(nm); return;}
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
