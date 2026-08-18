#!/usr/bin/env python3
"""VCT player-kill model with separate Map 1 and Maps 1-2 markets.

Commands:
  train    Build walk-forward evaluations and final production models.
  predict  Score future PrizePicks lines with probabilities and uncertainty.
  backtest Join timestamped historical lines to out-of-fold predictions.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math
from collections import Counter, defaultdict, deque
from pathlib import Path
import numpy as np

KEY_FIELDS=("Tournament","Stage","Match Type","Match Name")
TEAM_ALIASES={"Mega Minors":"NRG","Kru Esports":"KRÜ Esports","KRU Esports":"KRÜ Esports","KRU":"KRÜ Esports"}
AGENT_ROLES={
 "jett":"duelist","raze":"duelist","neon":"duelist","reyna":"duelist","phoenix":"duelist","yoru":"duelist","iso":"duelist","waylay":"duelist",
 "omen":"smokes","brimstone":"smokes","viper":"smokes","astra":"smokes","harbor":"smokes","clove":"smokes",
 "sova":"initiator","breach":"initiator","skye":"initiator","kay/o":"initiator","kayo":"initiator","fade":"initiator","gekko":"initiator","tejo":"initiator",
 "cypher":"sentinel","killjoy":"sentinel","sage":"sentinel","chamber":"sentinel","deadlock":"sentinel","vyse":"sentinel",
}
ROLES=["duelist","smokes","initiator","sentinel","flex"]

def clean(x): return (x or "").strip()
def team_name(x): return TEAM_ALIASES.get(clean(x),clean(x))
def match_key(r): return tuple(clean(r.get(x)) for x in KEY_FIELDS)
def number(x,default=0.0):
 try:return float(clean(x).replace("%",""))
 except (ValueError,TypeError):return default
def csv_rows(path):
 with Path(path).open(newline="",encoding="utf-8-sig") as f:yield from csv.DictReader(f)
def write_csv(path,rows,fields=None):
 rows=list(rows);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fields=fields or (list(rows[0]) if rows else [])
 with path.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
def fallback(prefix,value):return prefix+"_"+hashlib.sha1(clean(value).casefold().encode()).hexdigest()[:12]
def agent_role(agent):return AGENT_ROLES.get(clean(agent).casefold(),"flex")
def normal_cdf(x):return .5*(1+math.erf(x/math.sqrt(2)))

class Ridge:
 def fit(self,X,y,alpha=25.0,weights=None):
  X=np.asarray(X,float);y=np.asarray(y,float);w=np.ones(len(X)) if weights is None else np.asarray(weights,float)
  self.mean=np.average(X,axis=0,weights=w);self.scale=np.sqrt(np.average((X-self.mean)**2,axis=0,weights=w));self.scale[self.scale==0]=1
  Z=np.column_stack([np.ones(len(X)),(X-self.mean)/self.scale]);sw=np.sqrt(w);ZW=Z*sw[:,None];yw=y*sw
  penalty=np.eye(Z.shape[1])*alpha;penalty[0,0]=0;self.coef=np.linalg.solve(ZW.T@ZW+penalty,ZW.T@yw);return self
 def predict(self,X):
  X=np.asarray(X,float);Z=np.column_stack([np.ones(len(X)),(X-self.mean)/self.scale]);return Z@self.coef
 def one(self,x):return float(self.predict([x])[0])
 def dump(self,names):return {"features":names,"mean":self.mean.tolist(),"scale":self.scale.tolist(),"coef":self.coef.tolist()}
 @staticmethod
 def load(d):m=Ridge();m.mean=np.asarray(d["mean"]);m.scale=np.asarray(d["scale"]);m.coef=np.asarray(d["coef"]);return m

def nb_alpha(actual,pred):
 actual=np.asarray(actual,float);pred=np.maximum(np.asarray(pred,float),.1);m=float(pred.mean());mse=float(np.mean((actual-pred)**2))
 return max(.01,min(2.0,(mse-m)/(m*m)))
def probability_more(mean,line,alpha):
 """Negative-binomial tail P(X > line), with normal fallback for large sums."""
 mean=max(.05,float(mean));k=math.floor(float(line));variance=mean+alpha*mean*mean
 if mean>60 or k>120:return 1-normal_cdf((k+.5-mean)/math.sqrt(variance))
 r=1/alpha;p=r/(r+mean);cdf=0.0
 for x in range(max(0,k)+1):
  logp=math.lgamma(x+r)-math.lgamma(r)-math.lgamma(x+1)+r*math.log(p)+x*math.log(1-p)
  cdf+=math.exp(logp)
 return max(0.0,min(1.0,1-cdf))
def metrics(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);e=y-p
 return {"n":len(y),"mae":float(np.abs(e).mean()),"rmse":float(np.sqrt(np.mean(e*e))),"bias":float(e.mean())}

def weighted_profile(history,prospective_maps):
 hist=list(history);ages=range(len(hist)-1,-1,-1);w=[.5**(age/15) for age in ages]
 def kpr(sub):
  chosen=[(wi,h) for wi,h in zip(w,hist) if sub(h)];rd=sum(wi*h[1] for wi,h in chosen)
  return sum(wi*h[0] for wi,h in chosen)/rd if rd else .68
 def acs(sub=lambda h:True):
  chosen=[(wi,h) for wi,h in zip(w,hist) if sub(h)];den=sum(wi for wi,_ in chosen)
  return sum(wi*h[2] for wi,h in chosen)/den if den else 200.0
 def count(sub):return sum(1 for h in hist if sub(h))
 role_weight=defaultdict(float)
 for wi,h in zip(w,hist):role_weight[h[3]]+=wi
 role_total=sum(role_weight.values()) or 1
 map_rates=[kpr(lambda h,m=m:h[5]==m) for m in prospective_maps]
 return {
  "kpr5":kpr(lambda h:h in hist[-5:]),"kpr10":kpr(lambda h:h in hist[-10:]),"kpr25":kpr(lambda h:h in hist[-25:]),"kpr_decay":kpr(lambda h:True),
  "acs_decay":acs(),"team_kpr":kpr(lambda h:h[4]==hist[-1][4]) if hist else .68,"team_maps":count(lambda h:h[4]==hist[-1][4]) if hist else 0,
  "map_kpr":sum(map_rates)/len(map_rates) if map_rates else .68,"kill_share":sum(wi*h[6] for wi,h in zip(w,hist))/sum(w) if w else .20,
 "experience":len(hist),"role_props":[role_weight[r]/role_total for r in ROLES],
 }

def defensive_profile(history,prospective_maps):
 hist=list(history);weights=[.5**(age/20) for age in range(len(hist)-1,-1,-1)]
 def rate(index,subset=lambda h:True,default=0.0):
  chosen=[(w,h) for w,h in zip(weights,hist) if subset(h)];rounds=sum(w*h[0] for w,h in chosen)
  return sum(w*h[index] for w,h in chosen)/rounds if rounds else default
 map_rates=[rate(2,lambda h,m=m:h[4]==m,3.4) for m in prospective_maps]
 return {"opponent_deaths_per_round":rate(2,default=3.4),"opponent_kills_per_round":rate(1,default=3.4),"opponent_first_death_rate":rate(3,default=.5),"opponent_pace":rate(1,default=3.4)+rate(2,default=3.4),"opponent_map_deaths_per_round":sum(map_rates)/len(map_rates) if map_rates else rate(2,default=3.4)}

def role_defense_summary(history):
 """Calculate every role/map aggregate in one pass for efficient walk-forward use."""
 items=list(history);weights=[.5**(age/30) for age in range(len(items)-1,-1,-1)];summary=defaultdict(lambda:[0.0,0.0,0.0])
 for w,h in zip(weights,items):
  for key in ((h[3],None),(h[3],h[4])):
   summary[key][0]+=w*h[1];summary[key][1]+=w*h[0];summary[key][2]+=w*(h[1]-h[2]*h[0])
 return summary

def role_defense_table_from_summaries(team_summary,league_summary,maps,prior_rounds=150.0):
 def rate(role,map_name=None):
  tk,tr,tres=team_summary.get((role,map_name),(0,0,0));lk,lr,lres=league_summary.get((role,map_name),(0,0,0));league_kpr=lk/lr if lr else .68;league_residual=lres/lr if lr else 0.0;prior=prior_rounds*(1.5 if map_name is not None else 1.0)
  return {"kpr":(tk+prior*league_kpr)/(tr+prior),"residual":(tres+prior*league_residual)/(tr+prior),"sample_rounds":tr}
 overall={r:rate(r) for r in ROLES}
 return {"allowed_kpr_by_role":{r:overall[r]["kpr"] for r in ROLES},"residual_by_role":{r:overall[r]["residual"] for r in ROLES},"sample_rounds_by_role":{r:overall[r]["sample_rounds"] for r in ROLES},"map_allowed_kpr_by_role":{m:{r:rate(r,m)["kpr"] for r in ROLES} for m in maps}}

def role_defense_table(history,league_history,maps):
 return role_defense_table_from_summaries(role_defense_summary(history),role_defense_summary(league_history),maps)

def stored_role_matchup(table,role_props,prospective_maps):
 allowed=table.get("allowed_kpr_by_role",{});residual=table.get("residual_by_role",{});samples=table.get("sample_rounds_by_role",{});by_map=table.get("map_allowed_kpr_by_role",{})
 map_values=[]
 for r in ROLES:
  vals=[by_map.get(m,{}).get(r,allowed.get(r,.68)) for m in prospective_maps];map_values.append(sum(vals)/len(vals) if vals else allowed.get(r,.68))
 return {"opponent_role_allowed_kpr":sum(p*allowed.get(r,.68) for p,r in zip(role_props,ROLES)),"opponent_role_residual":sum(p*residual.get(r,0.0) for p,r in zip(role_props,ROLES)),"opponent_role_map_allowed_kpr":sum(p*x for p,x in zip(role_props,map_values)),"log_opponent_role_sample":math.log1p(sum(p*samples.get(r,0.0) for p,r in zip(role_props,ROLES)))}

def competition_profile(profiles):
 kprs=[p.get("kpr_decay",.68) for p in profiles]
 return {"teammate_avg_kpr":sum(kprs)/len(kprs) if kprs else .68,"teammate_max_kpr":max(kprs) if kprs else .68,"teammate_combined_kpr":sum(kprs) if kprs else 2.72,"high_usage_teammates":sum(x>=.72 for x in kprs)}

def vector(row,maps,market):
 base=[row[x] for x in ("expected_rounds","kpr5","kpr10","kpr25","kpr_decay","acs_decay","team_kpr","map_kpr","kill_share","log_team_maps","log_experience","elo_diff","win_probability","opponent_deaths_per_round","opponent_kills_per_round","opponent_first_death_rate","opponent_pace","opponent_map_deaths_per_round","opponent_role_allowed_kpr","opponent_role_residual","opponent_role_map_allowed_kpr","log_opponent_role_sample","teammate_avg_kpr","teammate_max_kpr","teammate_combined_kpr","high_usage_teammates")]
 base+=row["role_props"]
 base += [float(row["map1"]==m) for m in maps]
 if market=="maps_1_2":base += [float(row["map2"]==m) for m in maps]
 return base
def feature_names(maps,market):
 names=["expected_rounds","kpr5","kpr10","kpr25","kpr_decay","acs_decay","current_team_kpr","map_kpr","kill_share","log_current_team_maps","log_experience","elo_diff","win_probability","opponent_deaths_per_round","opponent_kills_per_round","opponent_first_death_rate","opponent_pace","opponent_map_deaths_per_round","opponent_role_allowed_kpr","opponent_role_residual","opponent_role_map_allowed_kpr","log_opponent_role_sample","teammate_avg_kpr","teammate_max_kpr","teammate_combined_kpr","high_usage_teammates"]+[f"role_prop={r}" for r in ROLES]+[f"map1={m}" for m in maps]
 if market=="maps_1_2":names += [f"map2={m}" for m in maps]
 return names

def build_training(archive,incremental_data=None):
 root=Path(archive);api_root=Path(incremental_data) if incremental_data else None
 api_matches=list(csv_rows(api_root/"matches.csv")) if api_root and (api_root/"matches.csv").exists() else []
 api_player_maps=list(csv_rows(api_root/"player_maps.csv")) if api_root and (api_root/"player_maps.csv").exists() else []
 years=sorted({int(p.name[-4:]) for p in root.glob("vct_20*") if (p/"matches/scores.csv").exists()}|{int(number(r.get("year"))) for r in api_matches if number(r.get("year"))})
 allids=root/"all_ids";player_ids={clean(r["Player"]):clean(r["Player ID"]) for r in csv_rows(allids/"all_players_ids.csv")}
 for r in api_player_maps:
  if clean(r.get("player")) and clean(r.get("player_id")):player_ids[clean(r["player"])]=clean(r["player_id"])
 games={};matches={};games_by_match=defaultdict(list)
 for r in csv_rows(allids/"all_matches_games_ids.csv"):
  y=int(r["Year"]);k=match_key(r);v={"match_id":clean(r["Match ID"]),"game_id":clean(r["Game ID"]),"map":clean(r["Map"])}
  games[(y,k,v["map"])]=v;matches[(y,k)]=v["match_id"];games_by_match[(y,k)].append(v)
 existing_match_ids=set(matches.values());api_match_meta={}
 for r in api_matches:
  y=int(number(r.get("year")));k=(clean(r.get("tournament")),clean(r.get("stage")),clean(r.get("match_type")),clean(r.get("match_name")));mid=clean(r.get("series_id"))
  if not y or not mid or mid in existing_match_ids or (y,k) in matches:continue
  api_match_meta[mid]=(y,k,r);matches[(y,k)]=mid
 for r in api_player_maps:
  mid=clean(r.get("series_id"));meta=api_match_meta.get(mid)
  if not meta:continue
  y,k,_=meta;map_name=clean(r.get("map"));q=(y,k,map_name)
  if q in games:continue
  v={"match_id":mid,"game_id":clean(r.get("game_id")),"map":map_name,"map_order":int(number(r.get("map_order"),999))}
  games[q]=v;games_by_match[(y,k)].append(v)
 for q,items in games_by_match.items():
  items.sort(key=lambda x:(int(x.get("map_order",10**12)),int(x["game_id"] or 10**12)))
  for i,v in enumerate(items,1):v["map_order"]=i
 scores={}
 for y in years:
  score_path=root/f"vct_{y}/matches/scores.csv"
  if score_path.exists():
   for r in csv_rows(score_path):
    scores[(y,match_key(r))]={"a":team_name(r["Team A"]),"b":team_name(r["Team B"]),"sa":number(r["Team A Score"]),"sb":number(r["Team B Score"])}
 for mid,(y,k,r) in api_match_meta.items():
  scores[(y,k)]={"a":team_name(r.get("team_a")),"b":team_name(r.get("team_b")),"sa":number(r.get("team_a_score")),"sb":number(r.get("team_b_score"))}
 def order_of(y,k):
  v=matches.get((y,k),"");return int(v) if v.isdigit() else 0
 ordered=sorted(scores,key=lambda q:(q[0],order_of(*q)));rating=defaultdict(lambda:1500.0);elo_pre={};last_year=None
 for y,k in ordered:
  if last_year is not None and y!=last_year:
   for t in list(rating):rating[t]=1500+.75*(rating[t]-1500)
  last_year=y;s=scores[(y,k)];ra,rb=rating[s["a"]],rating[s["b"]];pa=1/(1+10**((rb-ra)/400));result=float(s["sa"]>s["sb"]);elo_pre[(y,k)]=(ra,rb,pa)
  rating[s["a"]]=ra+48*(result-pa);rating[s["b"]]=rb+48*((1-result)-(1-pa))
 round_count=defaultdict(int)
 for y in years:
  round_path=root/f"vct_{y}/matches/win_loss_methods_round_number.csv"
  if round_path.exists():
   for r in csv_rows(round_path):
    q=(y,match_key(r),clean(r["Map"]));round_count[q]=max(round_count[q],int(number(r["Round Number"])))
 for r in api_player_maps:
  meta=api_match_meta.get(clean(r.get("series_id")))
  if meta:
   y,k,_=meta;q=(y,k,clean(r.get("map")));round_count[q]=max(round_count[q],int(number(r.get("rounds"))))
 map_history=defaultdict(list);expected_rounds={}
 for y,k in ordered:
  if (y,k) not in elo_pre:continue
  pa=elo_pre[(y,k)][2];closeness=1-2*abs(pa-.5)
  for g in games_by_match.get((y,k),[]):
   base=sum(map_history[g["map"]])/len(map_history[g["map"]]) if map_history[g["map"]] else 21.0
   expected_rounds[(y,k,g["map"])]=max(15,min(27,base+1.5*(closeness-.70)))
  for g in games_by_match.get((y,k),[]):
   rn=round_count.get((y,k,g["map"]),0)
   if rn:map_history[g["map"]].append(rn)
 obs=[]
 for y in years:
  overview_path=root/f"vct_{y}/matches/overview.csv"
  if overview_path.exists():
   for r in csv_rows(overview_path):
    if clean(r["Side"]).casefold()!="both" or clean(r["Map"])=="All Maps":continue
    k=match_key(r);g=games.get((y,k,clean(r["Map"])))
    if g and (y,k) in scores:
     obs.append({"year":y,"order":order_of(y,k),"key":k,"match_id":g["match_id"],"game_id":g["game_id"],"map_order":g["map_order"],"map":g["map"],"player":clean(r["Player"]),"team":team_name(r["Team"]),"agent":clean(r["Agents"]),"kills":number(r["Kills"]),"deaths":number(r.get("Deaths")),"first_deaths":number(r.get("First Deaths")),"acs":number(r["Average Combat Score"])})
 for r in api_player_maps:
  meta=api_match_meta.get(clean(r.get("series_id")))
  if not meta:continue
  y,k,_=meta;g=games.get((y,k,clean(r.get("map"))))
  if g and (y,k) in scores:
   obs.append({"year":y,"order":order_of(y,k),"key":k,"match_id":g["match_id"],"game_id":g["game_id"],"map_order":g["map_order"],"map":g["map"],"player":clean(r.get("player")),"team":team_name(r.get("team")),"agent":clean(r.get("agents")),"kills":number(r.get("kills")),"deaths":number(r.get("deaths")),"first_deaths":number(r.get("first_deaths")),"acs":number(r.get("acs"))})
 obs.sort(key=lambda x:(x["year"],x["order"],x["map_order"]));state=defaultdict(lambda:deque(maxlen=100));team_state=defaultdict(lambda:deque(maxlen=100));role_state=defaultdict(lambda:deque(maxlen=500));league_role_state=deque(maxlen=5000);records={"map_1":[],"maps_1_2":[]};team_map_counts=defaultdict(Counter);global_maps=Counter();i=0
 while i<len(obs):
  y,o,k=obs[i]["year"],obs[i]["order"],obs[i]["key"];j=i
  while j<len(obs) and (obs[j]["year"],obs[j]["order"],obs[j]["key"])==(y,o,k):j+=1
  group=obs[i:j];s=scores[(y,k)];ra,rb,pa=elo_pre[(y,k)];team_totals=Counter();team_deaths=Counter();team_first_deaths=Counter()
  for r in group:
   team_totals[(r["map_order"],r["team"])]+=r["kills"];team_deaths[(r["map_order"],r["team"])]+=r["deaths"];team_first_deaths[(r["map_order"],r["team"])]+=r["first_deaths"]
  by_player=defaultdict(dict)
  for r in group:by_player[r["player"]][r["map_order"]]=r
  match_maps=sorted({r["map"] for r in group});league_summary=role_defense_summary(league_role_state)
  role_tables={tm:role_defense_table_from_summaries(role_defense_summary(role_state[tm]),league_summary,match_maps) for tm in (s["a"],s["b"])}
  player_updates=[]
  for player,played in by_player.items():
   for market,orders in (("map_1",[1]),("maps_1_2",[1,2])):
    if not all(x in played for x in orders):continue
    chosen=[played[x] for x in orders];tm=chosen[0]["team"]
    if any(x["team"]!=tm for x in chosen):continue
    maps_used=[x["map"] for x in chosen];profile=weighted_profile(state[player],maps_used);te,oe,wp=(ra,rb,pa) if tm==s["a"] else (rb,ra,1-pa);opponent=s["b"] if tm==s["a"] else s["a"]
    # Recompute current-team fields against the future team rather than the last historical team.
    hist=list(state[player]);weights=[.5**(age/15) for age in range(len(hist)-1,-1,-1)];same=[(w,h) for w,h in zip(weights,hist) if h[4]==tm];rd=sum(w*h[1] for w,h in same)
    profile["team_kpr"]=sum(w*h[0] for w,h in same)/rd if rd else profile["kpr_decay"];profile["team_maps"]=sum(1 for h in hist if h[4]==tm)
    teammates=[weighted_profile(state[other],maps_used) for other,other_played in by_player.items() if other!=player and any(x["team"]==tm for x in other_played.values())]
    rec={"year":y,"match_order":o,"match_id":chosen[0]["match_id"],"game_id":chosen[0]["game_id"],"player":player,"player_id":player_ids.get(player) or fallback("P",player),"team":tm,"opponent":opponent,"market":market,"map1":maps_used[0],"map2":maps_used[1] if len(maps_used)>1 else "","actual_kills":sum(x["kills"] for x in chosen),"expected_rounds":sum(expected_rounds[(y,k,x["map"])] for x in chosen),"elo_diff":te-oe,"win_probability":wp,**profile,**defensive_profile(team_state[opponent],maps_used),**stored_role_matchup(role_tables[opponent],profile["role_props"],maps_used),**competition_profile(teammates)}
    rec["log_team_maps"]=math.log1p(rec["team_maps"]);rec["log_experience"]=math.log1p(rec["experience"]);records[market].append(rec)
   for order,r in played.items():
    rn=round_count.get((y,k,r["map"]),0) or expected_rounds[(y,k,r["map"])];share=r["kills"]/(team_totals[(order,r["team"])] or 1)
    player_updates.append((player,(r["kills"],rn,r["acs"],agent_role(r["agent"]),r["team"],r["map"],share)))
  role_updates=[]
  for r in group:
   rn=round_count.get((y,k,r["map"]),0) or expected_rounds[(y,k,r["map"])];pregame=weighted_profile(state[r["player"]],[r["map"]]);expected_kpr=.7*pregame["kpr_decay"]+.3*pregame["map_kpr"];opponent=s["b"] if r["team"]==s["a"] else s["a"]
   role_updates.append((opponent,(rn,r["kills"],expected_kpr,agent_role(r["agent"]),r["map"])))
  for opponent,h in role_updates:role_state[opponent].append(h);league_role_state.append(h)
  for player,h in player_updates:state[player].append(h)
  for order in sorted({r["map_order"] for r in group}):
   map_name=next(r["map"] for r in group if r["map_order"]==order);rn=round_count.get((y,k,map_name),0) or expected_rounds[(y,k,map_name)]
   for tm in {r["team"] for r in group if r["map_order"]==order}:team_state[tm].append((rn,team_totals[(order,tm)],team_deaths[(order,tm)],team_first_deaths[(order,tm)],map_name))
  if y==max(years):
   for r in group:team_map_counts[r["team"]][r["map"]]+=1;global_maps[r["map"]]+=1
  i=j
 latest_players={}
 for player,hist in state.items():
  p=weighted_profile(hist,[]);latest_players[player]={**p,"history_maps":len(hist)}
  byteam=defaultdict(list);bymap=defaultdict(list)
  for h in hist:byteam[h[4]].append(h);bymap[h[5]].append(h)
  latest_players[player]["team_stats"]={t:{"maps":len(v),"kpr":sum(x[0] for x in v)/(sum(x[1] for x in v) or 1)} for t,v in byteam.items()}
  latest_players[player]["map_kpr"]={m:sum(x[0] for x in v)/(sum(x[1] for x in v) or 1) for m,v in bymap.items()}
 latest_defense={team:defensive_profile(hist,[]) for team,hist in team_state.items()}
 for team,hist in team_state.items():latest_defense[team]["map_deaths_per_round"]={m:defensive_profile(hist,[m])["opponent_map_deaths_per_round"] for m in {h[4] for h in hist}}
 latest_role_defense={team:role_defense_table(hist,league_role_state,sorted(global_maps)) for team,hist in role_state.items()}
 current_rosters=defaultdict(list)
 for player,hist in state.items():
  if hist:current_rosters[hist[-1][4]].append(player)
 return years,rating,records,latest_players,team_map_counts,global_maps,sorted(global_maps),player_ids,latest_defense,latest_role_defense,dict(current_rosters)

def train_command(a):
 years,rating,records,players,team_maps,global_maps,maps,player_ids,team_defense,team_role_defense,current_rosters=build_training(a.archive,a.incremental_data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 models={};evaluation={};oof=[]
 for market in ("map_1","maps_1_2"):
  data=records[market];names=feature_names(maps,market);fold_metrics=[]
  for test_year in [y for y in years if y>=2023]:
   tr=[r for r in data if r["year"]<test_year and r["experience"]>=3];te=[r for r in data if r["year"]==test_year and r["experience"]>=3]
   if not tr or not te:continue
   weights=[.5**((test_year-1-r["year"])/1.5) for r in tr];m=Ridge().fit([vector(r,maps,market) for r in tr],[r["actual_kills"] for r in tr],weights=weights);pred=np.maximum(.1,m.predict([vector(r,maps,market) for r in te]));alpha=nb_alpha([r["actual_kills"] for r in tr],m.predict([vector(r,maps,market) for r in tr]));fold_metrics.append({"test_year":test_year,**metrics([r["actual_kills"] for r in te],pred),"dispersion_alpha":alpha})
   for r,p in zip(te,pred):
    r0=dict(r);r1=dict(r);r0["win_probability"]=0.0;r1["win_probability"]=1.0
    oof.append({"market":market,"year":test_year,"match_id":r["match_id"],"game_id":r["game_id"],"player":r["player"],"player_id":r["player_id"],"team":r["team"],"opponent":r["opponent"],"actual_kills":r["actual_kills"],"predicted_kills":round(float(p),4),"elo_win_probability":round(r["win_probability"],6),"prediction_wp0":round(max(.1,m.one(vector(r0,maps,market))),4),"prediction_wp1":round(max(.1,m.one(vector(r1,maps,market))),4),"dispersion_alpha":round(alpha,6)})
  usable=[r for r in data if r["experience"]>=3];weights=[.5**((max(years)-r["year"])/1.5) for r in usable];final=Ridge().fit([vector(r,maps,market) for r in usable],[r["actual_kills"] for r in usable],weights=weights);fit=final.predict([vector(r,maps,market) for r in usable]);alpha=nb_alpha([r["actual_kills"] for r in usable],fit)
  models[market]={"ridge":final.dump(names),"dispersion_alpha":alpha,"training_rows":len(usable)};evaluation[market]={"walk_forward":fold_metrics,"overall_oof":metrics([r["actual_kills"] for r in oof if r["market"]==market],[r["predicted_kills"] for r in oof if r["market"]==market])}
 total=sum(global_maps.values()) or 1;bundle={"version":5,"years":years,"maps":maps,"roles":ROLES,"models":models,"team_elo":dict(rating),"players":players,"team_defense":team_defense,"team_role_defense":team_role_defense,"current_rosters":current_rosters,"global_map_weights":{m:global_maps[m]/total for m in maps},"team_map_counts":{t:dict(v) for t,v in team_maps.items()},"player_ids":player_ids,"settings":{"player_map_half_life":15,"team_defense_half_life":20,"role_defense_half_life":30,"role_defense_prior_rounds":150,"season_half_life":1.5,"elo_k":48,"market_blend":.75,"minimum_probability":.57}}
 (out/"production_model.json").write_text(json.dumps(bundle));(out/"evaluation.json").write_text(json.dumps(evaluation,indent=2));write_csv(out/"walk_forward_predictions.csv",oof)
 print(json.dumps({m:{"training_rows":models[m]["training_rows"],"oof":evaluation[m]["overall_oof"]} for m in models},indent=2))

def resolve_case(value,items):
 lookup={x.casefold():x for x in items};return lookup.get(clean(value).casefold(),clean(value))
def map_weights(bundle,team,opponent):
 weights={};global_w=bundle["global_map_weights"];tc=bundle["team_map_counts"]
 for m in bundle["maps"]:weights[m]=5*global_w.get(m,0)+tc.get(team,{}).get(m,0)+tc.get(opponent,{}).get(m,0)
 total=sum(weights.values()) or 1;return {m:w/total for m,w in weights.items() if w>0}
def future_profile(bundle,player,team,opponent,maps_used):
 st=bundle["players"].get(player,{});team_st=st.get("team_stats",{}).get(team,{});map_rates=[st.get("map_kpr",{}).get(m,st.get("kpr_decay",.68)) for m in maps_used]
 defense=bundle.get("team_defense",{}).get(opponent,{});map_dpr=defense.get("map_deaths_per_round",{});dpr=[map_dpr.get(m,defense.get("opponent_deaths_per_round",3.4)) for m in maps_used]
 teammates=[bundle["players"].get(p,{}) for p in bundle.get("current_rosters",{}).get(team,[]) if p!=player]
 role_props=st.get("role_props",[0,0,0,0,1]);role_matchup=stored_role_matchup(bundle.get("team_role_defense",{}).get(opponent,{}),role_props,maps_used)
 return {"kpr5":st.get("kpr5",.68),"kpr10":st.get("kpr10",.68),"kpr25":st.get("kpr25",.68),"kpr_decay":st.get("kpr_decay",.68),"acs_decay":st.get("acs_decay",200),"team_kpr":team_st.get("kpr",st.get("kpr_decay",.68)),"team_maps":team_st.get("maps",0),"map_kpr":sum(map_rates)/len(map_rates),"kill_share":st.get("kill_share",.20),"experience":st.get("experience",st.get("history_maps",0)),"role_props":role_props,"opponent_deaths_per_round":defense.get("opponent_deaths_per_round",3.4),"opponent_kills_per_round":defense.get("opponent_kills_per_round",3.4),"opponent_first_death_rate":defense.get("opponent_first_death_rate",.5),"opponent_pace":defense.get("opponent_pace",6.8),"opponent_map_deaths_per_round":sum(dpr)/len(dpr),**role_matchup,**competition_profile(teammates)}
def expected_map_rounds(bundle,map_name,probability):
 # Current pool average inferred from the production model's training scale; capped strength adjustment.
 base=21.0;closeness=1-2*abs(probability-.5);return max(15,min(27,base+1.5*(closeness-.70)))
def predict_command(a):
 bundle=json.loads(Path(a.model).read_text());cal_path=Path(a.model).parent.parent/"line_calibration.json";calibration=json.loads(cal_path.read_text()) if cal_path.exists() else {};result=[];team_names=bundle["team_elo"].keys();player_names=bundle["players"].keys()
 for row in csv_rows(a.input):
  # Allow users to paste a complete CSV block underneath an existing header.
  if clean(row.get("market")).casefold()=="market" and clean(row.get("player")).casefold()=="player":continue
  market=clean(row.get("market")) or "map_1"
  if market not in bundle["models"]:raise SystemExit(f"Unknown market '{market}'. Use map_1 or maps_1_2.")
  team=resolve_case(team_name(row["team"]),team_names);opp=resolve_case(team_name(row["opponent"]),team_names);player=resolve_case(row["player"],player_names);line=number(row["line"]);te=bundle["team_elo"].get(team,1500);oe=bundle["team_elo"].get(opp,1500);elo_p=1/(1+10**((oe-te)/400));to=number(row.get("team_odds"));oo=number(row.get("opponent_odds"));market_p=(1/to)/((1/to)+(1/oo)) if to>1 and oo>1 else None;blend=calibration.get("market_blend",bundle["settings"]["market_blend"]);wp=blend*market_p+(1-blend)*elo_p if market_p is not None else elo_p
  supplied1=clean(row.get("map1") or row.get("map"));supplied2=clean(row.get("map2"));mw=map_weights(bundle,team,opp)
  if market=="map_1":choices=[(supplied1,1.0)] if supplied1 in bundle["maps"] else list(mw.items())
  else:
   if supplied1 in bundle["maps"] and supplied2 in bundle["maps"]:choices=[((supplied1,supplied2),1.0)]
   else:
    choices=[]
    for m1,w1 in mw.items():
     for m2,w2 in mw.items():
      if m1!=m2:choices.append(((m1,m2),w1*w2/(1-w1 if w1<1 else 1)))
  model=Ridge.load(bundle["models"][market]["ridge"]);pred=[]
  for maps_choice,w in choices:
   used=[maps_choice] if market=="map_1" else list(maps_choice);profile=future_profile(bundle,player,team,opp,used);override=clean(row.get("role_override")).casefold()
   if override in ROLES:
    profile["role_props"]=[float(override==r) for r in ROLES];profile.update(stored_role_matchup(bundle.get("team_role_defense",{}).get(opp,{}),profile["role_props"],used))
   rec={"expected_rounds":sum(expected_map_rounds(bundle,m,wp) for m in used),"elo_diff":te-oe,"win_probability":wp,"map1":used[0],"map2":used[1] if len(used)>1 else "",**profile};rec["log_team_maps"]=math.log1p(rec["team_maps"]);rec["log_experience"]=math.log1p(rec["experience"]);pred.append((w,max(.1,model.one(vector(rec,bundle["maps"],market))),profile))
  sw=sum(w for w,_,_ in pred) or 1;raw_mean=sum(w*p for w,p,_ in pred)/sw;mean=raw_mean
  if calibration.get("residual_model"):
   cm=Ridge.load(calibration["residual_model"]);mean=max(.1,cm.one([raw_mean,line,raw_mean-line,float(market=="map_1")]))
  alpha=bundle["models"][market]["dispersion_alpha"];p_more=probability_more(mean,line,alpha);p_less=1-p_more;direction="more" if mean>line else "less";minp=a.min_probability if a.min_probability is not None else calibration.get("minimum_probability",bundle["settings"]["minimum_probability"]);recommendation=direction if max(p_more,p_less)>=minp else "pass"
  context=lambda key:sum(w*p[key] for w,_,p in pred)/sw
  result.append({**row,"matched_player":player,"market":market,"elo_win_probability":round(elo_p,4),"market_win_probability":round(market_p,4) if market_p is not None else "","market_blend_used":round(blend,2),"combined_win_probability":round(wp,4),"map_assumption":supplied1 if supplied1 else "team-weighted map pool","opponent_deaths_per_round":round(context("opponent_deaths_per_round"),3),"opponent_map_deaths_per_round":round(context("opponent_map_deaths_per_round"),3),"opponent_pace":round(context("opponent_pace"),3),"opponent_role_allowed_kpr":round(context("opponent_role_allowed_kpr"),3),"opponent_role_residual":round(context("opponent_role_residual"),3),"opponent_role_map_allowed_kpr":round(context("opponent_role_map_allowed_kpr"),3),"opponent_role_sample":round(math.expm1(context("log_opponent_role_sample")),1),"teammate_avg_kpr":round(context("teammate_avg_kpr"),3),"teammate_max_kpr":round(context("teammate_max_kpr"),3),"high_usage_teammates":round(context("high_usage_teammates"),2),"raw_predicted_kills":round(raw_mean,2),"predicted_kills":round(mean,2),"line":line,"edge":round(mean-line,2),"probability_more":round(p_more,4),"probability_less":round(p_less,4),"direction":direction,"recommendation":recommendation,"uncertainty_sd":round(math.sqrt(mean+alpha*mean*mean),2),"calibration_applied":int(bool(calibration.get("residual_model"))),"warning":"" if player in bundle["players"] else "new player defaults used"})
 write_csv(a.output,result);print(f"wrote {len(result)} predictions")

def backtest_command(a):
 preds=list(csv_rows(a.predictions));lookup={(clean(r["market"]),clean(r["match_id"]),clean(r["player"]).casefold()):r for r in preds};joined=[]
 for line in csv_rows(a.lines):
  key=(clean(line["market"]),clean(line["match_id"]),clean(line["player"]).casefold());p=lookup.get(key)
  if not p:continue
  actual=number(p["actual_kills"]);mean=number(p["predicted_kills"]);threshold=number(line["line"]);alpha=number(p["dispersion_alpha"],.1);to=number(line.get("team_odds"));oo=number(line.get("opponent_odds"));market_p=(1/to)/((1/to)+(1/oo)) if to>1 and oo>1 else None
  joined.append({**line,"actual_kills":actual,"base_predicted_kills":mean,"elo_win_probability":number(p.get("elo_win_probability"),.5),"prediction_wp0":number(p.get("prediction_wp0"),mean),"prediction_wp1":number(p.get("prediction_wp1"),mean),"market_probability":market_p,"dispersion_alpha":alpha})
 # Calibrate the market/Elo blend only on rows that actually contain timestamped odds.
 blend_results=[]
 for blend in (0,.25,.50,.75,1.0):
  eligible=[r for r in joined if r["market_probability"] is not None];errors=[]
  for r in eligible:
   wp=blend*r["market_probability"]+(1-blend)*r["elo_win_probability"];pred=r["prediction_wp0"]+wp*(r["prediction_wp1"]-r["prediction_wp0"]);errors.append(abs(r["actual_kills"]-pred))
  blend_results.append({"market_weight":blend,"lines":len(errors),"mae":sum(errors)/len(errors) if errors else None})
 eligible_blends=[r for r in blend_results if r["lines"]>=50];best_blend=min(eligible_blends,key=lambda r:r["mae"])["market_weight"] if eligible_blends else .75
 for r in joined:
  wp=best_blend*r["market_probability"]+(1-best_blend)*r["elo_win_probability"] if r["market_probability"] is not None else r["elo_win_probability"]
  r["predicted_kills"]=r["prediction_wp0"]+wp*(r["prediction_wp1"]-r["prediction_wp0"]);r["edge"]=round(r["predicted_kills"]-number(r["line"]),3);r["probability_more"]=round(probability_more(r["predicted_kills"],number(r["line"]),r["dispersion_alpha"]),4);r["direction"]="more" if r["edge"]>0 else "less";r["outcome"]="push" if r["actual_kills"]==number(r["line"]) else ("more" if r["actual_kills"]>number(r["line"]) else "less");r["correct"]=int(r["direction"]==r["outcome"]) if r["outcome"]!="push" else ""
 calibration={"market_blend":best_blend,"minimum_probability":.57,"trained_lines":len(joined)};cal_eval=None
 if len(joined)>=100:
  ordered=sorted(joined,key=lambda r:clean(r.get("captured_at")));cut=max(50,int(.8*len(ordered)));tr,te=ordered[:cut],ordered[cut:]
  def cx(r):return [r["predicted_kills"],number(r["line"]),r["predicted_kills"]-number(r["line"]),float(clean(r["market"])=="map_1")]
  cm=Ridge().fit([cx(r) for r in tr],[r["actual_kills"] for r in tr],alpha=10);before=[r["predicted_kills"] for r in te];after=np.maximum(.1,cm.predict([cx(r) for r in te]));cal_eval={"test_lines":len(te),"before":metrics([r["actual_kills"] for r in te],before),"after":metrics([r["actual_kills"] for r in te],after)}
  final_cm=Ridge().fit([cx(r) for r in ordered],[r["actual_kills"] for r in ordered],alpha=10);calibration["residual_model"]=final_cm.dump(["model_projection","line","edge","is_map_1"])
 write_csv(a.output,joined)
 valid=[r for r in joined if r["outcome"]!="push"];summary={"joined_lines":len(joined),"model_mae":sum(abs(r["actual_kills"]-r["predicted_kills"]) for r in joined)/len(joined) if joined else None,"line_mae":sum(abs(r["actual_kills"]-number(r["line"])) for r in joined)/len(joined) if joined else None,"direction_accuracy":sum(r["correct"] for r in valid)/len(valid) if valid else None,"market_blend_grid":blend_results,"selected_market_weight":best_blend,"line_residual_calibration":cal_eval,"thresholds":[]}
 for t in (.50,.52,.54,.56,.58,.60,.62,.65,.70):
  selected=[r for r in valid if max(r["probability_more"],1-r["probability_more"])>=t];summary["thresholds"].append({"minimum_probability":t,"selections":len(selected),"accuracy":sum(r["correct"] for r in selected)/len(selected) if selected else None})
 Path(a.summary).write_text(json.dumps(summary,indent=2));Path(a.summary).with_name("line_calibration.json").write_text(json.dumps(calibration,indent=2));print(json.dumps(summary,indent=2))

if __name__=="__main__":
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
 t=sub.add_parser("train");t.add_argument("--archive",required=True);t.add_argument("--incremental-data",default="./api_data");t.add_argument("--output",required=True)
 q=sub.add_parser("predict");q.add_argument("--model",required=True);q.add_argument("--input",required=True);q.add_argument("--output",required=True);q.add_argument("--min-probability",type=float)
 b=sub.add_parser("backtest");b.add_argument("--predictions",required=True);b.add_argument("--lines",required=True);b.add_argument("--output",required=True);b.add_argument("--summary",required=True)
 args=p.parse_args();{"train":train_command,"predict":predict_command,"backtest":backtest_command}[args.command](args)
