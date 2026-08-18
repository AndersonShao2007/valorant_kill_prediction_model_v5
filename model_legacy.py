#!/usr/bin/env python3
import argparse,csv,json,math
from pathlib import Path
from collections import defaultdict,Counter,deque
import numpy as np
FIELDS=('Tournament','Stage','Match Type','Match Name'); ALIAS={'Mega Minors':'NRG'}
ROLES={'jett':'duelist','raze':'duelist','neon':'duelist','reyna':'duelist','phoenix':'duelist','yoru':'duelist','iso':'duelist','waylay':'duelist','omen':'smokes','brimstone':'smokes','viper':'smokes','astra':'smokes','harbor':'smokes','clove':'smokes','sova':'initiator','breach':'initiator','skye':'initiator','kay/o':'initiator','kayo':'initiator','fade':'initiator','gekko':'initiator','tejo':'initiator','cypher':'sentinel','killjoy':'sentinel','sage':'sentinel','chamber':'sentinel','deadlock':'sentinel','vyse':'sentinel'}
CATS=['duelist','smokes','initiator','sentinel','flex']
def c(x):return (x or '').strip()
def team(x):return ALIAS.get(c(x),c(x))
def key(r):return tuple(c(r.get(x)) for x in FIELDS)
def rows(p):
 with Path(p).open(newline='',encoding='utf-8-sig') as f:yield from csv.DictReader(f)
def num(x,d=0):
 try:return float(c(x).replace('%',''))
 except:return d
def agent_role(x):return ROLES.get(c(x).casefold(),'flex')
def primary(counts):
 total=sum(counts.values())
 if not total:return 'flex'
 role,n=max(counts.items(),key=lambda z:z[1]);return role if n/total>=.60 else 'flex'
class Ridge:
 def fit(self,X,y,a=25,weights=None):
  X=np.array(X,float);y=np.array(y,float);w=np.ones(len(X)) if weights is None else np.array(weights,float);self.m=np.average(X,axis=0,weights=w);self.s=np.sqrt(np.average((X-self.m)**2,axis=0,weights=w));self.s[self.s==0]=1;Z=np.c_[np.ones(len(X)),(X-self.m)/self.s];sw=np.sqrt(w);ZW=Z*sw[:,None];yw=y*sw;P=np.eye(Z.shape[1])*a;P[0,0]=0;self.b=np.linalg.solve(ZW.T@ZW+P,ZW.T@yw);return self
 def one(self,x):return float(np.r_[1,(np.array(x)-self.m)/self.s]@self.b)
 def dump(self):return {'mean':self.m.tolist(),'scale':self.s.tolist(),'coef':self.b.tolist()}
 @staticmethod
 def load(d):q=Ridge();q.m=np.array(d['mean']);q.s=np.array(d['scale']);q.b=np.array(d['coef']);return q
def save_csv(p,data):
 data=list(data)
 with Path(p).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def train(a):
 root=Path(a.archive);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);years=sorted(int(p.name[-4:]) for p in root.glob('vct_20*'))
 scores={};maprows={}
 for y in years:
  for r in rows(root/f'vct_{y}/matches/scores.csv'):scores[(y,key(r))]={'a':team(r['Team A']),'b':team(r['Team B']),'sa':num(r['Team A Score']),'sb':num(r['Team B Score'])}
  map_score_file=root/f'vct_{y}/matches/maps_scores.csv'
  if map_score_file.exists():
   for r in rows(map_score_file):maprows[(y,key(r),c(r['Map']))]={'map':c(r['Map']),'rounds':num(r['Team A Score'])+num(r['Team B Score'])}
  else:
   for r in rows(root/f'vct_{y}/matches/win_loss_methods_round_number.csv'):
    q=(y,key(r),c(r['Map']));maprows[q]={'map':c(r['Map']),'rounds':max(maprows.get(q,{}).get('rounds',0),num(r['Round Number']))}
 mids={}
 for r in rows(root/'all_ids/all_matches_games_ids.csv'):mids[(int(r['Year']),key(r))]=int(c(r['Match ID']) or 0)
 ordered=sorted(scores,key=lambda q:(q[0],mids.get(q,0)));ratings=defaultdict(lambda:1500.);pre={};last=None
 for y,k in ordered:
  if last and y!=last:
   for t in ratings:ratings[t]=1500+.75*(ratings[t]-1500)
  last=y;s=scores[(y,k)];ra,rb=ratings[s['a']],ratings[s['b']];p=1/(1+10**((rb-ra)/400));w=float(s['sa']>s['sb']);pre[(y,k)]=(ra,rb,p);ratings[s['a']]=ra+48*(w-p);ratings[s['b']]=rb+48*((1-w)-(1-p))
 maps=sorted({v['map'] for v in maprows.values()});X=[];Y=[]
 for (y,k,m),v in maprows.items():
  if (y,k) not in pre:continue
  p=pre[(y,k)][2];X.append([p,1-2*abs(p-.5)]+[float(m==z) for z in maps]);Y.append(v['rounds'])
 round_weights=[.5**((max(years)-y)/1.5) for (y,k,m),v in maprows.items() if (y,k) in pre]
 round_model=Ridge().fit(X,Y,weights=round_weights)
 obs=[]
 for y in years:
  for r in rows(root/f'vct_{y}/matches/overview.csv'):
   if c(r['Side']).casefold()=='both' and c(r['Map'])!='All Maps' and (y,key(r),c(r['Map'])) in maprows:obs.append((y,mids.get((y,key(r)),0),key(r),r))
 obs.sort(key=lambda z:(z[0],z[1]));state=defaultdict(lambda:{'history':deque(maxlen=60),'n':0});features=[];i=0
 while i<len(obs):
  y,mid,k=obs[i][:3];j=i
  while j<len(obs) and obs[j][:3]==(y,mid,k):j+=1
  s=scores[(y,k)];ra,rb,p=pre[(y,k)];pending=[]
  for _,_,_,r in obs[i:j]:
   pl=c(r['Player']);t=team(r['Team']);z=state[pl];hist=list(z['history']);weights=[.5**(age/15) for age in range(len(hist)-1,-1,-1)];wr=sum(weights) or 1;kr=sum(w*h[0] for w,h in zip(weights,hist));rd=sum(w*h[1] for w,h in zip(weights,hist));kpr=kr/rd if rd else .68;acs=sum(w*h[2] for w,h in zip(weights,hist))/wr if hist else 200;rc=defaultdict(float)
   for w,h in zip(weights,hist):rc[h[3]]+=w
   pr=primary(rc);mp=c(r['Map']);rn=maprows[(y,k,mp)]['rounds'];te,oe,wp=(ra,rb,p) if t==s['a'] else (rb,ra,1-p)
   q={'year':y,'rounds':rn,'kpr':kpr,'acs':acs,'n':z['n'],'diff':te-oe,'p':wp,'map':mp,'role':pr,'kills':num(r['Kills'])};features.append(q);pending.append((pl,q,agent_role(r['Agents']),num(r['Average Combat Score'])))
  for pl,q,ro,acs in pending:z=state[pl];z['history'].append((q['kills'],q['rounds'],acs,ro));z['n']+=1
  i=j
 def fx(q):return [q['rounds'],q['kpr'],q['acs'],math.log1p(q['n']),q['diff'],q['p']]+[float(q['map']==m) for m in maps]+[float(q['role']==r) for r in CATS]
 usable=[q for q in features if q['n']>=3];kill_weights=[.5**((max(years)-q['year'])/1.5) for q in usable];kill_model=Ridge().fit([fx(q) for q in usable],[q['kills'] for q in usable],weights=kill_weights);recent=Counter(v['map'] for (y,_,_),v in maprows.items() if y==max(years));total=sum(recent.values())
 players={}
 for p,z in state.items():
  hist=list(z['history']);weights=[.5**(age/15) for age in range(len(hist)-1,-1,-1)];wr=sum(weights) or 1;rd=sum(w*h[1] for w,h in zip(weights,hist));rc=defaultdict(float)
  for w,h in zip(weights,hist):rc[h[3]]+=w
  players[p]={'kpr':sum(w*h[0] for w,h in zip(weights,hist))/rd if rd else .68,'acs':sum(w*h[2] for w,h in zip(weights,hist))/wr if hist else 200,'maps':z['n'],'role':primary(rc),'role_weights':dict(rc)}
 recent_rounds=defaultdict(list)
 for (y,_,m),v in maprows.items():
  if y==max(years) and v['rounds']:recent_rounds[m].append(v['rounds'])
 map_round_means={m:(sum(recent_rounds[m])/len(recent_rounds[m]) if recent_rounds[m] else 21.0) for m in maps}
 bundle={'years':years,'maps':maps,'roles':CATS,'map_weights':{m:recent[m]/total for m in maps},'map_round_means':map_round_means,'round_model':round_model.dump(),'kill_model':kill_model.dump(),'team_elo':dict(ratings),'players':players,'settings':{'player_map_half_life':15,'season_sample_half_life':1.5,'elo_k':48,'market_blend':.75}}
 (out/'production_model.json').write_text(json.dumps(bundle));print(f"trained {len(scores)} matches and {len(features)} player-maps")
def predict(a):
 b=json.loads(Path(a.model).read_text());rm=Ridge.load(b['round_model']);km=Ridge.load(b['kill_model']);result=[]
 player_names={name.casefold():name for name in b['players']};team_names={name.casefold():name for name in b['team_elo']}
 for x in rows(a.input):
  if c(x.get('player')).casefold() in ('player_name','player name',''):
   raise SystemExit(f"ERROR: {a.input} still contains PLAYER_NAME. Replace it with the exact player name, save the file, and run again.")
  t0,o0,pl0=team(x['team']),team(x['opponent']),c(x['player']);t=team_names.get(t0.casefold(),t0);o=team_names.get(o0.casefold(),o0);pl=player_names.get(pl0.casefold(),pl0);line=float(x['line']);te,oe=b['team_elo'].get(t,1500),b['team_elo'].get(o,1500);elo_p=1/(1+10**((oe-te)/400));to=num(x.get('team_odds'));oo=num(x.get('opponent_odds'));market_p=(1/to)/((1/to)+(1/oo)) if to>1 and oo>1 else None;p=.75*market_p+.25*elo_p if market_p is not None else elo_p;st=b['players'].get(pl,{});role=c(x.get('role_override')) or st.get('role','flex');mp=c(x.get('map'));choices=[(mp,1)] if mp in b['maps'] else [(m,w) for m,w in b['map_weights'].items() if w];pred=[]
  for m,w in choices:
   onehot=[float(m==q) for q in b['maps']];base=b.get('map_round_means',{}).get(m,21.0);model_now=rm.one([p,1-2*abs(p-.5)]+onehot);model_even=rm.one([.5,1.0]+onehot);rn=base+max(-2,min(2,model_now-model_even));fx=[rn,st.get('kpr',.68),st.get('acs',200),math.log1p(st.get('maps',0)),te-oe,p]+onehot+[float(role==q) for q in b['roles']];pred.append((w,rn,max(0,km.one(fx))))
  sw=sum(w for w,_,_ in pred);rn=sum(w*r for w,r,_ in pred)/sw;pk=sum(w*k for w,_,k in pred)/sw;edge=pk-line
  result.append({**x,'matched_player_name':pl,'inferred_role':role,'elo_win_probability':round(elo_p,4),'market_win_probability':round(market_p,4) if market_p is not None else '','combined_win_probability':round(p,4),'recent_weighted_kpr':round(st.get('kpr',.68),4),'map_assumption':mp or 'weighted current map pool','predicted_rounds':round(rn,2),'predicted_kills':round(pk,2),'edge':round(edge,2),'recommendation':'more' if edge>=a.edge else ('less' if edge<=-a.edge else 'pass'),'warning':'' if pl in b['players'] else 'new player defaults used'})
 save_csv(a.output,result);print(f'wrote {len(result)} predictions')
if __name__=='__main__':
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True);t=s.add_parser('train');t.add_argument('--archive',required=True);t.add_argument('--output',required=True);q=s.add_parser('predict');q.add_argument('--model',required=True);q.add_argument('--input',required=True);q.add_argument('--output',required=True);q.add_argument('--edge',type=float,default=1.5);a=p.parse_args();train(a) if a.cmd=='train' else predict(a)
