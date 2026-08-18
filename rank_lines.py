#!/usr/bin/env python3
"""Rank model outputs by probability and standardized edge."""
import argparse,csv
from pathlib import Path
def rows(p):
 with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--minimum-probability',type=float,default=.50);a=p.parse_args();ranked=[]
 for r in rows(a.input):
  pm=float(r['probability_more']);pl=float(r['probability_less']);confidence=max(pm,pl);sd=max(float(r['uncertainty_sd']),.01);edge=float(r['edge']);warning=r.get('warning','')
  if confidence<a.minimum_probability:continue
  r['confidence']=round(confidence,4);r['standardized_edge']=round(abs(edge)/sd,4);r['quality']='review' if warning else 'ok';ranked.append(r)
 ranked.sort(key=lambda r:(r['quality']=='ok',float(r['confidence']),float(r['standardized_edge'])),reverse=True)
 for i,r in enumerate(ranked,1):r['rank']=i
 fields=['rank','player','matched_player','team','opponent','market','line','predicted_kills','edge','direction','recommendation','probability_more','probability_less','confidence','uncertainty_sd','standardized_edge','quality','warning']
 with Path(a.output).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(ranked)
 print(f'wrote {len(ranked)} ranked lines')
if __name__=='__main__':main()
