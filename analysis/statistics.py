from __future__ import annotations
import random

def cluster_bootstrap(clusters:dict[str,list[float]], statistic, *, seed:int, samples:int=2000)->tuple[float,float]:
    if not clusters: raise ValueError("clusters required")
    rng=random.Random(seed); keys=list(clusters); vals=[]
    for _ in range(samples):
        drawn=[v for _ in keys for v in clusters[rng.choice(keys)]]; vals.append(statistic(drawn))
    vals.sort(); return vals[int(.025*samples)],vals[min(samples-1,int(.975*samples))]

def mcnemar(a:list[bool],b:list[bool])->dict:
    if len(a)!=len(b): raise ValueError("paired outcomes required")
    b01=sum((not x) and y for x,y in zip(a,b)); b10=sum(x and (not y) for x,y in zip(a,b)); n=b01+b10
    # exact two-sided binomial without scipy
    import math
    tail=sum(math.comb(n,k) for k in range(0,min(b01,b10)+1))/(2**n) if n else 1
    return {"discordant_01":b01,"discordant_10":b10,"p_value":min(1,2*tail),"accuracy_delta":sum(b)/len(b)-sum(a)/len(a) if a else None}
