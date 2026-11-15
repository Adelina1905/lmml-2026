"""Realistic synthetic azmari takes with a splice, modelled on the training file.
Genuine take: harmonic instrument, repeated motifs in a random scale, background noise,
optional dynamics, mains hum, pauses, silent padding, fades.
Foreign phrase: other instrument / other register / out-of-scale notes / new melody,
optional own recording chain (rumble, noise, low-pass), crossfaded and RMS-matched."""
import numpy as np
from scipy import signal
SR=22050
def rms(v): return np.sqrt(np.mean(v**2))+1e-12
def colored_noise(rng,n,slope):  # slope 0 white, -1 pink, -2 brown
    w=np.fft.rfft(rng.standard_normal(n)); fr=np.fft.rfftfreq(n,1/SR); fr[0]=fr[1]
    z=np.fft.irfft(w*fr**(slope/2),n); return z/rms(z)
def lf_noise(rng,n,lo=20,hi=90):
    b,a=signal.butter(2,[lo,hi],btype='band',fs=SR); z=signal.lfilter(b,a,rng.standard_normal(n)); return z/rms(z)
def note(rng,f0,dur,nharm,decay,vib=0.004,breath=0.0,attack=0.04):
    n=int(dur*SR); tt=np.arange(n)/SR
    env=np.minimum(1,tt/attack)*np.minimum(1,(dur-tt+1e-4)/0.06)
    env*= (0.8+0.2*np.exp(-tt/0.3))
    ph=2*np.pi*np.cumsum(f0*(1+vib*np.sin(2*np.pi*rng.uniform(4,6)*tt+rng.uniform(0,6))))/SR
    y=np.zeros(n)
    for h in range(1,nharm+1):
        if h*f0>SR/2*0.9: break
        y+=decay**(h-1)*np.sin(h*ph+rng.uniform(0,6))
    if breath>0:
        b,a=signal.butter(2,[max(30,f0*0.7),min(SR/2*0.95,f0*4)],btype='band',fs=SR)
        z=signal.lfilter(b,a,rng.standard_normal(n)); y+=z/rms(z)*rms(y)*breath
    return y*env
def melody(rng,scale,n_notes,durs):
    return [(rng.choice(scale),rng.choice(durs)) for _ in range(n_notes)]
def render(rng,notes,inst):
    parts=[note(rng,f,d,**inst) for f,d in notes]
    cf=int(0.02*SR); out=parts[0]
    for p in parts[1:]:
        w=np.linspace(0,1,cf); out=np.concatenate([out[:-cf],out[-cf:]*(1-w)+p[:cf]*w,p[cf:]])
    return out
SEMI=2**(1/12)
def make_case(seed, hard=False, bg_after=None):
    rng=np.random.default_rng(seed)
    dur=rng.uniform(35,90)
    base=rng.uniform(180,330)
    steps=sorted(rng.choice(range(0,12),size=rng.integers(5,8),replace=False))
    scale=[base*SEMI**s for s in steps]
    instA=dict(nharm=int(rng.integers(6,12)),decay=rng.uniform(0.6,0.85),vib=rng.uniform(0,0.006),breath=rng.uniform(0,0.05),attack=rng.uniform(0.02,0.08))
    durs=[0.4,0.6,0.8,1.0,1.2,1.6]
    # phrases: a few motifs repeated with variation
    motifs=[melody(rng,scale,int(rng.integers(6,14)),durs) for _ in range(int(rng.integers(2,4)))]
    rep_strength=rng.uniform(0.3,1.0)   # how often motifs repeat vs new random notes
    notes=[]; total=0
    while total<dur+2:
        if rng.random()<rep_strength: ph=list(motifs[rng.integers(len(motifs))])
        else: ph=melody(rng,scale,int(rng.integers(6,14)),durs)
        # small variation
        ph=[(f if rng.random()>0.1 else rng.choice(scale),d) for f,d in ph]
        notes+=ph; total+=sum(d for _,d in ph)
    G=render(rng,notes,instA)[:int(dur*SR)]
    # dynamics (slow gain changes) -- distractor
    if rng.random()<0.5:
        g=np.interp(np.arange(len(G)),np.linspace(0,len(G),8),10**(rng.uniform(-4,4,8)/20)); G=G*g
    G=G/rms(G)
    noise_slope=rng.choice([0,-1,-2]); noise_lvl=10**(rng.uniform(-45,-28)/20)
    bg=colored_noise(rng,len(G),noise_slope)*noise_lvl
    if rng.random()<0.5: bg+=lf_noise(rng,len(G))*noise_lvl*rng.uniform(0.5,3)
    if rng.random()<0.3:  # mains hum
        hz=rng.choice([50,60]); tt=np.arange(len(G))/SR; bg+=noise_lvl*rng.uniform(0.5,3)*(np.sin(2*np.pi*hz*tt)+0.3*np.sin(2*np.pi*3*hz*tt))
    # ---- foreign phrase
    L=rng.uniform(3,10); Ln=int(L*SR)
    pos=int(rng.uniform(3,dur-L-3)*SR)
    kinds=['flute_up','flute_same','sameinst_outscale','sameinst_newmel','otherinst'] if hard else ['flute_up','flute_same','sameinst_outscale','otherinst']
    kind=rng.choice(kinds)
    if kind=='flute_up':
        sh=rng.choice([12,7,5,-12]); sc=[f*SEMI**sh for f in scale]+[base*SEMI**(sh+rng.integers(0,12))]
        instB=dict(nharm=int(rng.integers(2,4)),decay=rng.uniform(0.3,0.5),vib=rng.uniform(0.003,0.01),breath=rng.uniform(0.1,0.4),attack=rng.uniform(0.03,0.1))
    elif kind=='flute_same':
        sc=scale; instB=dict(nharm=int(rng.integers(2,4)),decay=rng.uniform(0.3,0.5),vib=rng.uniform(0.003,0.01),breath=rng.uniform(0.1,0.4),attack=rng.uniform(0.03,0.1))
    elif kind=='sameinst_outscale':
        outs=[s for s in range(12) if s not in steps]; sc=[base*SEMI**s for s in rng.choice(outs,size=min(3,len(outs)),replace=False)]+list(rng.choice(scale,2))
        instB=dict(instA)
    elif kind=='sameinst_newmel':
        sc=scale; instB=dict(instA)
    else:
        sc=[f*SEMI**rng.choice([0,12,-12]) for f in scale]
        instB=dict(nharm=int(rng.integers(3,16)),decay=rng.uniform(0.4,0.95),vib=rng.uniform(0,0.01),breath=rng.uniform(0,0.3),attack=rng.uniform(0.005,0.1))
    fn=[]; tot=0
    while tot<L+1: f,d=rng.choice(sc),rng.choice(durs); fn.append((f,d)); tot+=d
    Fseg=render(rng,fn,instB)[:Ln]
    # foreign recording chain
    chain=[]
    fb=np.zeros(Ln)
    if rng.random()<(0.5 if hard else 0.8):
        fb+=lf_noise(rng,Ln)*rng.uniform(0.02,0.3); chain.append('rumble')
    if rng.random()<0.4:
        fb+=colored_noise(rng,Ln,rng.choice([0,-1,-2]))*10**(rng.uniform(-40,-25)/20); chain.append('noise')
    if rng.random()<0.3:
        b,a=signal.butter(4,rng.uniform(3000,6000),fs=SR); Fseg=signal.lfilter(b,a,Fseg); chain.append('lowpass')
    Fseg=Fseg/rms(Fseg)
    # background of the genuine take continues under? no: replaced audio carries its own bg
    Fseg=Fseg+fb
    if bg_after is None: bg_after = rng.random()<0.5
    y=G.copy() if bg_after else G+bg
    Fseg=Fseg/rms(Fseg)*rms(y[pos:pos+Ln])
    cf=int(rng.uniform(0.02,0.2)*SR); w=np.ones(Ln); w[:cf]=np.linspace(0,1,cf); w[-cf:]=np.linspace(1,0,cf)
    y[pos:pos+Ln]=y[pos:pos+Ln]*(1-w)+Fseg*w
    if bg_after: y=y+bg*rms(G)/rms(G)
    # fades at file edges (distractor)
    if rng.random()<0.3:
        fi=int(rng.uniform(0.2,2)*SR); y[:fi]*=np.linspace(0,1,fi)
    if rng.random()<0.3:
        fo=int(rng.uniform(0.2,2)*SR); y[-fo:]*=np.linspace(1,0,fo)
    # pauses and silent padding (distractors)
    pad_a=pad_b=0
    if rng.random()<0.3:   # internal pause: performer stops, room tone only, outside the splice
        pl=int(rng.uniform(0.5,2.5)*SR)
        for _ in range(20):
            p0=int(rng.uniform(1,dur-3)*SR)
            if p0+pl<pos-SR or p0>pos+Ln+SR: break
        else: p0=None
        if p0 is not None: y[p0:p0+pl]=bg[p0:p0+pl]*rms(y)/rms(G+bg)*(1 if rng.random()<0.7 else 0)
    if rng.random()<0.3: pad_a=int(rng.uniform(0.3,3)*SR)
    if rng.random()<0.3: pad_b=int(rng.uniform(0.3,3)*SR)
    y=np.r_[np.zeros(pad_a),y,np.zeros(pad_b)]
    pos=pos+pad_a
    y=y/np.abs(y).max()*rng.uniform(0.04,0.6)*32767
    return np.round(y).astype(np.int16), pos/SR, (pos+Ln)/SR, kind+('+'+'+'.join(chain) if chain else '')+('|bgafter' if bg_after else '')
