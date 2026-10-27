import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
R=json.load(open('results.json')); d=pd.read_csv('mediastinitis_deidentified.csv'); d[d.columns[2:]]=d[d.columns[2:]].astype(float); H=d.htx==1
CH,CS='#C0392B','#2E5C8A'; T3={r['var']:r for r in R['table3']}
def ps(p): return '<0.001' if p<0.001 else f'{p:.3f}'
# ---- Supp Fig 1 with stars ----
cats=[('GPC','gpc'),('S. aureus','s_aureus'),('CoNS','cons'),('Entero-\ncocci','enterococci'),('Entero-\nbacterales','enterobacterales'),('NF-GNB','nf_gnb'),('P. aeru-\nginosa','p_aeruginosa'),('Fungi','fungal'),('ESBL-E','esbl'),('Any MDR','mdr_any')]
fig,ax=plt.subplots(1,2,figsize=(11,4.4),gridspec_kw={'width_ratios':[2.2,1]})
x=np.arange(len(cats)); w=.38; tops=np.zeros(len(cats))
for k,(g,c,lab) in enumerate([(H,CH,f'Heart transplant (n={H.sum()})'),(~H,CS,f'Standard surgery (n={(~H).sum()})')]):
    v=[100*d.loc[g,c2].mean() for _,c2 in cats]; ax[0].bar(x+(k-.5)*w,v,w,color=c,label=lab); tops=np.maximum(tops,v)
    for xi,vi in zip(x+(k-.5)*w,v): ax[0].text(xi,vi+1,f'{vi:.0f}',ha='center',fontsize=7)
for i,(_,c2) in enumerate(cats):
    r=T3[c2]; s='**' if r['q_raw']<0.05 else ('*' if r['p_raw']<0.05 else '')
    if s: ax[0].text(x[i],tops[i]+5,s,ha='center',fontsize=12,fontweight='bold')
ax[0].set_ylim(0,85); ax[0].set_xticks(x); ax[0].set_xticklabels([c for c,_ in cats],fontsize=7.5); ax[0].set_ylabel('% of patients'); ax[0].legend(frameon=False,fontsize=8,loc='upper right')
ax[0].set_title('A. Organism categories and resistance phenotypes',loc='left',fontsize=9.5,fontweight='bold')
ax[0].text(0.99,0.80,'* p < 0.05   ** q < 0.05 (Benjamini–Hochberg)',transform=ax[0].transAxes,ha='right',fontsize=7.5)
sp=[('C. albicans','c_albicans'),('C. glabrata','c_glabrata'),('C. parapsilosis','c_parapsilosis'),('C. lusitaniae','c_lusitaniae'),('C. orthopsilosis','c_orthopsilosis'),('C. tropicalis','c_tropicalis'),('Candida sp.','candida_unspec'),('Trichosporon','trichosporon'),('Aspergillus','aspergillus')]
y=np.arange(len(sp))[::-1]; h=.38
for k,(g,c) in enumerate([(H,CH),(~H,CS)]):
    v=[int(d.loc[g,c2].sum()) for _,c2 in sp]; ax[1].barh(y+(.5-k)*h,v,h,color=c)
    for yi,vi in zip(y+(.5-k)*h,v):
        if vi: ax[1].text(vi+.1,yi,str(vi),va='center',fontsize=7)
ax[1].set_yticks(y); ax[1].set_yticklabels([s for s,_ in sp],fontstyle='italic',fontsize=8); ax[1].set_xlabel('Number of patients'); ax[1].set_title('B. Fungal species',loc='left',fontsize=9.5,fontweight='bold')
plt.tight_layout(); plt.savefig('sfig1.png',dpi=200); plt.close()

# ---- generic forest with text columns ----
def forest(lines,fname,xlab,xlim,xt,height_per=0.27,groups=None,legend=None,colw=(0.25,0,0,0),qhead=None):
    n=len(lines); fig=plt.figure(figsize=(10,height_per*n+1.1))
    axl=fig.add_axes([0.0,0.07,colw[0],0.86]); axf=fig.add_axes([colw[0],0.07,0.36,0.86]); axt=fig.add_axes([colw[0]+0.38,0.07,1-colw[0]-0.38,0.86])
    ys=np.array([l['y'] for l in lines]); lo,hi=ys.min()-.7,ys.max()+.9
    for a in (axl,axf,axt): a.set_ylim(lo,hi)
    for l in lines:
        if l.get('o') is not None:
            axf.plot([l['l'],l['u']],[l['y']]*2,color=l['c'],lw=1.2); axf.plot(l['o'],l['y'],l.get('m','s'),color=l['c'],ms=4.5)
        if l.get('label'): axl.text(0.02+0.04*l.get('ind',0),l['y'],l['label'],va='center',fontsize=l.get('fs',8),fontweight=l.get('fw','normal'),fontstyle=l.get('st','normal'))
        if l.get('o') is not None:
            col='black'
            axt.text(0.00,l['y'],f"{l['o']:.2f} [{l['l']:.2f}–{l['u']:.2f}]",va='center',fontsize=7.5,color=l['c'] if l['c']!='#999999' else '#555555')
            axt.text(0.58,l['y'],l['p'],va='center',fontsize=7.5)
            if qhead: axt.text(0.80,l['y'],l['q'],va='center',fontsize=7.5,fontweight='bold' if l.get('sig') else 'normal')
    axt.text(0.00,hi-.35,'OR [95% CI]',fontweight='bold',fontsize=8,va='center'); axt.text(0.58,hi-.35,'p',fontweight='bold',fontsize=8,va='center'); (axt.text(0.80,hi-.35,qhead,fontweight='bold',fontsize=8,va='center') if qhead else None)
    axf.axvline(1,color='grey',ls='--',lw=.8); axf.set_xscale('log'); axf.set_xlim(*xlim); axf.set_xticks(xt); axf.set_xticklabels([f'{v:g}' for v in xt]); axf.set_xlabel(xlab,fontsize=8)
    axf.set_yticks([]); axf.spines['left'].set_visible(False)
    for a in (axl,axt): a.axis('off'); a.set_xlim(0,1)
    if legend: axf.legend(handles=legend,frameon=False,fontsize=7.5,loc='lower center',bbox_to_anchor=(0.5,1.0),ncol=3)
    fig.savefig(fname,dpi=200,bbox_inches='tight'); plt.close()

# ---- Supp Fig 2: unadjusted OR ----
rows=[r for r in R['table3'] if 'or_' in r]; lines=[]
for i,r in enumerate(rows):
    lines.append(dict(y=len(rows)-i,label=r['label'],ind=r['level'],st='italic' if r['level']==2 else 'normal',fw='bold' if r['level']==0 else 'normal',
      o=r['or_'],l=r['lo'],u=r['hi'],c='black' if r['p_raw']<0.05 else '#999999',p=r['p']))
forest(lines,'sfig2.png','Unadjusted OR, heart transplant vs standard (log scale)',(0.008,150),[0.01,0.1,1,10,100])

# ---- Figure 2: unadjusted / multivariable / OW ----
from matplotlib.lines import Line2D
T4=R['table4']; OW={r['var']:r for r in R['ow']}; lines=[]; y=0; K=R['k_micro']
for r in reversed(T4): pass
order=T4; n=len(order); yy=n*4
for r in order:
    t3=T3[r['var']]; ow=OW[r['var']]
    lines.append(dict(y=yy,label=r['label'].replace(' (primary)',''),fw='bold'))
    lines.append(dict(y=yy-.8,label='Unadjusted',ind=1,fs=7.5,o=t3['or_'],l=t3['lo'],u=t3['hi'],c='#999999',m='o',p=t3['p']))
    pm=r['p_raw']; lines.append(dict(y=yy-1.6,label='Multivariable (age + SAPS II)',ind=1,fs=7.5,o=r['o'],l=r['l'],u=r['u'],c='black',m='s',p=r['p']))
    lines.append(dict(y=yy-2.4,label='Overlap-weighted',ind=1,fs=7.5,o=ow['o'],l=ow['l'],u=ow['u'],c='#D35400',m='o',p=ow['p']))
    yy-=3.4
for l in lines: l['y']=l['y']
leg=[Line2D([],[],color='#999999',marker='o',label='Unadjusted'),Line2D([],[],color='black',marker='s',label='Multivariable (age + SAPS II)'),Line2D([],[],color='#D35400',marker='o',label='Overlap weights')]
forest(lines,'fig2_main.png','OR for heart-transplant status (log scale)',(0.04,30),[0.05,0.1,0.2,0.5,1,2,5,10,20],height_per=0.2,legend=leg)

# ---- Supp Fig 3: PS + balance with values ----
P=pd.read_pickle('ps.pkl'); psv,w,t=P['ps'],P['w'],P['t']; E=R['ess']
fig,ax=plt.subplots(1,2,figsize=(11,4.4)); bins=np.linspace(0,1,21)
h1,_=np.histogram(psv[t==1],bins); h0,_=np.histogram(psv[t==0],bins)
hw1,_=np.histogram(psv[t==1],bins,weights=w[t==1]*(t==1).sum()/w[t==1].sum()); hw0,_=np.histogram(psv[t==0],bins,weights=w[t==0]*(t==0).sum()/w[t==0].sum())
c=(bins[:-1]+bins[1:])/2
ax[0].bar(c,h1,.048,color=CH,alpha=.35,label='HTx, unweighted'); ax[0].bar(c,-h0,.048,color=CS,alpha=.35,label='Standard, unweighted')
ax[0].step(bins,np.r_[hw1,hw1[-1]],where='post',color=CH,lw=1.6,label='HTx, overlap-weighted (rescaled)'); ax[0].step(bins,-np.r_[hw0,hw0[-1]],where='post',color=CS,lw=1.6,label='Standard, overlap-weighted (rescaled)')
ax[0].axhline(0,color='k',lw=.6); ax[0].set_xlabel('Propensity score (probability of heart transplant)'); ax[0].set_ylabel('Patients (standard below axis)')
ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{abs(v):.0f}')); ax[0].legend(frameon=False,fontsize=7,loc='lower right')
ax[0].set_title('A. Propensity score distribution',loc='left',fontsize=9.5,fontweight='bold')
B=R['balance']; y=np.arange(len(B))[::-1]
for yi,b in zip(y,B):
    if abs(b['before']-b['after'])>0.02:
        ax[1].annotate('',xy=(b['after'],yi),xytext=(b['before'],yi),arrowprops=dict(arrowstyle='-|>',color='#888888',lw=1,shrinkA=5,shrinkB=5,mutation_scale=9))
ax[1].plot([b['before'] for b in B],y,'o',mfc='white',mec='k',label='Unweighted'); ax[1].plot([b['after'] for b in B],y,'o',color='#D35400',label='Overlap-weighted')
for yi,b in zip(y,B):
    ax[1].text(1.62,yi,f"{b['before']:.2f}",va='center',ha='right',fontsize=7.5); ax[1].text(1.85,yi,f"{b['after']:.2f}",va='center',ha='right',fontsize=7.5,color='#D35400')
ax[1].text(1.62,len(B)-.3,'Unw.',ha='right',fontsize=7.5,fontweight='bold'); ax[1].text(1.85,len(B)-.3,'OW',ha='right',fontsize=7.5,fontweight='bold',color='#D35400')
ax[1].set_xlim(-0.05,1.9); ax[1].set_xticks([0,.2,.4,.6,.8,1,1.2,1.4]); ax[1].set_ylim(-1.9,len(B)-.1)
ax[1].axvline(.1,color='grey',ls='--',lw=.8); ax[1].set_yticks(y); ax[1].set_yticklabels([b['label'] for b in B],fontsize=8); ax[1].set_xlabel('Absolute standardised mean difference')
ax[1].legend(frameon=False,fontsize=7.5,loc='center',bbox_to_anchor=(0.55,0.80))
ax[1].text(0.0,-1.55,f"* not in the PS model.\nEffective sample size: HTx {E['htx']:.1f}/{R['n']['htx']}, standard {E['std']:.1f}/{R['n']['std']}; maximum weight {E['w_max']:.2f}.",fontsize=7,va="center")
ax[1].set_title('B. Covariate balance',loc='left',fontsize=9.5,fontweight='bold')
plt.tight_layout(); plt.savefig('sfig3.png',dpi=200); plt.close()
from PIL import Image
for f in ['sfig1.png','sfig2.png','fig2_main.png','sfig3.png']: print(f,Image.open(f).size)
