"""flowchart.py — study flow diagram (Figure 1). Edit the counts in FLOW; included patients are read from the de-identified dataset."""
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
FLOW={'cpb_total':11744,'screened':256,
      'excl':[('Sternal wound infection without\n   mediastinitis',3),('Mediastinitis on prosthetic material\n   before surgery',7),('Medical records not found',5),('Duplicate record',1)]}
d=pd.read_csv('mediastinitis_deidentified.csv'); n_htx=int((d.htx==1).sum()); n_std=int((d.htx==0).sum())
n_excl=sum(n for _,n in FLOW['excl'])
if FLOW['screened']-n_excl!=n_htx+n_std:
    print(f"WARNING: screened {FLOW['screened']} - excluded {n_excl} = {FLOW['screened']-n_excl}, but {n_htx+n_std} patients are included")
fig,ax=plt.subplots(figsize=(7.5,6.2)); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
def box(x,y,w,h,txt,fs=9,bold=False,fc='white',left=False):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',fc=fc,ec='#333333',lw=1))
    ax.text(x-w/2+0.2 if left else x,y,txt,ha='left' if left else 'center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',linespacing=1.4)
def arrow(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color='#333333',lw=1))
tot=FLOW['cpb_total']
box(3.6,9.1,5.8,0.9,f"Cardiac surgery with cardiopulmonary bypass\nn = {tot:,}".replace(',',' '))
box(3.6,6.9,4.6,0.9,f"Screened for mediastinitis\nn = {FLOW['screened']}",bold=True)
arrow(3.6,8.65,3.6,7.35)
ex='\n'.join(f"• {l} (n = {n})" for l,n in FLOW['excl'])
box(7.6,5.35,4.6,2.1,f"Excluded (n = {n_excl})\n"+ex,fs=7.6,left=True)
ax.plot([3.6,3.6],[6.45,4.2],color='#333333',lw=1); arrow(3.6,5.35,5.3,5.35)
box(3.6,3.7,4.6,0.9,f"Patients included\nn = {n_htx+n_std}",bold=True)
arrow(3.6,4.4,3.6,4.15)
ax.plot([1.6,5.6],[2.75,2.75],color='#333333',lw=1); ax.plot([3.6,3.6],[3.25,2.75],color='#333333',lw=1)
arrow(1.6,2.75,1.6,2.2); arrow(5.6,2.75,5.6,2.2)
box(1.6,1.6,2.9,1.1,f"Heart transplantation\nn = {n_htx}",fc='#F6DDD9')
box(5.6,1.6,2.9,1.1,f"Other cardiac surgery\nn = {n_std}",fc='#DCE6F1')
plt.savefig('fig1_flowchart.png',dpi=200,bbox_inches='tight'); plt.close(); print('fig1_flowchart.png')
