"""analysis.py — descriptive, unadjusted OR, multivariable (MICE), overlap weights. Writes results.json + figures."""
import json, warnings, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
warnings.filterwarnings('ignore')
d=pd.read_csv('mediastinitis_deidentified.csv'); d[d.columns[2:]]=d[d.columns[2:]].astype(float); H=d.htx==1; R={}
nH,nS=int(H.sum()),int((~H).sum())

def fmt_p(p):
    if p is None or (isinstance(p,float) and np.isnan(p)): return '—'
    return '<0.001' if p<0.001 else f'{p:.3f}'
def ci(o,l,u): return f'{o:.2f} [{l:.2f}–{u:.2f}]'
def npct(x): x=x.dropna(); return f'{int(x.sum())} ({100*x.mean():.0f}%)' if len(x) else '—'
def miqr(x): x=x.dropna(); return f'{x.median():.1f} [{x.quantile(.25):.1f}–{x.quantile(.75):.1f}]' if len(x) else '—'
def p_bin(a,b):
    a,b=a.dropna(),b.dropna(); t=np.array([[a.sum(),len(a)-a.sum()],[b.sum(),len(b)-b.sum()]])
    if t.min()<0 or (t.sum(0)==0).any() or (t.sum(1)==0).any(): return np.nan
    e=stats.contingency.expected_freq(t)
    return stats.fisher_exact(t)[1] if (e<5).any() else stats.chi2_contingency(t,correction=False)[1]
def p_cont(a,b): a,b=a.dropna(),b.dropna(); return stats.mannwhitneyu(a,b).pvalue if len(a) and len(b) else np.nan
def smd(a,b,binary):
    a,b=a.dropna(),b.dropna()
    if binary: p1,p0=a.mean(),b.mean(); s=np.sqrt((p1*(1-p1)+p0*(1-p0))/2)
    else: p1,p0=a.mean(),b.mean(); s=np.sqrt((a.var()+b.var())/2)
    return abs(p1-p0)/s if s>0 else 0.0
CONT={'age','bmi','euroscore','cpb_min','rbc_units','ffp_units','plt_units','saps2','sofa_max','icu_los','surg_to_dx_days','wbc','ischaemic_time'}

# ---------- Table 1 ----------
T1=[('Patient',None),('Age, years','age'),('Female sex','female'),('BMI, kg/m²','bmi'),('EuroSCORE II, %','euroscore'),('Admitted from home','adm_home'),('Admitted from ICU','adm_icu'),
('Comorbidities',None),('Hypertension','hta'),('Diabetes','diabetes'),('Insulin-treated diabetes','insulin_dm'),('Dyslipidaemia','dyslipid'),('Smoking','smoking'),('COPD','copd'),('Chronic kidney disease','ckd'),('Chronic dialysis','dialysis'),('Previous cardiac surgery','prev_cardiac_surg'),('Preoperative colonisation','colonisation'),('Preoperative infection','preop_infection'),
('Treatments',None),('Antibiotics within 3 months','atb_3m'),('Proton pump inhibitor','ppi'),('Preoperative mechanical circulatory support','preop_mcs'),('  VA-ECMO','preop_ecmo'),('  LVAD / total artificial heart','preop_lvad_tah'),('Preoperative plasma exchange','preop_pe'),
('*Immunosuppression (transplant recipients)',None),('Corticosteroids','is_steroids'),('Anti-thymocyte globulin','is_atg'),('Basiliximab','is_basiliximab'),('Tacrolimus','is_tacrolimus'),('Ciclosporin','is_ciclosporin'),('Mycophenolate mofetil','is_mmf'),('Postoperative plasma exchange','postop_pe'),('IV immunoglobulins','is_ivig'),
('Surgery / transplantation',None),('*Standard surgery: procedure',None),('CABG','proc_cabg'),('Valve surgery','proc_valve'),('Major/combined procedure','proc_major'),('Tamponade','proc_tamponade'),('Mechanical support','proc_mcs'),('Other','proc_other'),
('*Transplantation: underlying cardiomyopathy',None),('Dilated','cm_dilated'),('Ischaemic','cm_ischaemic'),('Congenital','cm_congenital'),('Amyloidosis','cm_amyloid'),('Valvular','cm_valvular'),('Other / mixed','cm_other'),('Graft ischaemic time, min','ischaemic_time'),('Primary graft dysfunction','pgd'),('Rejection','rejection'),
('*Intraoperative (all patients)',None),('CPB duration, min','cpb_min'),('RBC units, n','rbc_units'),('FFP units, n','ffp_units'),('Platelet units, n','plt_units'),('Norepinephrine','norepi_intraop'),('Intraoperative VA-ECMO','ecmo_intraop'),
('ICU course (before diagnosis)',None),('SAPS II','saps2'),('SOFA max','sofa_max'),('Postoperative VA-ECMO','ecmo_postop'),('IABP','iabp'),('Haemorrhagic shock','haem_shock'),('Septic shock','septic_shock_icu'),('AKI (KDIGO ≥1)','aki'),('Kidney replacement therapy','krt'),
('Outcome',None),('ICU length of stay, days','icu_los'),('30-day mortality','mort30'),('1-year mortality','mort1y'),('Death during follow-up','death_followup')]
GROUPSPEC={'is_steroids','is_atg','is_basiliximab','is_tacrolimus','is_ciclosporin','is_mmf','postop_pe','is_ivig','proc_cabg','proc_valve','proc_major','proc_tamponade','proc_mcs','proc_other','cm_dilated','cm_ischaemic','cm_congenital','cm_amyloid','cm_valvular','cm_other','ischaemic_time','pgd','rejection'}
rows=[]
for lab,v in T1:
    if v is None: rows.append({'section':lab}); continue
    x=d[v]; f=miqr if v in CONT else npct
    if v in GROUPSPEC:
        g=d[H] if d.loc[H,v].notna().any() else d[~H]; miss=int(g[v].isna().sum())
        rows.append({'label':lab,'all':'—','htx':f(d.loc[H,v]) if v in GROUPSPEC and d.loc[H,v].notna().any() else '—','std':f(d.loc[~H,v]) if d.loc[~H,v].notna().any() else '—','p':'—','miss':miss})
    else:
        p=p_cont(x[H],x[~H]) if v in CONT else p_bin(x[H],x[~H])
        rows.append({'label':lab,'all':f(x),'htx':f(x[H]),'std':f(x[~H]),'p':fmt_p(p),'miss':int(x.isna().sum())})
R['table1']=rows

# ---------- Table 2 ----------
T2=[('Surgery-to-diagnosis interval, days','surg_to_dx_days'),('Fever','fever'),('Purulent discharge','purulent_discharge'),('Wound inflammation','wound_inflammation'),('Sternal instability','sternal_instability'),('WBC, ×10⁹/L','wbc'),('Positive blood cultures','blood_cultures_pos'),('CT: collection/abscess','ct_collection'),('CT: sternal dehiscence','ct_dehiscence'),('Diagnosis on surgical sample','dx_surgical_sample'),('Diagnosis on sternal puncture/drain','dx_sternal_puncture'),('Other infection before diagnosis','other_infection_before_dx'),('Antibiotics within 7 days before diagnosis','atb_7d_before_dx'),('Septic shock at diagnosis','septic_shock_dx'),('Norepinephrine at diagnosis','norepi_dx'),('Mechanical ventilation at diagnosis','mv_dx'),('Kidney replacement therapy at diagnosis','krt_dx'),('VA-ECMO at diagnosis','ecmo_dx')]
rows=[]
for lab,v in T2:
    c=v in CONT; x=d[v]; f=miqr if c else npct
    rows.append({'label':lab,'all':f(x),'htx':f(x[H]),'std':f(x[~H]),'p':fmt_p(p_cont(x[H],x[~H]) if c else p_bin(x[H],x[~H])),'smd':f'{smd(x[H],x[~H],not c):.2f}','miss':int(x.isna().sum())})
R['table2']=rows

# ---------- Table 3 micro ----------
def or_unadj(v):
    a=d.loc[H,v].sum(); b=nH-a; c=d.loc[~H,v].sum(); e=nS-c
    if min(a,b,c,e)==0: a,b,c,e=a+.5,b+.5,c+.5,e+.5
    o=(a*e)/(b*c); se=np.sqrt(1/a+1/b+1/c+1/e); return o,np.exp(np.log(o)-1.96*se),np.exp(np.log(o)+1.96*se)
MICRO=[('Gram-positive cocci','gpc',0),('S. aureus','s_aureus',1),('MSSA','mssa',2),('MRSA','mrsa',2),('Coagulase-negative staphylococci','cons',1),('Methicillin-resistant (when reported)','cons_mr',2),('Enterococci','enterococci',1),('E. faecalis','e_faecalis',2),('E. faecium','e_faecium',2),('E. gallinarum','e_gallinarum',2),('Streptococci','streptococci',1),
('Fermenting GNB (Enterobacterales)','enterobacterales',0),('Enterobacterales groups 1–2','entero_g12',1),('E. coli','e_coli',2),('K. pneumoniae','k_pneumoniae',2),('K. oxytoca','k_oxytoca',2),('Proteus spp.','proteus',2),('C. koseri','c_koseri',2),('Enterobacterales group 3 (AmpC)','entero_g3',1),('E. cloacae','e_cloacae',2),('K. (E.) aerogenes','k_aerogenes',2),('C. freundii','c_freundii',2),('Serratia marcescens','serratia',2),('Morganella morganii','morganella',2),('Other Enterobacterales','entero_other',2),
('Non-fermenting GNB','nf_gnb',0),('P. aeruginosa','p_aeruginosa',2),('S. maltophilia','s_maltophilia',2),('Acinetobacter','acinetobacter',2),
('Other bacteria','other_bacteria',0),('Corynebacterium','corynebacterium',2),('Cutibacterium / anaerobes','cutibacterium_anaerobes',2),('Other (Haemophilus, Neisseria, Ureaplasma)','other_misc',2),
('Fungi (fungal infection)','fungal',0),('C. albicans','c_albicans',2),('C. glabrata','c_glabrata',2),('C. parapsilosis','c_parapsilosis',2),('C. lusitaniae','c_lusitaniae',2),('C. orthopsilosis','c_orthopsilosis',2),('C. tropicalis','c_tropicalis',2),('Trichosporon','trichosporon',2),('Aspergillus','aspergillus',2),('Candida sp. (unspecified)','candida_unspec',2),
('Multidrug-resistant organisms (any)','mdr_any',0),('ESBL-producing Enterobacterales','esbl',1),('MRSA','mrsa',1),('MDR P. aeruginosa','mdr_pa',1)]
fam=[]; rows=[]
for i,(lab,v,lev) in enumerate(MICRO):
    tot=int(d[v].sum()); test=(tot>=5 and v!='cons_mr') or (i>=len(MICRO)-3)
    r={'label':lab,'level':lev,'htx':npct(d.loc[H,v]),'std':npct(d.loc[~H,v]),'or':'—','p':'—','q':'—','sig':False,'var':v}
    if test:
        o,l,u=or_unadj(v); p=p_bin(d.loc[H,v],d.loc[~H,v]); r.update(or_=o,lo=l,hi=u,p_raw=p,**{'or':ci(o,l,u),'p':fmt_p(p)}); fam.append(i)
    rows.append(r)
K=len(fam)
from statsmodels.stats.multitest import multipletests
QV=multipletests([rows[i]['p_raw'] for i in fam],method='fdr_bh')[1]   # Benjamini-Hochberg FDR, univariate family only
for i,q in zip(fam,QV): rows[i]['q']=fmt_p(q); rows[i]['q_raw']=float(q); rows[i]['sig']=bool(q<0.05)
R['table3']=rows; R['k_micro']=K
POLY=[('≥2 species','poly_2species'),('≥2 bacterial species','poly_2bact'),('Mixed bacterial + fungal','mixed_bact_fungus'),('Fungus only','fungus_only'),('No organism identified','no_organism')]
R['poly']=[{'label':l,'htx':npct(d.loc[H,v]),'std':npct(d.loc[~H,v]),'p':fmt_p(p_bin(d.loc[H,v],d.loc[~H,v]))} for l,v in POLY]

# ---------- MICE ----------
IMPVARS=['htx','age','female','saps2','euroscore','bmi','cpb_min','surg_to_dx_days','atb_7d_before_dx','septic_shock_dx','diabetes','mort30','mort1y','icu_los','fungal','esbl','mdr_any','s_aureus','cons','enterococci','enterobacterales','nf_gnb','p_aeruginosa']
M=20
def impute(seed):
    X=d[IMPVARS].astype(float)
    imp=IterativeImputer(estimator=BayesianRidge(),sample_posterior=True,max_iter=20,random_state=seed)
    Z=pd.DataFrame(imp.fit_transform(X),columns=IMPVARS,index=d.index)
    for b in ['female','atb_7d_before_dx','septic_shock_dx','diabetes','mort30','mort1y']: Z[b]=(Z[b]>0.5).astype(float)
    Z['saps2']=Z.saps2.clip(lower=0); Z['icu_los']=Z.icu_los.clip(lower=0)
    return Z
IMPS=[impute(s) for s in range(M)]
def rubin(ests,vars_):
    q=np.mean(ests); u=np.mean(vars_); b=np.var(ests,ddof=1); t=u+(1+1/M)*b
    r=(1+1/M)*b/u if u>0 else 0; df=(M-1)*(1+1/r)**2 if r>0 else 1e6
    p=2*stats.t.sf(abs(q)/np.sqrt(t),df); h=stats.t.ppf(.975,df)*np.sqrt(t)
    return np.exp(q),np.exp(q-h),np.exp(q+h),p
def logit_pooled(y,covs):
    E,V=[],[]
    for k,Z in enumerate(IMPS):
        D=Z.copy()
        for c in d.columns:
            if c not in D: D[c]=d[c]
        X=sm.add_constant(D[['htx']+covs].astype(float)); mod=sm.Logit(d[y] if d[y].notna().all() else D[y],X).fit(disp=0)
        E.append(mod.params['htx']); V.append(mod.bse['htx']**2)
    return rubin(E,V)
ENDP=[('Fungal infection','fungal'),('ESBL-producing Enterobacterales','esbl'),('Any MDR organism','mdr_any'),('S. aureus','s_aureus'),('CoNS','cons'),('Enterococci','enterococci'),('Fermenting GNB','enterobacterales'),('Non-fermenting GNB','nf_gnb'),('P. aeruginosa','p_aeruginosa')]
R['k_mv']=len(ENDP); rows=[]
for lab,v in ENDP:
    o,l,u,p=logit_pooled(v,['age','saps2']); ev=int(d[v].sum())
    o0,l0,u0=or_unadj(v)
    rows.append({'label':lab,'var':v,'events':ev,'epv':f'{ev/3:.1f}','aor':ci(o,l,u),'o':o,'l':l,'u':u,'p':fmt_p(p),'p_raw':p,'uo':o0,'ul':l0,'uu':u0})
R['table4']=rows
rows=[]
for lab,v in ENDP[:3]:
    b=logit_pooled(v,['age','saps2','atb_7d_before_dx'])   # age + SAPS II + antibiotics within 7 days
    rows.append({'label':lab,'aor':ci(*b[:3]),'p':fmt_p(b[3])})
R['table5']=rows
# mortality
rows=[]
for lab,v in [('30-day mortality','mort30'),('1-year mortality','mort1y')]:
    o,l,u,p=logit_pooled(v,['age','saps2']); rows.append({'label':lab,'events':int(d[v].sum()),'aor':ci(o,l,u),'p':fmt_p(p),'o':o,'l':l,'u':u})
R['table9']=rows

# ---------- Overlap weights ----------
PSV=['age','female','saps2']
def ps_ow(Z):
    X=sm.add_constant(Z[PSV]); ps=sm.Logit(Z.htx,X).fit(disp=0).predict(X)
    w=np.where(Z.htx==1,1-ps,ps); return ps.values,w
def wsmd(x,t,w,binary):
    m1=np.average(x[t==1],weights=w[t==1]); m0=np.average(x[t==0],weights=w[t==0])
    if binary: s=np.sqrt((x[t==1].mean()*(1-x[t==1].mean())+x[t==0].mean()*(1-x[t==0].mean()))/2)
    else: s=np.sqrt((x[t==1].var()+x[t==0].var())/2)
    return abs(m1-m0)/s
BAL=[('Age','age',0),('Female sex','female',1),('SAPS II','saps2',0),('EuroSCORE II*','euroscore',0),('SOFA max*','sofa_max',0),('BMI*','bmi',0),('Diabetes*','diabetes',1),('Antibiotics within 7 days*','atb_7d_before_dx',1),('CPB duration*','cpb_min',0)]
Z0=IMPS[0]; ps0,w0=ps_ow(Z0); t=d.htx.values
bal=[]
for lab,v,b in BAL:
    x=(Z0[v] if v in Z0 else d[v]).astype(float).values; ok=~np.isnan(x)
    bal.append({'label':lab,'before':smd(pd.Series(x[ok&(t==1)]),pd.Series(x[ok&(t==0)]),b),'after':wsmd(x[ok],t[ok],w0[ok],b)})
R['balance']=bal
ess=lambda w:(w.sum()**2)/(w**2).sum()
R['ess']={'htx':ess(w0[t==1]),'std':ess(w0[t==0]),'ps_med_htx':float(np.median(ps0[t==1])),'ps_med_std':float(np.median(ps0[t==0])),
          'w_max':float(w0.max()),'ps_range_htx':[float(ps0[t==1].min()),float(ps0[t==1].max())],'ps_range_std':[float(ps0[t==0].min()),float(ps0[t==0].max())]}
def ow_or(y):
    E,V,wp1,wp0=[],[],[],[]
    for Z in IMPS:
        ps,w=ps_ow(Z); yy=(d[y] if y in d and d[y].notna().all() else Z[y]).astype(float)
        X=sm.add_constant(Z[['htx']]); m=sm.GLM(yy,X,family=sm.families.Binomial(),freq_weights=None,var_weights=w).fit(cov_type='HC0')
        E.append(m.params['htx']); V.append(m.bse['htx']**2)
        wp1.append(np.average(yy[t==1],weights=w[t==1])); wp0.append(np.average(yy[t==0],weights=w[t==0]))
    o,l,u,p=rubin(E,V); return o,l,u,p,np.mean(wp1),np.mean(wp0)
rows=[]
for lab,v in ENDP+[('30-day mortality','mort30'),('1-year mortality','mort1y')]:
    o,l,u,p,p1,p0=ow_or(v)
    rows.append({'label':lab,'var':v,'w1':f'{100*p1:.1f}%','w0':f'{100*p0:.1f}%','or':ci(o,l,u),'o':o,'l':l,'u':u,'p':fmt_p(p)})
R['ow']=rows
R['n']={'htx':nH,'std':nS,'all':len(d)}
# ICU LOS weighted (descriptive)
# missing data table
MS=[('Age','age'),('Sex','female'),('SAPS II','saps2'),('EuroSCORE II','euroscore'),('BMI','bmi'),('SOFA max','sofa_max'),('CPB duration','cpb_min'),('Surgery-to-diagnosis interval','surg_to_dx_days'),('WBC at diagnosis','wbc'),('Antibiotics within 7 days before diagnosis','atb_7d_before_dx'),('Septic shock at diagnosis','septic_shock_dx'),('30-day mortality','mort30'),('1-year mortality','mort1y'),('ICU length of stay','icu_los')]
R['missing']=[{'label':l,'n':int(d[v].isna().sum()),'pct':f'{100*d[v].isna().mean():.1f}%','strat':'complete' if d[v].isna().sum()==0 else ('MICE' if v in IMPVARS else 'not imputed (≥20%)' if d[v].isna().mean()>=.2 else 'not used in models')} for l,v in MS]
json.dump(R,open('results.json','w'),indent=1,default=float)
pd.to_pickle({'ps':ps0,'w':w0,'t':t},'ps.pkl')
print('K micro',K); 
for r in R['table4']: print(r['label'],r['aor'],r['p'])
for r in R['table5']: print(r)
for r in R['table9']: print(r['label'],r['aor'],r['p'])
print(R['ess']); 
for b in bal: print(b['label'],round(b['before'],2),round(b['after'],2))
for r in R['ow']: print(r['label'],r['w1'],r['w0'],r['or'],r['p'])
for r in R['poly']: print(r)
