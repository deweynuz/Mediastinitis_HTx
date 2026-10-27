"""clean.py — raw CRF (Excel, sheet Feuil1) -> analysis-ready dataset (clean_full.pkl, identifiable, never distributed).
Usage: python clean.py path/to/crf.xlsx"""
import re, sys, unicodedata, numpy as np, pandas as pd
SRC=sys.argv[1] if len(sys.argv)>1 else 'crf.xlsx'
raw=pd.read_excel(SRC,'Feuil1')
raw.columns=[c.strip() for c in raw.columns]
raw=raw[raw['Date Dx Mediastinite'].notna()].reset_index(drop=True)   # confirmed mediastinitis

def num(s): return pd.to_numeric(s.replace({'DM':np.nan,'-':np.nan,'?':np.nan,'na':np.nan}),errors='coerce')
def b01(s, pos=(1,)):
    x=num(s); return x.where(x.isna(), x.isin(pos).astype(float))
def bpos(s):   # any value >0 -> 1 (aberrant codes binarised)
    x=num(s); return x.where(x.isna(), (x>0).astype(float))
def dt(s): return pd.to_datetime(s,errors='coerce')
def norm(t):
    if pd.isna(t): return ''
    t=unicodedata.normalize('NFKD',str(t)).encode('ascii','ignore').decode().lower()
    return ' '+re.sub(r'\s+',' ',t)+' '

d=pd.DataFrame()
d['group']=raw['Greffe'].map({1:'HTx',0:'Standard'})
d['htx']=raw['Greffe'].astype(int)
d['id_src']=raw['Greffe'].map({1:'H',0:'S'})+raw['ID'].map(str)
# patient
d['age']=num(raw['age']); d['female']=raw['sexe'].astype(float)
d['bmi']=num(raw['bmi']); d['euroscore']=num(raw['euroscore'])
prov=num(raw['Provenance patient'])
d['adm_home']=(prov==0).astype(float).where(prov.notna()); d['adm_icu']=(prov==2).astype(float).where(prov.notna())
# comorbidities
d['hta']=bpos(raw['hta']); d['diabetes']=bpos(raw['diabete']); d['insulin_dm']=bpos(raw['diabete sous insuline'])
d['dyslipid']=bpos(raw['dyslipid']); d['smoking']=bpos(raw['tabac']); d['copd']=bpos(raw['bpco'])
d['ckd']=bpos(raw['irc']); d['dialysis']=bpos(raw['dialysechronique']); d['prev_cardiac_surg']=bpos(raw['atcdchircard'])
d['colonisation']=bpos(raw['colonisation']); d['preop_infection']=bpos(raw['infectionPreop'])
# treatments
d['atb_3m']=bpos(raw['ATB 3mois']); d['ppi']=bpos(raw['IPP'])
ass=num(raw['AssistancePreop'])
d['preop_mcs']=(ass>0).astype(float).where(ass.notna()); d['preop_ecmo']=(ass==1).astype(float).where(ass.notna())
d['preop_lvad_tah']=ass.isin([4,5]).astype(float).where(ass.notna())
d['preop_pe']=bpos(raw['EP preop'])
H=d.htx==1
for c,src in [('is_steroids','corticoides'),('is_atg','Thymoglobulines'),('is_basiliximab','Simulect'),('is_tacrolimus','tacrolimus'),
              ('is_ciclosporin','Ciclosporine'),('is_mmf','cellcept'),('postop_pe','EP potop'),('rejection','rejet')]:
    d[c]=bpos(raw[src]).where(H)
oth=raw['Autre ttt IS'].map(str).str.upper()
d['is_ivig']=np.where(oth.str.contains('PRIVIGEN|IG IV'),1.0,np.where(raw['Autre ttt IS'].isna(),np.nan,0.0)); d.loc[~H,'is_ivig']=np.nan
# surgery
chir=raw['chir'].map(str).str.strip().str[0]
for k,v in {'1':'proc_cabg','2':'proc_valve','3':'proc_major','4':'proc_tamponade','5':'proc_mcs','6':'proc_other'}.items():
    d[v]=np.where(H,np.nan,(chir==k).astype(float))
tc=num(raw['TypeCardiopathie']).where(H).replace(0,np.nan)
for k,v in {2:'cm_dilated',1:'cm_ischaemic',3:'cm_congenital',4:'cm_amyloid',6:'cm_valvular'}.items(): d[v]=(tc==k).astype(float).where(H)
d['cm_other']=(~tc.isin([1,2,3,4,6])).astype(float).where(tc.notna())
for v in ['cm_dilated','cm_ischaemic','cm_congenital','cm_amyloid','cm_valvular']: d[v]=d[v].where(tc.notna())
d['ischaemic_time']=num(raw['dureeischemie']).where(H); d['pgd']=bpos(raw['DefaillancePrimGreffon']).where(H)
d['cpb_min']=num(raw['dureecec']); d['rbc_units']=num(raw['cgrperop (nbr)'])
d['ffp_units']=num(raw['pfcperop (nbr)']); d['plt_units']=num(raw['plqperop (nbr)'])
d['norepi_intraop']=bpos(raw['noradreperop']); d['ecmo_intraop']=bpos(raw['ECMOva perop'])
# ICU course before Dx
d['saps2']=num(raw['igs2']); d['sofa_max']=num(raw['sofa max'])
d['ecmo_postop']=bpos(raw['ecmova postop']); d['iabp']=bpos(raw['bcpia'])
d['haem_shock']=bpos(raw['chochemorragique']); d['septic_shock_icu']=bpos(raw['chocseptique'])
d['aki']=bpos(raw['ira (KDIGO)']); d['krt']=bpos(raw['eer'])
# diagnosis
datechir=dt(raw['datechir']); datedx=dt(raw['Date Dx Mediastinite'])
delai=(datedx-datechir).dt.days
dl=num(raw['DELAI']); d['surg_to_dx_days']=dl.where(dl.notna(), delai.where(delai>=0))
dxc=raw['dxclinque'].map(str).str.replace('.',' ',regex=False)
miss=raw['dxclinque'].isna()|raw['dxclinque'].map(str).str.strip().isin(['DM','-'])
for k,v in {'1':'sternal_instability','2':'fever','3':'purulent_discharge','4':'wound_inflammation'}.items():
    d[v]=dxc.apply(lambda s: float(k in re.findall(r'\d',s))).where(~miss)
gb=num(raw['GB']); d['wbc']=np.where(gb>100,gb/1000,gb)
d['blood_cultures_pos']=bpos(raw['hemocultures'])
tdm=raw['TDM'].map(str); tmiss=raw['TDM'].isna()|tdm.str.strip().eq('-')
d['ct_collection']=tdm.apply(lambda s:float('2' in s)).where(~tmiss); d['ct_dehiscence']=tdm.apply(lambda s:float('1' in s)).where(~tmiss)
d['dx_surgical_sample']=raw['prelevchir'].map(str).str.strip().str[0].map({'1':1.0,'0':0.0})
d['dx_sternal_puncture']=bpos(raw['prelevmed (ponction sternale)'])
d['other_infection_before_dx']=bpos(raw['infeavantdx']); d['atb_7d_before_dx']=bpos(raw['ATBavantdx'])
d['septic_shock_dx']=bpos(raw['Choc septique'])
ne=raw['Noradrénaline'].map(str)
d['norepi_dx']=np.where(ne.str.contains('mg'),1.0,num(raw['Noradrénaline']).clip(upper=1))
d['mv_dx']=np.where(raw['VM'].map(str).str.contains('sevrage'),1.0,bpos(raw['VM']))
d['krt_dx']=bpos(raw['RRT']); d['ecmo_dx']=bpos(raw['ECMOva'])
# outcome
d['icu_los']=num(raw['dureesejourrea']); d['death_followup']=(num(raw['Deces'])>0).astype(float)
d['mort30']=num(raw['mort30j']); d['mort1y']=num(raw['mort1an'])
d['_datechir']=datechir; d['_datedx']=datedx; d['_datedeath']=dt(raw['Date deces'])

# ---------------- microbiology ----------------
T=raw['Bacterie (nom)'].apply(norm)
def has(p): return T.str.contains(p.replace('(','(?:'),regex=True)
scn=num(raw['scn']); bg=raw['bgnf'].map(str); ent=raw['enterocoque'].map(str)
champ=raw['champignon'].map(str).str.strip(); fong=num(raw['fongique']); res=num(raw['Resistance'])
m=pd.DataFrame(index=d.index)
m['mssa']=((num(raw['sams'])==1)|has(r'sams|mssa|s\.? ?aureus|staph(ylococcus)? aureus')).astype(int)
m['mrsa']=((num(raw['sarm'])==1)|has(r'sarm|mrsa')).astype(int); m.loc[m.mrsa==1,'mssa']=0
m['s_aureus']=(m.mssa|m.mrsa)
m['cons']=((scn>=1)|has(r'epi|haemoly|haemoli|hominis|capitis|lugdun|caprae|serm')).astype(int)
m['cons_mr']=(has(r'meti ?-?r|\bmr\b|serm')&m.cons.astype(bool)).astype(int)
m['e_faecalis']=(has(r'faecalis|fecalis')).astype(int)
m['e_faecium']=(has(r'faecium')).astype(int)
m['e_gallinarum']=(has(r'gallinarum')).astype(int)
m['enterococci']=((ent.str.contains(r'[123]'))|m.e_faecalis.astype(bool)|m.e_faecium.astype(bool)|m.e_gallinarum.astype(bool)|has('enteroco')).astype(int)
m['streptococci']=((num(raw['strepto'])==1)|has(r'strepto|s\.? ?mitis')).astype(int)
m['e_coli']=has(r'e\.? ?coli|escherichia').astype(int)
m['k_pneumoniae']=has(r'\bkp\b|pneumoniae|pneum\b').astype(int)
m['k_oxytoca']=has(r'\bko\b|oxytoca').astype(int)
m['proteus']=has(r'proteus|mirabilis').astype(int)
m['c_koseri']=has(r'koser').astype(int)
m['e_cloacae']=has(r'cloac|cloaque|clocae|clocaie').astype(int)
m['k_aerogenes']=has(r'aerogenes').astype(int)
m['c_freundii']=has(r'freundi').astype(int)
m['serratia']=has(r'serratia|marces').astype(int)
m['morganella']=has(r'morgan').astype(int)
m['entero_other']=has(r'pantoea|hafnia|amalonaticus').astype(int)
m['entero_g12']=(m[['e_coli','k_pneumoniae','k_oxytoca','proteus','c_koseri']].max(axis=1)|has('amalonaticus')|bg.str.contains('2')).astype(int)
m['entero_g3']=(m[['e_cloacae','k_aerogenes','c_freundii','serratia','morganella']].max(axis=1)|has('hafnia')|bg.str.contains('1')).astype(int)
m['enterobacterales']=(m[['entero_g12','entero_g3','entero_other']].max(axis=1)|(bg.str.contains(r'[12]')&~bg.str.fullmatch(r'0|nan'))).astype(int)
m['p_aeruginosa']=((num(raw['pyo'])==1)|has(r'pyo\b|pseudomonas|aeruginosa|aeuruginosa')).astype(int)
m['s_maltophilia']=((num(raw['steno'])==1)|has(r'steno|maltophilia')).astype(int)
m['acinetobacter']=has('acinetobacter').astype(int)
m['nf_gnb']=m[['p_aeruginosa','s_maltophilia','acinetobacter']].max(axis=1)
m['corynebacterium']=has('coryne').astype(int)
m['cutibacterium_anaerobes']=has(r'cutibacterium|propioni|anaero|finegoldia').astype(int)
m['other_misc']=has(r'haemophilus|neisseria|ureaplasma').astype(int)
m['other_bacteria']=m[['corynebacterium','cutibacterium_anaerobes','other_misc']].max(axis=1)
m['gpc']=m[['s_aureus','cons','enterococci','streptococci']].max(axis=1)
# fungi: structured codes (1 albicans,2 glabrata,3 lusitaniae,4 tropicalis,5 parapsilosis,6 orthopsilosis,7 trichosporon,8 aspergillus)
codes=champ.apply(lambda s: set(re.findall(r'\d',s)) if s not in ('nan','') else set())
fmap={'1':'c_albicans','2':'c_glabrata','3':'c_lusitaniae','4':'c_tropicalis','5':'c_parapsilosis','6':'c_orthopsilosis','7':'trichosporon','8':'aspergillus'}
for k,v in fmap.items(): m[v]=codes.apply(lambda s:int(k in s))
m['c_albicans']=(m.c_albicans.astype(bool)|has('albicans')).astype(int)
anyf=(fong==1)|codes.apply(lambda s:len(s-{'0'})>0)|has(r'candida|levure|aspergill|trichospor')
m['fungal']=anyf.astype(int)
m['candida_unspec']=(m.fungal.astype(bool)&(m[list(fmap.values())].max(axis=1)==0)).astype(int)
m['esbl']=(res==1).astype(int)|has('blse').astype(int)
m['mdr_pa']=((res==2)&m.p_aeruginosa.astype(bool)|has(r'pyo.*multi ?r|pseudomonas multi')).astype(int)
m['mdr_any']=m[['esbl','mrsa','mdr_pa']].max(axis=1)
m['nb_species']=None
d=pd.concat([d,m.drop(columns='nb_species')],axis=1)
if __name__=='__main__':
    print(d.shape)
    cols=['gpc','s_aureus','mssa','mrsa','cons','cons_mr','enterococci','e_faecalis','e_faecium','e_gallinarum','streptococci','enterobacterales','entero_g12','e_coli','k_pneumoniae','k_oxytoca','proteus','c_koseri','entero_g3','e_cloacae','k_aerogenes','c_freundii','serratia','morganella','entero_other','nf_gnb','p_aeruginosa','s_maltophilia','acinetobacter','other_bacteria','corynebacterium','cutibacterium_anaerobes','other_misc','fungal','c_albicans','c_glabrata','c_parapsilosis','c_lusitaniae','c_orthopsilosis','c_tropicalis','trichosporon','aspergillus','candida_unspec','mdr_any','esbl','mrsa','mdr_pa']
    print(d.groupby('htx')[cols].sum().T)

# ---- polymicrobial (species-level count) ----
bact_species=['mssa','mrsa','e_faecalis','e_faecium','e_gallinarum','streptococci','e_coli','k_pneumoniae','k_oxytoca','proteus','c_koseri',
              'e_cloacae','k_aerogenes','c_freundii','serratia','morganella','entero_other','p_aeruginosa','s_maltophilia','acinetobacter',
              'corynebacterium','cutibacterium_anaerobes','other_misc']
fung_species=['c_albicans','c_glabrata','c_parapsilosis','c_lusitaniae','c_orthopsilosis','c_tropicalis','trichosporon','aspergillus','candida_unspec']
n_cons=np.where(num(raw['scn'])>=2,2,d['cons'])
d['n_bact']=d[bact_species].sum(axis=1)+n_cons
d.loc[(d.enterococci==1)&(d[['e_faecalis','e_faecium','e_gallinarum']].sum(axis=1)==0),'n_bact']+=1
d.loc[(d.enterobacterales==1)&(d[['e_coli','k_pneumoniae','k_oxytoca','proteus','c_koseri','e_cloacae','k_aerogenes','c_freundii','serratia','morganella','entero_other']].sum(axis=1)==0),'n_bact']+=1
d['n_fung']=d[fung_species].sum(axis=1)
d['poly_2species']=((d.n_bact+d.n_fung)>=2).astype(int)
d['poly_2bact']=(d.n_bact>=2).astype(int)
d['mixed_bact_fungus']=((d.n_bact>=1)&(d.n_fung>=1)).astype(int)
d['fungus_only']=((d.n_bact==0)&(d.n_fung>=1)).astype(int)
d['no_organism']=((d.n_bact==0)&(d.n_fung==0)).astype(int)
d.to_pickle('clean_full.pkl')
