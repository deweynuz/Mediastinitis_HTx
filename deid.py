"""deid.py — builds the publishable de-identified dataset + data dictionary from clean_full.pkl."""
import numpy as np, pandas as pd
d=pd.read_pickle('clean_full.pkl')
B='binary (0 = no, 1 = yes)'
DICT=[ # (variable, label, type/coding, unit, notes)
('study_id','Random study identifier','string','','Assigned at random; no link to hospital or CRF identifiers'),
('group','Study group','categorical (HTx / Standard)','','HTx = heart transplantation; Standard = other cardiac surgery'),
('htx','Heart-transplant status',B,'',''),
('age','Age at surgery','integer','years','Top-coded at 90 (patients aged ≥90 years recorded as 90)'),
('female','Female sex',B,'',''),
('bmi','Body mass index','continuous','kg/m²',''),
('euroscore','EuroSCORE II','continuous','%',''),
('adm_home','Admitted from home',B,'',''),('adm_icu','Admitted from ICU / intensive care',B,'',''),
('hta','Hypertension',B,'','Aberrant codes >1 recoded 1'),('diabetes','Diabetes',B,'',''),('insulin_dm','Insulin-treated diabetes',B,'',''),
('dyslipid','Dyslipidaemia',B,'',''),('smoking','Smoking (current or former)',B,'',''),('copd','COPD',B,'',''),('ckd','Chronic kidney disease',B,'',''),
('dialysis','Chronic dialysis before surgery',B,'','Aberrant code 3 recoded 1'),('prev_cardiac_surg','Previous cardiac surgery',B,'',''),
('colonisation','Preoperative colonisation',B,'',''),('preop_infection','Preoperative infection',B,'',''),
('atb_3m','Antibiotics within 3 months before surgery',B,'','Aberrant code 2 recoded 1'),('ppi','Proton pump inhibitor',B,'',''),
('preop_mcs','Preoperative mechanical circulatory support (any)',B,'',''),('preop_ecmo','Preoperative VA-ECMO',B,'',''),('preop_lvad_tah','Preoperative LVAD or total artificial heart',B,'',''),
('preop_pe','Preoperative plasma exchange',B,'',''),
('is_steroids','Corticosteroids',B,'','HTx only (missing in standard group)'),('is_atg','Anti-thymocyte globulin',B,'','HTx only'),('is_basiliximab','Basiliximab',B,'','HTx only'),
('is_tacrolimus','Tacrolimus',B,'','HTx only'),('is_ciclosporin','Ciclosporin',B,'','HTx only'),('is_mmf','Mycophenolate mofetil',B,'','HTx only'),
('is_ivig','IV immunoglobulins',B,'','HTx only; derived from free-text "other immunosuppression"'),('postop_pe','Postoperative plasma exchange',B,'','HTx only'),('rejection','Graft rejection',B,'','HTx only'),
('proc_cabg','Procedure: CABG',B,'','Standard group only'),('proc_valve','Procedure: valve surgery',B,'','Standard group only'),('proc_major','Procedure: major/combined',B,'','Standard group only'),
('proc_tamponade','Procedure: tamponade',B,'','Standard group only'),('proc_mcs','Procedure: mechanical support implantation',B,'','Standard group only'),('proc_other','Procedure: other',B,'','Standard group only'),
('cm_dilated','Cardiomyopathy: dilated',B,'','HTx only'),('cm_ischaemic','Cardiomyopathy: ischaemic',B,'','HTx only'),('cm_congenital','Cardiomyopathy: congenital',B,'','HTx only'),
('cm_amyloid','Cardiomyopathy: amyloid',B,'','HTx only'),('cm_valvular','Cardiomyopathy: valvular',B,'','HTx only'),('cm_other','Cardiomyopathy: other / mixed',B,'','HTx only'),
('ischaemic_time','Graft ischaemic time','continuous','min','HTx only'),('pgd','Primary graft dysfunction',B,'','HTx only; CRF dictionary 1/2/3 (LV/RV/biventricular) but column recorded 0/1 — coding to be confirmed'),
('cpb_min','Cardiopulmonary bypass duration','continuous','min',''),('rbc_units','Intraoperative red blood cell units','count','units',''),
('ffp_units','Intraoperative fresh frozen plasma units','count','units',''),('plt_units','Intraoperative platelet units','count','units',''),
('norepi_intraop','Intraoperative norepinephrine',B,'',''),('ecmo_intraop','Intraoperative VA-ECMO',B,'',''),
('saps2','SAPS II (24 h after ICU admission)','integer','points',''),('sofa_max','Maximum SOFA score before diagnosis','integer','points','34% missing'),
('ecmo_postop','Postoperative VA-ECMO',B,'',''),('iabp','Intra-aortic balloon pump',B,'',''),('haem_shock','Haemorrhagic shock in ICU',B,'',''),
('septic_shock_icu','Septic shock in ICU before diagnosis',B,'',''),('aki','Acute kidney injury (KDIGO ≥1)',B,'','KDIGO stage not recorded in HTx; presence only'),('krt','Kidney replacement therapy in ICU',B,'',''),
('surg_to_dx_days','Interval surgery to mediastinitis diagnosis','integer','days','CRF interval; computed from dates when missing'),
('sternal_instability','Clinical sign at diagnosis: sternal instability',B,'',''),('fever','Clinical sign at diagnosis: fever',B,'',''),
('purulent_discharge','Clinical sign at diagnosis: purulent discharge',B,'',''),('wound_inflammation','Clinical sign at diagnosis: wound inflammation',B,'',''),
('wbc','White blood cell count at diagnosis','continuous','×10⁹/L','Values >100 assumed entered per mm³ and divided by 1000; 41% missing'),
('blood_cultures_pos','Positive blood cultures',B,'',''),('ct_collection','CT: collection/abscess',B,'',''),('ct_dehiscence','CT: sternal dehiscence',B,'',''),
('dx_surgical_sample','Diagnosis on surgical sample',B,'',''),('dx_sternal_puncture','Diagnosis on sternal puncture or drain',B,'',''),
('other_infection_before_dx','Other infection before diagnosis',B,'',''),('atb_7d_before_dx','Antibiotics within 7 days before diagnosis',B,'',''),
('septic_shock_dx','Septic shock at diagnosis',B,'',''),('norepi_dx','Norepinephrine at diagnosis',B,'',''),('mv_dx','Mechanical ventilation at diagnosis',B,'',''),
('krt_dx','Kidney replacement therapy at diagnosis',B,'',''),('ecmo_dx','VA-ECMO at diagnosis',B,'',''),
('icu_los','ICU length of stay','integer','days',''),('mort30','30-day mortality (CRF column)',B,'','Intended reference: index surgery; coding to be verified (may be from diagnosis)'),
('mort1y','1-year mortality (CRF column)',B,'','Same caveat as mort30'),('death_followup','Death during follow-up (any cause)',B,'','Derived from cause-of-death codes >0'),
('gpc','Gram-positive cocci (any)',B,'','Microbiology derived from structured CRF columns + free text; all organisms isolated at diagnosis'),
('s_aureus','Staphylococcus aureus',B,'',''),('mssa','Methicillin-susceptible S. aureus',B,'',''),('mrsa','Methicillin-resistant S. aureus',B,'',''),
('cons','Coagulase-negative staphylococci',B,'',''),('cons_mr','Methicillin-resistant CoNS (when reported)',B,'',''),
('enterococci','Enterococci',B,'',''),('e_faecalis','Enterococcus faecalis',B,'',''),('e_faecium','Enterococcus faecium',B,'',''),('e_gallinarum','Enterococcus gallinarum',B,'',''),
('streptococci','Streptococci',B,'',''),('enterobacterales','Enterobacterales (fermenting GNB)',B,'',''),('entero_g12','Enterobacterales groups 1–2',B,'',''),
('e_coli','Escherichia coli',B,'',''),('k_pneumoniae','Klebsiella pneumoniae',B,'',''),('k_oxytoca','Klebsiella oxytoca',B,'',''),('proteus','Proteus spp.',B,'',''),('c_koseri','Citrobacter koseri',B,'',''),
('entero_g3','Enterobacterales group 3 (chromosomal AmpC)',B,'',''),('e_cloacae','Enterobacter cloacae complex',B,'',''),('k_aerogenes','Klebsiella (Enterobacter) aerogenes',B,'',''),
('c_freundii','Citrobacter freundii',B,'',''),('serratia','Serratia marcescens',B,'',''),('morganella','Morganella morganii',B,'',''),('entero_other','Other Enterobacterales (Pantoea, Hafnia, C. amalonaticus)',B,'',''),
('nf_gnb','Non-fermenting GNB',B,'',''),('p_aeruginosa','Pseudomonas aeruginosa',B,'',''),('s_maltophilia','Stenotrophomonas maltophilia',B,'',''),('acinetobacter','Acinetobacter spp.',B,'',''),
('other_bacteria','Other bacteria',B,'',''),('corynebacterium','Corynebacterium spp.',B,'',''),('cutibacterium_anaerobes','Cutibacterium / anaerobes',B,'',''),('other_misc','Haemophilus, Neisseria or Ureaplasma',B,'',''),
('fungal','Fungal infection (any fungus isolated)',B,'','Primary endpoint; fungal flag OR fungal species code OR free text'),
('c_albicans','Candida albicans',B,'',''),('c_glabrata','Candida glabrata',B,'',''),('c_parapsilosis','Candida parapsilosis',B,'',''),('c_lusitaniae','Candida lusitaniae',B,'',''),
('c_orthopsilosis','Candida orthopsilosis',B,'',''),('c_tropicalis','Candida tropicalis',B,'',''),('trichosporon','Trichosporon spp.',B,'',''),('aspergillus','Aspergillus spp.',B,'',''),('candida_unspec','Candida sp. (species not specified)',B,'',''),
('esbl','ESBL-producing Enterobacterales',B,'','Primary endpoint'),('mdr_pa','Multidrug-resistant P. aeruginosa',B,'',''),('mdr_any','Any multidrug-resistant organism (ESBL-E, MRSA or MDR P. aeruginosa)',B,'',''),
('n_bact','Number of bacterial species isolated','count','',''),('n_fung','Number of fungal species isolated','count','',''),
('poly_2species','Polymicrobial: ≥2 species',B,'',''),('poly_2bact','≥2 bacterial species',B,'',''),('mixed_bact_fungus','Mixed bacterial + fungal infection',B,'',''),
('fungus_only','Fungus only',B,'',''),('no_organism','No organism identified',B,'',''),
]
cols=[v for v,*_ in DICT if v!='study_id']
out=d[cols].copy(); out['age']=out.age.clip(upper=90)
rng=np.random.default_rng(20261026); out=out.sample(frac=1,random_state=20261026).reset_index(drop=True)
out.insert(0,'study_id',[f'MED-{i:03d}' for i in range(1,len(out)+1)])
for c in out.columns:
    if c not in ('study_id','group') and (out[c].dropna().astype(float)%1==0).all(): out[c]=out[c].astype('Int64')
out.to_csv('mediastinitis_deidentified.csv',index=False)
dd=pd.DataFrame(DICT,columns=['variable','label','type_coding','unit','notes'])
dd.to_csv('mediastinitis_data_dictionary.csv',index=False)
print(out.shape, dd.shape)
