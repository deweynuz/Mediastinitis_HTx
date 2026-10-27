"""tables.py — writes every report table as markdown (tables_report.md), in the layout of the report.
Reads results.json (analysis.py). p values are unadjusted; bold = p < 0.05."""
import json, re
R=json.load(open('results.json')); n=R['n']
def sig(p): return p not in ('—','') and (p.startswith('<') or float(p)<0.05)
def row(c,bold=False): return '| '+' | '.join(f'**{x}**' if bold and x.strip() else x for x in c)+' |'
def head(c): return '| '+' | '.join(c)+' |\n|'+'---|'*len(c)
HTX_ONLY={'Corticosteroids','Anti-thymocyte globulin','Basiliximab','Tacrolimus','Ciclosporin','Mycophenolate mofetil','Postoperative plasma exchange','IV immunoglobulins',
          'Dilated','Ischaemic','Congenital','Amyloidosis','Valvular','Other / mixed','Graft ischaemic time, min','Primary graft dysfunction','Rejection'}
STD_ONLY={'CABG','Valve surgery','Major/combined procedure','Tamponade','Mechanical support','Other'}
SKIP_SECT={'*Immunosuppression (transplant recipients)','*Standard surgery: procedure','*Transplantation: underlying cardiomyopathy'}
out=[]
# Table 1 (both groups only)
t=[head(['Characteristic',f"Overall (n={n['all']})",f"Heart transplant (n={n['htx']})",f"Standard surgery (n={n['std']})",'p-value'])]
for r in R['table1']:
    if 'section' in r:
        if r['section'] in SKIP_SECT: continue
        s=r['section']; t.append(f"| {'*'+s[1:]+'*' if s.startswith('*') else '**'+s+'**'} |  |  |  |  |"); continue
    lab=r['label'].strip()
    if lab in HTX_ONLY or lab in STD_ONLY: continue
    t.append(row([lab,r['all'],r['htx'],r['std'],r['p']],sig(r['p'])))
out+=['## Table 1 — Characteristics of the study population','\n'.join(t)]
# Supplementary Table 1 (HTx only) / Supplementary Table 2 (standard only)
T1={r['label'].strip():r for r in R['table1'] if 'label' in r}
g=lambda labs,col: '\n'.join(row([l,T1[l][col]]) for l in labs)
IS=['Corticosteroids','Anti-thymocyte globulin','Basiliximab','Tacrolimus','Ciclosporin','Mycophenolate mofetil','Postoperative plasma exchange','IV immunoglobulins']
CM=['Dilated','Ischaemic','Congenital','Amyloidosis','Valvular','Other / mixed']; GR=['Graft ischaemic time, min','Primary graft dysfunction','Rejection']
out+=['## Supplementary Table 1 — Transplant-specific characteristics',head(['Characteristic',f"Heart transplant (n={n['htx']})"])+'\n| **Immunosuppression** |  |\n'+g(IS,'htx')+'\n| **Transplantation** |  |\n| *Underlying cardiomyopathy* |  |\n'+g(CM,'htx')+'\n| *Graft* |  |\n'+g(GR,'htx')]
out+=['## Supplementary Table 2 — Type of procedure (standard surgery)',head(['Procedure',f"Standard surgery (n={n['std']})"])+'\n'+g(sorted(STD_ONLY,key=['CABG','Valve surgery','Major/combined procedure','Tamponade','Mechanical support','Other'].index),'std')]
# Table 2
out+=['## Table 2 — Presentation at diagnosis',head(['Variable',f"Overall (n={n['all']})",f"Heart transplant (n={n['htx']})",f"Standard (n={n['std']})",'p-value'])+'\n'+'\n'.join(row([r['label'],r['all'],r['htx'],r['std'],r['p']]) for r in R['table2'])]
# Supplementary Table 3 (unadjusted p only; OR in Supplementary Figure 2)
t=[head(['Organism',f"HTx (n={n['htx']})",f"Standard (n={n['std']})",'p'])]
for r in R['table3']:
    lab=f"*{r['label']}*" if r['level']==2 else r['label']
    if sig(r['p']): t.append(row([lab,r['htx'],r['std'],r['p']],True))
    else: t.append(row([f'**{lab}**' if r['level']==0 else lab,r['htx'],r['std'],r['p']]))
out+=['## Supplementary Table 3 — Isolated organisms','\n'.join(t)]
out+=['## Supplementary Table 4 — Polymicrobial patterns',head(['Pattern',f"HTx (n={n['htx']})",f"Standard (n={n['std']})",'p'])+'\n'+'\n'.join(row([r['label'],r['htx'],r['std'],r['p']]) for r in R['poly'])]
out+=['## Supplementary Table 5 — Role of prior antibiotic exposure (age + SAPS II + antibiotics within 7 days)',head(['Endpoint','aOR [95% CI]','p'])+'\n'+'\n'.join(row([r['label'],r['aor'],r['p']]) for r in R['table5'])]
ow={r['var']:r for r in R['ow']}; t9=R['table9']; T1o={r['label']:r for r in R['table1'] if 'label' in r}
out+=['## Table 3 — Mortality',head(['Endpoint','HTx','Standard','aOR [95% CI]','p','OW OR [95% CI]','p'])+'\n'+'\n'.join(row([m['label'],T1o[m['label']]['htx'],T1o[m['label']]['std'],m['aor'],m['p'],ow[v]['or'],ow[v]['p']]) for m,v in zip(t9,['mort30','mort1y']))]
out+=['## Supplementary Table 6 — Missing data and handling strategy',head(['Variable','n missing','% missing','Strategy'])+'\n'+'\n'.join(row([r['label'],str(r['n']),r['pct'],r['strat']]) for r in R['missing'])]
open('tables_report.md','w').write('\n\n'.join(out)+'\n'); print('tables_report.md')
