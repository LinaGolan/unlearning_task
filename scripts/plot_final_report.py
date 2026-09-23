"""Create compact scientific report plots from verified saved tables; no model work."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report/figures'
OUT.mkdir(parents=True, exist_ok=True)

def read(path, fallback):
    source = ROOT / path
    if not source.exists():
        source = ROOT / 'report/evidence' / fallback
    with source.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

layers = read('outputs/stage4/verified_2026-09-20/outputs/stage4/full/layer_scores.csv','layer_scores.csv')
curves = read('outputs/stage5/verified_2026-09-21/outputs/stage5/test/accuracy.csv','test_accuracy.csv')
plt.rcParams.update({'font.size': 10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axs = plt.subplots(1,3,figsize=(11,2.5),sharey=True)
for ax, key, title, color in zip(axs,('forget_score','retain_score','selectivity_score'),('Forget sensitivity F','Retain sensitivity R','Selectivity F minus R'),('#a54032','#356994','#247d69')):
    ax.bar([int(r['layer']) for r in layers],[float(r[key]) for r in layers],color=color)
    ax.axhline(0,color='#777777',linewidth=.7); ax.set_title(title); ax.set_xlabel('Layer'); ax.set_xticks([0,3,6,9,12,15])
axs[0].set_ylabel('Mean gate gradient'); fig.tight_layout()
fig.savefig(OUT/'localization.png',dpi=220); plt.close(fig)
methods = [('top_forget','Top forget','#a54032'),('top_selective','Top selective','#247d69'),('bottom_forget','Bottom forget','#8264a4'),('random_mean','Random mean','#356994')]
def values(role, method):
    base = next(float(r['accuracy']) for r in curves if r['role']==role and r['method']=='baseline')
    result = [base]
    for alpha in (.25,.5,.75,1):
        found = [float(r['accuracy']) for r in curves if r['role']==role and float(r['alpha'])==alpha and (r['method']==method or (method=='random_mean' and r['method'].startswith('random_') and r['method']!='random_mean'))]
        # Saved CSV contains individual pairs, not an aggregate row in this run.
        aggregate = [float(r['accuracy']) for r in curves if r['role']==role and float(r['alpha'])==alpha and r['method']==method]
        result.append(aggregate[0] if aggregate else sum(found)/len(found))
    return [v*100 for v in result]
fig, axs = plt.subplots(1,2,figsize=(10.5,3.3),sharey=True)
for ax, role in zip(axs,('forget','retain')):
    for method,label,color in methods:
        ax.plot([0,.25,.5,.75,1],values(role,method),'-o',label=label,color=color,markersize=4)
    ax.axvline(.75,color='#777',linestyle=':',linewidth=1)
    ax.set_title('WMDP Bio' if role=='forget' else 'General retain'); ax.set_xlabel('Strength α'); ax.set_ylim(15,61); ax.set_xticks([0,.25,.5,.75,1])
axs[0].set_ylabel('Accuracy (%)'); axs[1].legend(fontsize=9); fig.tight_layout()
fig.savefig(OUT/'strength.png',dpi=220); plt.close(fig)
fig, ax = plt.subplots(figsize=(8,4.2))
for method,label,color in methods:
    f,r=values('forget',method),values('retain',method)
    x,y=[r[0]-v for v in r],[f[0]-v for v in f]
    ax.plot(x,y,'-o',label=label,color=color,markersize=4)
    ax.scatter([x[3]],[y[3]],marker='s',s=65,color=color)
ax.axhline(0,color='#aaa',lw=.7); ax.axvline(0,color='#aaa',lw=.7)
ax.set_xlabel('Retain accuracy drop (percentage points)'); ax.set_ylabel('WMDP accuracy drop (percentage points)'); ax.legend(fontsize=9)
ax.set_title('Squares mark the frozen strength 0.75'); fig.tight_layout()
fig.savefig(OUT/'tradeoff.png',dpi=220); plt.close(fig)
print('Created three report figures from saved numeric evidence.')
