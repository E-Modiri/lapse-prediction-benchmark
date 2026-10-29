# ════════════════════════════════════════════════════════
# ALL EIGHT PAPER FIGURES — self-contained: the final reported values are typed in, so no data files are needed.
# Usage: python scripts/make_figures.py   (writes PNG and PDF to ./paper_figures)
# Saves PNG (300 dpi) and PDF of every figure to a folder "paper_figures" next to the notebook.
# ════════════════════════════════════════════════════════
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score

FIG_DIR = 'paper_figures'; os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#6b6b6b',
                     'axes.grid': True, 'grid.color': '#e6e6e3', 'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'legend.frameon': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
INK, MUTED = '#2b2b2a', '#6b6b6b'
FAMILY = {'LR': 'Traditional', 'RF': 'Traditional', 'XGB': 'Traditional', 'SVM': 'Traditional', 'LSTM': 'Recurrent',
          'GRU': 'Recurrent', 'GRUD': 'Recurrent', 'TCN': 'Convolutional', 'GRUTCN': 'Convolutional', 'TFM': 'Attention',
          'SSLGRU': 'Self-supervised', 'SSLGRUD': 'Self-supervised'}
FAM_COLOR = {'Traditional': '#2a78d6', 'Recurrent': '#eb6834', 'Convolutional': '#1baf7a', 'Attention': '#eda100',
             'Self-supervised': '#e87ba4'}
FAM_MARKER = {'Traditional': 'o', 'Recurrent': 's', 'Convolutional': '^', 'Attention': 'D', 'Self-supervised': 'P'}
LABEL = {'LR': 'Logistic regression', 'RF': 'Random forest', 'XGB': 'XGBoost', 'SVM': 'SVM', 'LSTM': 'LSTM', 'GRU': 'GRU',
         'GRUD': 'GRU-D', 'TCN': 'TCN', 'GRUTCN': 'GRU-TCN', 'TFM': 'Transformer', 'SSLGRU': 'SSL-GRU', 'SSLGRUD': 'SSL-GRU-D'}
TASK_NAME = {'classify': 'Same-entry classification', 'next': 'Next-entry prediction'}

def save(fig, name):
    for ext in ('png', 'pdf'): fig.savefig(os.path.join(FIG_DIR, f'{name}.{ext}'))
    plt.close(fig)

def fam_legend(ax, loc):
    ax.legend(handles=[Line2D([], [], marker=FAM_MARKER[f], color=FAM_COLOR[f], ls='', ms=6, label=f) for f in FAM_COLOR],
              loc=loc, fontsize=8)

# ── final results (Table 3: AUC [95% CI]; per-participant medians from Supplement S4) ──
T3 = pd.DataFrame([
    # config, cls, cls_lo, cls_hi, next, next_lo, next_hi
    ('GRUD_exp', .962, .947, .977, .947, .927, .966), ('TFM_exp', .961, .946, .976, .945, .924, .966),
    ('TFM_orig', .959, .941, .976, .940, .914, .965), ('TCN_exp', .954, .936, .972, .939, .917, .960),
    ('GRU_exp', .959, .943, .975, .938, .915, .960), ('LSTM_exp', .958, .941, .975, .935, .911, .959),
    ('SSLGRUD_exp', .956, .939, .973, .934, .907, .960), ('GRU_orig', .955, .936, .974, .934, .907, .961),
    ('LSTM_orig', .953, .933, .973, .933, .907, .960), ('TCN_orig', .950, .930, .971, .933, .909, .958),
    ('GRUTCN_exp', .955, .936, .973, .932, .903, .961), ('GRUD_orig', .959, .943, .975, .931, .902, .960),
    ('SSLGRU_exp', .954, .936, .971, .929, .898, .960), ('SSLGRU_orig', .948, .923, .972, .928, .896, .961),
    ('GRUTCN_orig', .952, .931, .973, .926, .895, .957), ('RF_exp', .951, .930, .971, .924, .891, .957),
    ('SSLGRUD_orig', .939, .910, .968, .922, .887, .957), ('XGB_exp', .952, .930, .974, .918, .883, .953),
    ('LR_exp', .945, .921, .970, .913, .873, .952), ('SVM_exp', .934, .904, .964, .905, .862, .948),
    ('RF_orig', .919, .883, .954, .876, .820, .932), ('XGB_orig', .921, .885, .957, .873, .814, .932),
    ('LR_orig', .904, .858, .950, .856, .784, .927), ('SVM_orig', .878, .816, .940, .848, .775, .922)],
    columns=['config', 'classify', 'classify_lo', 'classify_hi', 'next', 'next_lo', 'next_hi'])
T3[['arch', 'feats']] = T3.config.str.rsplit('_', n=1, expand=True)
PP_NEXT_SF = {'LR': .583, 'RF': .584, 'XGB': .572, 'GRU': .551, 'GRUD': .565}      # expanded features

# ════════ FIGURE 1 — leakage-free protocol ════════
fig, ax = plt.subplots(figsize=(8.6, 3.9)); ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
TRAIN, VAL, TEST, BOX = '#cfe0f5', '#fbe3c6', '#f6cfdc', '#f2f2ef'
def box(x, y, w, h, fc, text='', fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc, ec=MUTED, lw=0.8))
    if text: ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=INK)
ax.text(0.1, 5.65, 'Outer loop: 10 folds of participants (each fold is the test set once)', fontsize=9, color=INK, fontweight='bold')
for k in range(10): box(0.1 + k * 0.97, 4.75, 0.9, 0.6, TEST if k == 3 else TRAIN, 'test' if k == 3 else 'train', fs=7.5)
ax.text(0.1, 4.25, 'Inner split of the training participants (test participants never used)', fontsize=9, color=INK, fontweight='bold')
box(0.1, 3.35, 7.0, 0.6, TRAIN, '80%: fit candidate models'); box(7.2, 3.35, 2.7, 0.6, VAL, '20%: choose settings\nand epochs')
ax.text(0.1, 2.75, 'Fitted on training participants only', fontsize=9, color=INK, fontweight='bold')
for i, t in enumerate(['Standardisation\n(means, SDs)', 'Hyperparameters,\nearly stopping', 'Self-supervised\npretraining',
                       'Features: current and\nearlier entries only']):
    box(0.1 + i * 2.47, 1.6, 2.3, 0.9, BOX, t)
ax.text(0.1, 1.05, 'Then: refit on the full training fold → predict the test fold once → AUC per fold → mean and 95% CI over 10 folds',
        fontsize=8.5, color=INK)
for lab, fc, x in [('Training participants', TRAIN, 0.1), ('Validation participants', VAL, 2.6), ('Test participants', TEST, 5.2)]:
    box(x, 0.2, 0.35, 0.3, fc); ax.text(x + 0.45, 0.35, lab, va='center', fontsize=8, color=INK)
save(fig, 'fig1_leakage_free_protocol')

# ════════ FIGURE 2 — all 24 configurations with 95% CIs ════════
order = T3.sort_values('next', ascending=False).config.tolist()
fig, axes = plt.subplots(1, 2, figsize=(9.5, 7), sharey=True)
for ax, task in zip(axes, ['classify', 'next']):
    d = T3.set_index('config').loc[order]
    for i, (name, r) in enumerate(d.iterrows()):
        f = FAMILY[r.arch]; filled = r.feats == 'exp'
        ax.plot([r[f'{task}_lo'], r[f'{task}_hi']], [i, i], color=FAM_COLOR[f], lw=1.4, alpha=0.7)
        ax.scatter(r[task], i, marker=FAM_MARKER[f], s=44, zorder=3, facecolor=FAM_COLOR[f] if filled else 'white',
                   edgecolor='white' if filled else FAM_COLOR[f], lw=0.8 if filled else 1.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{LABEL[n.rsplit('_', 1)[0]]} ({'exp' if n.endswith('exp') else 'orig'})" for n in order])
    ax.set_xlabel('AUC (mean of 10 participant-level folds, 95% CI)'); ax.grid(axis='y', visible=False)
    ax.set_xlim(0.76, 1.0); ax.set_title(TASK_NAME[task], loc='left')
axes[0].invert_yaxis(); fam_legend(axes[1], 'upper left')
plt.tight_layout(); save(fig, 'fig2_all_configurations')

DELTA = {'classify': dict(LR=.042, SVM=.056, XGB=.031, RF=.032, LSTM=.005, GRU=.003, GRUD=.003, TCN=.004, GRUTCN=.003, TFM=.002, SSLGRU=.006, SSLGRUD=.018),   # Table 7 paired differences
         'next': dict(LR=.057, SVM=.056, XGB=.046, RF=.048, LSTM=.002, GRU=.004, GRUD=.015, TCN=.005, GRUTCN=.006, TFM=.005, SSLGRU=.001, SSLGRUD=.012)}

# ════════ FIGURE 3 — original vs expanded features (pooled AUC) ════════
ORDER = ['LR', 'SVM', 'XGB', 'RF', 'LSTM', 'GRU', 'GRUD', 'TCN', 'GRUTCN', 'TFM', 'SSLGRU', 'SSLGRUD']
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
for ax, task in zip(axes, ['classify', 'next']):
    d = T3.set_index('config')[task]
    for i, a in enumerate(ORDER):
        o, e = d[f'{a}_orig'], d[f'{a}_exp']; f = FAMILY[a]
        ax.plot([o, e], [i, i], color=FAM_COLOR[f], lw=2, alpha=0.6, solid_capstyle='round')
        ax.scatter(o, i, marker=FAM_MARKER[f], s=46, facecolor='white', edgecolor=FAM_COLOR[f], lw=1.6, zorder=3)
        ax.scatter(e, i, marker=FAM_MARKER[f], s=46, color=FAM_COLOR[f], edgecolor='white', lw=0.8, zorder=4)
        ax.text(max(o, e) + 0.003, i, f"{DELTA[task][a]:+.3f}", va='center', fontsize=7, color=MUTED)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([LABEL[a] for a in ORDER])
    ax.set_xlabel('AUC (mean of 10 folds)'); ax.grid(axis='y', visible=False)
    ax.set_xlim(d.min() - 0.01, d.max() + 0.03); ax.set_title(TASK_NAME[task], loc='left')
    for k in (3.5, 6.5, 8.5, 9.5): ax.axhline(k, color='#e6e6e3', lw=0.6)
axes[0].invert_yaxis()
fig.legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', mec=MUTED, ms=7, label='Original features (33)'),
                    Line2D([], [], marker='o', ls='', mfc=MUTED, mec='white', ms=7, label='Expanded features (46)')],
           loc='lower center', ncol=2, fontsize=8)
plt.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, 'fig3_feature_engineering')

# ════════ FIGURE 4 — survival ════════
SURV = pd.DataFrame({'regime': ['From scratch', 'Self-supervised, fine-tuned', 'Self-supervised, frozen'],
                     'c_mean': [0.778, 0.776, 0.742], 'c_sd': [0.091, 0.100, 0.087],
                     '1 h': [0.889, 0.891, 0.851], '6 h': [0.912, 0.913, 0.890],
                     '24 h': [0.909, 0.909, 0.878], '72 h': [0.878, 0.873, 0.840]})
RC = {'From scratch': ('#2a78d6', 'o'), 'Self-supervised, fine-tuned': ('#e87ba4', 'P'), 'Self-supervised, frozen': ('#6b6b6b', 's')}
H = ['1 h', '6 h', '24 h', '72 h']
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={'width_ratios': [1, 1.5]})
for i, r in SURV.iterrows():
    c, mk = RC[r.regime]
    a1.errorbar(r.c_mean, i, xerr=r.c_sd, fmt=mk, color=c, mfc=c, mec='white', ms=8, capsize=3, lw=1.4)
    a1.text(r.c_mean, i - 0.28, f"{r.c_mean:.3f}", ha='center', fontsize=7.5, color=INK)
    a2.plot(range(4), [r[h] for h in H], color=c, marker=mk, ms=7, lw=2, label=r.regime)
a1.set_yticks(range(3)); a1.set_yticklabels(SURV.regime); a1.invert_yaxis(); a1.set_ylim(2.6, -0.6)
a1.set_xlabel('Concordance index (mean ± SD over folds)'); a1.set_title('A  Ranking by time to next lapse', loc='left')
a1.grid(axis='y', visible=False)
a2.set_xticks(range(4)); a2.set_xticklabels(H); a2.set_xlabel('Prediction horizon'); a2.set_ylabel('AUC')
a2.set_title('B  Lapse within each horizon', loc='left'); a2.legend(loc='lower center', fontsize=8)
plt.tight_layout(); save(fig, 'fig4_survival')

# ════════ FIGURE 5 — five models in all three studies (next-entry, expanded features) ════════
ALL5 = pd.DataFrame([('Smoke Free', a, T3.set_index('config').loc[f'{a}_exp', 'next']) for a in ['LR', 'RF', 'XGB', 'GRU', 'GRUD']] +
                    [('EMASENS', 'LR', .869), ('EMASENS', 'RF', .867), ('EMASENS', 'XGB', .805), ('EMASENS', 'GRU', .821),
                     ('EMASENS', 'GRUD', .820), ('Alcohol', 'LR', .884), ('Alcohol', 'RF', .875), ('Alcohol', 'XGB', .832),
                     ('Alcohol', 'GRU', .886), ('Alcohol', 'GRUD', .890)], columns=['study', 'arch', 'auc'])
archs = ['LR', 'RF', 'XGB', 'GRU', 'GRUD']
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True, sharey=True)
for ax, st in zip(axes, ['Smoke Free', 'EMASENS', 'Alcohol']):
    d = ALL5[ALL5.study == st].set_index('arch').reindex(archs)
    for i, a in enumerate(archs):
        f = FAMILY[a]; v = d.loc[a, 'auc']
        ax.plot([0, v], [i, i], color='#e1e1de', lw=1, zorder=1)
        ax.scatter(v, i, s=50, marker=FAM_MARKER[f], color=FAM_COLOR[f], edgecolor='white', lw=0.8, zorder=3)
        ax.text(v + 0.006, i, f"{v:.3f}", va='center', fontsize=7.5, color=INK)
    ax.set_yticks(range(5)); ax.set_yticklabels([LABEL[a] for a in archs]); ax.invert_yaxis()
    ax.set_title(st, loc='left'); ax.set_xlabel('AUC'); ax.grid(axis='y', visible=False)
axes[0].set_xlim(0.75, 1.0)
plt.tight_layout(); save(fig, 'fig5_three_studies')

# ════════ FIGURE 6 — transfer to small new studies (pooled AUC) ════════
P9D = pd.DataFrame([('EM', 3, .620, .834, .730), ('EM', 5, .750, .835, .793), ('EM', 10, .777, .838, .819), ('EM', 15, .823, .839, .830),
                    ('AL', 3, .683, .851, .771), ('AL', 5, .760, .852, .834), ('AL', 10, .840, .855, .853),
                    ('AL', 20, .853, .858, .858), ('AL', 40, .860, .861, .862)],
                   columns=['target', 'n', 'target_only', 'pool2', 'aug2'])
SHOW = [('target_only', 'Target study only', '#2a78d6', 'o'), ('pool2', 'Naive pooling with the other study', '#eb6834', 's'),
        ('aug2', 'Partial pooling with the other study', '#1baf7a', '^')]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, (tgt, nm) in zip(axes, [('EM', 'EMASENS'), ('AL', 'Alcohol')]):
    d = P9D[P9D.target == tgt].sort_values('n')
    for col, lab, colr, mk in SHOW: ax.plot(d.n, d[col], color=colr, marker=mk, ms=6, lw=2, label=lab)
    ax.set_xscale('log'); ax.minorticks_off(); ax.set_xticks(d.n); ax.set_xticklabels(d.n)
    ax.set_xlabel('Target-study participants used for training (n)'); ax.set_title(nm, loc='left')
axes[0].set_ylabel('Pooled AUC (fold mean)')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=8)
plt.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, 'fig6_small_study_transfer')

# ════════ FIGURE 7 — why pooled AUC can be high while per-participant AUC is not (simulated) ════════
rng = np.random.default_rng(3)
people = {'Person A\n(never lapses)': (rng.normal(0.08, 0.03, 14), np.zeros(14)),
          'Person B\n(lapses often)': (rng.normal(0.62, 0.05, 14), (rng.random(14) < 0.45).astype(int)),
          'Person C\n(never lapses)': (rng.normal(0.15, 0.04, 14), np.zeros(14))}
fig, ax = plt.subplots(figsize=(6.2, 3.0)); allp, ally = [], []
for i, (lab, (p, y)) in enumerate(people.items()):
    x = i + rng.uniform(-0.18, 0.18, len(p))
    ax.scatter(x[y == 0], p[y == 0], s=36, facecolor='white', edgecolor=MUTED, lw=1.2, zorder=3)
    ax.scatter(x[y == 1], p[y == 1], s=36, color='#eb6834', edgecolor='white', lw=0.8, zorder=4)
    allp += list(p); ally += list(y)
pb, yb = people['Person B\n(lapses often)']
ax.set_xticks(range(3)); ax.set_xticklabels(list(people)); ax.set_ylabel('Predicted lapse risk'); ax.set_ylim(-0.05, 0.85)
ax.grid(axis='x', visible=False)
ax.text(0.98, 0.50, f"Pooled AUC, all entries = {roc_auc_score(ally, allp):.2f}\nWithin Person B = {roc_auc_score(yb, pb):.2f}",
        transform=ax.transAxes, ha='right', va='center', fontsize=8.5, color=INK)
ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='#eb6834', mec='white', ms=7, label='Entry followed by a lapse'),
                   Line2D([], [], marker='o', ls='', mfc='white', mec=MUTED, ms=7, label='Entry not followed by a lapse')],
          loc='upper left', fontsize=8)
save(fig, 'fig7_pooled_vs_within_illustration')

# ════════ FIGURE 8 — pooled vs median per-participant AUC, three studies (next-entry, expanded features) ════════
PP8 = pd.DataFrame([('Smoke Free', a, T3.set_index('config').loc[f'{a}_exp', 'next'], PP_NEXT_SF[a]) for a in archs] +
                   [('EMASENS', 'LR', .869, .653), ('EMASENS', 'RF', .867, .642), ('EMASENS', 'XGB', .805, .621),
                    ('EMASENS', 'GRU', .821, .621), ('EMASENS', 'GRUD', .820, .615),
                    ('Alcohol', 'LR', .884, .698), ('Alcohol', 'RF', .875, .678), ('Alcohol', 'XGB', .832, .617),
                    ('Alcohol', 'GRU', .886, .686), ('Alcohol', 'GRUD', .890, .703)], columns=['study', 'arch', 'pooled', 'pp'])
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, sharey=True)
for ax, st in zip(axes, ['Smoke Free', 'EMASENS', 'Alcohol']):
    d = PP8[PP8.study == st].set_index('arch').reindex(archs)
    for i, a in enumerate(archs):
        c = FAM_COLOR[FAMILY[a]]
        ax.plot([d.loc[a, 'pp'], d.loc[a, 'pooled']], [i, i], color='#cfcfcb', lw=3, solid_capstyle='round', zorder=1)
        ax.scatter(d.loc[a, 'pp'], i, s=160, marker='|', color=c, lw=2.5, zorder=3)
        ax.scatter(d.loc[a, 'pooled'], i, s=46, marker=FAM_MARKER[FAMILY[a]], color=c, edgecolor='white', lw=0.8, zorder=4)
    ax.set_yticks(range(5)); ax.set_yticklabels([LABEL[a] for a in archs]); ax.invert_yaxis()
    ax.set_title(st, loc='left'); ax.set_xlabel('AUC'); ax.grid(axis='y', visible=False); ax.set_xlim(0.5, 1.0)
fig.legend(handles=[Line2D([], [], marker='|', ls='', color=MUTED, ms=12, mew=2.5, label='Per-participant median'),
                    Line2D([], [], marker='o', ls='', mfc=MUTED, mec='white', ms=7, label='Pooled')],
           loc='lower center', ncol=2, fontsize=8)
plt.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, 'fig8_pooled_vs_per_participant')

print("✅ All 8 figures saved to", os.path.abspath(FIG_DIR))
