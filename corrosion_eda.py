import sys
sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size':        13,
    'axes.titlesize':   14,
    'axes.labelsize':   13,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
    'figure.titlesize': 16,
})
PALETTE = sns.color_palette("husl", 15)
ACCENT  = '#2c7bb6'
TARGET_COLOR = '#d7191c'

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def _find_excel(folder: Path) -> Path:
    for name in [
        "Synthetic_Corrosion_Data_10000_Samples.xlsx",
        "Synthetic_Corrosion_Data_10000_Samples (1).xlsx",
    ]:
        p = folder / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"Excel dataset not found in {folder}")


print("=" * 70)
print("CORROSION EDA - Exploratory Data Analysis")
print("=" * 70)

script_dir = Path(__file__).resolve().parent
df = pd.read_excel(_find_excel(script_dir), engine="openpyxl")
print(f"\n✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

FEATURES = [
    'SiO2_Content_%', 'Epoxy_Type_Code', 'Curing_Agent_Ratio',
    'Coating_Thickness_um', 'Coating_Age_days', 'Surface_Prep_Quality_0_100',
    'Temperature_C', 'Humidity_percent', 'pH', 'Chloride_Concentration_ppm',
    'Dissolved_Oxygen_ppm', 'Sulfide_H2S_ppm', 'Applied_Potential_mV_vs_SCE',
    'Exposure_Time_hours', 'Substrate_Type_Code'
]
TARGET = 'Corrosion_Rate_uA_cm2'
CATEGORICAL = ['Epoxy_Type_Code', 'Substrate_Type_Code']
CONTINUOUS  = [f for f in FEATURES if f not in CATEGORICAL]

# Friendly display names for axis labels
LABELS = {
    'SiO2_Content_%':             'SiO₂ Content (%)',
    'Epoxy_Type_Code':            'Epoxy Type',
    'Curing_Agent_Ratio':         'Curing Agent Ratio',
    'Coating_Thickness_um':       'Coating Thickness (μm)',
    'Coating_Age_days':           'Coating Age (days)',
    'Surface_Prep_Quality_0_100': 'Surface Prep Quality',
    'Temperature_C':              'Temperature (°C)',
    'Humidity_percent':           'Humidity (%)',
    'pH':                         'pH',
    'Chloride_Concentration_ppm': 'Chloride Conc. (ppm)',
    'Dissolved_Oxygen_ppm':       'Dissolved O₂ (ppm)',
    'Sulfide_H2S_ppm':            'H₂S Sulfide (ppm)',
    'Applied_Potential_mV_vs_SCE':'Applied Potential (mV)',
    'Exposure_Time_hours':        'Exposure Time (hrs)',
    'Substrate_Type_Code':        'Substrate Type',
    TARGET:                       'Corrosion Rate (μA/cm²)',
}

# ---------------------------------------------------------------------------
# 1. Dataset overview
# ---------------------------------------------------------------------------
print("\n--- Dataset Overview ---")
print(f"Shape         : {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates    : {df.duplicated().sum()}")
print(f"\nData types:\n{df[FEATURES + [TARGET]].dtypes.to_string()}")

# ---------------------------------------------------------------------------
# 2. Descriptive statistics → CSV
# ---------------------------------------------------------------------------
desc = df[FEATURES + [TARGET]].describe(percentiles=[.05, .25, .5, .75, .95]).T
desc['skewness'] = df[FEATURES + [TARGET]].skew()
desc['kurtosis'] = df[FEATURES + [TARGET]].kurt()
desc.to_csv('EDA_Summary_Statistics.csv')
print("\n✓ Saved: EDA_Summary_Statistics.csv")
print(f"\nDescriptive statistics (continuous features + target):\n")
print(desc[['mean','std','min','5%','50%','95%','max','skewness','kurtosis']].to_string())

# ===========================================================================
# FIGURE 1: Target Variable Distribution
# ===========================================================================
print("\n[EDA 1/7] Target variable distribution...")
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Target Variable: Corrosion Rate Distribution', fontsize=17, fontweight='bold', y=1.02)

y = df[TARGET]

# Histogram + KDE
axes[0].hist(y, bins=60, color=TARGET_COLOR, alpha=0.7, edgecolor='white', linewidth=0.4, density=True)
y_kde = np.linspace(y.min(), y.max(), 300)
kde = stats.gaussian_kde(y)
axes[0].plot(y_kde, kde(y_kde), color='black', linewidth=2.5)
axes[0].axvline(y.mean(),   color='blue',  linestyle='--', linewidth=2, label=f'Mean = {y.mean():.1f}')
axes[0].axvline(y.median(), color='green', linestyle='-.', linewidth=2, label=f'Median = {y.median():.1f}')
axes[0].set_xlabel(LABELS[TARGET])
axes[0].set_ylabel('Density')
axes[0].set_title('Histogram & KDE', fontweight='bold')
axes[0].legend()

# Boxplot
axes[1].boxplot(y, vert=True, patch_artist=True,
                boxprops=dict(facecolor=TARGET_COLOR, alpha=0.6),
                medianprops=dict(color='black', linewidth=2.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.4))
axes[1].set_ylabel(LABELS[TARGET])
axes[1].set_title('Boxplot', fontweight='bold')
axes[1].set_xticks([])

# Q-Q plot
(osm, osr), (slope, intercept, r) = stats.probplot(y, dist='norm')
axes[2].scatter(osm, osr, s=6, alpha=0.4, color=TARGET_COLOR)
axes[2].plot(osm, slope * np.array(osm) + intercept, 'k-', linewidth=2)
axes[2].set_xlabel('Theoretical Quantiles')
axes[2].set_ylabel('Sample Quantiles')
axes[2].set_title(f'Q-Q Plot  (Skew={y.skew():.2f}, Kurt={y.kurt():.2f})', fontweight='bold')

plt.tight_layout(pad=2.5)
plt.savefig('EDA_01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: EDA_01_target_distribution.png")

# ===========================================================================
# FIGURE 2: Feature Distributions (continuous only)
# ===========================================================================
print("[EDA 2/7] Feature distributions...")
ncols = 3
nrows = int(np.ceil(len(CONTINUOUS) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 5))
fig.suptitle('Continuous Feature Distributions', fontsize=17, fontweight='bold')
axes_flat = axes.flatten()

for i, feat in enumerate(CONTINUOUS):
    ax = axes_flat[i]
    ax.hist(df[feat], bins=50, color=PALETTE[i % len(PALETTE)],
            alpha=0.75, edgecolor='white', linewidth=0.3, density=True)
    ax.axvline(df[feat].mean(), color='black', linestyle='--', linewidth=1.8)
    ax.set_title(LABELS[feat], fontweight='bold')
    ax.set_ylabel('Density')

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
plt.savefig('EDA_02_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: EDA_02_feature_distributions.png")

# ===========================================================================
# FIGURE 3: Correlation Heatmap
# ===========================================================================
print("[EDA 3/7] Correlation heatmap...")
corr_cols = CONTINUOUS + [TARGET]
corr_matrix = df[corr_cols].corr()

nice_labels = [LABELS.get(c, c) for c in corr_cols]

fig, ax = plt.subplots(figsize=(18, 15))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(
    corr_matrix,
    annot=True, fmt='.2f', annot_kws={'size': 11},
    cmap='RdYlGn', center=0, vmin=-1, vmax=1,
    linewidths=0.8, linecolor='white',
    xticklabels=nice_labels, yticklabels=nice_labels,
    ax=ax
)
ax.set_title('Pearson Correlation Matrix — All Continuous Features & Target',
             fontsize=16, fontweight='bold', pad=18)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout(pad=2.5)
plt.savefig('EDA_03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: EDA_03_correlation_heatmap.png")

# ===========================================================================
# FIGURE 4: Feature–Target Correlation Bar Chart
# ===========================================================================
print("[EDA 4/7] Feature-target correlations...")
target_corr = df[corr_cols].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)

colors_bar = [TARGET_COLOR if v > 0 else '#1a9641' for v in target_corr.values]

fig, ax = plt.subplots(figsize=(13, 8))
bars = ax.barh(
    [LABELS.get(f, f) for f in target_corr.index],
    target_corr.values,
    color=colors_bar, edgecolor='black', linewidth=0.8, alpha=0.85,
    height=0.6
)
ax.axvline(0, color='black', linewidth=1.0)
ax.set_xlabel('Pearson Correlation Coefficient (r)', fontweight='bold')
ax.set_title('Feature Correlation with Corrosion Rate', fontweight='bold')
ax.invert_yaxis()

for bar, val in zip(bars, target_corr.values):
    xpos = val + 0.007 if val >= 0 else val - 0.007
    ha = 'left' if val >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f'{val:.3f}', va='center', ha=ha, fontsize=11, fontweight='bold')

plt.tight_layout(pad=2.5)
plt.savefig('EDA_04_feature_target_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: EDA_04_feature_target_correlations.png")

# ===========================================================================
# FIGURE 5: Scatter Plots - Top 6 Continuous Features vs Target
# ===========================================================================
print("[EDA 5/7] Top feature scatter plots...")
top6 = target_corr.index[:6].tolist()

fig, axes = plt.subplots(2, 3, figsize=(21, 13))
fig.suptitle('Top Features vs. Corrosion Rate — Scatter Plots', fontsize=17, fontweight='bold')

for ax, feat in zip(axes.flatten(), top6):
    r, p = stats.pearsonr(df[feat], df[TARGET])
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    ax.scatter(sample[feat], sample[TARGET],
               alpha=0.35, s=18, color=ACCENT, edgecolors='none')

    m, b = np.polyfit(df[feat], df[TARGET], 1)
    x_line = np.linspace(df[feat].min(), df[feat].max(), 200)
    ax.plot(x_line, m * x_line + b, color=TARGET_COLOR, linewidth=2.5, label=f'r = {r:.3f}')

    ax.set_xlabel(LABELS.get(feat, feat))
    ax.set_ylabel(LABELS[TARGET])
    ax.set_title(LABELS.get(feat, feat), fontweight='bold')
    ax.legend()

plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
plt.savefig('EDA_05_scatter_top_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: EDA_05_scatter_top_features.png")

# ===========================================================================
# FIGURE 6: Outlier Detection (Boxplots, continuous features)
# ===========================================================================
print("[EDA 6/7] Outlier detection boxplots...")
fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 5))
fig.suptitle('Boxplots — Outlier Detection per Feature', fontsize=17, fontweight='bold')
axes_flat = axes.flatten()

outlier_summary = {}
for i, feat in enumerate(CONTINUOUS):
    ax = axes_flat[i]
    bp = ax.boxplot(df[feat], vert=True, patch_artist=True, notch=False,
                    boxprops=dict(facecolor=PALETTE[i % len(PALETTE)], alpha=0.6),
                    medianprops=dict(color='black', linewidth=2.5),
                    flierprops=dict(marker='o', markersize=3, alpha=0.35, color='gray'))
    ax.set_title(LABELS[feat], fontweight='bold')
    ax.set_xticks([])

    q1, q3 = df[feat].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[feat] < q1 - 1.5 * iqr) | (df[feat] > q3 + 1.5 * iqr)).sum()
    outlier_summary[feat] = {'Q1': q1, 'Q3': q3, 'IQR': iqr, 'Outliers': n_out, '%': round(n_out / len(df) * 100, 2)}
    ax.set_xlabel(f'Outliers: {n_out} ({n_out/len(df)*100:.1f}%)')

for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)

plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
plt.savefig('EDA_06_outlier_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

outlier_df = pd.DataFrame(outlier_summary).T
outlier_df.to_csv('EDA_Outlier_Summary.csv')
print("  ✓ Saved: EDA_06_outlier_boxplots.png")
print("  ✓ Saved: EDA_Outlier_Summary.csv")

# ===========================================================================
# FIGURE 7: Categorical Feature Analysis
# ===========================================================================
print("[EDA 7/7] Categorical feature analysis...")
fig, axes = plt.subplots(2, 2, figsize=(18, 13))
fig.suptitle('Categorical Feature Analysis — Distribution & Effect on Corrosion Rate',
             fontsize=17, fontweight='bold')

for row_i, cat in enumerate(CATEGORICAL):
    # Count bar chart
    ax_cnt = axes[row_i][0]
    counts = df[cat].value_counts().sort_index()
    bars = ax_cnt.bar(counts.index.astype(str), counts.values,
                      color=PALETTE[:len(counts)], edgecolor='black', linewidth=0.8, alpha=0.85)
    ax_cnt.set_xlabel(LABELS[cat])
    ax_cnt.set_ylabel('Count')
    ax_cnt.set_title(f'{LABELS[cat]} — Class Distribution', fontweight='bold')
    for b in bars:
        ax_cnt.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                    f'{int(b.get_height()):,}', ha='center', va='bottom', fontsize=12)

    # Box plot per category vs target
    ax_box = axes[row_i][1]
    cats_sorted = sorted(df[cat].unique())
    data_by_cat = [df[df[cat] == c][TARGET].values for c in cats_sorted]
    bp = ax_box.boxplot(data_by_cat, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2.5),
                        flierprops=dict(marker='o', markersize=3, alpha=0.3))
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax_box.set_xticks(range(1, len(cats_sorted) + 1))
    ax_box.set_xticklabels([str(c) for c in cats_sorted])
    ax_box.set_xlabel(LABELS[cat])
    ax_box.set_ylabel(LABELS[TARGET])
    ax_box.set_title(f'Corrosion Rate by {LABELS[cat]}', fontweight='bold')

plt.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0)
plt.savefig('EDA_07_categorical_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: EDA_07_categorical_analysis.png")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EDA SUMMARY")
print("=" * 70)
print(f"\nDataset: {df.shape[0]:,} samples, {len(FEATURES)} features, 1 target")
print(f"Missing values : {df.isnull().sum().sum()}")
print(f"Duplicate rows : {df.duplicated().sum()}")
print(f"\nTarget — Corrosion Rate (μA/cm²):")
print(f"  Min / Max  : {df[TARGET].min():.2f} / {df[TARGET].max():.2f}")
print(f"  Mean ± Std : {df[TARGET].mean():.2f} ± {df[TARGET].std():.2f}")
print(f"  Skewness   : {df[TARGET].skew():.3f}")
print(f"  Kurtosis   : {df[TARGET].kurt():.3f}")

print(f"\nTop correlations with Corrosion Rate:")
for feat, val in target_corr.items():
    direction = "↑ positive" if val > 0 else "↓ negative"
    print(f"  {LABELS.get(feat, feat):<35} r = {val:+.3f}  ({direction})")

print(f"\nOutlier counts (IQR method):")
for feat, row in outlier_df.iterrows():
    print(f"  {LABELS.get(feat, feat):<35} {int(row['Outliers']):>4} outliers ({row['%']:.1f}%)")

print("\nGenerated Files:")
for f in [
    'EDA_01_target_distribution.png',
    'EDA_02_feature_distributions.png',
    'EDA_03_correlation_heatmap.png',
    'EDA_04_feature_target_correlations.png',
    'EDA_05_scatter_top_features.png',
    'EDA_06_outlier_boxplots.png',
    'EDA_07_categorical_analysis.png',
    'EDA_Summary_Statistics.csv',
    'EDA_Outlier_Summary.csv',
]:
    print(f"  ✓ {f}")
print()
