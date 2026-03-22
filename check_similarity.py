# check_similarity.py
import pandas as pd
import numpy as np

print("="*60)
print("  FILE SIMILARITY ANALYSIS")
print("  DDoS-ICMP_Flood vs DDoS-ICMP_Flood1")
print("="*60)

f1 = pd.read_csv('data/DDoS-ICMP_Flood.pcap.csv',  nrows=5000)
f2 = pd.read_csv('data/DDoS-ICMP_Flood1.pcap.csv', nrows=5000)

# Keep only numeric columns
f1 = f1.select_dtypes(include=[np.number])
f2 = f2.select_dtypes(include=[np.number])

# Remove infinite and NaN
f1 = f1.replace([np.inf, -np.inf], np.nan).dropna()
f2 = f2.replace([np.inf, -np.inf], np.nan).dropna()

print(f"\n  File 1 rows : {len(f1):,}")
print(f"  File 2 rows : {len(f2):,}")
print(f"  Features    : {len(f1.columns)}")

# Compare means
means1   = f1.mean()
means2   = f2.mean()
diff     = (means1 - means2).abs()

# Compare standard deviations
std1     = f1.std()
std2     = f2.std()
std_diff = (std1 - std2).abs()

# Compare medians
med1     = f1.median()
med2     = f2.median()
med_diff = (med1 - med2).abs()

print(f"\n{'─'*60}")
print(f"  MEAN COMPARISON")
print(f"{'─'*60}")
print(f"  Average difference across all features : {diff.mean():.4f}")
print(f"  Max difference in any single feature   : {diff.max():.4f}")
print(f"  Min difference in any single feature   : {diff.min():.6f}")
print(f"  Features with zero difference          : {(diff == 0).sum()}")
print(f"  Features with <0.1 difference          : {(diff < 0.1).sum()}")
print(f"  Features with >1.0 difference          : {(diff > 1.0).sum()}")
print(f"  Features with >10.0 difference         : {(diff > 10.0).sum()}")

print(f"\n  Top 5 most different features (by mean):")
for feat, val in diff.nlargest(5).items():
    print(f"    {feat:<35} : {val:.4f}")

print(f"\n  Top 5 most similar features (by mean):")
for feat, val in diff.nsmallest(5).items():
    print(f"    {feat:<35} : {val:.6f}")

print(f"\n{'─'*60}")
print(f"  STANDARD DEVIATION COMPARISON")
print(f"{'─'*60}")
print(f"  Average std difference : {std_diff.mean():.4f}")
print(f"  Max std difference     : {std_diff.max():.4f}")

print(f"\n{'─'*60}")
print(f"  MEDIAN COMPARISON")
print(f"{'─'*60}")
print(f"  Average median difference : {med_diff.mean():.4f}")
print(f"  Max median difference     : {med_diff.max():.4f}")

print(f"\n{'─'*60}")
print(f"  CORRELATION BETWEEN FILES")
print(f"{'─'*60}")
# Flatten both into 1D and correlate means
corr = np.corrcoef(means1.values, means2.values)[0, 1]
print(f"  Pearson correlation of feature means : {corr:.6f}")
print(f"  (1.0 = identical, 0.0 = no relation)")

print(f"\n{'─'*60}")
print(f"  CONCLUSION")
print(f"{'─'*60}")
if corr > 0.99 and diff.mean() < 5.0:
    print(f"  STATUS : HIGHLY SIMILAR")
    print(f"  The two files share nearly identical statistical")
    print(f"  distributions. Using one representative file per")
    print(f"  attack type is scientifically justified.")
    print(f"  Adding more numbered variants would increase")
    print(f"  dataset size without adding pattern diversity.")
elif corr > 0.95:
    print(f"  STATUS : MODERATELY SIMILAR")
    print(f"  The files share similar distributions with some")
    print(f"  variation. One file per type is still reasonable")
    print(f"  but including more could add marginal diversity.")
else:
    print(f"  STATUS : MEANINGFULLY DIFFERENT")
    print(f"  The files show significant statistical differences.")
    print(f"  Including more numbered variants could improve")
    print(f"  model generalization.")
print(f"{'='*60}\n")