# check_similarity2.py
# Compare two COMPLETELY DIFFERENT attack types
# to show the contrast
import pandas as pd
import numpy as np

print("="*60)
print("  CROSS-CATEGORY COMPARISON")
print("  DDoS-ICMP_Flood vs Mirai-greeth_flood")
print("  (different attack categories)")
print("="*60)

f1 = pd.read_csv('data/DDoS-ICMP_Flood.pcap.csv',
                 nrows=5000)
f2 = pd.read_csv('data/Mirai-greeth_flood.pcap.csv',
                 nrows=5000)

f1 = f1.select_dtypes(include=[np.number])
f2 = f2.select_dtypes(include=[np.number])
f1 = f1.replace([np.inf, -np.inf], np.nan).dropna()
f2 = f2.replace([np.inf, -np.inf], np.nan).dropna()

means1 = f1.mean()
means2 = f2.mean()
diff   = (means1 - means2).abs()
corr   = np.corrcoef(means1.values, means2.values)[0, 1]

print(f"\n  Pearson correlation : {corr:.6f}")
print(f"  Avg mean difference : {diff.mean():.4f}")
print(f"  Features <0.1 diff  : {(diff < 0.1).sum()} / {len(diff)}")
print(f"\n  Top 5 most different features:")
for feat, val in diff.nlargest(5).items():
    print(f"    {feat:<35} : {val:.4f}")