# INCOMPLETE
# uncomment the evaluate script
# # read the log.txt and extract the rewards per episode

# script to run the models for eval for 1000 episodes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('returns_MiniGrid-DistShift2-v0', header=None)
df = df.sort_values(by=0, ignore_index=True)

# plt.figure(figsize=(10,10))
plt.clf()
plt.grid()
plt.errorbar(np.arange(len(df)), df[1], df[2], fmt='ok', lw=5)
plt.errorbar(np.arange(len(df)), df[1], [df[1] - df[3], df[4] - df[1]],
             fmt='.k', ecolor='gray', lw=2)
plt.xticks(np.arange(len(df)), df[0], rotation=0)
plt.title('DistShift2 Eval returns per episode (for 1000 episodes)')
plt.ylabel('Returns per episode')
plt.xlabel('Models with different SHAP regularizer coeff')
plt.savefig('plot_eval.png')

