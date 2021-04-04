
import pickle as pkl
import matplotlib.pyplot as plt


data = 'phish'
cols = ['f1_train', 'f1_test', 'accuracy_train', 'accuracy_test']
df, = pkl.load(file=open('results/ann_clusters_phish.pkl', 'rb'))


fig = plt.figure(figsize=(5, 8))
axs = fig.subplots(2, 1, sharex=False).flatten()
# -----------------------------------------------------------------------
df[cols].transpose().plot.bar(ax=axs[0])
axs[0].tick_params(axis='x', labelrotation=15)
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.28), ncol=3)
axs[0].set_title('Performance Comparison ({})'.format(data))
# axs[0].set_xlabel('Metrics')
axs[0].set_ylabel('Scores')
# ------------------------------------------------------------------------
df['time'].transpose().plot.bar(ax=axs[1])
axs[1].tick_params(axis='x', labelrotation=30)
axs[1].set_title('Efficiency Comparison ({})'.format(data))
# axs[1].set_xlabel('Feature Type')
axs[1].set_ylabel('Time (s)')
# ------------------------------------------------------------------------
fig.tight_layout()
fig.savefig('figures/5_ANN_Clusters_{}.png'.format(data))

