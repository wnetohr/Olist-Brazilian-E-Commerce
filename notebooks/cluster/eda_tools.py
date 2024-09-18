import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def barplot_clusters_stacked(dfall : dict[str, pd.DataFrame], title="plot de clusters stackadados", l2_loc: list=[1.01, 0.1], H: str="/", **kwargs):
    n_df = len(list(dfall.keys()))
    n_col = len(dfall[list(dfall.keys())[0]].columns) 
    n_ind = len(dfall[list(dfall.keys())[0]].index)
    plt.figure(figsize=(18, 10))
    axe = plt.subplot(111)

    for df in dfall.values() :
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=False,
                      grid=False,
                      **kwargs) 

    h,l = axe.get_legend_handles_labels()
    for i in range(0, n_df * n_col, n_col):
        for j, pa in enumerate(h[i:i+n_col]):
            for rect in pa.patches:
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_hatch(H * int(i / n_col))
                rect.set_width(1 / float(n_df + 1))

    axe.set_xticks((np.arange(0, 2 * n_ind, 2) + 1 / float(n_df + 1)) / 2.)
    axe.set_xticklabels(df.index, rotation = 0)
    axe.set_title(title)

    n=[]        
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    l1 = axe.legend(h[:n_col], l[:n_col], loc=[1.01, 0.5])
    which_df = list(dfall.keys())
    l2 = plt.legend(n, which_df, loc=l2_loc) 
    axe.add_artist(l1)
    plt.show()
    return 