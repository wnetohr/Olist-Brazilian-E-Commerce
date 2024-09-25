import numpy as np
import pandas as pd
from math import ceil
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import matplotlib.pyplot as plt
import seaborn as sns

def cluster_scores(df_cluster : pd.DataFrame, initial_range : int, final_range : int, random:int=42, 
    score_types:list=["silhouette"], width_per_ax : int=5, height_per_ax : int=5): 
    
    km = KMeans(random_state=random)
    visualizer = KElbowVisualizer(km, k=(initial_range, final_range))
    visualizer.fit(df_cluster)
    visualizer.show()

    fitted_kmeans = {}
    labels_kmeans = {}
    df_scores = []
    for n_clusters in np.arange(initial_range, final_range):
        tmp_scores = {}
        tmp_scores["n_clusters"] = n_clusters
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=random)
        labels_clusters = kmeans.fit_predict(df_cluster)
        
        fitted_kmeans[n_clusters] = kmeans
        labels_kmeans[n_clusters] = labels_clusters
        
        if "silhouette" in score_types:
            silhouette = silhouette_score(df_cluster, labels_clusters)
            tmp_scores["silhouette"] = silhouette
        
        if "calinski_harabasz" in score_types:
            ch = calinski_harabasz_score(df_cluster, labels_clusters)
            tmp_scores["calinski_harabasz"] = ch
        
        if "davies_bouldin" in score_types:
            db = davies_bouldin_score(df_cluster, labels_clusters)
            tmp_scores["davies_bouldin"] = db
                    
        df_scores.append(tmp_scores)

    df_scores = pd.DataFrame(df_scores)
    df_scores.set_index("n_clusters", inplace=True)
    df_scores.plot(subplots=True, layout=(1,len(score_types)), figsize=(len(score_types) * width_per_ax, height_per_ax), xticks=np.arange(initial_range, final_range+1))

    return

def multi_visualize_silhoutte(df_cluster : pd.DataFrame, initial_range : int, final_range : int, 
        per_col : int=2, random : int=42, width_per_ax : int=6, height_per_ax : int=5):
    
    lines = ceil((final_range-initial_range)/per_col)
    
    fig, axes = plt.subplots(lines, per_col, figsize=(per_col*width_per_ax,lines*height_per_ax))
    for pos, nCluster in enumerate(range(initial_range, final_range)):
        km = KMeans(n_clusters=nCluster, random_state=random)
        
        if lines == 1:
            ax=axes[pos]
        else: 
            q, mod = divmod(pos, per_col)
            ax = axes[q][mod]
        
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax)
        visualizer.fit(df_cluster)
        ax.set_title(f"clusters = {nCluster}\nscore = {visualizer.silhouette_score_}") 

    return

def visualize_silhoutte(df_cluster : pd.DataFrame, n_cluster : int, random : int=42, width_per_ax : int=6, height_per_ax : int=5, versao_cluster : str="clusters"):
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(width_per_ax, height_per_ax)
    km = KMeans(n_clusters=n_cluster, random_state=random)
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax1)
    visualizer.fit(df_cluster)
    ax1.set_title(f"{versao_cluster} = {n_cluster}\nscore = {visualizer.silhouette_score_}") 

    return 


def visualize_scores(
        df : pd.DataFrame,
        initial_range : int, 
        final_range : int, 
        sample_percentual : int=100,  
        scaler : str="original",
        random :int=42, 
        score_types : list=["silhouette"], 
        width_per_ax : int=5, 
        height_per_ax : int=5, 
        per_col : int=2
    ):

    df = df.sample(ceil(df.shape[0]*(sample_percentual/100)), random_state=random)
    
    match scaler:
        case "standard":
            scaler = StandardScaler()
            df = scaler.fit_transform(df)
        case "minmax":
            scaler = MinMaxScaler()
            df = scaler.fit_transform(df)
        case _:
            pass

    cluster_scores(df_cluster=df, initial_range=initial_range, final_range=final_range, 
        random=random, score_types=score_types, width_per_ax=width_per_ax, height_per_ax=height_per_ax)
   
    multi_visualize_silhoutte(df_cluster=df, initial_range=initial_range, final_range=final_range, 
        random=random, width_per_ax=width_per_ax, height_per_ax=height_per_ax, per_col=per_col)

    return 

def visualize_all_features(df : pd.DataFrame, n_clusters : int, cluster_colors : list):
    for cluster in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster]

        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(df.columns):
            plt.subplot(5, 4, i + 1)
            sns.histplot(cluster_data[feature], kde=True, color=cluster_colors[cluster])
            plt.title(f'{feature} - Cluster {cluster}')
            plt.xlabel('')
            plt.ylabel('')

        plt.tight_layout()
        plt.show()

    for feature in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='cluster', y=feature, data=df, palette=cluster_colors, hue='cluster', legend=False)
        plt.title(f'Boxplot of {feature} by Cluster (Original Data)')
        plt.show()

    for feature in df.columns:
        plt.figure(figsize=(8, 6))
        sns.violinplot(x='cluster', y=feature, data=df, palette=cluster_colors, hue='cluster', legend=False)
        plt.title(f'Violin Plot of {feature} by Cluster (Original Data)')
        plt.show()