"""
Advanced Clustering Analysis
Hierarchical clustering and correlation analysis for portfolio construction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict


class ClusteringAnalyzer:
    """Performs advanced clustering analysis on portfolio assets."""
    
    def __init__(self, returns: pd.DataFrame, tickers: List[str]):
        self.returns = returns
        self.tickers = tickers
        self.corr_matrix = returns.corr()
        
    def perform_hierarchical_clustering(self, n_clusters: int = 3) -> Tuple[np.ndarray, Dict]:
        """
        Perform hierarchical clustering on assets based on correlation.
        
        Returns:
            cluster_labels: Array of cluster assignments
            cluster_info: Dictionary with cluster statistics
        """
        # Convert correlation to distance
        distance_matrix = np.sqrt(2 * (1 - self.corr_matrix))
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        
        # Get cluster labels
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Calculate cluster statistics
        cluster_info = self._calculate_cluster_stats(cluster_labels)
        
        return cluster_labels, cluster_info, linkage_matrix
    
    def _calculate_cluster_stats(self, cluster_labels: np.ndarray) -> Dict:
        """Calculate statistics for each cluster."""
        cluster_info = {}
        
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_tickers = [self.tickers[i] for i, m in enumerate(mask) if m]
            cluster_returns = self.returns[cluster_tickers]
            
            # Calculate cluster metrics
            avg_return = cluster_returns.mean().mean() * 252 * 100
            avg_vol = cluster_returns.std().mean() * np.sqrt(252) * 100
            avg_corr = cluster_returns.corr().values[np.triu_indices_from(
                cluster_returns.corr().values, k=1)].mean()
            
            cluster_info[f'Cluster {cluster_id}'] = {
                'tickers': cluster_tickers,
                'size': len(cluster_tickers),
                'avg_return': avg_return,
                'avg_volatility': avg_vol,
                'avg_correlation': avg_corr
            }
        
        return cluster_info
    
    def create_clustering_visualization(self, cluster_labels: np.ndarray,
                                       linkage_matrix: np.ndarray,
                                       save_path: str) -> None:
        """Create comprehensive clustering visualization."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. Dendrogram
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_dendrogram(ax1, linkage_matrix)
        
        # 2. Correlation heatmap with clusters
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_correlation_heatmap(ax2, cluster_labels)
        
        # 3. Cluster composition
        ax3 = fig.add_subplot(gs[1, 2])
        self._plot_cluster_composition(ax3, cluster_labels)
        
        # 4. PCA visualization
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_pca_clusters(ax4, cluster_labels)
        
        # 5. Cluster risk-return
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_cluster_risk_return(ax5, cluster_labels)
        
        # 6. Intra-cluster correlation
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_intracluster_correlation(ax6, cluster_labels)
        
        plt.suptitle('Análisis de Clustering Jerárquico de Activos', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_dendrogram(self, ax, linkage_matrix):
        """Plot hierarchical clustering dendrogram."""
        dendrogram(linkage_matrix, labels=self.tickers, ax=ax,
                  leaf_font_size=10, color_threshold=0)
        ax.set_title('Dendrograma de Clustering Jerárquico', fontweight='bold', fontsize=12)
        ax.set_xlabel('Activos')
        ax.set_ylabel('Distancia (basada en correlación)')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_correlation_heatmap(self, ax, cluster_labels):
        """Plot correlation matrix heatmap ordered by clusters."""
        # Sort by cluster
        sorted_idx = np.argsort(cluster_labels)
        sorted_corr = self.corr_matrix.iloc[sorted_idx, sorted_idx]
        sorted_tickers = [self.tickers[i] for i in sorted_idx]
        
        sns.heatmap(sorted_corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   xticklabels=sorted_tickers, yticklabels=sorted_tickers,
                   ax=ax, cbar_kws={'label': 'Correlación'}, vmin=-1, vmax=1)
        ax.set_title('Matriz de Correlación (Ordenada por Clusters)', 
                    fontweight='bold', fontsize=12)
    
    def _plot_cluster_composition(self, ax, cluster_labels):
        """Plot cluster composition pie chart."""
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        colors_palette = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
        
        ax.pie(counts, labels=[f'Cluster {c}' for c in unique_clusters],
              autopct='%1.1f%%', colors=colors_palette, startangle=90)
        ax.set_title('Composición de Clusters', fontweight='bold', fontsize=11)
    
    def _plot_pca_clusters(self, ax, cluster_labels):
        """Plot PCA visualization of clusters."""
        # Perform PCA
        pca = PCA(n_components=2)
        returns_pca = pca.fit_transform(self.returns.T)
        
        # Plot each cluster
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            ax.scatter(returns_pca[mask, 0], returns_pca[mask, 1],
                      label=f'Cluster {cluster_id}', s=100, alpha=0.7,
                      edgecolors='black', linewidths=1)
            
            # Add labels
            for i, (x, y) in enumerate(returns_pca[mask]):
                ticker_idx = np.where(mask)[0][i]
                ax.annotate(self.tickers[ticker_idx], (x, y), fontsize=8,
                           xytext=(5, 5), textcoords='offset points')
        
        ax.set_title(f'PCA de Activos por Cluster\n(Var. Explicada: {pca.explained_variance_ratio_.sum()*100:.1f}%)', 
                    fontweight='bold', fontsize=11)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_cluster_risk_return(self, ax, cluster_labels):
        """Plot risk-return profile by cluster."""
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_tickers = [self.tickers[i] for i, m in enumerate(mask) if m]
            
            for ticker in cluster_tickers:
                annual_ret = self.returns[ticker].mean() * 252 * 100
                annual_vol = self.returns[ticker].std() * np.sqrt(252) * 100
                
                ax.scatter(annual_vol, annual_ret, s=120, alpha=0.7,
                          label=f'Cluster {cluster_id}' if ticker == cluster_tickers[0] else "",
                          edgecolors='black', linewidths=1)
                ax.annotate(ticker, (annual_vol, annual_ret), fontsize=8,
                           xytext=(3, 3), textcoords='offset points')
        
        ax.set_title('Perfil Riesgo-Retorno por Cluster', fontweight='bold', fontsize=11)
        ax.set_xlabel('Volatilidad Anualizada (%)')
        ax.set_ylabel('Retorno Anualizado (%)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_intracluster_correlation(self, ax, cluster_labels):
        """Plot average intra-cluster correlation."""
        avg_corrs = []
        cluster_names = []
        
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            cluster_tickers = [self.tickers[i] for i, m in enumerate(mask) if m]
            
            if len(cluster_tickers) > 1:
                cluster_corr = self.returns[cluster_tickers].corr()
                # Get upper triangle (excluding diagonal)
                avg_corr = cluster_corr.values[np.triu_indices_from(cluster_corr.values, k=1)].mean()
                avg_corrs.append(avg_corr)
                cluster_names.append(f'Cluster {cluster_id}')
        
        bars = ax.bar(cluster_names, avg_corrs, color='steelblue', alpha=0.7,
                     edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, avg_corrs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('Correlación Promedio Intra-Cluster', fontweight='bold', fontsize=11)
        ax.set_ylabel('Correlación Promedio')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Umbral 0.5')
        ax.legend(fontsize=8)
    
    def generate_cluster_report_data(self, cluster_labels: np.ndarray,
                                    cluster_info: Dict) -> pd.DataFrame:
        """Generate cluster analysis summary table."""
        report_data = []
        
        for cluster_name, info in cluster_info.items():
            report_data.append({
                'Cluster': cluster_name,
                'Activos': ', '.join(info['tickers']),
                'Cantidad': info['size'],
                'Retorno Promedio (%)': info['avg_return'],
                'Volatilidad Promedio (%)': info['avg_volatility'],
                'Correlación Intra-Cluster': info['avg_correlation']
            })
        
        return pd.DataFrame(report_data)
