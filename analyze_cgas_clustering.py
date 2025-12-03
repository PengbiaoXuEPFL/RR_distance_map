#!/usr/bin/env python3
"""
Enhanced cGAS clustering analysis with differential contact maps vs apo cGAS.

New features:
- Calculate average contact map per condition
- Generate difference maps (each condition - apo_cgas)
- Interactive HTML visualizations for difference maps
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

try:
    import gemmi
    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False
    from Bio.PDB import MMCIFParser

def parse_cif(cif_file):
    """Parse .cif file and extract all chains."""
    if HAS_GEMMI:
        structure = gemmi.read_structure(str(cif_file))
        chains_data = {}
        
        for model in structure:
            for chain in model:
                if chain.name not in chains_data:
                    chains_data[chain.name] = []
                
                for residue in chain:
                    if residue.name == 'HOH':
                        continue
                    ca_atom = None
                    for atom in residue:
                        if atom.name == 'CA':
                            ca_atom = atom
                            break
                    if ca_atom:
                        chains_data[chain.name].append({
                            'resname': residue.name,
                            'resid': residue.seqid.num,
                            'coords': np.array([ca_atom.pos.x, ca_atom.pos.y, ca_atom.pos.z])
                        })
        return chains_data
    else:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('structure', str(cif_file))
        chains_data = {}
        
        for model in structure:
            for chain in model:
                if chain.id not in chains_data:
                    chains_data[chain.id] = []
                
                for residue in chain:
                    if residue.id[0] != ' ':
                        continue
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        chains_data[chain.id].append({
                            'resname': residue.resname,
                            'resid': residue.id[1],
                            'coords': ca_atom.get_coord()
                        })
        return chains_data

def identify_cgas_chain(chains_data, expected_length=367):
    """Automatically identify cGAS chain."""
    cgas_candidates = []
    
    for chain_id, residues in chains_data.items():
        chain_len = len(residues)
        if abs(chain_len - expected_length) < expected_length * 0.1:
            cgas_candidates.append((chain_id, chain_len))
    
    if not cgas_candidates:
        longest = max(chains_data.items(), key=lambda x: len(x[1]))
        print(f"  Warning: No chain matches expected length {expected_length}")
        print(f"  Using longest chain: {longest[0]} ({len(longest[1])} residues)")
        return longest[0]
    
    cgas_chain = cgas_candidates[0][0]
    print(f"  Detected cGAS chain: {cgas_chain} ({cgas_candidates[0][1]} residues)")
    
    return cgas_chain

def calculate_monomer_contact_map(residues, cutoff=8.0):
    """Calculate internal contact map for a single monomer."""
    n = len(residues)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(residues[i]['coords'] - residues[j]['coords'])
            dist_matrix[i, j] = dist
    
    contact_map = (dist_matrix <= cutoff).astype(int)
    
    # Exclude diagonal and sequential neighbors
    for i in range(n):
        for j in range(max(0, i-2), min(n, i+3)):
            contact_map[i, j] = 0
    
    return contact_map, dist_matrix

def process_all_structures(base_dir):
    """Process all .cif files in AF3 prediction folder."""
    results = []
    
    af3_dir = base_dir / "AF3 prediction"
    subdirs = sorted([d for d in af3_dir.iterdir() if d.is_dir()])
    
    print(f"Found {len(subdirs)} prediction folders")
    
    for subdir in subdirs:
        condition = subdir.name.replace('fold_', '')
        print(f"\n{'='*60}")
        print(f"Processing: {condition}")
        print(f"{'='*60}")
        
        cif_files = sorted(subdir.glob("*.cif"))
        
        for cif_file in cif_files:
            model_id = cif_file.stem.split('_')[-1]
            print(f"\n  {cif_file.name}")
            
            try:
                chains_data = parse_cif(cif_file)
                cgas_chain = identify_cgas_chain(chains_data)
                cgas_residues = chains_data[cgas_chain]
                contact_map, dist_matrix = calculate_monomer_contact_map(cgas_residues)
                
                n_contacts = contact_map.sum()
                print(f"  Contact map: {contact_map.shape}, {n_contacts} contacts")
                
                results.append({
                    'condition': condition,
                    'model_id': model_id,
                    'file': cif_file,
                    'chain': cgas_chain,
                    'contact_map': contact_map,
                    'n_residues': len(cgas_residues),
                    'n_contacts': n_contacts
                })
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
    
    return results

def calculate_average_contact_maps(results):
    """Calculate average contact map for each condition."""
    from collections import defaultdict
    
    condition_maps = defaultdict(list)
    
    for r in results:
        condition_maps[r['condition']].append(r['contact_map'])
    
    avg_maps = {}
    for condition, maps in condition_maps.items():
        avg_maps[condition] = np.mean(maps, axis=0)
        print(f"{condition}: averaged {len(maps)} models, mean contacts = {avg_maps[condition].sum():.1f}")
    
    return avg_maps

def create_hover_text_simple(n_res):
    """Create simple hover text for contact map."""
    hover_text = []
    for i in range(n_res):
        row = []
        for j in range(n_res):
            text = f"Res {i+1} ↔ Res {j+1}"
            row.append(text)
        hover_text.append(row)
    return hover_text

def plot_difference_map_html(diff_map, condition, output_dir):
    """Create interactive HTML difference map."""
    n_res = diff_map.shape[0]
    hover_text = create_hover_text_simple(n_res)
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=diff_map,
        text=hover_text,
        hovertemplate='%{text}<br>Δ Contact: %{z:.3f}<extra></extra>',
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title=dict(
                text='Δ Contact Frequency<br>(Blue: More<br>Red: Fewer)',
                side='right'
            )
        ),
        showscale=True
    ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>{condition} - apo_cgas</b><br>Contact Map Difference',
            font=dict(size=18, family="Arial Black")
        ),
        xaxis=dict(
            title='cGAS Residue Position',
            tickmode='linear',
            tick0=0,
            dtick=50
        ),
        yaxis=dict(
            title='cGAS Residue Position',
            tickmode='linear',
            tick0=0,
            dtick=50,
            autorange='reversed'
        ),
        width=900,
        height=850,
        font=dict(size=11)
    )
    
    save_path = output_dir / f"diff_{condition}_vs_apo.html"
    fig.write_html(str(save_path))
    print(f"  Saved: {save_path.name}")
    
    return fig

def flatten_contact_map(contact_map, region_start=50, region_end=80):
    """
    Flatten contact map region into 1D feature vector.
    
    Args:
        contact_map: Full contact map (N x N)
        region_start: Start residue (1-indexed, will convert to 0-indexed)
        region_end: End residue (1-indexed, will convert to 0-indexed)
    
    Returns:
        Flattened upper triangle of the specified region
    """
    # Convert to 0-indexed
    start_idx = region_start - 1
    end_idx = region_end
    
    # Extract region submatrix
    region = contact_map[start_idx:end_idx, start_idx:end_idx]
    
    # Get upper triangle (excluding diagonal)
    n = region.shape[0]
    indices = np.triu_indices(n, k=1)  # k=1 to exclude diagonal
    
    return region[indices]

def perform_clustering(results, output_dir):
    """Perform clustering analysis."""
    print(f"\n{'='*80}")
    print("CLUSTERING ANALYSIS (Based on residues 50-80 region)")
    print(f"{'='*80}")
    
    labels = [f"{r['condition']}_model{r['model_id']}" for r in results]
    conditions = [r['condition'] for r in results]
    
    print("\nFlattening contact maps (region 50-80)...")
    features = np.array([flatten_contact_map(r['contact_map'], 50, 80) for r in results])
    print(f"Feature matrix shape: {features.shape}")
    print(f"Region analyzed: residues 50-80 (31x31 submatrix = {31*30//2} features)")
    
    print("\nPerforming PCA...")
    pca = PCA(n_components=min(10, len(results)))
    pca_features = pca.fit_transform(features)
    
    print(f"Explained variance (first 3 PCs): {pca.explained_variance_ratio_[:3]}")
    print(f"Cumulative variance (first 3 PCs): {pca.explained_variance_ratio_[:3].sum():.2%}")
    
    print("\nPerforming hierarchical clustering...")
    linkage_matrix = linkage(features, method='ward', metric='euclidean')
    
    n_clusters = 3
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    print(f"\nPerforming k-means clustering (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(features)
    
    return {
        'labels': labels,
        'conditions': conditions,
        'pca_features': pca_features,
        'hierarchical_clusters': cluster_labels,
        'kmeans_clusters': kmeans_labels,
        'linkage_matrix': linkage_matrix,
        'pca': pca,
        'features': features
    }

def plot_dendrogram(results, clustering_results, output_dir):
    """Plot hierarchical clustering dendrogram."""
    labels = clustering_results['labels']
    linkage_matrix = clustering_results['linkage_matrix']
    conditions = clustering_results['conditions']
    
    colors_dict = {
        'apo_cgas': '#808080',
        'd2': '#FFA500',
        'dna_cgas': '#00FF00',
        'f4_q': '#FF00FF',
        'sa3': '#FF0000',
        'sa3_3e': '#0000FF'
    }
    
    # Add more colors for any new conditions
    unique_conditions = set(conditions)
    extra_colors = ['#00FFFF', '#FFFF00', '#FF69B4', '#8B4513', '#4B0082']
    for i, cond in enumerate(sorted(unique_conditions)):
        if cond not in colors_dict:
            colors_dict[cond] = extra_colors[i % len(extra_colors)]
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    dendr = dendrogram(
        linkage_matrix,
        labels=labels,
        ax=ax,
        leaf_font_size=9,
        leaf_rotation=90
    )
    
    xlabels = ax.get_xmajorticklabels()
    for label in xlabels:
        text = label.get_text()
        cond = text.split('_model')[0]
        if cond in colors_dict:
            label.set_color(colors_dict[cond])
            label.set_fontweight('bold')
    
    ax.set_title('Hierarchical Clustering - cGAS Monomer Contacts (Residues 50-80)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Structure (colored by condition)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance (Ward linkage)', fontsize=12, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors_dict[c], label=c) 
                      for c in sorted(colors_dict.keys()) if c in conditions]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dendrogram.png', dpi=300, bbox_inches='tight')
    print(f"Saved: dendrogram.png")
    plt.close()

def plot_pca_scatter(clustering_results, output_dir):
    """Create PCA scatter plot."""
    pca_features = clustering_results['pca_features']
    conditions = clustering_results['conditions']
    labels = clustering_results['labels']
    
    colors_dict = {
        'apo_cgas': '#808080',
        'd2': '#FFA500',
        'dna_cgas': '#00FF00',
        'f4_q': '#FF00FF',
        'sa3': '#FF0000',
        'sa3_3e': '#0000FF'
    }
    
    unique_conditions = set(conditions)
    extra_colors = ['#00FFFF', '#FFFF00', '#FF69B4', '#8B4513', '#4B0082']
    for i, cond in enumerate(sorted(unique_conditions)):
        if cond not in colors_dict:
            colors_dict[cond] = extra_colors[i % len(extra_colors)]
    
    fig = go.Figure()
    
    for cond in sorted(set(conditions)):
        mask = np.array(conditions) == cond
        fig.add_trace(go.Scatter(
            x=pca_features[mask, 0],
            y=pca_features[mask, 1],
            mode='markers+text',
            name=cond,
            marker=dict(
                size=12,
                color=colors_dict.get(cond, '#000000'),
                line=dict(width=1, color='white')
            ),
            text=[f"M{l.split('_model')[1]}" for l, c in zip(labels, conditions) if c == cond],
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate='%{hovertext}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>',
            hovertext=[l for l, c in zip(labels, conditions) if c == cond]
        ))
    
    fig.update_layout(
        title='PCA of cGAS Monomer Contacts (Residues 50-80)',
        xaxis_title='PC1',
        yaxis_title='PC2',
        width=1100,
        height=750,
        font=dict(size=12)
    )
    
    fig.write_html(str(output_dir / 'pca_scatter.html'))
    print(f"Saved: pca_scatter.html")

def analyze_cluster_composition(clustering_results):
    """Analyze cluster composition."""
    conditions = clustering_results['conditions']
    hier_clusters = clustering_results['hierarchical_clusters']
    
    print(f"\n{'='*80}")
    print("CLUSTER COMPOSITION ANALYSIS")
    print(f"{'='*80}\n")
    
    unique_clusters = sorted(set(hier_clusters))
    
    for cluster_id in unique_clusters:
        mask = hier_clusters == cluster_id
        cluster_conditions = [c for c, m in zip(conditions, mask) if m]
        cluster_labels = [l for l, m in zip(clustering_results['labels'], mask) if m]
        
        print(f"Cluster {cluster_id} ({len(cluster_labels)} structures):")
        print(f"{'-'*60}")
        
        from collections import Counter
        cond_counts = Counter(cluster_conditions)
        for cond, count in sorted(cond_counts.items()):
            print(f"  {cond}: {count}")
        
        print(f"\n  Members: {', '.join(cluster_labels)}\n")

def main():
    base_dir = Path(".")
    output_dir = base_dir / "clustering_analysis"
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("cGAS MONOMER CLUSTERING & DIFFERENTIAL ANALYSIS")
    print("="*80)
    print(f"\nOutput directory: {output_dir}\n")
    
    # Process all structures
    results = process_all_structures(base_dir)
    
    if not results:
        print("\nERROR: No structures processed!")
        return
    
    print(f"\n{'='*80}")
    print(f"Successfully processed {len(results)} structures from {len(set(r['condition'] for r in results))} conditions")
    print(f"{'='*80}")
    
    # Calculate average contact maps per condition
    print(f"\n{'='*80}")
    print("CALCULATING AVERAGE CONTACT MAPS PER CONDITION")
    print(f"{'='*80}\n")
    
    avg_maps = calculate_average_contact_maps(results)
    
    # Generate difference maps vs apo_cgas
    if 'apo_cgas' in avg_maps:
        print(f"\n{'='*80}")
        print("GENERATING DIFFERENCE MAPS (vs apo_cgas)")
        print(f"{'='*80}\n")
        
        apo_map = avg_maps['apo_cgas']
        
        for condition, avg_map in sorted(avg_maps.items()):
            if condition == 'apo_cgas':
                continue
            
            print(f"Creating difference map: {condition} - apo_cgas")
            diff_map = avg_map - apo_map
            plot_difference_map_html(diff_map, condition, output_dir)
    else:
        print("\nWarning: No apo_cgas found, skipping difference maps")
    
    # Perform clustering
    clustering_results = perform_clustering(results, output_dir)
    
    # Analyze clusters
    analyze_cluster_composition(clustering_results)
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING CLUSTERING VISUALIZATIONS")
    print(f"{'='*80}\n")
    
    plot_dendrogram(results, clustering_results, output_dir)
    plot_pca_scatter(clustering_results, output_dir)
    
    print(f"\n{'='*80}")
    print("✅ ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  • dendrogram.png - Hierarchical clustering tree")
    print(f"  • pca_scatter.html - Interactive PCA plot")
    print(f"  • diff_*.html - Interactive difference maps vs apo_cgas")

if __name__ == "__main__":
    main()
