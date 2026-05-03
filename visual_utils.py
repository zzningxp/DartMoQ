import numpy as np
import matplotlib.pyplot as plt

def plot_diff_wbits_correlation(model_type, layer_idx, expert_num, rates_2, rates_3, rates_4):
    for x in range(expert_num):
        rates_2_x = rates_2[x].cpu().numpy()
        rates_3_x = rates_3[x].cpu().numpy()
        rates_4_x = rates_4[x].cpu().numpy()
        
        n_neurons = len(rates_2_x)
        bins = len(rates_3_x) // n_neurons
        # print(all_rates_x, rates_3_x)
        rank_x_2 = np.argsort(np.argsort(rates_2_x)) // bins
        rank_x_3 = np.argsort(np.argsort(rates_3_x)) // bins
        rank_x_4 = np.argsort(np.argsort(rates_4_x)) // bins
        # print(rank_x_2[:40], rank_x_3[:40], rank_x_4[:40],)

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        ax1 = axes[0]
        ax1.scatter(rank_x_2, rank_x_3, s=5, alpha=0.5)
        ax1.plot([1, n_neurons], [1, n_neurons], 'r--', linewidth=1)
        ax1.set_xlabel('2bit Rank (1=most important)')
        ax1.set_ylabel('3bit Rank (1=most important)')
        ax1.set_title('2bit vs 3bit Neuron Rank\n', fontsize=12, fontweight='bold')
        ax1.set_xlim(1, n_neurons)
        ax1.set_ylim(1, n_neurons)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.scatter(rank_x_2, rank_x_4, s=5, alpha=0.5)
        ax2.plot([1, n_neurons], [1, n_neurons], 'r--', linewidth=1)
        ax2.set_xlabel('2bit Rank (1=most important)')
        ax2.set_ylabel('4bit Rank (1=most important)')
        ax2.set_title('2bit vs 4bit Neuron Rank', fontsize=12, fontweight='bold')
        ax2.set_xlim(1, n_neurons)
        ax2.set_ylim(1, n_neurons)
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        ax3.scatter(rank_x_3, rank_x_4, s=5, alpha=0.5)
        ax3.plot([1, n_neurons], [1, n_neurons], 'r--', linewidth=1)
        ax3.set_xlabel('3bit Rank (1=most important)')
        ax3.set_ylabel('4bit Rank (1=most important)')
        ax3.set_title('3bit vs 4bit Neuron Rank', fontsize=12, fontweight='bold')
        ax3.set_xlim(1, n_neurons)
        ax3.set_ylim(1, n_neurons)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"plot/_rank_compare_{model_type}_{layer_idx}_{x}.png")
        plt.close()

def plot_spearman_rank_correlation(model_type, layer_idx, expert_num, rates_2, rates_3, rates_4):
    from scipy.stats import spearmanr
    import seaborn as sns

    corr_matrix_list = []
    for x in range(expert_num):
        rates_2_x = rates_2[x].cpu().numpy()
        rates_3_x = rates_3[x].cpu().numpy()
        rates_4_x = rates_4[x].cpu().numpy()
        
        ranks = {
            "2-bit": np.argsort(np.argsort(-rates_2_x)) + 1,
            "3-bit": np.argsort(np.argsort(-rates_3_x)) + 1,
            "4-bit": np.argsort(np.argsort(-rates_4_x)) + 1
        }

        corr_matrix = np.zeros((3, 3))
        methods = list(ranks.keys())
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                corr_matrix[i, j], _ = spearmanr(ranks[m1], ranks[m2])
        corr_matrix_list.append(corr_matrix)
        
    cols = min(expert_num, 8) 
    rows = (expert_num + cols - 1) // cols  
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes) 

    for x, corr_matrix in enumerate(corr_matrix_list):
        row = x // cols
        col = x % cols
        ax = axes[row, col]

        sns.heatmap(corr_matrix, 
                annot=True, 
                fmt=".3f", 
                xticklabels=methods, 
                yticklabels=methods,
                cmap="coolwarm", 
                vmin=0.8, vmax=1.0,
                # annot_kws={"size": 10},
                ax=ax) 
        ax.set_title(f'Spearman Rank Correlation Between Bitwidths, Expert {x}', fontsize=8)

    for x in range(expert_num, rows * cols):
        row = x // cols
        col = x % cols
        axes[row, col].axis('off')
    
    fig.suptitle(f'Spearman Rank Correlation (Layer {layer_idx})', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"plot/_spearman_rank_compare_{model_type}_{layer_idx}.png")
    plt.close()
    # assert False, "stoprank compare"
