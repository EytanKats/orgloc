import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results_3d = '/home/kats/storage/staff/eytankats/projects/orgloc/experiments/202507082234_multidim_multilabel_alllabels_weighted3_unet_processed_aggmasksv2/results_100.csv'
results_25d = '/home/kats/storage/staff/eytankats/projects/orgloc/experiments/202507141821_multidim_multilabel_alllabels_weighted5_25dim_unet_processed_aggmasksv2/results_100.csv'
results_mm = '/home/kats/storage/staff/eytankats/projects/orgloc/experiments/mean_model/results_100.csv'

df_3d = pd.read_csv(results_3d)#[:-2]
df_25d = pd.read_csv(results_25d)#[:-2]
df_mm = pd.read_csv(results_mm)

def prepare_long_format_grouped_by_prefix(df, prefixes, suffixes=None, dataset_label='A'):
    rows = []
    for col in df.columns:
        for p in prefixes:
            if col.startswith(p + '_'):
                suffix = col[len(p) + 1:]
                if suffixes is not None and suffix not in suffixes:
                    continue
                rows.append({
                    'prefix': p,
                    'value': df[col].abs(),  # Absolute value
                    'dataset': dataset_label
                })
                break
    long_df = pd.concat([
        pd.DataFrame({
            'prefix': row['prefix'],
            'value': row['value'],
            'dataset': row['dataset']
        }) for row in rows
    ], ignore_index=True)
    return long_df


def plot_prefix_comparison_barplot(df_list, prefixes, suffixes=None,
                                     title='Prefix-Level Comparison',
                                     figsize=None,
                                     labels=('A', 'B', 'C'),
                                     y_label='',
                                     save_path=''):

    # Prepare long format for all datasets
    long_dfs = [
        prepare_long_format_grouped_by_prefix(df, prefixes, suffixes, dataset_label=label)
        for df, label in zip(df_list, labels)
    ]
    combined_df = pd.concat(long_dfs, ignore_index=True)

    # Compute mean and std per prefix and dataset
    agg_df = combined_df.groupby(['prefix', 'dataset']).agg(
        mean_value=('value', 'mean'),
        std_value=('value', 'std')
    ).reset_index()

    # Plotting
    ordered_prefixes = [p for p in prefixes if p in agg_df['prefix'].unique()]
    datasets = labels
    x = np.arange(len(ordered_prefixes))

    if figsize is None:
        figsize = (4, 6)
    plt.figure(figsize=figsize)

    ax = sns.boxplot(
        data=combined_df,
        x='prefix',
        y='value',
        hue='dataset',
        order=ordered_prefixes,
        hue_order=labels,
        palette="muted",
        showfliers=False,
        whis=(10, 90)
    )

    ax.set_ylim(bottom=0)
    ax.set_ylabel(y_label)
    ax.set_xlabel('')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_prefixes)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(title='Method')

    plt.tight_layout()

    # --- Save to file if path is provided ---
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

    # --- Print mean ± std summary ---
    print("\nMean ± Std per Prefix:")
    for dataset in datasets:
        print(f"\nDataset: {dataset}")
        subset = agg_df[agg_df['dataset'] == dataset].set_index('prefix').reindex(ordered_prefixes)
        for prefix in ordered_prefixes:
            mean_val = subset.loc[prefix, 'mean_value']
            std_val = subset.loc[prefix, 'std_value']
            print(f"  {prefix:<15}: {mean_val:.2f} ± {std_val:.2f}")

plot_prefix_comparison_barplot(
    [df_3d, df_mm, df_25d ],
    prefixes=['right', 'left', 'inferior', 'superior', 'anterior', 'posterior'],
    suffixes = None,
    labels=('Pix2Vox',  'Mean model', 'CNN 2.5D'),
    title='',
    y_label='Mean Absolute Detection Offset (mm)',
    figsize=(8, 4),
    save_path='/home/kats/storage/staff/eytankats/projects/orgloc/plots/detetection_offset_2.pdf'
)

