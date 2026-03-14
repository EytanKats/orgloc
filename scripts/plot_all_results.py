import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths to the results CSV files (taken from plot_vol_metrics.py)
results_3d = '/home/kats/storage/staff/eytankats/projects/orgloc/experiments/202507082234_multidim_multilabel_alllabels_weighted3_unet_processed_aggmasksv2/results_100.csv'
results_25d = '/home/kats/storage/staff/eytankats/projects/orgloc/experiments/202507141821_multidim_multilabel_alllabels_weighted5_25dim_unet_processed_aggmasksv2/results_100.csv'
results_mm = '/home/kats/storage/staff/eytankats/projects/orgloc/experiments/mean_model/results_100.csv'

def merge_left_right_columns(df, filter_outliers=False):
    """
    Merge columns with '_left' and '_right' suffixes in a DataFrame by calculating their mean.
    If only one side exists, use it as-is. The new column is named by removing the side suffix.
    If filter_outliers is True, 10% of samples with the highest absolute values are removed
    per each side before computing the mean.
    """
    df = df.copy()
    columns = set(df.columns)
    merged = {}
    used_cols = set()

    for col in columns:
        if col.endswith('_left'):
            base = col[:-5]
            right_col = base + '_right'
            if right_col in columns:
                if filter_outliers:
                    # Remove top 10% from each side
                    s_left = df[col].copy()
                    s_right = df[right_col].copy()
                    
                    # Compute thresholds
                    q_left = s_left.abs().quantile(0.9)
                    q_right = s_right.abs().quantile(0.9)
                    
                    # Mark outliers as NaN so they don't contribute to the mean
                    s_left[s_left.abs() > q_left] = np.nan
                    s_right[s_right.abs() > q_right] = np.nan
                    
                    # Mean of the non-NaN values
                    merged[base] = pd.concat([s_left, s_right], axis=1).mean(axis=1)
                else:
                    merged[base] = df[[col, right_col]].mean(axis=1)
                used_cols.update([col, right_col])
            else:
                if filter_outliers:
                    s = df[col].copy()
                    q = s.abs().quantile(0.9)
                    s[s.abs() > q] = np.nan
                    merged[base] = s
                else:
                    merged[base] = df[col]
                used_cols.add(col)
        elif col.endswith('_right'):
            base = col[:-6]
            left_col = base + '_left'
            if left_col not in columns:
                if filter_outliers:
                    s = df[col].copy()
                    q = s.abs().quantile(0.9)
                    s[s.abs() > q] = np.nan
                    merged[base] = s
                else:
                    merged[base] = df[col]
                used_cols.add(col)

    for col in df.columns:
        if col not in used_cols:
            if filter_outliers:
                # Also filter non-merged columns if they are being processed here
                # But wait, we only want to filter directional ones.
                # However, this function is called once per DF.
                # We can check if it's likely a directional column.
                # But to be safe and match prepare_long_format, let's only filter if requested.
                s = df[col].copy()
                try:
                    # Only filter if numeric
                    if pd.api.types.is_numeric_dtype(s):
                        q = s.abs().quantile(0.9)
                        s[s.abs() > q] = np.nan
                except:
                    pass
                merged[col] = s
            else:
                merged[col] = df[col]

    new_cols_order = []
    for col in df.columns:
        if col.endswith('_left') or col.endswith('_right'):
            base = re.sub(r'_(left|right)$', '', col)
            if base not in new_cols_order and base in merged:
                new_cols_order.append(base)
        elif col in merged and col not in new_cols_order:
            new_cols_order.append(col)

    for col in merged:
        if col not in new_cols_order:
            new_cols_order.append(col)

    return pd.DataFrame(merged)[new_cols_order]

def prepare_long_format(df, prefixes, suffixes=None, dataset_label='A', filter_outliers=False):
    rows = []
    # Identify unique suffixes present in the dataframe that match our interest
    found_suffixes = set()
    for col in df.columns:
        for p in prefixes:
            if col.startswith(p + '_'):
                suffix = col[len(p) + 1:]
                if suffixes is None or suffix in suffixes:
                    found_suffixes.add(suffix)
                break
    
    # If suffixes were provided, use them to maintain order, otherwise use found ones
    active_suffixes = suffixes if suffixes is not None else sorted(list(found_suffixes))

    for suffix in active_suffixes:
        relevant_cols = []
        for p in prefixes:
            col = f"{p}_{suffix}"
            if col in df.columns:
                relevant_cols.append(col)
        
        if not relevant_cols:
            continue

        # Get values for all matching prefixes for this suffix
        # We take the mean across the directions (prefixes) for each sample
        # Note: df[relevant_cols] might have NaN if filter_outliers was True in merge_left_right
        # but the requirement says do not filter outliers now.
        values = df[relevant_cols].abs().mean(axis=1)

        if filter_outliers:
            threshold = values.quantile(0.9)
            values = values[values <= threshold]

        rows.append({
            'suffix': suffix,
            'value': values,
            'dataset': dataset_label
        })

    if not rows:
        return pd.DataFrame(columns=['suffix', 'value', 'dataset'])
    
    all_rows = []
    for row in rows:
        temp_df = pd.DataFrame({
            'suffix': row['suffix'],
            'value': row['value'].dropna() if filter_outliers else row['value'],
            'dataset': row['dataset']
        })
        all_rows.append(temp_df)
        
    long_df = pd.concat(all_rows, ignore_index=True)
    return long_df

def get_ordered_columns(df, prefixes, suffixes=None):
    ordered = []
    for col in df.columns:
        for p in prefixes:
            if col.startswith(p + '_'):
                suffix = col[len(p) + 1:]
                if suffixes is None or suffix in suffixes:
                    ordered.append(col)
                    break
    # Sort suffixes to maintain a consistent order if suffixes were provided
    if suffixes is not None:
        ordered = sorted(ordered, key=lambda x: suffixes.index(x.split('_', 1)[1]) if x.split('_', 1)[1] in suffixes else 999)
    return ordered

def plot_boxplots(df_list, prefixes, suffixes=None, title='', aspect_ratio=2.5, labels=None, y_label='DICE', prefix_to_cut='', column_name_map=None, save_path='', filter_outliers=False):
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(df_list))]

    # Merge left/right columns
    df_list_merged = [merge_left_right_columns(df, filter_outliers=filter_outliers) for df in df_list]

    # Prepare long format
    long_dfs = [
        prepare_long_format(df, prefixes, suffixes, dataset_label=label, filter_outliers=filter_outliers)
        for df, label in zip(df_list_merged, labels)
    ]
    combined_df = pd.concat(long_dfs, ignore_index=True)

    if combined_df.empty:
        print(f"No data found for suffixes: {suffixes}")
        return

    # Determine order of suffixes for plotting
    unique_suffixes = combined_df['suffix'].unique()
    if suffixes is not None:
        ordered_suffixes = [s for s in suffixes if s in unique_suffixes]
    else:
        ordered_suffixes = sorted(list(unique_suffixes))

    # Renaming of suffixes for display
    if column_name_map is not None:
        # Try to find a mapping for the suffix.
        # Since column_name_map keys are like "dice_spleen", we try a few common prefixes.
        common_prefixes = ['dice', 'sd', 'left', 'right', 'inferior', 'superior', 'anterior', 'posterior']
        display_names = []
        for s in ordered_suffixes:
            name = s
            # Try specific prefixes first if they are in the list of prefixes being plotted
            for p in prefixes + common_prefixes:
                if f"{p}_{s}" in column_name_map:
                    name = column_name_map[f"{p}_{s}"]
                    break
            display_names.append(name)
    else:
        display_names = ordered_suffixes

    plt.figure(figsize=(aspect_ratio * 4, 6))
    ax = sns.boxplot(
        data=combined_df,
        x='suffix',
        y='value',
        hue='dataset',
        order=ordered_suffixes,
        hue_order=labels,
        palette="muted",
        showfliers=False,
        whis=(10, 90)
    )

    if prefixes[0] in ['dice']:
        ax.set_ylim(0, 1)
    elif any(p in ['sd', 'left', 'right', 'inferior', 'superior', 'anterior', 'posterior'] for p in prefixes):
        ax.set_ylim(bottom=0)
    ax.set_ylabel(y_label)
    ax.set_xlabel('')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(ordered_suffixes)))
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.legend(title='Method')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

def calculate_and_print_stats(df_list, labels, prefixes, suffixes, metric_label):
    """
    Calculate and print mean and std value for a metric averaged across all organs and directions.
    """
    print(f"\n--- Global Stats for {metric_label} ---")
    
    # Merge left/right columns
    df_list_merged = [merge_left_right_columns(df, filter_outliers=False) for df in df_list]

    for df, label in zip(df_list_merged, labels):
        # Prepare long format
        # prepare_long_format already averages across prefixes (directions) for each sample and organ
        long_df = prepare_long_format(df, prefixes, suffixes, dataset_label=label, filter_outliers=False)
        
        if long_df.empty:
            print(f"{label}: No data")
            continue
            
        mean_val = long_df['value'].mean()
        std_val = long_df['value'].std()
        print(f"{label}: {mean_val:.4f} ± {std_val:.4f}")

def main():
    try:
        df_3d = pd.read_csv(results_3d)
        df_25d = pd.read_csv(results_25d)
        df_mm = pd.read_csv(results_mm)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    labels = ('Pix2Vox', 'Mean model', 'CNN 2.5D')
    column_name_map = {}
    for prefix in ['dice', 'sd', 'left', 'right', 'inferior', 'superior', 'anterior', 'posterior']:
        column_name_map.update({
            f"{prefix}_spleen": "Spleen",
            f"{prefix}_kidney": "Kidney",
            f"{prefix}_liver": "Liver",
            f"{prefix}_stomach": "Stomach",
            f"{prefix}_pancreas": "Pancreas",
            f"{prefix}_lung": "Lung",
            f"{prefix}_trachea": "Trachea",
            f"{prefix}_thyroid_gland": "Thyroid Gland",
            f"{prefix}_duodenum": "Duodenum",
            f"{prefix}_urinary_bladder": "Urinary Bladder",
            f"{prefix}_heart": "Heart",
            f"{prefix}_aorta": "Aorta",
            f"{prefix}_scapula": "Scapula",
            f"{prefix}_clavicula": "Clavicula",
            f"{prefix}_femur": "Femur",
            f"{prefix}_hip": "Hip",
            f"{prefix}_sacrum": "Sacrum",
            f"{prefix}_vertebrae_L5": "Vertebrae L5",
            f"{prefix}_vertebrae_L4": "Vertebrae L4",
            f"{prefix}_vertebrae_L3": "Vertebrae L3",
            f"{prefix}_vertebrae_L2": "Vertebrae L2",
            f"{prefix}_vertebrae_L1": "Vertebrae L1",
            f"{prefix}_vertebrae_T12": "Vertebrae T12",
            f"{prefix}_vertebrae_T11": "Vertebrae T11",
            f"{prefix}_vertebrae_T10": "Vertebrae T10",
            f"{prefix}_vertebrae_T9": "Vertebrae T9",
            f"{prefix}_vertebrae_T8": "Vertebrae T8",
            f"{prefix}_vertebrae_T7": "Vertebrae T7",
            f"{prefix}_vertebrae_T6": "Vertebrae T6",
            f"{prefix}_vertebrae_T5": "Vertebrae T5",
            f"{prefix}_vertebrae_T4": "Vertebrae T4",
            f"{prefix}_vertebrae_T3": "Vertebrae T3",
            f"{prefix}_vertebrae_T2": "Vertebrae T2",
            f"{prefix}_vertebrae_T1": "Vertebrae T1",
        })

    # Group 1: Sacrum and all vertebrae
    vertebrae_suffixes = [
        "sacrum",
        "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
        "vertebrae_T12", "vertebrae_T11", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8",
        "vertebrae_T7", "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3",
        "vertebrae_T2", "vertebrae_T1"
    ]

    # Group 2: All other organs
    other_suffixes = [
        "spleen", "kidney", "liver", "stomach", "pancreas", "lung", "trachea",
        "thyroid_gland", "duodenum", "urinary_bladder", "aorta", "scapula",
        "clavicula", "femur", "hip", "heart"
    ]

    # Dice Plots
    plot_boxplots(
        [df_3d, df_mm],
        prefixes=['dice'],
        suffixes=vertebrae_suffixes,
        labels=labels[:2],
        title='',
        y_label='DICE',
        prefix_to_cut='',
        column_name_map=column_name_map,
        aspect_ratio=4.0,
        save_path='/home/kats/storage/staff/eytankats/projects/orgloc/plots/dice_vertebrae.pdf'
    )

    plot_boxplots(
        [df_3d, df_mm],
        prefixes=['dice'],
        suffixes=other_suffixes,
        labels=labels[:2],
        title='',
        y_label='DICE',
        prefix_to_cut='',
        column_name_map=column_name_map,
        aspect_ratio=4.0,
        save_path='/home/kats/storage/staff/eytankats/projects/orgloc/plots/dice_others.pdf'
    )

    # SD Plots
    plot_boxplots(
        [df_3d, df_mm],
        prefixes=['sd'],
        suffixes=vertebrae_suffixes,
        labels=labels[:2],
        title='',
        y_label='Average Symmetric Surface Distance (mm)',
        prefix_to_cut='',
        column_name_map=column_name_map,
        aspect_ratio=4.0,
        save_path='/home/kats/storage/staff/eytankats/projects/orgloc/plots/sd_vertebrae.pdf'
    )

    plot_boxplots(
        [df_3d, df_mm],
        prefixes=['sd'],
        suffixes=other_suffixes,
        labels=labels[:2],
        title='',
        y_label='Surface Distance (mm)',
        prefix_to_cut='',
        column_name_map=column_name_map,
        aspect_ratio=4.0,
        save_path='/home/kats/storage/staff/eytankats/projects/orgloc/plots/sd_others.pdf'
    )

    # Directional Offset Plots
    all_directions = ['left', 'right', 'inferior', 'superior', 'anterior', 'posterior']
    
    plot_boxplots(
        [df_3d, df_mm, df_25d],
        prefixes=all_directions,
        suffixes=vertebrae_suffixes,
        labels=labels,
        title='',
        y_label='Mean Absolute Detection Offset (mm)',
        prefix_to_cut='',  # Handled by column_name_map or suffixes
        column_name_map=column_name_map,
        aspect_ratio=4.0,
        save_path='/home/kats/storage/staff/eytankats/projects/orgloc/plots/avg_offset_vertebrae.pdf',
        filter_outliers=False
    )

    plot_boxplots(
        [df_3d, df_mm, df_25d],
        prefixes=all_directions,
        suffixes=other_suffixes,
        labels=labels,
        title='',
        y_label='Mean Absolute Detection Offset (mm)',
        prefix_to_cut='',  # Handled by column_name_map or suffixes
        column_name_map=column_name_map,
        aspect_ratio=4.0,
        save_path='/home/kats/storage/staff/eytankats/projects/orgloc/plots/avg_offset_others.pdf',
        filter_outliers=False
    )

    # Calculate and print global statistics
    all_suffixes = vertebrae_suffixes + other_suffixes
    
    # DICE Statistics
    calculate_and_print_stats(
        [df_3d, df_mm],
        labels[:2],
        prefixes=['dice'],
        suffixes=all_suffixes,
        metric_label='DICE'
    )
    
    # SD Statistics
    calculate_and_print_stats(
        [df_3d, df_mm],
        labels[:2],
        prefixes=['sd'],
        suffixes=all_suffixes,
        metric_label='Surface Distance (mm)'
    )
    
    # Offset Statistics
    calculate_and_print_stats(
        [df_3d, df_mm, df_25d],
        labels,
        prefixes=all_directions,
        suffixes=all_suffixes,
        metric_label='Detection Offset (mm)'
    )

if __name__ == "__main__":
    main()
