import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


results_3d = '/home/kats/storage/staff/eytankats/projects/orgloc/experiments/202507082234_multidim_multilabel_alllabels_weighted3_unet_processed_aggmasksv2/results_100.csv'
results_25d = '/home/kats/storage/staff/eytankats/projects/orgloc/experiments/202507141821_multidim_multilabel_alllabels_weighted5_25dim_unet_processed_aggmasksv2/results_100.csv'
results_mm = '/home/kats/storage/staff/eytankats/projects/orgloc/experiments/mean_model/results_100.csv'

df_3d = pd.read_csv(results_3d)
df_25d = pd.read_csv(results_25d)
df_mm = pd.read_csv(results_mm)

def merge_left_right_columns(df):
    """
    Merge columns with '_left' and '_right' suffixes in a DataFrame by calculating their mean.
    If only one side exists, use it as-is. The new column is named by removing the side suffix.

    Parameters:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Modified DataFrame with merged columns
    """
    df = df.copy()
    columns = set(df.columns)
    merged = {}
    used_cols = set()

    # Find all _left/_right pairs
    for col in columns:
        if col.endswith('_left'):
            base = col[:-5]
            right_col = base + '_right'
            if right_col in columns:
                # Both sides exist
                merged[base] = df[[col, right_col]].mean(axis=1)
                used_cols.update([col, right_col])
            else:
                # Only left side
                merged[base] = df[col]
                used_cols.add(col)

        elif col.endswith('_right'):
            base = col[:-6]
            left_col = base + '_left'
            if left_col not in columns:
                # Only right side
                merged[base] = df[col]
                used_cols.add(col)

    # Add non-merged columns
    for col in df.columns:
        if col not in used_cols:
            merged[col] = df[col]

    # Construct new DataFrame in original column order where possible
    new_cols_order = []
    for col in df.columns:
        if col.endswith('_left') or col.endswith('_right'):
            base = re.sub(r'_(left|right)$', '', col)
            if base not in new_cols_order and base in merged:
                new_cols_order.append(base)
        elif col in merged and col not in new_cols_order:
            new_cols_order.append(col)

    # Include any additional merged keys
    for col in merged:
        if col not in new_cols_order:
            new_cols_order.append(col)

    return pd.DataFrame(merged)[new_cols_order]

def merge_vertebrae_columns_multiple_prefixes(df, prefixes):
    """
    For each prefix, merges vertebrae columns like '<prefix>_vertebrae_T*' and '<prefix>_vertebrae_L*'
    by computing the row-wise mean. Adds new columns like '<prefix>_vertebrae_t' and '<prefix>_vertebrae_l'.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        prefixes (list of str): List of prefixes to match (e.g., ['dice', 'haus'])

    Returns:
        pd.DataFrame: Modified DataFrame with merged vertebrae columns
    """
    df = df.copy()
    all_to_drop = []
    merged_data = {}

    for prefix in prefixes:
        # Find all matching thoracic and lumbar columns
        t_cols = [col for col in df.columns if re.fullmatch(f"{prefix}_vertebrae_T\\d+", col)]
        l_cols = [col for col in df.columns if re.fullmatch(f"{prefix}_vertebrae_L\\d+", col)]

        # Compute means
        if t_cols:
            merged_data[f"{prefix}_vertebrae_T"] = df[t_cols].mean(axis=1)
            all_to_drop.extend(t_cols)

        if l_cols:
            merged_data[f"{prefix}_vertebrae_L"] = df[l_cols].mean(axis=1)
            all_to_drop.extend(l_cols)

    # Drop the original vertebrae columns
    df.drop(columns=all_to_drop, inplace=True)

    # Add the merged columns
    df = pd.concat([df, pd.DataFrame(merged_data)], axis=1)

    return df

def prepare_long_format_2(df, prefixes, suffixes=None, dataset_label='A'):
    rows = []
    for col in df.columns:
        for p in prefixes:
            if col.startswith(p + '_'):
                suffix = col[len(p) + 1:]

                if suffixes is not None and suffix not in suffixes:
                    continue

                rows.append({
                    'column': col,
                    'value': df[col].abs(),
                    'dataset': dataset_label
                })
                break
    long_df = pd.concat([
        pd.DataFrame({
            'column': row['column'],
            'value': row['value'],
            'dataset': row['dataset']
        }) for row in rows
    ], ignore_index=True)
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
    return ordered

def calculate_and_print_vol_stats(df_list, labels, prefixes, suffixes, metric_label, column_name_map=None):
    """
    Calculate and print mean and std value for a metric averaged across all organs,
    as well as separately for each organ.
    """
    print(f"\n--- Statistics for {metric_label} ---")
    
    # Preprocess dataframes to match plotting logic
    df_list_processed = []
    for df in df_list:
        df_p = merge_left_right_columns(df)
        df_p = merge_vertebrae_columns_multiple_prefixes(df_p, prefixes)
        df_list_processed.append(df_p)

    for df, label in zip(df_list_processed, labels):
        print(f"\nDataset: {label}")
        
        all_values = []
        organ_stats = []
        
        for suffix in suffixes:
            # Find the actual column name for this prefix and suffix
            col = None
            for p in prefixes:
                potential_col = f"{p}_{suffix}"
                if potential_col in df.columns:
                    col = potential_col
                    break
            
            if col and col in df.columns:
                vals = df[col].dropna()
                if not vals.empty:
                    mean_val = vals.mean()
                    std_val = vals.std()
                    display_name = column_name_map.get(col, suffix) if column_name_map else suffix
                    organ_stats.append(f"  {display_name:<25}: {mean_val:.4f} ± {std_val:.4f}")
                    all_values.extend(vals.tolist())
        
        # Print separate organ stats
        for stat in organ_stats:
            print(stat)
            
        # Print global mean
        if all_values:
            global_mean = np.mean(all_values)
            global_std = np.std(all_values)
            print(f"  {'OVERALL MEAN':<25}: {global_mean:.4f} ± {global_std:.4f}")
        else:
            print("  No data available for overall mean.")

def plot_columns_stacked_barplot(df_list, prefixes, suffixes=None,
                                 title='Mean Absolute Error by Column',
                                 aspect_ratio=2.5,
                                 labels=None,
                                 y_label='',
                                 prefix_to_cut='',
                                 merge=False,
                                 column_name_map=None,
                                 save_path=''):

    if merge:
        df_list = [merge_left_right_columns(df) for df in df_list]
        df_list = [merge_vertebrae_columns_multiple_prefixes(df, prefixes) for df in df_list]

    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(df_list))]

    ordered_columns = get_ordered_columns(df_list[0], prefixes, suffixes)

    # Optional renaming of columns
    stripped_columns = [col[len(prefix_to_cut):] if col.startswith(prefix_to_cut) else col for col in ordered_columns]
    if column_name_map is not None:
        stripped_columns = [column_name_map.get(col, col) for col in ordered_columns]

    # Prepare long format
    long_dfs = [
        prepare_long_format_2(df, prefixes, suffixes, dataset_label=label)
        for df, label in zip(df_list, labels)
    ]
    combined_df = pd.concat(long_dfs, ignore_index=True)

    x = np.arange(len(ordered_columns))  # positions for each group

    plt.figure(figsize=(aspect_ratio * 4, 6))

    ax = sns.boxplot(
        data=combined_df,
        x='column',
        y='value',
        hue='dataset',
        order=ordered_columns,
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
    ax.set_xticklabels(stripped_columns, rotation=45, ha='right')
    ax.legend(title='Method')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    # --- Save to file if path is provided ---
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


plot_columns_stacked_barplot(
    [df_3d, df_mm],
    prefixes=['dice'],
    suffixes = [
    "spleen",
    "kidney",
    "liver",
    "stomach",
    "pancreas",
    "lung",
    "trachea",
    "thyroid_gland",
    "duodenum",
    "urinary_bladder",
    "heart",
    "aorta",
    "scapula",
    "clavicula",
    "femur",
    "hip",
    "sacrum",
    "vertebrae_L",
    "vertebrae_T",
],
    labels=('Pix2Vox', 'Mean model'),
    title='',
    y_label='DICE',
    prefix_to_cut='dice_',
    aspect_ratio=3.2,
    merge=True,
    column_name_map={
    "dice_spleen": "Spleen",
    "dice_kidney": "Kidney",
    "dice_liver": "Liver",
    "dice_stomach": "Stomach",
    "dice_pancreas": "Pancreas",
    "dice_lung": "Lung",
    "dice_trachea": "Trachea",
    "dice_thyroid_gland": "Thyroid Gland",
    "dice_duodenum": "Duodenum",
    "dice_urinary_bladder": "Urinary Bladder",
    "dice_heart": "Heart",
    "dice_aorta": "Aorta",
    "dice_scapula": "Scapula",
    "dice_clavicula": "Clavicula",
    "dice_femur": "Femur",
    "dice_hip":"Hip",
    "dice_sacrum": "Sacrum",
    "dice_vertebrae_L": "Vertebrae Lumbar",
    "dice_vertebrae_T": "Vertebrae Thoracic",
    },
    save_path='/home/kats/storage/staff/eytankats/projects/orgloc/plots/dice_2.pdf'
)

calculate_and_print_vol_stats(
    [df_3d, df_mm],
    labels=('Pix2Vox', 'Mean model'),
    prefixes=['dice'],
    suffixes = [
    "spleen",
    "kidney",
    "liver",
    "stomach",
    "pancreas",
    "lung",
    "trachea",
    "thyroid_gland",
    "duodenum",
    "urinary_bladder",
    "heart",
    "aorta",
    "scapula",
    "clavicula",
    "femur",
    "hip",
    "sacrum",
    "vertebrae_L",
    "vertebrae_T",
    ],
    metric_label='DICE',
    column_name_map={
    "dice_spleen": "Spleen",
    "dice_kidney": "Kidney",
    "dice_liver": "Liver",
    "dice_stomach": "Stomach",
    "dice_pancreas": "Pancreas",
    "dice_lung": "Lung",
    "dice_trachea": "Trachea",
    "dice_thyroid_gland": "Thyroid Gland",
    "dice_duodenum": "Duodenum",
    "dice_urinary_bladder": "Urinary Bladder",
    "dice_heart": "Heart",
    "dice_aorta": "Aorta",
    "dice_scapula": "Scapula",
    "dice_clavicula": "Clavicula",
    "dice_femur": "Femur",
    "dice_hip":"Hip",
    "dice_sacrum": "Sacrum",
    "dice_vertebrae_L": "Vertebrae Lumbar",
    "dice_vertebrae_T": "Vertebrae Thoracic",
    }
)

plot_columns_stacked_barplot(
    [df_3d, df_mm],
    prefixes=['sd'],
    suffixes = [
    "spleen",
    "kidney",
    "liver",
    "stomach",
    "pancreas",
    "lung",
    "trachea",
    "thyroid_gland",
    "duodenum",
    "urinary_bladder",
    "heart",
    "aorta",
    "scapula",
    "clavicula",
    "femur",
    "hip",
    "sacrum",
    "vertebrae_L",
    "vertebrae_T",
],
    labels=('Pix2Vox', 'Mean model'),
    title='',
    y_label='Average Symmetric Surface Distance (mm)',
    prefix_to_cut='sd_',
    aspect_ratio=3.2,
    merge=True,
    column_name_map={
    "sd_spleen": "Spleen",
    "sd_kidney": "Kidney",
    "sd_liver": "Liver",
    "sd_stomach": "Stomach",
    "sd_pancreas": "Pancreas",
    "sd_lung": "Lung",
    "sd_trachea": "Trachea",
    "sd_thyroid_gland": "Thyroid Gland",
    "sd_duodenum": "Duodenum",
    "sd_urinary_bladder": "Urinary Bladder",
    "sd_heart": "Heart",
    "sd_aorta": "Aorta",
    "sd_scapula": "Scapula",
    "sd_clavicula": "Clavicula",
    "sd_femur": "Femur",
    "sd_hip":"Hip",
    "sd_sacrum": "Sacrum",
    "sd_vertebrae_L": "Vertebrae Lumbar",
    "sd_vertebrae_T": "Vertebrae Thoracic",
    },
    save_path='/home/kats/storage/staff/eytankats/projects/orgloc/plots/sd_2.pdf'
)

calculate_and_print_vol_stats(
    [df_3d, df_mm],
    labels=('Pix2Vox', 'Mean model'),
    prefixes=['sd'],
    suffixes = [
    "spleen",
    "kidney",
    "liver",
    "stomach",
    "pancreas",
    "lung",
    "trachea",
    "thyroid_gland",
    "duodenum",
    "urinary_bladder",
    "heart",
    "aorta",
    "scapula",
    "clavicula",
    "femur",
    "hip",
    "sacrum",
    "vertebrae_L",
    "vertebrae_T",
    ],
    metric_label='Surface Distance (SD)',
    column_name_map={
    "sd_spleen": "Spleen",
    "sd_kidney": "Kidney",
    "sd_liver": "Liver",
    "sd_stomach": "Stomach",
    "sd_pancreas": "Pancreas",
    "sd_lung": "Lung",
    "sd_trachea": "Trachea",
    "sd_thyroid_gland": "Thyroid Gland",
    "sd_duodenum": "Duodenum",
    "sd_urinary_bladder": "Urinary Bladder",
    "sd_heart": "Heart",
    "sd_aorta": "Aorta",
    "sd_scapula": "Scapula",
    "sd_clavicula": "Clavicula",
    "sd_femur": "Femur",
    "sd_hip":"Hip",
    "sd_sacrum": "Sacrum",
    "sd_vertebrae_L": "Vertebrae Lumbar",
    "sd_vertebrae_T": "Vertebrae Thoracic",
    }
)