import pandas as pd
import yaml


def process_benchmark(
    df: pd.DataFrame,
    tasklist_path: str = "configs/eval/tasklist.yml",
) -> pd.DataFrame:
    """Process CLIP benchmark results: parse model_fullname, clean dataset names,
    and add preferred_metric / preferred_metric_value columns.

    For retrieval tasks, rows are melted into two rows (image and text retrieval).
    For linear_probe tasks, the lp_ prefix is applied to the metric name.
    Datasets not in the tasklist default to acc1.
    """
    df = df.copy()

    # --- Parse model_fullname ---
    parts = df["model_fullname"].str.split("/", expand=True)
    df['checkpoint_path'] = df['pretrained'].apply(lambda x: x.rsplit('/', maxsplit=1)[0])
    df["pretrain_dataset"] = parts.iloc[:, -3].str.split("_", n=1).str[0]
    df["ablation"] = parts.iloc[:, -3].str.split("_", n=1).str[1].str.rsplit("_", n=1).str[0]
    df["ablation_type"] = df["ablation"].apply(lambda x: "architecture" if x.startswith("architecture") else "caption")
    df["checkpoint_type"] = parts.iloc[:, -1].str.replace("epoch_", "").str.replace(".pt", "")
    df["compression_target"] = (
        df["ablation"]
        .apply(lambda x: x.split("_", maxsplit=1)[1] if "caption" in x else pd.NA)
        .replace({"caption": "0"})
    )

    # --- Clean dataset name (strip wds/ prefix) ---
    df["dataset"] = df["dataset"].str.replace("wds/", "", n=1)

    # --- Add preferred metric ---
    with open(tasklist_path) as f:
        tasklist = yaml.safe_load(f)

    metric_lookup = {key: info.get("main_metric", "acc1") for key, info in tasklist.items()}
    tags_lookup = {key: "|".join(info.get("tags", [])) for key, info in tasklist.items()}
    df["tags"] = df["dataset"].map(tags_lookup).fillna("")

    is_retrieval = df["task"] == "zeroshot_retrieval"
    df_non_ret = df[~is_retrieval].copy()
    df_ret = df[is_retrieval].copy()

    # Non-retrieval
    base_metrics = df_non_ret["dataset"].map(metric_lookup).fillna("acc1")
    prefix = df_non_ret["task"].map({"linear_probe": "lp_"}).fillna("")
    df_non_ret["preferred_metric"] = prefix + base_metrics
    df_non_ret["preferred_metric_value"] = df_non_ret.apply(
        lambda r: r.get(r["preferred_metric"]), axis=1
    )

    # Retrieval: melt image & text recall into separate rows
    parts_list = [df_non_ret]
    if len(df_ret):
        for col in ["image_retrieval_recall@1", "text_retrieval_recall@1"]:
            tmp = df_ret.copy()
            tmp["preferred_metric"] = col
            tmp["preferred_metric_value"] = tmp[col]
            parts_list.append(tmp)

    return pd.concat(parts_list, ignore_index=True)
