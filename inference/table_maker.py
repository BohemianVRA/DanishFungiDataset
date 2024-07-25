from typing import Set

import pandas as pd
import wandb


def get_results_df(tags: Set[str]) -> pd.DataFrame:
    """
    Fetches results from Weights & Biases runs filtered by specified tags.

    Parameters
    ----------
    tags : set of str
        Set of tags to filter the runs.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the final metrics of filtered runs.
    """
    api = wandb.Api()
    entity, project = "zcu_cv", "DanishFungi2024"

    filtered_tags = {"tags": {"$in": ["Production"]}}

    runs = api.runs(f"{entity}/{project}", filters=filtered_tags)
    results_map = {}

    for run in runs:
        if run.state.lower() != "finished":
            continue

        run_tags = run.tags
        if not tags.issubset(run_tags):
            continue

        history = run.history()
        metrics = history[["Val. Accuracy", "Val. Recall@3", "Val. F1"]]
        final_metrics = metrics.dropna().iloc[-1]

        results_map[run.name] = final_metrics

    results_df = pd.DataFrame.from_dict(
        results_map,
        columns=["Val. Accuracy", "Val. Recall@3", "Val. F1"],
        orient="index",
    )

    results_df *= 100
    results_df = results_df.round(decimals=2)

    return results_df


def main():
    """
    Main function to fetch results from Weights & Biases, process, and save them.
    """
    resolution_tag = "224x224"
    dataset_tag = "DF24_FIX"
    output_path = f"../output/{dataset_tag}_{resolution_tag}.txt"
    tags = {resolution_tag, "Production", dataset_tag}

    results_df = get_results_df(tags)
    results_df = results_df.sort_values(["Val. Accuracy", "Val. Recall@3", "Val. F1"])
    results_df.sort_index(inplace=True)

    result_message = (
        f"\n{dataset_tag}-{resolution_tag}: Production\n{results_df.to_markdown()}\n"
    )
    print(result_message)

    with open(output_path, "a") as f:
        f.write(result_message)


if __name__ == "__main__":
    main()
