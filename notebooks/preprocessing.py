from typing import Optional, Tuple
import pandas as pd


def get_models_and_soups_df(path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the soups and models dataframes from the specified path.
    Args:
        path (Optional[str]): The path to the parquet file containing the data.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the soups dataframe and the models dataframe.
    """
    if path is None:
        path = "../analysis/combined_analysis.parquet"
    df = pd.read_parquet(path)
    df.rename(columns={"acc_corr_3": "corrupted_accuracy", "loss_corr_3": "corrupted_loss"}, inplace=True)

    same_df = df[df.key_a == df.key_b].copy()
    soups = df[~(df.key_a == df.key_b)].copy()

    models_df = same_df[["key_a", "clean_loss", "clean_accuracy", "corrupted_accuracy", "corrupted_loss"]].copy()
    models_df = models_df.rename(columns={"key_a": "key"})
    models_df = models_df.drop_duplicates(subset="key")
    models_df[["epoch", "variant"]] = models_df["key"].str.split("_", expand=True).astype(int)

    soups[["epoch_a", "variant_a"]] = soups["key_a"].str.split("_", expand=True).astype(int)
    soups[["epoch_b", "variant_b"]] = soups["key_b"].str.split("_", expand=True).astype(int)
    soups["shared_epochs"] = soups[["epoch_a", "epoch_b"]].min(axis=1)

    soups = soups.merge(
        models_df[
            [
                "key",
                "clean_loss",
                "clean_accuracy",
                "corrupted_loss",
            ]
        ].rename(
            columns={
                "key": "key_a",
                "clean_loss": "clean_loss_a",
                "clean_accuracy": "clean_accuracy_a",
                "corrupted_loss": "corrupted_loss_a",
            }
        ),
        on="key_a",
        how="left",
    )

    soups = soups.merge(
        models_df[["key", "clean_loss", "clean_accuracy", "corrupted_loss"]].rename(
            columns={
                "key": "key_b",
                "clean_loss": "clean_loss_b",
                "clean_accuracy": "clean_accuracy_b",
                "corrupted_loss": "corrupted_loss_b",
            }
        ),
        on="key_b",
        how="left",
    )

    # Souping gain in loss: min(L(A), L(B)) - L(soup)
    soups["soup_gain"] = soups[["clean_loss_a", "clean_loss_b"]].min(axis=1) - soups["clean_loss"]
    soups["permutated_gain"] = soups[["clean_loss_a", "clean_loss_b"]].min(axis=1) - soups["clean_loss_permuted"]
    soups["corrupted_gain"] = soups[["corrupted_loss_a", "corrupted_loss_b"]].min(axis=1) - soups["corrupted_loss"]

    soups = soups[
        [
            "key_a",
            "key_b",
            "epoch_a",
            "variant_a",
            "epoch_b",
            "variant_b",
            "shared_epochs",
            "clean_accuracy",
            "clean_loss",
            "corrupted_accuracy",
            "corrupted_loss",
            "clean_accuracy_permuted",
            "clean_loss_permuted",
            "l2_distance",
            "cosine_similarity",
            "cka_logits",
            "mse_logits",
            "kl_logits",
            "cka_features",
            "clean_loss_a",
            "clean_accuracy_a",
            "clean_loss_b",
            "clean_accuracy_b",
            "corrupted_loss_a",
            "corrupted_loss_b",
            "soup_gain",
            "permutated_gain",
            "corrupted_gain",
        ]
    ]
    return soups, models_df
