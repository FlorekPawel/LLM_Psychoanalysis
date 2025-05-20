import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def generate_distribution_plots(result_path, model_name):
    def generate_plot(data, name):
        length = data.shape[0]
        data = data.reset_index()
        exploded = data[f"{name.upper()}"].apply(pd.Series)
        data = pd.concat([data, exploded], axis=1)
        data = data.drop(f"{name.upper()}", axis=1)
        data = data.melt(
            id_vars="index",
            var_name="key",
            value_name="value",
            value_vars=["1", "2", "3", "4", "5"],
        )

        fig, axes = plt.subplots(1, length, figsize=(4 * length, 4))

        for i, index in enumerate(np.unique(data["index"])):
            axes[i].bar(
                data[data["index"] == index]["key"],
                data[data["index"] == index]["value"],
            )
            axes[i].set_title(index)
            axes[i].set_ylabel("Proportion")
            axes[i].set_ylim(0, 100)
            axes[i].label_outer()

        fig.suptitle(
            f"{name.upper()} data proportion by trait for {model}", fontsize=16
        )
        fig.supxlabel("Answer")
        plt.tight_layout()

        for i, index in enumerate(np.unique(data["index"])):
            for bar in axes[i].patches:
                height = bar.get_height()
                axes[i].annotate(
                    f"{height:.2f}",
                    (bar.get_x() + bar.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        fig.savefig(os.path.join(f"./Prompt_code/plots/{model_name}", f"{name}_distribution.png"))
        plt.close()

    model = model_name.replace("-", " ")
    df = pd.read_json(os.path.join(result_path, "distribution.json"))

    bfi = df.iloc[0:5, 0]
    generate_plot(bfi, "bfi")

    panas = df.iloc[5:7, 1]
    generate_plot(panas, "panas")

    bpaq = df.iloc[7:11, 2]
    generate_plot(bpaq, "bpaq")

    sscs = df.iloc[11:, 3]
    generate_plot(sscs, "sscs")


def generate_reliability_coeff_plots(result_path, model_name):
    def generate_plot(data, name):
        length = data.shape[0]
        data = data.reset_index()
        exploded = data[f"{name.upper()}"].apply(pd.Series)
        data = pd.concat([data, exploded], axis=1)
        data = data.drop(f"{name.upper()}", axis=1)
        data = data.melt(
            id_vars="index",
            var_name="key",
            value_name="value",
            value_vars=["Cronbach's Alpha", "GLB", "Omega"],
        )

        fig, axes = plt.subplots(1, length, figsize=(4 * length, 4))

        for i, index in enumerate(np.unique(data["index"])):
            axes[i].bar(
                data[data["index"] == index]["key"],
                data[data["index"] == index]["value"],
            )
            axes[i].set_title(index)
            axes[i].set_ylabel("Reliability coefficient's value")
            axes[i].set_ylim(0, 1)
            axes[i].label_outer()

        fig.suptitle(
            f"{name.upper()} data reliability coefficients by trait for {model}",
            fontsize=16,
        )
        fig.supxlabel("Coefficient")
        plt.tight_layout()

        for i, index in enumerate(np.unique(data["index"])):
            for bar in axes[i].patches:
                height = bar.get_height()
                axes[i].annotate(
                    f"{height:.2f}",
                    (bar.get_x() + bar.get_width() / 2, height),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        fig.savefig(os.path.join(f"./Prompt_code/plots/{model_name}", f"{name}_reliability.png"))
        plt.close()

    model = model_name.replace("-", " ")
    df = pd.read_json(os.path.join(result_path, "reliability_stats.json"))

    bfi = df.iloc[0:5, 0]
    generate_plot(bfi, "bfi")

    panas = df.iloc[5:7, 1]
    generate_plot(panas, "panas")

    bpaq = df.iloc[7:11, 2]
    generate_plot(bpaq, "bpaq")

    sscs = df.iloc[11:, 3]
    generate_plot(sscs, "sscs")


def generate_correlation_plots(result_path, model_name):
    def generate_plot(data, name):
        length = data.shape[0]
        data = data.reset_index()
        exploded = data[f"{name.upper()}"].apply(pd.Series)
        data = pd.concat([data, exploded], axis=1)
        data = data.drop(f"{name.upper()}", axis=1)
        data.index = data["index"]
        data = data.drop("index", axis=1)

        plt.figure(figsize=(12, 9))
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            cbar=True,
            vmin=-1,
            vmax=1,
        )
        plt.gca().invert_xaxis()
        plt.title(f"{name.upper()} traits correlation for {model}")
        plt.ylabel("")
        plt.tight_layout()

        plt.savefig(os.path.join(f"./Prompt_code/plots/{model_name}", f"{name}_correlation.png"))
        plt.close()

    model = model_name.replace("-", " ")
    df = pd.read_json(os.path.join(result_path, "subscale_correlations.json"))

    bfi = df.iloc[0:5, 0]
    generate_plot(bfi, "bfi")

    panas = df.iloc[5:7, 1]
    generate_plot(panas, "panas")

    bpaq = df.iloc[7:11, 2]
    generate_plot(bpaq, "bpaq")

    sscs = df.iloc[11:, 3]
    generate_plot(sscs, "sscs")


def generate_crit_val_plot(result_path, model_name):
    model = model_name.replace("-", " ")
    df = pd.read_json(os.path.join(result_path, "criterion_validity.json"))
    df = df.pivot(index="Construct", columns="Big Five Trait", values="Correlation")

    plt.figure(figsize=(9, 10))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar=True,
        vmin=-1,
        vmax=1,
    )
    plt.title(f"Criterion validity correlations for {model}")
    plt.tight_layout()
    plt.gca().invert_xaxis()
    plt.ylabel("Constructs")
    plt.xlabel("Big Five Traits")
    plt.savefig(os.path.join(f"./Prompt_code/plots/{model_name}", f"crit_val_correlation.png"))
    plt.close()


def main():
    for model_name in os.listdir("./Prompt_code/persona_results"):
        results_path = os.path.join("./Prompt_code/persona_results", model_name, "results")
        os.makedirs(os.path.join("plots", model_name), exist_ok=True)

        generate_distribution_plots(results_path, model_name)
        generate_reliability_coeff_plots(results_path, model_name)
        generate_correlation_plots(results_path, model_name)
        generate_crit_val_plot(results_path, model_name)


if __name__ == "__main__":
    main()
