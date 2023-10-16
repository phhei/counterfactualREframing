import json
from typing import List

import plotly.graph_objects as go
from pathlib import Path

with Path("annotations_extended.json").open("r", encoding="utf-8") as annotation_file:
    data = json.load(annotation_file)
    data.pop("_agreement_krippendorff_alpha", None)

from typing import List
import plotly.graph_objs as go


def plot_grouped_inffos(group_names: List[str], result_name:str):
    # Separate strategies into specified group names
    strategy_groups = {group_name: {} for group_name in group_names}

    for strategy_name, strategy_data in data.items():
        for group_name in group_names:
            if group_name in strategy_name:
                strategy_groups[group_name][strategy_name] = strategy_data

    # Extract bins and counts for a_meaning for each strategy group
    def extract_bins_and_counts(strategy_group):
        valid_bins = ['-1', '0', '1', '10']
        bins = {bin_value: [0] * len(group_names) for bin_value in valid_bins}

        for group_index, (group_name, group_data) in enumerate(strategy_group.items()):
            a_meaning_bins = group_data[list(group_data.keys())[group_index]][result_name]["bins"]

            for bin_value in valid_bins:
                if bin_value in a_meaning_bins:
                    bin_count = a_meaning_bins[bin_value]
                    bins[bin_value][group_index] += bin_count

        return bins

    bins_data = extract_bins_and_counts(strategy_groups)

    # Create a grouped bar plot using Plotly
    fig = go.Figure()

    for bin_value, bin_counts in bins_data.items():
        fig.add_trace(
            go.Bar(
                x=group_names,
                y=bin_counts,
                name=f"Bin {bin_value}",
                hoverinfo="y+name",
            )
        )

    fig.update_layout(
        barmode="group",
        title=f"Bins of {result_name} for {group_names}",
        xaxis_title="Strategy Group",
        yaxis_title="Counts",
        legend_title="Bin Value"
    )

    fig.show()


if __name__ == "__main__":
    # Specify the group names you want to plot
    #plot_grouped_inffos(['remove', 'exchange'], "a_meaning")
    plot_grouped_inffos(['t5-small', 't5-large'], "a_meaning")
