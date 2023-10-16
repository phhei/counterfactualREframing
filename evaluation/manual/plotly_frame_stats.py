import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pathlib import Path


def create_frame_stats_plots(group_names):
    with Path("annotations_extended.json").open("r", encoding="utf-8") as annotation_file:
        data = json.load(annotation_file)
        data.pop("_agreement_krippendorff_alpha", None)

    strategy_group_data = {group_name: {} for group_name in group_names}

    for strategy_name, strategy_data in data.items():
        for group_name in group_names:
            if group_name in strategy_name:
                strategy_group_data[group_name][strategy_name] = strategy_data
            if group_name == "no_framed_decoding":
                if "framed_decoding" not in strategy_name:
                    strategy_group_data[group_name][strategy_name] = strategy_data

    # TODO: Define how many plots you want to see
    min_votes_values = [1, 2, 3]
    metrics = ["bool_successful_added_frames", "bool_successful_removed_frames", "bool_successful_hit_frame_set"]

    fig = make_subplots(rows=len(metrics), cols=len(min_votes_values), shared_xaxes=True, shared_yaxes=True,
                        vertical_spacing=0.1, subplot_titles=[f"min_votes = {mv}" for mv in min_votes_values])

    for i, metric in enumerate(metrics, start=1):
        for j, min_votes in enumerate(min_votes_values, start=1):
            for group_name in group_names:
                metric_data_group = extract_voted_data(strategy_group_data[group_name], metric, group_names, group_name,
                                                       min_votes)

                bins = list(set(list(metric_data_group.keys())))

                # Initialize an empty list to store counts for the current group and bins
                counts_group = []

                # Loop through each bin_value in the list of bins
                for bin_value in bins:
                    # Get the count data for the current bin_value, or use [0, 0, ...] for all groups if not available
                    counts_for_bin = metric_data_group.get(bin_value, [0] * len(group_names))

                    # Find the index of the current group within the group_names list
                    group_index = group_names.index(group_name)

                    # Get the count for the current group from the counts_for_bin list
                    count_for_group = counts_for_bin[group_index]

                    # Add the current group's count to the counts_group list
                    counts_group.append(count_for_group)

                fig.add_trace(
                    go.Bar(
                        x=bins,
                        y=counts_group,
                        name=group_name,
                        hoverinfo="y+name"
                    ),
                    row=i, col=j
                )

            fig.update_yaxes(title_text=f"{metric}", row=i, col=j)
            fig.update_xaxes(title_text="Bin Value", row=i, col=j)

    fig.update_layout(
        title=f"Frame Stats for {group_names} and {min_votes_values} min_votes Values",
        title_x=0.5,
        title_font_size=20
    )

    fig.show()


# Call the function with desired group names


def extract_voted_data(strategy_group, metric, group_names, strategy_group_name, min_votes=1):
    aggregated_bins = {}
    min_votes_string = f"min_{min_votes}x_voted"

    for strategy_name, strategy_data in strategy_group.items():
        metric_bins = strategy_data["frame_stats"][min_votes_string][metric]["bins"]

        for bin_value, bin_count in metric_bins.items():
            if bin_value not in aggregated_bins:
                aggregated_bins[bin_value] = [0] * len(group_names)
            aggregated_bins[bin_value][group_names.index(strategy_group_name)] += bin_count

    return aggregated_bins


if __name__ == "__main__":
    # Call function with distinct groups so that there is no overlap in the data
    # create_frame_stats_plots(["t5-small", "t5-large"])
    # create_frame_stats_plots(["remove", "exchange"])
    # 'no_framed_decoding' is a special case, since it is the only strategy that does not have framed_decoding in its name
    # and is handled separately in the code.
    create_frame_stats_plots(["no_framed_decoding", "framed_decoding0.1", "framed_decoding0.2", ])
