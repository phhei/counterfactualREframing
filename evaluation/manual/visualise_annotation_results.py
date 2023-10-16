import json

import pandas as pd
from matplotlib import pyplot as plt


def plot_strategies_and_result(result_name: str, plot_bins=False):
    # Load the json data
    data = json.load(open(
        '/Users/dimitrymindlin/UniProjects/counterfactuals-for-frame-identification/evaluation/manual/annotations_extended.json'))
    result_values = []

    for strategy_name, content in data.items():
        if not plot_bins:
            result_values.append(content[result_name]['mean'])
        else:
            result_values.append(content[result_name]['bins'])

    # Plot the data
    if not plot_bins:
        plt.bar(data.keys(), result_values)
        plt.xlabel('Strategy')
        plt.ylabel(f'{result_name}')
        plt.title(f'{result_name}')
        # Rotate the x-axis labels
        plt.xticks(rotation=90)
        # Show the plot
        plt.show()
    else:
        df = pd.DataFrame(result_values)
        df.plot.bar(stacked=True)
        plt.xlabel('Strategy')
        plt.ylabel(f'{result_name}')
        plt.title(f'{result_name}')
        # Rotate the x-axis labels
        plt.xticks(rotation=90)
        # Show the plot
        plt.show()


import json
import matplotlib.pyplot as plt


def plot_metrics_for_strategy(strategy_name):
    data = json.load(open(
        '/Users/dimitrymindlin/UniProjects/counterfactuals-for-frame-identification/evaluation/manual/annotations_extended.json'))

    strategy_data = data.get(strategy_name)
    if strategy_data is None:
        print(f"Strategy '{strategy_name}' not found in the data.")
        return

    a_fluency_values = strategy_data['a_fluency']['mean']
    a_meaning_values = strategy_data['a_meaning']['mean']

    frame_stats = strategy_data['frame_stats']
    # Extract specific sub-metrics from frame_stats as needed

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    # Plot a_fluency
    axes[0].bar(['a_fluency'], [a_fluency_values])
    axes[0].set_ylabel('a_fluency')
    axes[0].set_title(f'a_fluency for Strategy: {strategy_name}')

    # Plot a_meaning
    # Split between meaningful bins and not meaningful
    # Sinnvoll? -> 0, 10
    # Passt zum Ursprung? -> 1, 10
    axes[1].bar(['a_meaning'], [a_meaning_values])
    axes[1].set_ylabel('a_meaning')
    axes[1].set_title(f'a_meaning for Strategy: {strategy_name}')

    # Plot frame_stats sub-metrics
    # Customize this section to extract and plot specific sub-metrics from frame_stats

    plt.tight_layout()
    plt.show()


# Call the function to plot metrics for a specific strategy
plot_metrics_for_strategy("ne/exchange_random_labels_1_by_t5-large_TOP10")
print("Hey")

def plot_frame_stats(min_x_voted: str, target_set_strategy: str):
    data = json.load(open(
        '/Users/dimitrymindlin/UniProjects/counterfactuals-for-frame-identification/evaluation/manual/annotations_extended.json'))

    result_values = []
    for strategy_name, content in data.items():
        result_values.append(content['frame_stats'][min_x_voted][target_set_strategy]['mean'])

    plt.bar(data.keys(), result_values)
    plt.xlabel('Strategy')
    plt.ylabel(f'{target_set_strategy}')
    plt.title(f'{min_x_voted}, {target_set_strategy}')
    # Rotate the x-axis labels
    plt.xticks(rotation=90)
    plt.tight_layout()
    # Show the plot
    plt.show()

def plot_frame_stats_for_all_strategies(min_x_voted):
    data = json.load(open(
        '/Users/dimitrymindlin/UniProjects/counterfactuals-for-frame-identification/evaluation/manual/annotations_extended.json'))

    fig, axes = plt.subplots(len(data), 1, figsize=(10, 6 * len(data)),
                             sharex=True)  # Create a single figure with subplots

    for idx, (strategy_name, content) in enumerate(data.items()):
        result_values = {}
        for target_frame_set_strategy, target_frame_set_content in content['frame_stats'][min_x_voted].items():
            result_values[target_frame_set_strategy] = target_frame_set_content['mean']

        ax = axes[idx]  # Get the current subplot
        ax.bar(result_values.keys(), result_values.values())
        ax.set_ylabel(f'{min_x_voted}')
        ax.set_title(f'{min_x_voted}, {strategy_name}')
        ax.tick_params(axis='x', rotation=90)
        ax.grid(True)

    plt.xlabel('Target Frame Set Strategy')
    plt.tight_layout()
    plt.show()


# Call the function for each value of min_x_voted
min_x_voted_values = ['min_1x_voted', 'min_2x_voted', 'min_3x_voted']
for min_x_voted in min_x_voted_values:
    plot_frame_stats_for_all_strategies(min_x_voted)

import json
import matplotlib.pyplot as plt


def plot_metrics_for_all_strategies(metric_name):
    data = json.load(open(
        '/Users/dimitrymindlin/UniProjects/counterfactuals-for-frame-identification/evaluation/manual/annotations_extended.json'))

    fig, axes = plt.subplots(len(data), 1, figsize=(10, 6 * len(data)), sharex=True)

    for idx, (strategy_name, content) in enumerate(data.items()):
        metric_values = content[metric_name]
        bins = metric_values['bins']

        ax = axes[idx]
        ax.bar(bins.keys(), bins.values())
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}, {strategy_name}')
        ax.tick_params(axis='x', rotation=0)  # Set rotation to 0 for better visibility
        ax.grid(True)

    plt.xlabel('Bins')
    plt.tight_layout()
    plt.show()


# Call the function for each metric
metrics_to_plot = ['a_fluency', 'a_meaning']
for metric in metrics_to_plot:
    plot_metrics_for_all_strategies(metric)

plot_frame_stats_for_all_strategies("min_1x_voted")
plot_strategies_and_result("a_fluency")
plot_strategies_and_result("a_fluency", plot_bins=True)
plot_strategies_and_result("a_meaning")
plot_strategies_and_result("a_meaning", plot_bins=True)
plot_frame_stats("min_1x_voted", "bool_successful_added_frames")
plot_frame_stats("min_1x_voted", "bool_successful_hit_frame_set")
plot_frame_stats("min_1x_voted", "bool_successful_removed_frames")
plot_frame_stats("min_1x_voted", "number_overlap_frames")
plot_frame_stats("min_1x_voted", "number_removed_frames_should_stay")
plot_frame_stats("min_1x_voted", "number_successful_added_frames")
plot_frame_stats("min_1x_voted", "number_successful_removed_frames")
