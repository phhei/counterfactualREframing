import json
import plotly.graph_objects as go

def plot_interactive_metrics(data):
    strategy_names = list(data.keys())
    a_fluency_values = [entry['a_fluency']['mean'] for entry in data.values()]
    a_meaning_values = [entry['a_meaning']['mean'] for entry in data.values()]

    fig = go.Figure()

    # Add a_fluency trace
    fig.add_trace(go.Bar(
        x=strategy_names,
        y=a_fluency_values,
        name='a_fluency'
    ))

    # Add a_meaning trace
    fig.add_trace(go.Bar(
        x=strategy_names,
        y=a_meaning_values,
        name='a_meaning'
    ))

    fig.update_layout(
        title='Interactive Metrics Comparison',
        xaxis_title='Strategy',
        yaxis_title='Metric Value',
        barmode='group'  # Change to 'overlay' for overlapping bars
    )

    # Display the interactive plot
    fig.show()

# Load your data from JSON
data = json.load(open('/Users/dimitrymindlin/UniProjects/counterfactuals-for-frame-identification/evaluation/manual/annotations_extended.json'))

# Call the function to create the interactive plot
#plot_interactive_metrics(data)

import json
import plotly.graph_objects as go

def plot_interactive_frame_stats(data):
    strategy_names = list(data.keys())
    sub_metric_names = list(data[strategy_names[0]]['frame_stats']['min_1x_voted'].keys())  # Choose a sub-metric set

    fig = go.Figure()

    for sub_metric_name in sub_metric_names:
        sub_metric_values = [entry['frame_stats']['min_1x_voted'][sub_metric_name]['mean'] for entry in data.values()]
        fig.add_trace(go.Bar(
            x=strategy_names,
            y=sub_metric_values,
            name=sub_metric_name
        ))

    fig.update_layout(
        title='Interactive Frame Stats Comparison',
        xaxis_title='Strategy',
        yaxis_title='Sub-Metric Value',
        barmode='group'  # Change to 'overlay' for overlapping bars
    )

    # Display the interactive plot
    fig.show()

# Call the function to create the interactive plot
plot_interactive_frame_stats(data)

