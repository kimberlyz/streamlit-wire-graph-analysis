import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objects as go
import streamlit as st
from classes.filegraph import FileGraph
from classes.point import Point

# Update Matplotlib default font size
plt.rcParams.update({"font.size": 15})

x_col = "time"
y_col = "ohm"
plot_font_size = 20
max_x_threshold = 2 * 1e-10

if 'matched_filenames' not in st.session_state:
    st.session_state.matched_filenames = []

def match_filenames(filenames):
    st.session_state.matched_filenames.append((filenames[0], filenames[1]))

def clear_data():
    st.cache_data.clear()
    st.session_state.matched_filenames = []

# ===========================================================
# Reading CSV Files from Uploaded Files
# ===========================================================
@st.cache_data
def load_csvs_from_uploaded_files(uploaded_files):
    graphs_dict = {}
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file, skiprows=2)
        filename = uploaded_file.name
        file_graph_data = FileGraph(
            filename, df.rename(columns={df.columns[0]: x_col, df.columns[1]: y_col})
        )
        graphs_dict[filename] = file_graph_data
    return graphs_dict


# ===========================================================
# Filter out matched graphs
# ===========================================================
def filter_out_matched_graphs(graphs_dict):
    filenames = set()
    for pair in st.session_state.matched_filenames:
        filenames.add(pair[0])
        filenames.add(pair[1])

    return {k: v for k, v in graphs_dict.items() if k not in filenames}

# ===========================================================
# Building a Histogram of Y-Values at Target X
# ===========================================================
def interpolate_y_value(coord1, coord2, target_x):
    x1 = coord1.x
    y1 = coord1.y

    x2 = coord2.x
    y2 = coord2.y

    y_at_target_x = y1 + (y2 - y1) * (target_x - x1) / (x2 - x1)

    return y_at_target_x

# Find index where x is >= some value, and then get the two closest points + or - 1 index
def get_closest_points_to_x(df, x_col, y_col, target_x):
    # Find the index where x is greater than or equal to the target x
    idx = df[df[x_col] >= target_x].index[0]

    # Get the two closest points to the target x
    closest_points_df = df.iloc[idx - 1 : idx + 1]

    return [
        Point(row[x_col], row[y_col]) for _, row in closest_points_df.iterrows()
    ]

def set_y_values_at_target_x(graphs_dict, target_x):
    for graph_data in graphs_dict.values():
        df = graph_data.df
        closest_points = get_closest_points_to_x(df, x_col, y_col, target_x)
        y_at_target_x = interpolate_y_value(
            closest_points[0], closest_points[1], target_x
        )
        graph_data.y_at_target_x = y_at_target_x

# ===========================================================
# Building a Histogram of X-Values at Target Y
# ===========================================================
def interpolate_x_value(coord1, coord2, target_y):
    x1 = coord1.x
    y1 = coord1.y

    x2 = coord2.x
    y2 = coord2.y

    slope = (y2 - y1) / (x2 - x1)

    # Calculate the x value at the target y using the slope
    x_at_target_y = x1 + (target_y - y1) / slope

    return x_at_target_y


# Find index where y is >= 52, and then get the two closest points + or - 1 index
def get_closest_points_to_y(df, x_col, y_col, target_y):
    # Find the index where y is greater than or equal to the target y
    idx = df[(df[y_col] >= target_y) & (df[x_col] > max_x_threshold)].index[0]

    # Get the two closest points to the target y
    closest_points_df = df.iloc[idx - 1 : idx + 1]

    return [
        Point(row[x_col], row[y_col]) for _, row in closest_points_df.iterrows()
    ]


def set_x_values_at_target_y(graphs_dict, target_y):
    for graph_data in graphs_dict.values():
        df = graph_data.df
        closest_points = get_closest_points_to_y(df, x_col, y_col, target_y)
        x_at_target_y = interpolate_x_value(
            closest_points[0], closest_points[1], target_y
        )
        graph_data.x_at_target_y = x_at_target_y

def get_bin_data(graphs, bin_indices, num_bins):
    bin_data = {i: [] for i in range(1, num_bins)}

    for i, bin_index in enumerate(bin_indices):
        if bin_index < num_bins:
            bin_data[bin_index].append(graphs[i])
        elif bin_index == num_bins:
            # Edge case for point on the right-most bin edge
            bin_data[bin_index - 1].append(graphs[i])
    return bin_data

# Function to create a histogram and find x, y coordinates within each bin
def get_histogram_and_bins(all_graphs_dict, graphs_dict, num_bins):
    x_values = [graph.x_at_target_y for graph in all_graphs_dict.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{round(x * 1e12, 2)} ps"))

    counts, bins, patches = ax.hist(
        x_values, bins=num_bins, edgecolor="black", alpha=0.7
    )

    # Update histogram if we've already removed some graphs
    if len(graphs_dict) != len(all_graphs_dict):
        remaining_x_values = [graph.x_at_target_y for graph in graphs_dict.values()]
        counts, bins, patches = ax.hist(
            remaining_x_values, bins=bins, edgecolor="black", alpha=0.5, color="purple"
        )

    ax.bar_label(patches, labels=[f"Bin{i+1}\n{int(c)}" for i, c in enumerate(counts)])
    ax.set_xlabel("X-axis Value at Y=52")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of X-axis Values at Y=52\n")
    ax.grid(True)

    # Find graphs in each bin
    graphs = [graph for graph in graphs_dict.values()]
    x_values = [graph.x_at_target_y for graph in graphs]
    bin_indices = np.digitize(x_values, bins)
    bin_data = get_bin_data(graphs, bin_indices, len(bins))

    return {"histogram": fig, "bin_data": bin_data}

def get_y_at_target_x_histogram_and_bins(graphs, num_bins):
    y_values = [graph.y_at_target_x for graph in graphs]

    fig, ax = plt.subplots(figsize=(10, 6))

    counts, bins, patches = ax.hist(
        y_values, bins=num_bins, edgecolor="black", alpha=0.7
    )

    ax.bar_label(patches, labels=[f"Bin{i+1}\n{int(c)}" for i, c in enumerate(counts)])
    ax.set_xlabel("Y Value at X=250ps")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Y Values at X=250ps\n")
    ax.grid(True)

    # Find graphs in each bin
    bin_indices = np.digitize(y_values, bins)
    bin_data = get_bin_data(graphs, bin_indices, len(bins))

    return {"histogram": fig, "bin_data": bin_data}

# ===========================================================
# Building Scatterplot of Peak XY Coordinate
# ===========================================================
def get_peak_xy_coord(df, x_col, y_col, x_threshold):
    filtered_df = df[df[x_col] < x_threshold]

    if not filtered_df.empty:
        index_of_peak_y = filtered_df[y_col].idxmax()
        peak_row = filtered_df.loc[index_of_peak_y]
        peak_y = peak_row[y_col]
        peak_x = peak_row[x_col]
        return Point(peak_x, peak_y)
    else:
        raise Exception(f"No x values found less than {x_threshold}")


def set_peak_xy_coords(graphs_dict):
    for graph in graphs_dict.values():
        df = graph.df
        peak_xy = get_peak_xy_coord(df, "time", "ohm", max_x_threshold)
        graph.xy_at_peak = peak_xy


def get_surrounding_bin_nums(bin_data, bin_num):
    bin_nums = []
    for delta in (-1, 0, 1):
        new_bin_num = bin_num + delta
        if new_bin_num > 0 and new_bin_num <= len(bin_data):
            bin_nums.append(new_bin_num)
    return bin_nums


def get_axis_ranges_for_scatter_plot():
    xys_at_peak = [graph.xy_at_peak for graph in graphs_dict.values()]
    x_min = min(xys_at_peak, key=lambda point: point.x).x
    x_max = max(xys_at_peak, key=lambda point: point.x).x
    y_min = min(xys_at_peak, key=lambda point: point.y).y
    y_max = max(xys_at_peak, key=lambda point: point.y).y

    x_padding = (x_max - x_min) * 0.1  # 10% padding
    y_padding = (y_max - y_min) * 0.1  # 10% padding

    xaxis_range = [x_min - x_padding, x_max + x_padding]
    yaxis_range = [y_min - y_padding, y_max + y_padding]

    return (xaxis_range, yaxis_range)


def get_scatter_plot(bin_data, bin_num, should_see_neighboring_bins):
    fig = go.Figure()

    custom_colors = ["orange", "green", "blue"]

    if should_see_neighboring_bins:
        bin_nums = get_surrounding_bin_nums(bin_data, bin_num)
    else:
        bin_nums = [bin_num]

    for i in range(len(bin_nums)):
        bin_num = bin_nums[i]

        if should_see_neighboring_bins:
            color = custom_colors[i % len(custom_colors)]
        else:
            color = custom_colors[1]
        x = []
        y = []
        labels = []

        graphs = bin_data[bin_num]
        for graph in graphs:
            peak_xy = graph.xy_at_peak
            x.append(peak_xy.x)
            y.append(peak_xy.y)
            labels.append(graph.filename)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                text=labels,
                textposition="top center",
                marker=dict(size=10, color=color),
                name=bin_num,
                showlegend=True,
            )
        )

    xaxis_range, yaxis_range = get_axis_ranges_for_scatter_plot()

    fig.update_layout(
        title=dict(
            text="Interactive Scatter Plot with Plotly",
            font=dict(size=plot_font_size + 2),  # Title font size
        ),
        xaxis=dict(
            title="X values",
            tickmode="linear",
            dtick=10 * 1e-12,
            range=xaxis_range,
            tickfont=dict(size=plot_font_size),
            title_font=dict(size=plot_font_size),
        ),
        yaxis=dict(
            title="Y values",
            range=yaxis_range,
            tickmode="linear",
            dtick=0.1,
            tickfont=dict(size=plot_font_size),
            title_font=dict(size=plot_font_size),
        ),
        legend=dict(
            font=dict(
                size=plot_font_size,
            )
        ),
    )

    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    fig.update_traces(
        hoverlabel=dict(
            font=dict(
                size=plot_font_size - 3,
            )
        )
    )

    return fig


# ===========================================================
# Plot Line Graphs
# ===========================================================
def get_image_filename(filenames):
    filenames = [filename.removesuffix('.CSV') for filename in filenames]
    filenames.sort()
    return '_'.join(filenames)

def get_line_graph(graphs_dict, filenames):
    fig = go.Figure()

    for filename in filenames:
        df = graphs_dict[filename].df
        fig.add_trace(
            go.Scatter(
                x=df[x_col], y=df[y_col], mode="lines", name=filename, showlegend=True
            )
        )

    fig.update_layout(
        title=dict(
            text="Line Graphs of Ohms vs Time",
            font=dict(size=plot_font_size + 2),  # Title font size
        ),
        xaxis=dict(
            title="Time(s)",
            tickfont=dict(size=plot_font_size),
            title_font=dict(size=plot_font_size),
            range=[-100 * 1e-12, 400 * 1e-12],
        ),
        yaxis=dict(
            title="Ohms",
            range=[46, 56],
            tickfont=dict(size=plot_font_size),
            title_font=dict(size=plot_font_size),
        ),
        legend=dict(
            font=dict(
                size=plot_font_size,
            )
        ),
        colorway=[
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "lightblue",
            "pink",
            "lightgreen",
            "gray",
        ],
    )

    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)

    fig.update_traces(
        hoverlabel=dict(
            font=dict(
                size=plot_font_size - 2,
            )
        )
    )

    return fig


# ===========================================================
# App Start
# ===========================================================
st.title("Wire Graph Analysis")
st.sidebar.title("Histogram Bin Values")

uploaded_files = st.file_uploader(
    "Choose CSV files", accept_multiple_files=True, type="csv"
)

if uploaded_files:
    try:
        # Load data
        all_graphs_dict = load_csvs_from_uploaded_files(uploaded_files)
        num_all_files = len(all_graphs_dict)
        st.markdown(f"#### Loaded {num_all_files} CSV files.")

        num_matched_files = len(st.session_state.matched_filenames) * 2
        graphs_dict = filter_out_matched_graphs(all_graphs_dict)
        num_remaining_files = len(graphs_dict)
        st.markdown(f'##### Matched {num_matched_files} Files. {num_remaining_files} Remaining.')

        st.button("Clear Data", on_click=clear_data)

        # Extract features from graphs
        set_x_values_at_target_y(all_graphs_dict, 52)
        set_y_values_at_target_x(all_graphs_dict, 250*1e-12)
        set_peak_xy_coords(all_graphs_dict)

        # Display histogram and bin data
        st.markdown(f"#### How many bins do you want to see?")
        num_bins = st.slider(
            "How many bins do you want to see?", 1, 20, 10, label_visibility="hidden"
        )

        bin_contents = get_histogram_and_bins(all_graphs_dict, graphs_dict, num_bins)
        histogram = bin_contents["histogram"]
        bin_data = bin_contents["bin_data"]

        st.pyplot(histogram)

        for bin_num, graph_data in bin_data.items():
            st.sidebar.write(bin_num, [graph.filename for graph in graph_data])

        should_further_segment_bins = st.checkbox("Enable Further Bin Segmentation")
        if should_further_segment_bins:
            st.markdown(
                f"#### Which bin would you like to see segmented further?"
            )
            selected_bin_num = st.selectbox(
                "Which bin with would you like to see segmented further?",
                bin_data.keys(),
                label_visibility="hidden",
            )

            num_bins = st.slider(
                "How many bins do you want to see?",
                1,
                20,
                10,
                label_visibility="hidden",
                key="num_bins_2"
            )

            bin_contents = get_y_at_target_x_histogram_and_bins(bin_data[selected_bin_num], num_bins)
            histogram = bin_contents["histogram"]
            bin_data = bin_contents["bin_data"]

            st.pyplot(histogram)

        # Display scatterPlot for selected bin
        st.markdown(
            f"#### Which bin would you like to see in a scatterplot?"
        )
        selected_bin_num = st.selectbox(
            "Which bin would you like to see in a scatterplot?",
            bin_data.keys(),
            label_visibility="hidden",
        )
        should_see_neighboring_bins = st.checkbox("See neighboring bins")

        scatter_plot = get_scatter_plot(bin_data, selected_bin_num, should_see_neighboring_bins)
        st.plotly_chart(scatter_plot)

        # Display line graphs for filenames
        filenames = sorted(graphs_dict.keys())
        st.markdown(f"#### Which graphs would you like to see in a line graph?")
        selected_filenames = st.multiselect(
            "Which graphs would you like to see in a line graph?",
            filenames,
            label_visibility="hidden",
        )

        line_graph = get_line_graph(graphs_dict, selected_filenames)
        img_bytes = line_graph.to_image(format="jpeg", width=800, height=400)

        button_name = 'Match Cables and Save Image: ' + ', '.join(selected_filenames)
        sorted_filenames = sorted(selected_filenames)
        st.download_button(
            label=button_name,
            data=img_bytes,
            file_name=get_image_filename(sorted_filenames),
            mime="image/jpeg",
            disabled=len(sorted_filenames) != 2,
            on_click=match_filenames,
            args=(sorted_filenames,)
        )

        st.plotly_chart(line_graph, config={'modeBarButtonsToRemove': ['toImage']})

        st.write(st.session_state.matched_filenames)

    except Exception as e:
        print("got here", e)
        st.error(e)
