import pandas as pd
import plotly.graph_objects as go


def plot_interactive(target: pd.Series, predictions: pd.DataFrame) -> go.Figure:
    # Create figure data
    fig_dict = {"data": [], "layout": {}, "frames": []}

    # Fill in most of layout
    fig_dict["layout"]["xaxis"] = {"range": [target.index.min(), predictions.ds.max()]}
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 500, "redraw": False},
                            "fromcurrent": True,
                            "transition": {"duration": 500, "easing": "quadratic-in-out"},
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}},
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 25, "t": 78},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {"font": {"size": 20}, "prefix": "Cutoff date: ", "visible": True, "xanchor": "right"},
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    def create_history_data_dict(target, cutoff_date):
        data = target.loc[target.index < cutoff_date]
        return {
            "x": data.index,
            "y": data.values,
            "name": "history",
        }

    def create_prediction_dict(prediction):
        return {
            "x": prediction.ds,
            "y": prediction.yhat,
            "mode": "lines",
            "marker": {"color": "rgba(239, 85, 59, 0.75)"},
            "name": "yhat",
        }

    def create_target_dict(prediction, target):
        data = prediction.merge(target, left_on="ds", right_index=True, how="left")
        return {
            "x": data.ds,
            "y": data.y,
            "mode": "lines",
            "marker": {"color": "rgba(0, 204, 150, 0.75)"},
            "name": "y",
        }

    first_prediction_date = predictions.cutoff_date.min()
    first_prediction = predictions[predictions.ds == first_prediction_date]
    fig_dict["data"].append(create_history_data_dict(target, first_prediction_date))
    fig_dict["data"].append(create_prediction_dict(first_prediction))
    fig_dict["data"].append(create_target_dict(first_prediction, target))

    for cutoff_date, prediction in predictions.groupby("cutoff_date"):
        frame = {"data": [], "name": cutoff_date.isoformat()}
        frame["data"].append(create_history_data_dict(target, cutoff_date))
        frame["data"].append(create_prediction_dict(prediction))
        frame["data"].append(create_target_dict(prediction, target))
        fig_dict["frames"].append(frame)

        slider_step = {
            "args": [
                [cutoff_date],
                {"frame": {"duration": 300, "redraw": True}, "mode": "immediate", "transition": {"duration": 500}},
            ],
            "label": cutoff_date.isoformat(),
            "method": "animate",
        }
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]
    fig = go.Figure(fig_dict)
    return fig
