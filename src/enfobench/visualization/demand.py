import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from enfobench.evaluation.filters import hampel

try:
    import seaborn as sns
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.ticker import LinearLocator

    from statsmodels.graphics import tsaplots
except ImportError as e:
    msg = f"Missing optional dependency '{e.name}'. Use pip or conda to install it."
    raise ImportError(msg) from e


def plot_monthly_box(
    data: pd.DataFrame,
    figsize: tuple[float, float] = (12, 5),
) -> tuple[Figure, Axes]:
    """Plot annual seasonality of demand data.

    Args:
        data: Demand data.
        figsize: Figure size.

    Returns:
        fig: Figure object.
        ax: Axes object.
    """
    # Make a copy of the data to avoid modifying the original data
    data = data.copy()
    data['month'] = data.index.month

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the boxplot
    data.boxplot(column='y', by='month', ax=ax)

    # Plot the median values
    data.groupby('month')['y'].median().plot(style='o-', linewidth=0.8, ax=ax)

    # Remove the title and the super title
    plt.title('')
    fig.suptitle('')

    # Set the labels
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Demand distribution by month', fontsize='large', loc='left')
    return fig, ax


def plot_weekly_box(
    data: pd.DataFrame,
    figsize: tuple[float, float] = (12, 5),
) -> tuple[Figure, Axes]:
    """Plot weekly seasonality of demand data.

    Args:
        data: Demand data.
        figsize: Figure size.

    Returns:
        fig: Figure object.
        ax: Axes object.
    """
    # Make a copy of the data to avoid modifying the original data
    data = data.copy()
    data['dayofweek'] = data.index.dayofweek + 1

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the boxplot
    data.boxplot(column='y', by='dayofweek', ax=ax)

    # Plot the median values
    data.groupby('dayofweek')['y'].median().plot(style='o-', linewidth=0.8, ax=ax)

    # Remove the title and the super title
    plt.title('')
    fig.suptitle('')

    # Set the labels
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Demand distribution by day of week', loc='left', fontsize='large')
    return fig, ax


def plot_daily_box(
    data: pd.DataFrame,
    figsize: tuple[float, float] = (12, 5),
) -> tuple[Figure, Axes]:
    """Plot daily seasonality of demand data.

    Args:
        data: Demand data.
        figsize: Figure size.

    Returns:
        fig: Figure object.
        ax: Axes object.
    """
    # Make a copy of the data to avoid modifying the original data
    data = data.copy()
    data['hour'] = data.index.hour + 1

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the boxplot
    data.boxplot(column='y', by='hour', ax=ax)

    # Plot the median values
    data.groupby('hour')['y'].median().plot(style='o-', linewidth=0.8, ax=ax)

    # Remove the title and the super title
    plt.title('')
    fig.suptitle('')

    # Set the labels
    ax.set_ylabel('Energy (kWh)')
    ax.set_title('Demand distribution by hour', loc='left', fontsize='large')
    return fig, ax


def plot_histogram(
    data: pd.DataFrame,
    figsize: tuple[float, float] = (12, 5),
    n_bins: int = 100,
) -> tuple[Figure, Axes]:
    """Plot a histogram of demand data.

    Args:
            data: Demand data.
            figsize: Figure size.
            n_bins: Number of bins for the histogram.

        Returns:
            fig: Figure object.
            ax: Axes object.

    """
    # define the energy intervals to use for the histogram
    bins = np.linspace(0, data.y.max(), n_bins + 1)
    # calculate the distribution of energy values
    hist_df, _ = np.histogram(data['y'], bins=bins)

    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # plot the histogram
    ax[0].plot(bins[:-1], hist_df)
    ax[0].set_xlabel('Energy (kWh)')
    ax[0].set_ylabel('Count')
    ax[0].grid()

    # Set title
    ax[0].set_title('Demand distribution', loc='left', fontsize='large')

    # plot the histogram on a logarithmic scale
    ax[1].plot(bins[:-1], hist_df)
    ax[1].set_yscale('log')
    ax[1].set_ylim(10, None)
    ax[1].set_xlabel('Energy (kWh)')
    ax[1].set_ylabel('Count')
    ax[1].grid(which='both')
    return fig, ax


def plot_heatmap(
    data: pd.DataFrame,
    figsize: tuple[float, float] = (12, 5),
    **kwargs,
) -> tuple[Figure, Axes]:
    """Plot a heatmap of demand data.

    Args:
        data: Demand data.
        figsize: Figure size.
        **kwargs: Additional arguments to pass to the seaborn heatmap function.

    Returns:
        fig: Figure object.
        ax: Axes object.
    """
    # Make a copy of the data to avoid modifying the original data
    data = data.copy()

    # Add columns for date and time
    data["date"] = data.index.date
    data["time"] = data.index.time

    # Pivot dates and times to create a two-dimensional representation
    data_hm = data.pivot_table(index='time', columns='date', values='y', aggfunc=lambda x: x.iloc[0], dropna=False)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title('Demand heatmap', loc='left', fontsize='large')

    sns.heatmap(data_hm, ax=ax, cbar_kws={'label': 'Energy (kWh)'}, **kwargs)
    return fig, ax


def plot_3d(data: pd.DataFrame, figsize: tuple[float, float] = (10, 8)) -> tuple[Figure, Axes]:
    # Make a copy of the data to avoid modifying the original data
    data = data.copy()

    # Add columns for date and time
    data["date"] = data.index.date
    data["time"] = data.index.time

    # Pivot dates and times to create a two-dimensional representation
    data_hm = data.pivot_table(index='time', columns='date', values='y', aggfunc=lambda x: x.iloc[0], dropna=False)

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection='3d')

    # Create meshgrid
    X, Y = np.meshgrid(np.arange(data_hm.shape[1]), np.arange(data_hm.shape[0]))

    # Plot the surface
    plot = ax.plot_surface(X=X, Y=Y, Z=np.nan_to_num(data_hm.values), cmap='YlGnBu_r', vmin=0)

    # set tick every three hours for the time axis
    ax.set_yticks(ticks=np.linspace(0, data_hm.shape[0], 8 + 1))
    ax.set_yticklabels(labels=range(0, 25, 3))
    ax.set_ylabel('time')

    # set ticks for the date axis based on the heatmap above
    tick_index = range(0, data_hm.shape[1], int(data_hm.shape[1] / 10))
    tick_labels = [data_hm.columns[i] for i in tick_index]
    ax.set_xticks(ticks=tick_index)
    ax.set_xticklabels(labels=tick_labels, rotation=45)
    ax.set_xlabel('')

    # set z-axis limits and ticks
    ax.set_zlim(0, round(data['y'].max()))
    ax.zaxis.set_major_locator(LinearLocator(11))
    ax.set_zlabel('Energy (kWh)')

    # Remove gray panes and axis grid
    ax.xaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.fill = False
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.fill = False
    ax.zaxis.pane.set_edgecolor('white')
    ax.grid(False)

    fig.colorbar(plot, ax=ax, label='Energy (kWh)')
    return fig, ax


def plot_data_quality(data: pd.DataFrame, figsize: tuple[float, float] = (12, 5)) -> tuple[Figure, Axes]:
    """Plot the data quality of demand data.

    Args:
        data: Demand data.
        figsize: Figure size.

    Returns:
        fig: Figure object.
        ax: Axes object.
    """
    # Make a copy of the data to avoid modifying the original data
    data = data.copy()
    data['quality'] = 3
    data.loc[data['y'].isna(), 'quality'] = 0
    data.loc[data['y'].eq(0), 'quality'] = 1
    _, is_outlier = hampel(data['y'])
    data.loc[is_outlier, 'quality'] = 2

    # Add columns for date and time
    data["date"] = data.index.date
    data["time"] = data.index.time

    # Pivot dates and times to create a two-dimensional representation
    data_hm = data.pivot_table(
        index='time', columns='date', values='quality', aggfunc=lambda x: x.iloc[0], dropna=False
    )
    data_hm.columns = [str(col) for col in data_hm.columns]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title('Data quality of demand data', loc='left', fontsize='large')

    cmap = plt.get_cmap('Reds', 4)
    sns.heatmap(
        data_hm,
        ax=ax,
        cmap=cmap,
        cbar_kws={'label': 'Quality metric', 'ticks': [0.5, 1.5, 2.5, 3.5]},
        vmin=0,
        vmax=4,
    )
    ax.collections[0].colorbar.set_ticklabels(['Missing', 'Zero', 'Outlier', 'Good Data'])
    return fig, ax


def plot_acf(
    data: pd.DataFrame,
    figsize: tuple[float, float] = (12, 5),
) -> tuple[Figure, Axes]:
    """Plot the autocorrelation function of demand data.

    Args:
        data: Demand data.
        figsize: Figure size.

    Returns:
        fig: Figure object.
        ax: Axes object.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    periods = periods_in_
    tsaplots.plot_acf(data['y'], ax=ax, lags=periods)

    ax.set_title('Autocorrelation', loc='left', fontsize='large')
    ax.set_xlabel('Lag')
    return fig, ax


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from enfobench.evaluation.filters import hampel

    data = pd.DataFrame(
        data=np.random.randint(0, 100, 100),
        index=pd.date_range(start='2020-01-01', periods=100, freq='H'),
        columns=['y']
    )

    fig, ax = plot_acf(data)
    plt.show()
