from __future__ import annotations  # we enable future annotations for forward refs
from pathlib import Path  # we import path handling
from typing import Dict, Mapping, Optional, Sequence, Union, Literal  # we import precise typing

import pandas as pd  # we import pandas
import plotly.graph_objects as go  # we import plotly graph objects
from plotly.subplots import make_subplots  # we import subplot factory


__all__ = [
    "make_stooq_figure",
    "make_stooq_figures",
    "save_figure_html",
]  # we expose public api


def _validate_price_df(df: pd.DataFrame) -> None:
    """
    we validate the dataframe structure for price visualization
    - we require a pandas dataframe with a datetimeindex
    - we require columns: open, high, low, close
    - we allow optional column: volume
    """
    if not isinstance(df, pd.DataFrame):  # we check type
        raise TypeError("df must be a pandas DataFrame")  # we raise explicit error
    if df.empty:  # we check non-empty
        raise ValueError("df must not be empty")  # we raise explicit error
    if not isinstance(df.index, pd.DatetimeIndex):  # we check datetime index
        raise TypeError("df.index must be a pandas DatetimeIndex")  # we raise explicit error
    required = {"open", "high", "low", "close"}  # we set required columns
    missing = required.difference(df.columns)  # we compute missing columns
    if missing:  # we validate
        raise ValueError(f"missing required columns: {sorted(missing)}")  # we raise explicit error


def _maybe_convert_display_tz(df: pd.DataFrame, tz_display: Optional[str]) -> pd.DataFrame:
    """
    we convert or localize the index to the target display timezone if requested
    """
    if tz_display is None:  # we keep as-is if no timezone requested
        return df  # we return original df
    out = df.copy()  # we avoid mutating caller data
    if out.index.tz is None:  # we localize naive timestamps
        out.index = out.index.tz_localize(tz_display, nonexistent="shift_forward", ambiguous="NaT")  # we localize safely
    else:  # we convert tz-aware timestamps
        out.index = out.index.tz_convert(tz_display)  # we convert timezone
    return out  # we return tz-adjusted df


def _build_price_trace(
    df: pd.DataFrame,
    ohlc_type: Literal["candlestick", "ohlc"],
    name: str,
) -> go.BaseTraceType:
    """
    we build either a candlestick or ohlc trace from the dataframe
    """
    if ohlc_type == "candlestick":  # we choose candlestick
        return go.Candlestick(  # we create candlestick trace
            x=df.index,  # we set x as timestamp
            open=df["open"],  # we set open
            high=df["high"],  # we set high
            low=df["low"],  # we set low
            close=df["close"],  # we set close
            name=name,  # we set legend name
            showlegend=True,  # we show legend
        )  # we return trace
    return go.Ohlc(  # we create ohlc trace
        x=df.index,  # we set x
        open=df["open"],  # we set open
        high=df["high"],  # we set high
        low=df["low"],  # we set low
        close=df["close"],  # we set close
        name=name,  # we set name
        showlegend=True,  # we show legend
    )  # we return trace


def _build_ma_traces(
    df: pd.DataFrame,
    windows: Sequence[int],
    color_cycle: Sequence[str],
) -> list[go.Scatter]:
    """
    we compute and build moving average line traces for given windows
    """
    traces: list[go.Scatter] = []  # we prepare output traces
    if not windows:  # we handle empty windows
        return traces  # we return no traces
    close = pd.to_numeric(df["close"], errors="coerce")  # we coerce close to numeric
    for i, w in enumerate(windows):  # we iterate windows
        if not isinstance(w, int) or w <= 0:  # we validate window
            raise ValueError(f"ma window must be positive int, got {w}")  # we raise error
        ma = close.rolling(window=w, min_periods=w).mean()  # we compute simple moving average
        traces.append(  # we append scatter trace
            go.Scatter(
                x=df.index,  # we set x
                y=ma,  # we set y
                mode="lines",  # we draw lines
                name=f"MA{w}",  # we set legend
                line=dict(width=1.6, color=color_cycle[i % len(color_cycle)]),  # we style line
                hovertemplate="MA" + str(w) + ": %{y:.4f}<extra></extra>",  # we set hover
            )
        )  # we add trace
    return traces  # we return traces


def _build_volume_trace(df: pd.DataFrame) -> Optional[go.Bar]:
    """
    we build a colored volume bar trace if 'volume' column exists
    """
    if "volume" not in df.columns:  # we check availability
        return None  # we skip volume
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)  # we clean volume
    up = df["close"] >= df["open"]  # we compute up-move mask
    colors = ["#26a69a" if is_up else "#ef5350" for is_up in up]  # we color up/down
    return go.Bar(  # we create bar trace
        x=df.index,  # we set x
        y=vol,  # we set y
        marker_color=colors,  # we set colors
        name="Volume",  # we set legend
        opacity=0.7,  # we set opacity
        showlegend=False,  # we hide legend for volume
        hovertemplate="Vol: %{y}<extra></extra>",  # we set hover
    )  # we return trace


def make_stooq_figure(
    df: pd.DataFrame,
    *,
    ohlc_type: Literal["candlestick", "ohlc"] = "candlestick",
    title: Optional[str] = None,
    add_mas: Sequence[int] = (20, 50),
    show_volume: bool = True,
    hide_weekends: bool = True,
    tz_display: Optional[str] = None,
    height: int = 600,
    template: str = "plotly_white",
) -> go.Figure:
    """
    we build a ready-to-use plotly figure from a stooq-like dataframe
    args:
        df: dataframe with index=datetime and columns open, high, low, close, optional volume and ticker  # we define input
        ohlc_type: "candlestick" or "ohlc"  # we select price trace type
        title: optional title, default derived from ticker if present  # we define title behavior
        add_mas: moving average windows to overlay on price  # we define ma windows
        show_volume: whether to show a volume subplot if available  # we toggle volume panel
        hide_weekends: whether to hide weekends with rangebreaks  # we toggle rangebreaks
        tz_display: timezone name to display timestamps  # we set display tz
        height: total figure height in pixels  # we set figure size
        template: plotly template name  # we set styling
    returns:
        a plotly figure with price (and optional volume)  # we describe return
    """
    _validate_price_df(df)  # we validate inputs
    data = _maybe_convert_display_tz(df, tz_display)  # we adjust timezone for display
    has_volume = "volume" in data.columns  # we check volume availability

    ticker = None  # we init ticker name
    if "ticker" in data.columns and not data["ticker"].empty:  # we test ticker column
        try:
            ticker = str(data["ticker"].iloc[0])  # we read ticker
        except Exception:
            ticker = None  # we fallback if any issue

    nrows = 2 if (show_volume and has_volume) else 1  # we decide number of rows
    row_heights = [0.76, 0.24] if nrows == 2 else None  # we set row heights
    fig = make_subplots(  # we create subplot figure
        rows=nrows,  # we set rows
        cols=1,  # we set one column
        shared_xaxes=True,  # we share x axis
        vertical_spacing=0.03 if nrows == 2 else 0.04,  # we set spacing
        row_heights=row_heights,  # we set heights
    )  # we build base figure

    price_name = ticker if ticker is not None else "Price"  # we derive trace name
    fig.add_trace(_build_price_trace(data, ohlc_type=ohlc_type, name=price_name), row=1, col=1)  # we add price trace

    ma_colors = ["#2E86AB", "#E67E22", "#8E44AD", "#16A085", "#C0392B"]  # we define palette
    for tr in _build_ma_traces(data, list(add_mas), ma_colors):  # we create ma traces
        fig.add_trace(tr, row=1, col=1)  # we add ma trace

    if nrows == 2:  # we handle volume
        vol_trace = _build_volume_trace(data)  # we build volume trace
        if vol_trace is not None:  # we check existence
            fig.add_trace(vol_trace, row=2, col=1)  # we add volume subplot
            fig.update_yaxes(title_text="Volume", row=2, col=1, rangemode="tozero")  # we label y2

    derived_title = title if title is not None else (f"{ticker}" if ticker else "Timeseries")  # we compute title
    fig.update_layout(  # we update layout
        title=dict(text=derived_title, x=0.02, xanchor="left"),  # we set title
        height=height,  # we set height
        template=template,  # we set template
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01),  # we set legend
        margin=dict(l=50, r=20, t=60, b=40),  # we set margins
    )  # we apply layout

    fig.update_yaxes(title_text="Price", row=1, col=1)  # we label price axis
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")  # we enhance hover spikes

    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)  # we remove default range slider

    if hide_weekends:  # we optionally hide weekends
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=nrows, col=1)  # we add rangebreaks

    return fig  # we return the figure


def make_stooq_figures(
    data: Mapping[str, pd.DataFrame],
    *,
    ohlc_type: Literal["candlestick", "ohlc"] = "candlestick",
    add_mas: Sequence[int] = (20, 50),
    show_volume: bool = True,
    hide_weekends: bool = True,
    tz_display: Optional[str] = None,
    height: int = 600,
    template: str = "plotly_white",
) -> Dict[str, go.Figure]:
    """
    we build one figure per ticker from a mapping {ticker: dataframe}
    args:
        data: mapping of ticker to curated dataframe (from utils.data.curating_stooq)  # we define input
    returns:
        dict {ticker: figure}  # we describe return
    """
    out: Dict[str, go.Figure] = {}  # we init output mapping
    for key, df in data.items():  # we iterate items
        fig = make_stooq_figure(  # we delegate single figure build
            df,
            ohlc_type=ohlc_type,
            title=str(key),
            add_mas=add_mas,
            show_volume=show_volume,
            hide_weekends=hide_weekends,
            tz_display=tz_display,
            height=height,
            template=template,
        )  # we build one figure
        out[str(key)] = fig  # we store figure
    return out  # we return mapping


def save_figure_html(fig: go.Figure, output: Union[str, Path], auto_open: bool = False) -> None:
    """
    we save the figure to a self-contained html file
    """
    p = Path(output)  # we normalize path
    p.parent.mkdir(parents=True, exist_ok=True)  # we ensure folder exists
    fig.write_html(str(p), include_plotlyjs="cdn", full_html=True, auto_open=auto_open)  # we write html