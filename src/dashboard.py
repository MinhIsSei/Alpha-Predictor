"""Read-only Streamlit dashboard over the predictions database.

Launched through the repository-root entry point, which is also what
Streamlit Community Cloud looks for:
    streamlit run streamlit_app.py
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.config import (
    DEMO_DB_PATH,
    DEMO_MODEL_COMPARISON_PATH,
    FEATURE_COLS,
    PREDICTIONS_DB_PATH,
    PROCESSED_PATH,
    TARGET_COL,
    WALK_FORWARD_TEST_SESSIONS,
    WALK_FORWARD_TRAIN_SESSIONS,
)
from src.dashboard_data import (
    STATUS_AWAITING,
    STATUS_CORRECT,
    STATUS_INCORRECT,
    STATUS_UNRESOLVED,
    cumulative_accuracy,
    load_model_comparison_csv,
    load_predictions,
    model_comparison_long,
    summarize,
)

DATABASES = {
    "Live database": PREDICTIONS_DB_PATH,
    "Demo database": DEMO_DB_PATH,
}

# Validated with the dataviz skill's validate_palette.js against Streamlit's
# own surfaces (#ffffff light, #0e1117 dark): all-pairs CVD and normal-vision
# checks pass in both modes. Aqua is below 3:1 on the light surface, which is
# why every chart below ships with a table view.
CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a"],
    "dark": ["#3987e5", "#d95926", "#199e70"],
}
# Primary text ink for labels drawn inside a chart: Vega-Lite's default mark
# text is black, which disappears on Streamlit's dark surface.
INK = {"light": "#0b0b0b", "dark": "#fafafa"}
# Status colors are reserved for right/wrong and never reused for a series;
# each status also gets its own shape so meaning never rests on color alone.
MUTED = "#898781"
STATUS_STYLE = {
    STATUS_CORRECT: ("#0ca30c", "circle"),
    STATUS_INCORRECT: ("#d03b3b", "cross"),
    STATUS_AWAITING: (MUTED, "triangle-up"),
    STATUS_UNRESOLVED: (MUTED, "diamond"),
}
SERIES_ORDER = ["logistic_regression", "gradient_boosting", "baseline"]
SERIES_LABELS = {
    "logistic_regression": "Logistic Regression",
    "gradient_boosting": "Gradient boosting",
    "baseline": "Baseline (most frequent class)",
}


def theme_mode() -> str:
    theme_type = getattr(st.context.theme, "type", None)
    return "dark" if theme_type == "dark" else "light"


@st.cache_data(show_spinner="Running walk-forward model comparison...")
def cached_model_comparison(path: str, modified: float) -> pd.DataFrame:
    # `modified` is part of the cache key only: a re-run of features.py
    # changes the file's mtime, which invalidates this cached result.
    from src.compare_models import compare_models

    df = pd.read_parquet(path).sort_index()
    return compare_models(
        df,
        FEATURE_COLS,
        TARGET_COL,
        train_sessions=WALK_FORWARD_TRAIN_SESSIONS,
        test_sessions=WALK_FORWARD_TEST_SESSIONS,
    )


def probability_chart(df: pd.DataFrame) -> alt.Chart:
    statuses = list(STATUS_STYLE)
    color = alt.Color(
        "status:N",
        title="Outcome",
        scale=alt.Scale(domain=statuses, range=[STATUS_STYLE[s][0] for s in statuses]),
    )
    shape = alt.Shape(
        "status:N",
        scale=alt.Scale(domain=statuses, range=[STATUS_STYLE[s][1] for s in statuses]),
    )
    # Ordinal, one slot per candle, not a continuous time axis: predictions
    # only exist during market hours, so on a real time scale the overnight
    # and weekend gaps take up most of the width and a session's worth of
    # predictions collapses into one overlapping clump at the edge.
    df = df.assign(candle_label=df["candle_start"].dt.strftime("%b %d %H:%M"))
    x = alt.X(
        "candle_label:O",
        title="Candle start (New York time)",
        sort=alt.SortField("candle_start"),
        axis=alt.Axis(labelAngle=-45, labelOverlap="greedy"),
    )

    threshold = (
        alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=MUTED, strokeWidth=1).encode(y="y:Q")
    )
    threshold_label = (
        alt.Chart(pd.DataFrame({"y": [0.5], "text": ["0.5 decision threshold"]}))
        .mark_text(align="left", dx=4, dy=-6, color=MUTED, fontSize=11)
        .encode(y="y:Q", text="text:N", x=alt.value(0))
    )

    points = (
        alt.Chart(df)
        .mark_point(size=90, filled=True, opacity=0.95)
        .encode(
            x=x,
            y=alt.Y(
                "probability_up:Q",
                title="Model probability of Up",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=color,
            shape=shape,
            tooltip=[
                alt.Tooltip("candle_start:T", title="Candle start (NY)", format="%b %d %H:%M"),
                alt.Tooltip("mode:N", title="Mode"),
                alt.Tooltip("predicted_label:N", title="Predicted"),
                alt.Tooltip("probability_up:Q", title="Probability of Up", format=".1%"),
                alt.Tooltip("actual_label:N", title="Actual"),
                alt.Tooltip("status:N", title="Outcome"),
            ],
        )
    )
    return (threshold + threshold_label + points).properties(height=320)


def running_accuracy_chart(running: pd.DataFrame, line_color: str) -> alt.Chart:
    base = alt.Chart(running).encode(
        x=alt.X(
            "n:Q", title="Evaluated predictions (in candle order)", axis=alt.Axis(tickMinStep=1)
        ),
        y=alt.Y(
            "cumulative_accuracy:Q",
            title="Running accuracy",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format="%"),
        ),
    )
    reference = (
        alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=MUTED, strokeWidth=1).encode(y="y:Q")
    )
    reference_label = (
        alt.Chart(pd.DataFrame({"y": [0.5], "text": ["50% · coin flip"]}))
        .mark_text(align="left", dx=4, dy=-6, color=MUTED, fontSize=11)
        .encode(y="y:Q", text="text:N", x=alt.value(0))
    )

    hover = alt.selection_point(fields=["n"], nearest=True, on="pointerover", empty=False)
    line = base.mark_line(strokeWidth=2, color=line_color)
    points = (
        base.mark_point(filled=True, color=line_color)
        .encode(
            size=alt.condition(hover, alt.value(140), alt.value(70)),
            tooltip=[
                alt.Tooltip("n:Q", title="Evaluated #"),
                alt.Tooltip("candle_start:T", title="Candle start (NY)", format="%b %d %H:%M"),
                alt.Tooltip("status:N", title="This outcome"),
                alt.Tooltip("cumulative_accuracy:Q", title="Running accuracy", format=".1%"),
            ],
        )
        .add_params(hover)
    )
    crosshair = (
        base.mark_rule(color=MUTED)
        .encode(opacity=alt.condition(hover, alt.value(0.6), alt.value(0)))
        .transform_filter(hover)
    )
    end_label = (
        base.transform_window(rank="rank()", sort=[alt.SortField("n", order="descending")])
        .transform_filter("datum.rank == 1")
        .mark_text(
            # Above-left of the last point: the last point sits on the plot's
            # right edge, so a label to its right would be clipped.
            align="right",
            dx=-6,
            dy=-12,
            fontSize=12,
            fontWeight="bold",
            color=INK[theme_mode()],
        )
        .encode(text=alt.Text("cumulative_accuracy:Q", format=".0%"))
    )

    return (reference + reference_label + line + crosshair + points + end_label).properties(
        height=280
    )


def model_comparison_chart(long_df: pd.DataFrame, palette: list[str]) -> alt.Chart:
    labels = [SERIES_LABELS[s] for s in SERIES_ORDER]
    long_df = long_df.assign(series_label=long_df["series"].map(SERIES_LABELS))

    return (
        alt.Chart(long_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, width={"band": 0.75})
        .encode(
            x=alt.X(
                "fold_label:N",
                title="Walk-forward test window",
                sort=None,
                scale=alt.Scale(paddingInner=0.35),
                axis=alt.Axis(labelAngle=0, labelLimit=300),
            ),
            xOffset=alt.XOffset("series_label:N", sort=labels),
            y=alt.Y(
                "accuracy:Q",
                title="Test accuracy",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            # A fixed domain -> color mapping: each series keeps its color no
            # matter which others are present, so a hue always means one model.
            color=alt.Color(
                "series_label:N",
                title="Series",
                scale=alt.Scale(domain=labels, range=palette),
                legend=alt.Legend(orient="top", labelLimit=400),
            ),
            tooltip=[
                alt.Tooltip("fold_label:N", title="Window"),
                alt.Tooltip("series_label:N", title="Series"),
                alt.Tooltip("accuracy:Q", title="Accuracy", format=".1%"),
            ],
        )
        .properties(height=320)
    )


def render_predictions_section(df: pd.DataFrame, mode: str) -> None:
    stats = summarize(df)

    tile_1, tile_2, tile_3, tile_4 = st.columns(4)
    tile_1.metric("Predictions", stats["n_predictions"], border=True)
    if stats["accuracy"] is None:
        tile_2.metric(
            "Accuracy (evaluated only)",
            "—",
            border=True,
            help="No prediction in this selection has an outcome yet.",
        )
    else:
        tile_2.metric(
            "Accuracy (evaluated only)",
            f"{stats['accuracy']:.0%}",
            delta=f"{stats['n_correct']} of {stats['n_evaluated']} correct",
            delta_color="off",
            delta_arrow="off",
            border=True,
            help="Computed only over predictions with a stored outcome. With a small "
            "sample, one new outcome can move this a lot.",
        )
    tile_3.metric(
        "Pending",
        stats["n_awaiting"] + stats["n_unresolved"],
        delta=f"{stats['n_awaiting']} awaiting target · {stats['n_unresolved']} not evaluated",
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    tile_4.metric(
        "Predicted Up share",
        f"{stats['predicted_up_rate']:.0%}",
        border=True,
        help="Share of predictions that said Up. Compare it with how often the price "
        "actually rose — a model that almost always says Up behaves like an "
        "always-Up rule.",
    )

    latest = df.iloc[-1]
    st.caption(
        f"Latest prediction: **{latest['predicted_label']}** for the candle starting "
        f"{latest['candle_start']:%b %d, %H:%M} NY time — model probability of Up "
        f"{latest['probability_up']:.1%} ({latest['status'].lower()}). "
        "The probability is the model's score, not its accuracy."
    )

    if stats["n_predictions"] >= 5 and stats["predicted_up_rate"] in (0.0, 1.0):
        only = "Up" if stats["predicted_up_rate"] == 1.0 else "Not up"
        st.warning(
            f"Every prediction in this selection is **{only}**. On this sample the model "
            f"is behaving like a constant always-{only} rule, so its accuracy here says "
            "more about how often the market rose than about the model."
        )
    if stats["n_unresolved"]:
        st.info(
            f"{stats['n_unresolved']} prediction(s) are past their target time but have no "
            "stored outcome. Run `python -m src.evaluate_predictions --mode "
            f"{'live' if mode != 'replay' else 'replay'}` to score them — Yahoo Finance only "
            "keeps 5-minute candles for about 60 days, after which they can't be evaluated."
        )

    st.subheader("Predictions over time")
    st.altair_chart(probability_chart(df), width="stretch")

    st.subheader("Running accuracy")
    running = cumulative_accuracy(df)
    if len(running) < 2:
        st.info("At least two evaluated predictions are needed to draw running accuracy.")
    else:
        st.caption(
            f"Over {len(running)} evaluated predictions. The line at 50% is a coin flip; "
            "with a sample this small, expect large swings."
        )
        st.altair_chart(
            running_accuracy_chart(running, CATEGORICAL[theme_mode()][0]),
            width="stretch",
        )

    with st.expander("Table view — all predictions in this selection"):
        st.dataframe(
            df[
                [
                    "candle_start",
                    "mode",
                    "model_version",
                    "predicted_label",
                    "probability_up",
                    "actual_label",
                    "status",
                    "reference_close",
                    "target_close",
                ]
            ].rename(columns={"candle_start": "candle_start (NY)"}),
            hide_index=True,
            column_config={
                "probability_up": st.column_config.NumberColumn("probability_up", format="percent"),
            },
        )


def render_model_comparison_section() -> None:
    st.header("Model comparison (walk-forward)")
    st.caption(
        "Logistic Regression (the deployed model) and gradient boosting, each re-fit on "
        f"identical rolling windows of {WALK_FORWARD_TRAIN_SESSIONS} sessions and tested on "
        f"the next {WALK_FORWARD_TEST_SESSIONS}, against a most-frequent-class baseline. "
        "Same computation as `python -m src.compare_models`."
    )

    if PROCESSED_PATH.is_file():
        results = cached_model_comparison(str(PROCESSED_PATH), PROCESSED_PATH.stat().st_mtime)
    elif DEMO_MODEL_COMPARISON_PATH.is_file():
        # data/processed/ is gitignored, so a fresh clone or a cloud deploy
        # has no training table to re-fit on; show the checked-in snapshot.
        results = load_model_comparison_csv(DEMO_MODEL_COMPARISON_PATH)
        st.info(
            "Showing a precomputed snapshot (`data/demo/model_comparison_demo.csv`). "
            "Run `python -m src.ingest` and `python -m src.features` locally to recompute "
            "it from fresh data."
        )
    else:
        st.info(
            "No processed training table found. Run `python -m src.ingest` and "
            "`python -m src.features` to create it, then reload."
        )
        return

    if results.empty:
        st.info("Not enough trading sessions in the processed table for a walk-forward.")
        return

    long_df = model_comparison_long(results)

    pivot = results.pivot(index="fold", columns="model_name", values="model_accuracy")
    lr_wins = int((pivot["logistic_regression"] > pivot["gradient_boosting"]).sum())
    beats_baseline = int(
        (
            results.query("model_name == 'logistic_regression'").eval(
                "model_accuracy > baseline_accuracy"
            )
        ).sum()
    )
    n_folds = len(pivot)

    tile_1, tile_2, tile_3 = st.columns(3)
    tile_1.metric(
        "Logistic Regression mean accuracy",
        f"{pivot['logistic_regression'].mean():.1%}",
        delta=f"±{pivot['logistic_regression'].std():.1%} across folds",
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    tile_2.metric("LR beats gradient boosting", f"{lr_wins} of {n_folds} folds", border=True)
    tile_3.metric("LR beats baseline", f"{beats_baseline} of {n_folds} folds", border=True)

    st.altair_chart(
        model_comparison_chart(long_df, CATEGORICAL[theme_mode()]),
        width="stretch",
    )

    with st.expander("Table view — accuracy per fold"):
        table = long_df.pivot(index="fold_label", columns="series", values="accuracy")
        table = table[SERIES_ORDER].rename(columns=SERIES_LABELS)
        st.dataframe(
            table,
            column_config={
                col: st.column_config.NumberColumn(col, format="percent") for col in table.columns
            },
        )


def main() -> None:
    st.set_page_config(page_title="Alpha-Predictor", page_icon="📈", layout="wide")
    st.title("Alpha-Predictor")
    st.caption(
        "AAPL · will the close rise over the next 30 minutes? · all times New York time. "
        "Read-only view of the predictions database — an educational experiment, not "
        "trading advice."
    )

    available = {name: path for name, path in DATABASES.items() if path.is_file()}
    if not available:
        st.error("No predictions database found (neither the live nor the demo file exists).")
        return

    filter_1, filter_2, filter_3 = st.columns([2, 2, 3])
    db_name = filter_1.segmented_control(
        "Database", list(available), default=list(available)[0], required=True
    )
    df = load_predictions(available[db_name])

    if df.empty:
        st.info("This database has no predictions yet.")
        render_model_comparison_section()
        return

    mode = filter_2.segmented_control(
        "Mode", ["all", *sorted(df["mode"].unique())], default="all", required=True
    )
    versions = sorted(df["model_version"].unique())
    version = filter_3.selectbox("Model version", versions, index=len(versions) - 1)

    selection = df[df["model_version"] == version]
    if mode != "all":
        selection = selection[selection["mode"] == mode]

    if selection.empty:
        st.info("No predictions match this selection.")
    else:
        render_predictions_section(selection, mode)

    st.divider()
    render_model_comparison_section()


if __name__ == "__main__":
    main()
