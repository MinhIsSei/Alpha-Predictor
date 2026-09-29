import pandas as pd

def target_fits_session(
    prediction_end: pd.Timestamp,
    session_close: pd.Timestamp,
) -> bool:
    """Return whether the target ends by the session close."""
    return prediction_end <= session_close