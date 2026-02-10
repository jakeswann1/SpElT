"""Session metadata loading from Google Sheets."""

import pandas as pd

from spelt.utils import gs_to_df


def load_session_from_sheet(
    sheet_url: str, animal: str, date_short: str, area: str | None, pos_only: bool
) -> pd.DataFrame:
    """
    Load and filter session data from Google Sheets.

    Args:
        sheet_url: URL of the Google Sheet with session metadata
        animal: Animal ID
        date_short: Recording date in YYMMDD format
        area: Brain area to filter by (optional)
        pos_only: If True, allows thresholded recordings

    Returns:
        DataFrame row with session information

    Raises:
        ValueError: If session not found or has no valid recordings
    """
    df = gs_to_df(sheet_url)
    df = df[df["Include"] == "Y"]
    df = df[df["Areas"] == area] if area else df
    session = df.loc[df["Session"] == f"{animal}_{date_short}"]

    if session.empty:
        raise ValueError(f"Session {animal}_{date_short} not found in Google Sheet")

    if "Format" in session.columns and not pos_only:
        session = session[session["Format"] != "thresholded"]
        if session.empty:
            raise ValueError(
                f"Session {animal}_{date_short} has no non-thresholded recordings "
                f"(all marked as thresholded)"
            )

    return session


def extract_session_metadata(
    session: pd.DataFrame,
) -> tuple[int | None, str | None, str | None, str]:
    """
    Extract key metadata fields from session DataFrame.

    Args:
        session: Session DataFrame loaded from Google Sheets

    Returns:
        Tuple of (age, probe_type, area, recording_type):
            - age: Animal age at recording (None if not in DataFrame)
            - probe_type: Probe type identifier (None if not in DataFrame)
            - area: Brain area (None if not in DataFrame)
            - recording_type: Recording system type (always present)

    Raises:
        KeyError: If recording_type is not in DataFrame
    """
    age = int(session["Age"].iloc[0]) if "Age" in session.columns else None
    probe_type = (
        session["probe_type"].iloc[0] if "probe_type" in session.columns else None
    )
    area = session["Areas"].iloc[0] if "Areas" in session.columns else None
    recording_type = session["recording_type"].iloc[0]

    return age, probe_type, area, recording_type
