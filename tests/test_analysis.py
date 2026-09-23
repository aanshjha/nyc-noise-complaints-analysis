from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis import clean_complaints, response_summary


def sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Unique Key": ["1", "1", "2", "3", "4"],
            "Created Date": [
                "2024-06-30T20:00:00",
                "2024-06-30T20:00:00",
                "bad-date",
                "2024-07-01T10:00:00",
                "2024-07-01T10:00:00",
            ],
            "Closed Date": [
                "2024-06-30T23:00:00",
                "2024-06-30T23:00:00",
                "2024-07-01T01:00:00",
                "2024-07-01T09:00:00",
                "2024-07-01T11:30:00",
            ],
            "Complaint Type": ["Noise - Residential"] * 5,
            "Borough": ["BROOKLYN", "BROOKLYN", "QUEENS", "BRONX", "QUEENS"],
            "Incident Zip": ["11201", "11201", "11101", "10451", "01101"],
            "Open Data Channel Type": ["ONLINE"] * 5,
            "Latitude": [40.7] * 5,
            "Longitude": [-73.9] * 5,
        }
    )


def test_clean_complaints_audits_bad_rows_and_derives_features() -> None:
    cleaned, audit = clean_complaints(sample_rows())

    assert audit.raw_rows == 5
    assert audit.duplicate_rows == 1
    assert audit.invalid_date_rows == 1
    assert audit.negative_response_rows == 1
    assert audit.analysis_rows == 2
    assert cleaned["response_time_hours"].tolist() == [3.0, 1.5]
    assert cleaned["over_two_hours"].tolist() == [1, 0]
    assert cleaned["incident_zip"].tolist() == ["11201", "01101"]


def test_response_summary_uses_group_counts_and_rates() -> None:
    cleaned, _ = clean_complaints(sample_rows())
    summary = response_summary(cleaned, "borough").set_index("borough")

    assert summary.loc["BROOKLYN", "n"] == 1
    assert summary.loc["BROOKLYN", "over_two_hour_rate"] == 1
    assert summary.loc["QUEENS", "median_hours"] == 1.5
