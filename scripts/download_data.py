"""Download the fixed analysis window from the NYC Open Data Socrata API."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "nyc_311_noise_2024-06-30_to_2024-07-06.csv"
LIMIT = 50_000
FIELDS = [
    "unique_key",
    "created_date",
    "closed_date",
    "agency",
    "agency_name",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "status",
    "resolution_action_updated_date",
    "community_board",
    "police_precinct",
    "borough",
    "open_data_channel_type",
    "latitude",
    "longitude",
]
WHERE = (
    "agency='NYPD' "
    "AND created_date >= '2024-06-30T00:00:00' "
    "AND created_date < '2024-07-07T00:00:00' "
    "AND starts_with(complaint_type, 'Noise')"
)


def main() -> None:
    query = urlencode(
        {
            "$select": ",".join(FIELDS),
            "$where": WHERE,
            "$order": "unique_key",
            "$limit": LIMIT,
        }
    )
    with urlopen(f"{API_URL}?{query}", timeout=120) as response:
        records = json.load(response)

    if not records or len(records) >= LIMIT:
        raise RuntimeError(f"Unexpected row count {len(records)}; review query/pagination")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records):,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
