"""Integration tests for the ``/api/stats`` aggregate-statistics endpoints.

Detections are seeded at controlled *naive* timestamps (mirroring production,
where the detector writes ``datetime.now()``), then the aggregated buckets are
asserted. June 2024 anchors the day-of-week checks: 2024-06-01 is a Saturday,
06-02 a Sunday, 06-03 a Monday.
"""

from datetime import datetime

from birdscanner.db.models import DetectionRecord


def _insert_raw(session_factory, **kwargs) -> None:
    """Insert a bare ``DetectionRecord`` (no on-disk files) for stats-only tests."""
    defaults = {
        "image_path": "x/img.jpg",
        "thumbnail_path": "x/img_thumb.jpg",
        "confidence": 0.9,
    }
    defaults.update(kwargs)
    with session_factory() as session:
        session.add(DetectionRecord(**defaults))
        session.commit()


class TestSummary:
    def test_totals_and_span(self, detection_factory, make_client):
        detection_factory(
            species="Robin", track_id=1, ts=datetime(2024, 6, 1, 12, 0, 0)
        )
        detection_factory(species="Robin", track_id=2, ts=datetime(2024, 6, 2, 9, 0, 0))
        detection_factory(
            species="Sparrow", track_id=3, ts=datetime(2024, 6, 3, 18, 0, 0)
        )
        client = make_client()

        data = client.get("/api/stats/summary").json()
        assert data["total"] == 3
        assert data["distinct_species"] == 2
        assert data["first_detection"].startswith("2024-06-01T12:00:00")
        assert data["last_detection"].startswith("2024-06-03T18:00:00")
        assert data["corrected_count"] == 0

    def test_counts_corrections(self, session_factory, make_client):
        _insert_raw(
            session_factory,
            species="Robin",
            timestamp=datetime(2024, 6, 1, 12, 0, 0),
            corrected=True,
        )
        _insert_raw(
            session_factory, species="Wren", timestamp=datetime(2024, 6, 1, 13, 0, 0)
        )
        client = make_client()

        data = client.get("/api/stats/summary").json()
        assert data["total"] == 2
        assert data["corrected_count"] == 1

    def test_empty_db(self, make_client):
        client = make_client()
        data = client.get("/api/stats/summary").json()
        assert data == {
            "total": 0,
            "distinct_species": 0,
            "first_detection": None,
            "last_detection": None,
            "corrected_count": 0,
        }

    def test_respects_range(self, detection_factory, make_client):
        detection_factory(track_id=1, ts=datetime(2024, 6, 1, 12, 0, 0))
        detection_factory(track_id=2, ts=datetime(2024, 6, 5, 12, 0, 0))
        client = make_client()

        data = client.get("/api/stats/summary", params={"from": "2024-06-03"}).json()
        assert data["total"] == 1
        assert data["first_detection"].startswith("2024-06-05")


class TestTimeOfDay:
    def test_bins_by_bucket_start(self, detection_factory, make_client):
        # 08:45 -> minute-of-day 525 -> bucket 17 (30 min) -> start minute 510.
        # A precedence bug ((h*60)+(m/bin)) would mis-place this at 480.
        detection_factory(track_id=1, ts=datetime(2024, 6, 1, 8, 45, 0))
        client = make_client()

        data = client.get("/api/stats/time-of-day", params={"bin_minutes": 30}).json()
        assert data["bin_minutes"] == 30
        assert data["bins"] == [{"minute": 510, "count": 1}]

    def test_multiple_buckets_ordered(self, detection_factory, make_client):
        detection_factory(track_id=1, ts=datetime(2024, 6, 1, 6, 5, 0))
        detection_factory(
            track_id=2, ts=datetime(2024, 6, 1, 6, 20, 0)
        )  # same 6:00 bin
        detection_factory(track_id=3, ts=datetime(2024, 6, 1, 18, 10, 0))
        client = make_client()

        bins = client.get("/api/stats/time-of-day", params={"bin_minutes": 60}).json()[
            "bins"
        ]
        assert bins == [{"minute": 360, "count": 2}, {"minute": 1080, "count": 1}]

    def test_species_filter(self, detection_factory, make_client):
        detection_factory(species="Robin", track_id=1, ts=datetime(2024, 6, 1, 9, 0, 0))
        detection_factory(
            species="Sparrow", track_id=2, ts=datetime(2024, 6, 1, 14, 0, 0)
        )
        client = make_client()

        bins = client.get(
            "/api/stats/time-of-day", params={"bin_minutes": 60, "species": "Robin"}
        ).json()["bins"]
        assert bins == [{"minute": 540, "count": 1}]

    def test_empty(self, make_client):
        client = make_client()
        data = client.get("/api/stats/time-of-day").json()
        assert data["bins"] == []


class TestActivity:
    def test_day_of_week_normalised_to_monday(self, detection_factory, make_client):
        detection_factory(track_id=1, ts=datetime(2024, 6, 1, 10, 0, 0))  # Sat
        detection_factory(track_id=2, ts=datetime(2024, 6, 2, 11, 0, 0))  # Sun
        detection_factory(track_id=3, ts=datetime(2024, 6, 3, 12, 0, 0))  # Mon
        client = make_client()

        cells = {
            (c["dow"], c["hour"]): c["count"]
            for c in client.get("/api/stats/activity").json()["cells"]
        }
        assert cells == {(5, 10): 1, (6, 11): 1, (0, 12): 1}

    def test_aggregates_same_slot(self, detection_factory, make_client):
        detection_factory(track_id=1, ts=datetime(2024, 6, 3, 8, 5, 0))
        detection_factory(track_id=2, ts=datetime(2024, 6, 3, 8, 50, 0))
        client = make_client()

        cells = client.get("/api/stats/activity").json()["cells"]
        assert cells == [{"dow": 0, "hour": 8, "count": 2}]

    def test_empty(self, make_client):
        client = make_client()
        assert client.get("/api/stats/activity").json()["cells"] == []


class TestTimeline:
    def test_day_interval_points(self, detection_factory, make_client):
        detection_factory(species="Robin", track_id=1, ts=datetime(2024, 6, 1, 9, 0))
        detection_factory(species="Robin", track_id=2, ts=datetime(2024, 6, 1, 10, 0))
        detection_factory(species="Sparrow", track_id=3, ts=datetime(2024, 6, 2, 9, 0))
        client = make_client()

        data = client.get("/api/stats/timeline").json()
        assert data["interval"] == "day"
        assert data["species"] == ["Robin", "Sparrow"]  # Robin (2) before Sparrow (1)
        assert data["points"] == [
            {
                "date": "2024-06-01",
                "total": 2,
                "distinct_species": 1,
                "counts": {"Robin": 2},
            },
            {
                "date": "2024-06-02",
                "total": 1,
                "distinct_species": 1,
                "counts": {"Sparrow": 1},
            },
        ]

    def test_distinct_species_per_point(self, detection_factory, make_client):
        detection_factory(species="Robin", track_id=1, ts=datetime(2024, 6, 1, 9, 0))
        detection_factory(species="Sparrow", track_id=2, ts=datetime(2024, 6, 1, 10, 0))
        client = make_client()

        point = client.get("/api/stats/timeline").json()["points"][0]
        assert point["distinct_species"] == 2
        assert point["total"] == 2

    def test_top_n_folds_into_other(self, detection_factory, make_client):
        detection_factory(species="Robin", track_id=1, ts=datetime(2024, 6, 1, 9, 0))
        detection_factory(species="Robin", track_id=2, ts=datetime(2024, 6, 1, 10, 0))
        detection_factory(species="Sparrow", track_id=3, ts=datetime(2024, 6, 1, 11, 0))
        client = make_client()

        data = client.get("/api/stats/timeline", params={"top": 1}).json()
        assert data["species"] == ["Robin"]
        assert data["points"][0]["counts"] == {"Robin": 2, "Other": 1}

    def test_week_interval_groups_days(self, detection_factory, make_client):
        # 2024-06-03..05 all fall in the same ISO-ish %W week bucket.
        detection_factory(track_id=1, ts=datetime(2024, 6, 3, 9, 0))
        detection_factory(track_id=2, ts=datetime(2024, 6, 5, 9, 0))
        client = make_client()

        points = client.get("/api/stats/timeline", params={"interval": "week"}).json()[
            "points"
        ]
        assert len(points) == 1
        assert points[0]["total"] == 2

    def test_rejects_bad_interval(self, make_client):
        client = make_client()
        assert (
            client.get("/api/stats/timeline", params={"interval": "month"}).status_code
            == 422
        )

    def test_empty(self, make_client):
        client = make_client()
        data = client.get("/api/stats/timeline").json()
        assert data["species"] == []
        assert data["points"] == []


class TestDailyWindow:
    def test_first_last_minute(self, detection_factory, make_client):
        detection_factory(track_id=1, ts=datetime(2024, 6, 1, 6, 30, 0))  # 390
        detection_factory(track_id=2, ts=datetime(2024, 6, 1, 18, 15, 0))  # 1095
        detection_factory(track_id=3, ts=datetime(2024, 6, 1, 12, 0, 0))  # 720
        client = make_client()

        days = client.get("/api/stats/daily-window").json()["days"]
        assert days == [
            {
                "date": "2024-06-01",
                "first_minute": 390,
                "last_minute": 1095,
                "count": 3,
            }
        ]

    def test_ordered_by_day(self, detection_factory, make_client):
        detection_factory(track_id=1, ts=datetime(2024, 6, 3, 9, 0))
        detection_factory(track_id=2, ts=datetime(2024, 6, 1, 9, 0))
        client = make_client()

        dates = [
            d["date"] for d in client.get("/api/stats/daily-window").json()["days"]
        ]
        assert dates == ["2024-06-01", "2024-06-03"]

    def test_empty(self, make_client):
        client = make_client()
        assert client.get("/api/stats/daily-window").json()["days"] == []


class TestFirstSightings:
    def test_ordered_oldest_first(self, detection_factory, make_client):
        detection_factory(species="Robin", track_id=1, ts=datetime(2024, 6, 2, 9, 0))
        detection_factory(species="Sparrow", track_id=2, ts=datetime(2024, 6, 1, 9, 0))
        detection_factory(species="Robin", track_id=3, ts=datetime(2024, 6, 3, 9, 0))
        client = make_client()

        data = client.get("/api/stats/first-sightings").json()
        assert [d["species"] for d in data] == ["Sparrow", "Robin"]
        assert data[0]["first_seen"].startswith("2024-06-01")
        assert data[1]["first_seen"].startswith("2024-06-02")  # earliest Robin

    def test_is_all_time_ignoring_range(self, detection_factory, make_client):
        detection_factory(track_id=1, ts=datetime(2024, 6, 1, 9, 0))
        client = make_client()

        # No range params are accepted; a passed ``from`` is simply ignored.
        data = client.get(
            "/api/stats/first-sightings", params={"from": "2025-01-01"}
        ).json()
        assert len(data) == 1

    def test_empty(self, make_client):
        client = make_client()
        assert client.get("/api/stats/first-sightings").json() == []
