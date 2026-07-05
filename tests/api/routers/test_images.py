"""Integration tests for the image + video + ZIP-download routes."""

import io
import zipfile


class TestImages:
    def _first_id(self, client):
        return client.get("/api/detections").json()[-1]["id"]

    def test_thumbnail_returns_jpeg(self, client):
        det_id = self._first_id(client)
        resp = client.get(f"/api/images/{det_id}/thumbnail")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == b"FAKETHUMB"

    def test_full_image_returns_jpeg(self, client):
        det_id = self._first_id(client)
        resp = client.get(f"/api/images/{det_id}/full")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == b"FAKEJPEG"

    def test_thumbnail_not_found(self, client):
        resp = client.get("/api/images/99999/thumbnail")
        assert resp.status_code == 404

    def test_full_image_not_found(self, client):
        resp = client.get("/api/images/99999/full")
        assert resp.status_code == 404

    def test_video_returns_mp4(self, client, detection_factory):
        record = detection_factory(species="Finch", track_id=42, with_video=True)
        resp = client.get(f"/api/images/{record.id}/video")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"
        assert resp.content == b"FAKEMP4"

    def test_video_404_when_no_clip(self, client):
        # Seeded records have no video_path.
        det_id = self._first_id(client)
        resp = client.get(f"/api/images/{det_id}/video")
        assert resp.status_code == 404

    def test_video_404_when_file_missing(self, client, detection_factory):
        record = detection_factory(
            species="Wren", track_id=43, with_video=True, video_missing=True
        )
        resp = client.get(f"/api/images/{record.id}/video")
        assert resp.status_code == 404

    def test_video_not_found_missing_detection(self, client):
        resp = client.get("/api/images/99999/video")
        assert resp.status_code == 404


class TestDownload:
    def test_download_returns_valid_zip(self, client):
        all_data = client.get("/api/detections").json()
        ids = ",".join(str(d["id"]) for d in all_data)
        resp = client.get(f"/api/images/download?ids={ids}")
        assert resp.status_code == 200
        assert "zip" in resp.headers["content-type"]
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = zf.namelist()
        assert len(names) == 3
        assert all(n.endswith(".jpg") for n in names)

    def test_download_skips_missing_ids(self, client):
        all_data = client.get("/api/detections").json()
        first_id = all_data[0]["id"]
        resp = client.get(f"/api/images/download?ids={first_id},99999")
        assert resp.status_code == 200
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        assert len(zf.namelist()) == 1

    def test_download_invalid_ids(self, client):
        resp = client.get("/api/images/download?ids=abc,def")
        assert resp.status_code == 400
