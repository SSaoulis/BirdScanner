"""Integration tests for the camera-proxy routes (snapshot + crop control).

The API cannot open the camera itself, so these routes proxy to the detector's
control server via ``httpx``. The tests monkeypatch ``httpx`` on the router module
and assert the proxying + status relaying, using the shared ``fake_httpx_response``.
"""


class TestCamera:
    def test_snapshot_proxies_detector_jpeg(
        self, client, monkeypatch, fake_httpx_response
    ):
        from birdscanner.api.routers import camera

        captured = {}

        def _fake_get(url, timeout):
            captured["url"] = url
            return fake_httpx_response(b"JPEGBYTES")

        monkeypatch.setattr(camera.httpx, "get", _fake_get)
        resp = client.get("/api/camera/snapshot")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert resp.content == b"JPEGBYTES"
        assert captured["url"].endswith("/capture")

    def test_snapshot_returns_503_when_detector_unreachable(self, client, monkeypatch):
        import httpx

        from birdscanner.api.routers import camera

        def _fake_get(url, timeout):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(camera.httpx, "get", _fake_get)
        resp = client.get("/api/camera/snapshot")
        assert resp.status_code == 503

    def test_full_snapshot_proxies_detector_jpeg(
        self, client, monkeypatch, fake_httpx_response
    ):
        from birdscanner.api.routers import camera

        captured = {}

        def _fake_get(url, timeout):
            captured["url"] = url
            return fake_httpx_response(b"FULLJPEG")

        monkeypatch.setattr(camera.httpx, "get", _fake_get)
        resp = client.get("/api/camera/snapshot/full")
        assert resp.status_code == 200
        assert resp.content == b"FULLJPEG"
        assert captured["url"].endswith("/capture/full")

    def test_get_crop_proxies_detector_json(
        self, client, monkeypatch, fake_httpx_response
    ):
        from birdscanner.api.routers import camera

        state = {"x": 1, "y": 2, "w": 900, "h": 900, "sensor_w": 4056}

        def _fake_get(url, timeout):
            return fake_httpx_response(content_type="application/json", json_body=state)

        monkeypatch.setattr(camera.httpx, "get", _fake_get)
        resp = client.get("/api/camera/crop")
        assert resp.status_code == 200
        assert resp.json() == state

    def test_set_crop_proxies_post_body(self, client, monkeypatch, fake_httpx_response):
        from birdscanner.api.routers import camera

        captured = {}
        state = {"x": 1, "y": 2, "w": 900, "h": 900}

        def _fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return fake_httpx_response(content_type="application/json", json_body=state)

        monkeypatch.setattr(camera.httpx, "post", _fake_post)
        resp = client.post(
            "/api/camera/crop", json={"nx": 0.1, "ny": 0.2, "nw": 0.3, "nh": 0.4}
        )
        assert resp.status_code == 200
        assert resp.json() == state
        assert captured["url"].endswith("/crop")
        assert captured["json"] == {"nx": 0.1, "ny": 0.2, "nw": 0.3, "nh": 0.4}

    def test_set_crop_relays_detector_400(
        self, client, monkeypatch, fake_httpx_response
    ):
        from birdscanner.api.routers import camera

        def _fake_post(url, json, timeout):
            return fake_httpx_response(b"bad body", status_code=400)

        monkeypatch.setattr(camera.httpx, "post", _fake_post)
        resp = client.post("/api/camera/crop", json={"nx": 0.1})
        assert resp.status_code == 400
