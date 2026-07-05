"""Integration tests for the settings-proxy routes.

The API cannot change settings itself (read-only data volume), so these routes
proxy to the detector's control server via ``httpx``.  The tests monkeypatch
``httpx`` on the router module and assert the proxying + error relaying, using the
shared ``fake_httpx_response``.
"""

_STATE = {
    "settings": {"detection_threshold": 0.55, "ignore_species": []},
    "needs_restart": False,
    "restart_fields": ["stability_seconds"],
    "live_fields": ["detection_threshold"],
}


class TestSettings:
    def test_get_proxies_detector_state(self, client, monkeypatch, fake_httpx_response):
        from birdscanner.api.routers import settings

        captured = {}

        def _fake_get(url, timeout):
            captured["url"] = url
            return fake_httpx_response(
                content_type="application/json", json_body=_STATE
            )

        monkeypatch.setattr(settings.httpx, "get", _fake_get)
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        assert resp.json() == _STATE
        assert captured["url"].endswith("/settings")

    def test_get_returns_503_when_detector_unreachable(self, client, monkeypatch):
        import httpx

        from birdscanner.api.routers import settings

        def _fake_get(url, timeout):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(settings.httpx, "get", _fake_get)
        assert client.get("/api/settings").status_code == 503

    def test_post_proxies_update_body(self, client, monkeypatch, fake_httpx_response):
        from birdscanner.api.routers import settings

        captured = {}

        def _fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return fake_httpx_response(
                content_type="application/json", json_body=_STATE
            )

        monkeypatch.setattr(settings.httpx, "post", _fake_post)
        resp = client.post("/api/settings", json={"detection_threshold": 0.8})
        assert resp.status_code == 200
        assert resp.json() == _STATE
        assert captured["url"].endswith("/settings")
        assert captured["json"] == {"detection_threshold": 0.8}

    def test_post_relays_validation_error_message(
        self, client, monkeypatch, fake_httpx_response
    ):
        from birdscanner.api.routers import settings

        def _fake_post(url, json, timeout):
            return fake_httpx_response(
                status_code=400,
                content_type="application/json",
                json_body={"error": "detection_threshold must be between 0 and 1"},
            )

        monkeypatch.setattr(settings.httpx, "post", _fake_post)
        resp = client.post("/api/settings", json={"detection_threshold": 9})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "detection_threshold must be between 0 and 1"

    def test_post_returns_503_when_detector_unreachable(self, client, monkeypatch):
        import httpx

        from birdscanner.api.routers import settings

        def _fake_post(url, json, timeout):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(settings.httpx, "post", _fake_post)
        assert client.post("/api/settings", json={}).status_code == 503

    def test_restart_proxies_detector(self, client, monkeypatch, fake_httpx_response):
        from birdscanner.api.routers import settings

        captured = {}

        def _fake_post(url, timeout):
            captured["url"] = url
            return fake_httpx_response(
                content_type="application/json", json_body={"status": "restarting"}
            )

        monkeypatch.setattr(settings.httpx, "post", _fake_post)
        resp = client.post("/api/settings/restart")
        assert resp.status_code == 200
        assert resp.json() == {"status": "restarting"}
        assert captured["url"].endswith("/restart")

    def test_restart_returns_503_when_detector_unreachable(self, client, monkeypatch):
        import httpx

        from birdscanner.api.routers import settings

        def _fake_post(url, timeout):
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(settings.httpx, "post", _fake_post)
        assert client.post("/api/settings/restart").status_code == 503
