import pytest

import app as app_module


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


@pytest.mark.parametrize("model", list(app_module.MODEL_CHOICES))
def test_past_season_page_renders_for_every_model(client, model):
    r = client.get(f"/?season=2016-17&model={model}")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Russell Westbrook" in body
    assert "Model comparison" in body
    assert "Every model&#39;s pick" in body or "Every model's pick" in body


def test_unknown_model_falls_back_to_shipped(client):
    r = client.get("/?season=2016-17&model=nope")
    assert r.status_code == 200
    assert 'value="shipped" selected' in r.get_data(as_text=True)


def test_api_returns_backtest_for_past_season(client):
    data = client.get("/api/2024-25").get_json()
    assert data["season"] == "2024-25"
    assert data["actual_mvp"]
    assert len(data["top10"]) == 10


def test_api_rejects_unknown_season(client):
    assert client.get("/api/1890-91").status_code == 404


def test_model_comparison_lists_all_models():
    keys = [row["key"] for row in app_module.read_model_comparison()]
    assert set(keys) == set(app_module.MODEL_CHOICES)
