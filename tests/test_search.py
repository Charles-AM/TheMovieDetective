import search


def test_distance_similarity_normalization():
    assert search._distance_similarity(0.2, 0.2, 0.8) == 1.0
    assert search._distance_similarity(0.8, 0.2, 0.8) == 0.0
    mid = search._distance_similarity(0.5, 0.2, 0.8)
    assert 0.49 <= mid <= 0.51


def test_contains_any_token_and_phrase_matching():
    haystack = "Animated Bible movie about Moses"
    assert search._contains_any(haystack, ["moses", "bible movie"]) == 2
    assert search._contains_any("program", ["ram"]) == 0


def test_rerank_prefers_lower_distance_and_attribute_hits():
    candidates = [
        {
            "rank": 1,
            "doc": "Animated epic about Moses leading an exodus.",
            "meta": {
                "title": "The Prince of Egypt",
                "year": "1998",
                "genres": "Animation, Drama, Family",
                "keywords": "Moses, Exodus, Egypt"
            },
            "dist": 0.2,
        },
        {
            "rank": 2,
            "doc": "A space adventure with robots.",
            "meta": {
                "title": "Space Bots",
                "year": "2019",
                "genres": "Sci-Fi",
                "keywords": "space, robot"
            },
            "dist": 0.6,
        },
    ]

    attributes = {
        "title_hint": "",
        "genres": ["Animation"],
        "themes": ["faith"],
        "setting": ["egypt"],
        "time_period": "1990s",
        "characters": ["moses"],
        "keywords": ["exodus"],
        "exclude": [],
    }

    results = search.rerank("animated bible movie about Moses", candidates, attributes)
    assert results[0]["title"] == "The Prince of Egypt"
    assert results[0]["score"] >= results[1]["score"]
    assert "semantic similarity" in results[0]["why"]


def test_rerank_prefers_k_drama_when_requested():
    candidates = [
        {
            "rank": 1,
            "doc": "A melodrama set in Seoul.",
            "meta": {
                "title": "Seoul Hearts",
                "year": "2021",
                "genres": "Drama",
                "countries": "South Korea",
                "languages": "Korean",
                "media_type": "tv",
                "media_label": "TV Series",
            },
            "dist": 0.5,
        },
        {
            "rank": 2,
            "doc": "A US crime series.",
            "meta": {
                "title": "City Detectives",
                "year": "2020",
                "genres": "Crime",
                "countries": "United States",
                "languages": "English",
                "media_type": "tv",
                "media_label": "TV Series",
            },
            "dist": 0.2,
        },
    ]

    results = search.rerank("k drama about lost memories", candidates, {})
    assert results[0]["title"] == "Seoul Hearts"
    assert "k-drama match" in results[0]["why"]


def test_rerank_prefers_asian_movie_when_requested():
    candidates = [
        {
            "rank": 1,
            "doc": "A Japanese crime thriller.",
            "meta": {
                "title": "Midnight Harbor",
                "year": "2018",
                "genres": "Thriller",
                "countries": "Japan",
                "languages": "Japanese",
                "media_type": "movie",
                "media_label": "Movie",
            },
            "dist": 0.6,
        },
        {
            "rank": 2,
            "doc": "A US courtroom drama.",
            "meta": {
                "title": "Final Verdict",
                "year": "2016",
                "genres": "Drama",
                "countries": "United States",
                "languages": "English",
                "media_type": "movie",
                "media_label": "Movie",
            },
            "dist": 0.2,
        },
    ]

    results = search.rerank("asian crime thriller", candidates, {})
    assert results[0]["title"] == "Midnight Harbor"
    assert "asian match" in results[0]["why"]
