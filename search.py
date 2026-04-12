import json
import re
from difflib import SequenceMatcher


GENERIC_TERMS = {
    "dog", "cat", "man", "woman", "boy", "girl", "kid", "kids", "baby",
    "love", "life", "death", "war", "city", "house", "home", "school",
    "friend", "friends", "family", "king", "queen", "robot", "monster",
    "cop", "police", "detective", "doll", "fish", "shark", "ghost"
}

ANIMATION_TOKENS = {
    "animation", "animated", "anime", "cartoon", "cartoons", "toon",
    "pixar", "disney", "dreamworks", "ghibli"
}

ANIMATION_SIGNAL_TERMS = {
    "animation", "animated", "anime", "cartoon", "pixar", "disney", "dreamworks", "ghibli"
}

K_DRAMA_TOKENS = {
    "kdrama", "k-drama", "k drama", "korean drama", "korean series",
    "korean tv", "korean tv show"
}

KOREAN_SIGNAL_TERMS = {
    "korea", "korean", "south korea", "seoul", "busan"
}

ASIAN_INTENT_TOKENS = {
    "asian", "asia", "japanese", "korean", "chinese", "taiwanese",
    "thai", "vietnamese", "indian", "pakistani", "bangladeshi",
    "filipino", "indonesian", "malaysian", "singaporean", "nepali",
    "sri lankan", "cambodian", "burmese", "laotian"
}

ASIAN_SIGNAL_TERMS = {
    "asia", "asian", "japan", "japanese", "korea", "korean", "south korea",
    "china", "chinese", "hong kong", "taiwan", "taiwanese", "thailand",
    "thai", "vietnam", "vietnamese", "india", "indian", "pakistan",
    "pakistani", "bangladesh", "bangladeshi", "philippines", "filipino",
    "indonesia", "indonesian", "malaysia", "malaysian", "singapore",
    "singaporean", "sri lanka", "sri lankan", "nepal", "nepali",
    "myanmar", "burma", "burmese", "cambodia", "cambodian", "laos", "laotian"
}


def _safe_json_from_text(text: str, fallback: dict) -> dict:
    if not text:
        return fallback

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return fallback

    try:
        data = json.loads(match.group())
        if isinstance(data, dict):
            return data
    except Exception:
        return fallback

    return fallback


def hyde_expand_query(client_ai, query: str) -> str:
    prompt = f"""
You are helping a movie retrieval system search more accurately.

Rewrite the user's memory into a richer search description.
Do not guess a specific title.
Focus on likely plot, motivations, relationships, setting, tone, and memorable scenes.
Avoid overemphasizing a single generic noun unless it is central to the story.

User memory:
{query}

Return only the rewritten search description.
""".strip()

    try:
        res = client_ai.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=180,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.content[0].text.strip()
    except Exception:
        return ""


def extract_attributes(client_ai, query: str) -> dict:
    fallback = {
        "title_hint": "",
        "franchise": "",
        "genres": [],
        "themes": [],
        "setting": [],
        "release_period": "",
        "setting_period": "",
        "characters": [],
        "keywords": [],
        "exclude": []
    }

    prompt = f"""
Extract structured movie-search clues from this user memory.

Memory:
"{query}"

Return JSON only with this schema:
{{
  "title_hint": "",
  "franchise": "",
  "genres": [],
  "themes": [],
  "setting": [],
  "release_period": "",
  "setting_period": "",
  "characters": [],
  "keywords": [],
  "exclude": []
}}

Rules:
- title_hint should only be filled if the user likely remembers part of a real title
- franchise should contain a likely franchise/series only if strongly implied
- genres should be standard genres like Animation, Horror, Thriller, Drama, Family, Romance, Action, Adventure, Sci-Fi, Fantasy, Crime, Mystery, Comedy
- themes should describe story structure like revenge, parent-child, survival, faith, redemption, assassin-retirement, rescue-mission, coming-of-age, time-loop
- setting should include environments like ocean, desert, spaceship, church, prison, jungle, city, village, school
- release_period should only be used if the user means when the MOVIE came out
- setting_period should only be used if the user means when the STORY takes place, like "set in the 1980s"
- characters should include distinctive entities like assassin, prophet, doll, shark, fish, alien, robot, detective, soldier
- keywords should be memorable plot clues, but avoid generic filler words
- exclude should contain things the user says it is not

Important:
- If the user says "set in the 1980s", that belongs in setting_period, not release_period.
- If the user says "a movie from the 1980s", that belongs in release_period.
- Do not treat single generic nouns like "dog" or "man" as strong title clues.

Return JSON only.
""".strip()

    try:
        res = client_ai.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=320,
            messages=[{"role": "user", "content": prompt}]
        )
        return _safe_json_from_text(res.content[0].text, fallback)
    except Exception:
        return fallback


def _norm(text: str) -> str:
    return (text or "").strip().lower()


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-zA-Z0-9\-']+", _norm(text)) if len(tok) > 2]


def _wants_animation(query: str, attributes: dict) -> bool:
    query_tokens = set(_tokenize(query))
    if query_tokens.intersection(ANIMATION_TOKENS):
        return True

    genres = attributes.get("genres", []) or []
    for genre in genres:
        genre_n = _norm(genre)
        if "animation" in genre_n or "anime" in genre_n:
            return True

    return False


def _has_any_phrase(text: str, phrases: set[str]) -> bool:
    if not text:
        return False
    text_n = _norm(text)
    tokens = set(_tokenize(text_n))
    for phrase in phrases:
        phrase_n = _norm(phrase)
        if not phrase_n:
            continue
        if " " in phrase_n:
            pattern = r"\b" + re.escape(phrase_n).replace("\\ ", r"\s+") + r"\b"
            if re.search(pattern, text_n):
                return True
        else:
            if phrase_n in tokens:
                return True
    return False


def _wants_k_drama(query: str) -> bool:
    return _has_any_phrase(query, K_DRAMA_TOKENS)


def _wants_asian(query: str) -> bool:
    return _has_any_phrase(query, ASIAN_INTENT_TOKENS)


def _is_animation_candidate(text: str) -> bool:
    text_n = _norm(text)
    return any(term in text_n for term in ANIMATION_SIGNAL_TERMS)


def _has_korean_signal(text: str) -> bool:
    return _has_any_phrase(text, KOREAN_SIGNAL_TERMS)


def _has_asian_signal(text: str) -> bool:
    return _has_any_phrase(text, ASIAN_SIGNAL_TERMS)


def _contains_any(haystack: str, items: list[str]) -> int:
    haystack_n = _norm(haystack)
    tokens = set(_tokenize(haystack_n))
    hits = 0
    for item in items:
        item_n = _norm(item)
        if not item_n:
            continue
        if " " in item_n:
            pattern = r"\b" + re.escape(item_n).replace("\\ ", r"\s+") + r"\b"
            if re.search(pattern, haystack_n):
                hits += 1
        else:
            if item_n in tokens:
                hits += 1
    return hits


def _distance_similarity(dist: float, min_dist: float, max_dist: float) -> float:
    if max_dist <= min_dist:
        return 0.0
    if dist <= min_dist:
        return 1.0
    if dist >= max_dist:
        return 0.0
    return 1.0 - ((dist - min_dist) / (max_dist - min_dist))


def _count_query_overlap(text: str, query_tokens: list[str], generic_terms: set[str]) -> int:
    text_n = _norm(text)
    hits = 0
    for tok in query_tokens:
        if tok in text_n:
            hits += 0.5 if tok in generic_terms else 1.0
    return hits


def _fuzzy_title_bonus(title: str, title_hint: str, original_title: str = "") -> float:
    t = _norm(title)
    ot = _norm(original_title)
    hint = _norm(title_hint)

    if not hint:
        return 0.0

    hint_tokens = _tokenize(hint)
    if hint_tokens and all(tok in GENERIC_TERMS for tok in hint_tokens):
        return 0.0

    bonus = 0.0
    for candidate_title in [t, ot]:
        if not candidate_title:
            continue
        ratio = SequenceMatcher(None, hint, candidate_title).ratio()
        if hint in candidate_title:
            bonus = max(bonus, 18.0)
        elif ratio >= 0.88:
            bonus = max(bonus, 13.0)
        elif ratio >= 0.75:
            bonus = max(bonus, 7.0)

    return bonus


def _franchise_bonus(franchise: str, collection_name: str, title: str, doc: str) -> float:
    f = _norm(franchise)
    if not f:
        return 0.0

    search_space = " ".join([collection_name or "", title or "", doc or ""]).lower()
    return 10.0 if f in search_space else 0.0


def _period_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return " ".join(parts)
    return ""


def _release_period_bonus(release_period, year_value: str) -> float:
    release_period_text = _period_text(release_period)
    if not release_period_text or not year_value:
        return 0.0

    year_text = str(year_value).strip()
    if not year_text.isdigit():
        return 0.0

    year = int(year_text)
    rp = release_period_text.lower()

    if "1990" in rp or "90" in rp:
        return 6.0 if 1990 <= year <= 1999 else 0.0
    if "1980" in rp or "80" in rp:
        return 6.0 if 1980 <= year <= 1989 else 0.0
    if "2000" in rp:
        return 6.0 if 2000 <= year <= 2009 else 0.0
    if "2010" in rp:
        return 6.0 if 2010 <= year <= 2019 else 0.0
    if "2020" in rp:
        return 6.0 if 2020 <= year <= 2029 else 0.0

    return 0.0


def _setting_period_bonus(setting_period, combined_text: str) -> float:
    """
    Match story setting period ONLY against overview/doc text, not release year.
    """
    setting_period_text = _period_text(setting_period)
    if not setting_period_text:
        return 0.0

    text = _norm(combined_text)
    sp = setting_period_text.lower()

    hints = []

    if "1980" in sp or "80" in sp:
        hints = ["1980s", "80s", "eighties", "cold war", "arcade", "cassette", "soviet"]
    elif "1990" in sp or "90" in sp:
        hints = ["1990s", "90s", "nineties", "dial-up", "boy band", "vhs"]
    elif "2000" in sp:
        hints = ["2000s", "early 2000s", "y2k"]
    elif "2010" in sp:
        hints = ["2010s"]
    elif "2020" in sp:
        hints = ["2020s"]

    if not hints:
        return 0.0

    hits = sum(1 for hint in hints if hint in text)
    return min(6.0, hits * 2.5)


def rerank(query: str, candidates: list[dict], attributes: dict) -> list[dict]:
    results = []

    title_hint = attributes.get("title_hint", "")
    franchise = attributes.get("franchise", "")
    genres = attributes.get("genres", []) or []
    themes = attributes.get("themes", []) or []
    setting = attributes.get("setting", []) or []
    characters = attributes.get("characters", []) or []
    keywords = attributes.get("keywords", []) or []
    exclude = attributes.get("exclude", []) or []
    release_period = attributes.get("release_period", "")
    setting_period = attributes.get("setting_period", "")
    wants_animation = _wants_animation(query, attributes)
    wants_k_drama = _wants_k_drama(query)
    wants_asian = _wants_asian(query)

    query_tokens = _tokenize(query)
    named_entities = [tok.lower() for tok in re.findall(r"\b[A-Z][a-z]+\b", query or "")]
    specific_tokens = [tok for tok in query_tokens if tok not in GENERIC_TERMS]

    for c in candidates:
        meta = c.get("meta", {}) or {}
        doc = c.get("doc", "") or ""

        title = meta.get("title", "Unknown")
        original_title = meta.get("original_title", "")
        year = str(meta.get("year", "N/A"))
        poster = meta.get("poster", "")
        overview = meta.get("overview", "") or doc
        genres_text = meta.get("genres", "") or ""
        keywords_text = meta.get("keywords", "") or ""
        collection_name = meta.get("collection", "") or ""
        tagline = meta.get("tagline", "") or ""
        media_type = meta.get("media_type", "movie")
        media_label = meta.get("media_label", "Movie")
        spoiler_excerpt = meta.get("spoiler_excerpt", "") or ""
        spoiler_source_url = meta.get("spoiler_source_url", "") or ""
        countries_text = meta.get("countries", "") or ""
        languages_text = meta.get("languages", "") or ""

        # Separate title text from plot text
        title_text = " ".join([title, original_title, collection_name]).lower()
        plot_text = " ".join([overview, tagline, keywords_text, spoiler_excerpt, doc]).lower()
        combined_text = " ".join([title_text, plot_text, genres_text, countries_text, languages_text]).lower()
        signal_text = " ".join([genres_text, keywords_text, title_text, plot_text, countries_text, languages_text])

        rank_index = max(1, int(c.get("rank", 1)))

        # Lower base so bonuses matter, but not too much
        base_score = max(14.0, 62.0 - ((rank_index - 1) * 1.75))
        score = base_score
        why = []

        # Title bonus only if user likely remembers actual title fragment
        title_bonus = _fuzzy_title_bonus(title, title_hint, original_title)
        if title_bonus > 0:
            score += title_bonus
            why.append("title clue matched")

        franchise_bonus = _franchise_bonus(franchise, collection_name, title, combined_text)
        if franchise_bonus > 0:
            score += franchise_bonus
            why.append("franchise clue matched")

        # Genre bonus
        genre_hits = _contains_any(genres_text, genres)
        if genre_hits:
            score += min(8.0, genre_hits * 3.5)
            why.append("genre matched")

        # Favor animated content when the user is explicit about cartoons/anime
        is_animation = _is_animation_candidate(signal_text)
        if wants_animation:
            if is_animation:
                score += 12.0
                why.append("animation intent matched")
            else:
                score -= 14.0
                why.append("non-animated downweighted")

        # Favor K-drama content when the user is explicit about it
        if wants_k_drama:
            is_korean = _has_korean_signal(signal_text)
            if media_type == "tv" and is_korean:
                score += 14.0
                why.append("k-drama match")
            elif media_type == "tv":
                score += 5.0
                why.append("tv series matched")
            elif is_korean:
                score += 7.0
                why.append("korean match")
            else:
                score -= 14.0
                why.append("non-kdrama downweighted")
        elif wants_asian:
            is_asian = _has_asian_signal(signal_text)
            if is_asian:
                score += 9.0
                why.append("asian match")
            else:
                score -= 9.0
                why.append("non-asian downweighted")

        # Theme/setting/character/keyword should reward plot text more than title text
        theme_hits = _contains_any(plot_text, themes)
        if theme_hits:
            score += min(10.0, theme_hits * 4.0)
            why.append("theme matched")

        setting_hits = _contains_any(plot_text, setting)
        if setting_hits:
            score += min(8.0, setting_hits * 3.5)
            why.append("setting matched")

        character_hits = _contains_any(plot_text, characters)
        if character_hits:
            score += min(8.0, character_hits * 3.5)
            why.append("character clue matched")

        keyword_hits = _contains_any(plot_text, keywords)
        if keyword_hits:
            score += min(7.0, keyword_hits * 3.0)
            why.append("keyword matched")

        # Overlap: plot overlap is stronger than title overlap
        plot_overlap = _count_query_overlap(plot_text, query_tokens, GENERIC_TERMS)
        title_overlap = _count_query_overlap(title_text, query_tokens, GENERIC_TERMS)

        if plot_overlap:
            score += min(10.0, plot_overlap * 1.4)
            why.append("plot overlap found")

        if title_overlap:
            score += min(2.5, title_overlap * 0.5)

        entity_hits = _contains_any(plot_text, named_entities)
        if entity_hits:
            score += min(14.0, entity_hits * 7.0)
            why.append("named character matched")

        specific_hits = _contains_any(plot_text, specific_tokens)
        if specific_hits:
            score += min(10.0, specific_hits * 2.5)
            why.append("specific clue matched")

        # Penalize cases where only a vague generic title word matched
        if title_overlap > 0 and plot_overlap == 0:
            generic_overlap_tokens = [tok for tok in query_tokens if tok in title_text and tok in GENERIC_TERMS]
            if generic_overlap_tokens:
                score -= min(8.0, len(generic_overlap_tokens) * 3.0)
                why.append("generic title-only clue downweighted")

        release_bonus = _release_period_bonus(release_period, year)
        if release_bonus:
            score += release_bonus
            why.append("release period matched")

        setting_period_bonus = _setting_period_bonus(setting_period, plot_text)
        if setting_period_bonus:
            score += setting_period_bonus
            why.append("story period matched")

        exclude_hits = _contains_any(combined_text, exclude)
        if exclude_hits:
            score -= min(18.0, exclude_hits * 9.0)
            why.append("some excluded clue appeared")

        dist = c.get("dist")
        if isinstance(dist, (float, int)):
            score -= min(18.0, max(0.0, dist) * 8.0)
            why.append("semantic similarity")

        # Soft cap
        if score > 95:
            score = 95 + ((score - 95) * 0.12)

        score = round(max(0.0, min(99.0, score)), 1)

        results.append({
            "title": title,
            "year": year,
            "poster": poster,
            "overview": overview,
            "genres": genres_text,
            "keywords": keywords_text,
            "media_type": media_type,
            "media_label": media_label,
            "spoiler_excerpt": spoiler_excerpt,
            "spoiler_source_url": spoiler_source_url,
            "score": score,
            "why": why
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results
