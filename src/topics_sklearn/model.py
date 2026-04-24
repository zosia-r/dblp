import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


CUSTOM_STOPWORDS = {
    "based", "using", "approach", "method", "model",
    "systems", "system", "study", "analysis", "new",
    "proposed", "results", "paper", "performance",
}

STOPWORDS = list(set(ENGLISH_STOP_WORDS).union(CUSTOM_STOPWORDS))


class TopicModel:
    def __init__(self, n_clusters: int = 8, max_features: int = 5000, random_state: int = 42):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(
            stop_words=STOPWORDS,
            max_features=max_features,
            ngram_range=(1, 2),
        )
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        self._top_words: dict[int, list[str]] = {}

    def fit(self, titles: list[str]) -> "TopicModel":
        X = self.vectorizer.fit_transform(titles)
        self.kmeans.fit(X)

        terms = self.vectorizer.get_feature_names_out()
        for i in range(self.n_clusters):
            center = self.kmeans.cluster_centers_[i]
            self._top_words[i] = [terms[j] for j in center.argsort()[-10:]]

        return self

    def predict_batch(self, titles: list[str]) -> np.ndarray:
        X = self.vectorizer.transform(titles)
        return self.kmeans.predict(X)

    def top_words(self) -> dict[int, list[str]]:
        return self._top_words

    def name_topics_with_gemini(self) -> dict[int, str]:
        """
        Calls Gemini to assign human-readable names to topics.
        Falls back to 'Topic N' if the API key is missing or the call fails.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("  GEMINI_API_KEY not set — using fallback names.")
            return self._fallback_names()

        try:
            from google import genai  # type: ignore

            client = genai.Client(api_key=api_key)

            topics_text = "\n".join(
                f"Topic {i}: {', '.join(words)}"
                for i, words in self._top_words.items()
            )
            prompt = f"""You are given {self.n_clusters} research topics from a computer science \
paper clustering algorithm. Each topic is represented by its 10 most characteristic keywords.

{topics_text}

Assign a short, distinct name (2-5 words) to each topic.
Requirements:
- Names must be distinct from each other
- Use standard CS/academic terminology
- Be specific, not generic (avoid names like "Computer Science Research")
- Return ONLY a JSON object mapping topic number to name, \
e.g. {{"0": "Deep Learning Optimization", "1": "Network Security Protocols"}}
"""
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])

            return {int(k): v for k, v in json.loads(raw).items()}

        except Exception as exc:
            print(f"  Gemini naming failed ({exc}) — using fallback names.")
            return self._fallback_names()

    def _fallback_names(self) -> dict[int, str]:
        return {i: f"Topic {i}" for i in range(self.n_clusters)}