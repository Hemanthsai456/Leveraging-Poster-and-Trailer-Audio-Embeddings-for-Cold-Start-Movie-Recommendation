"""Inference wrapper around the saved notebook artifacts.

This file intentionally follows the recommendation functions from the notebook.
It does not retrain or modify the models.
"""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras


class MovieRecommender:
    def __init__(self, artifact_dir):
        self.artifact_dir = Path(artifact_dir)
        if not self.artifact_dir.exists():
            raise FileNotFoundError(f"Artifact directory not found: {self.artifact_dir.resolve()}")

        required = [
            "multimodal_model.keras",
            "coldstart_multimodal_model.keras",
            "user_to_index.pkl",
            "movie_to_index.pkl",
            "multimodal_movie_index.pkl",
            "user_seen_movies.pkl",
            "aligned_clip.npy",
            "aligned_audio.npy",
            "aligned_movie_ids.npy",
            "multimodal_movies.parquet",
            "config.json",
        ]
        missing = [x for x in required if not (self.artifact_dir / x).exists()]
        if missing:
            raise FileNotFoundError("Missing artifacts: " + ", ".join(missing))

        self.multimodal_model = keras.models.load_model(self.artifact_dir / "multimodal_model.keras")
        self.coldstart_model = keras.models.load_model(self.artifact_dir / "coldstart_multimodal_model.keras")

        self.user_to_index = self._load_pickle("user_to_index.pkl")
        self.movie_to_index = self._load_pickle("movie_to_index.pkl")
        self.multimodal_movie_index = self._load_pickle("multimodal_movie_index.pkl")
        self.user_seen_movies = self._load_pickle("user_seen_movies.pkl")

        self.aligned_clip = np.load(self.artifact_dir / "aligned_clip.npy")
        self.aligned_audio = np.load(self.artifact_dir / "aligned_audio.npy")
        self.aligned_movie_ids = np.load(self.artifact_dir / "aligned_movie_ids.npy")
        self.multimodal_movies = pd.read_parquet(self.artifact_dir / "multimodal_movies.parquet")

        with open(self.artifact_dir / "config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.unk_movie_index = self.config.get("unk_movie_index", 16517)
        self.movie_meta_map = self.multimodal_movies.set_index("movieId").to_dict(orient="index")

    def _load_pickle(self, name):
        with open(self.artifact_dir / name, "rb") as f:
            return pickle.load(f)

    @property
    def stats(self):
        return {
            "users": len(self.user_to_index),
            "known_movies": len(self.movie_to_index),
            "multimodal_movies": len(self.multimodal_movie_index),
            "clip_dim": self.config.get("clip_dim", self.aligned_clip.shape[1]),
            "audio_dim": self.config.get("audio_dim", self.aligned_audio.shape[1]),
            "embedding_dim": self.config.get("embedding_dim", 32),
        }

    def search_movies(self, query, top_n=20):
        query_str = str(query).strip()
        if not query_str:
            return pd.DataFrame()

        if query_str.isdigit():
            mid = int(query_str)
            res = self.multimodal_movies[self.multimodal_movies["movieId"] == mid]
            if not res.empty:
                return res.head(top_n).reset_index(drop=True)

        res = self.multimodal_movies[
            self.multimodal_movies["title"].str.contains(query_str, case=False, na=False)
            | self.multimodal_movies["genres"].str.contains(query_str, case=False, na=False)
        ].reset_index(drop=True)
        return res.head(top_n).reset_index(drop=True)

    def predict_movie_rating(self, user_id, movie_id):
        user_id = int(user_id)
        movie_id = int(movie_id)
        if user_id not in self.user_to_index:
            return {"status": "error", "message": f"User ID {user_id} not found."}
        if movie_id not in self.multimodal_movie_index:
            return {"status": "error", "message": "Movie does not have the required CLIP and audio embeddings."}

        user_idx = self.user_to_index[user_id]
        mm_idx = self.multimodal_movie_index[movie_id]
        clip_vec = self.aligned_clip[mm_idx:mm_idx + 1]
        audio_vec = self.aligned_audio[mm_idx:mm_idx + 1]
        u_in = np.array([[user_idx]], dtype=np.int32)
        meta = self.movie_meta_map.get(movie_id, {"title": "Unknown", "genres": "Unknown"})

        if movie_id in self.movie_to_index:
            m_in = np.array([[self.movie_to_index[movie_id]]], dtype=np.int32)
            pred = self.multimodal_model.predict([u_in, m_in, clip_vec, audio_vec], verbose=0)[0][0]
            model_type = "Multimodal Model"
            training_status = "Known (In Training)"
        else:
            m_in = np.array([[self.unk_movie_index]], dtype=np.int32)
            pred = self.coldstart_model.predict([u_in, m_in, clip_vec, audio_vec], verbose=0)[0][0]
            model_type = "Cold-Start Multimodal Model"
            training_status = "Unseen (0 Training Ratings)"

        return {
            "status": "success",
            "user_id": user_id,
            "movie_id": movie_id,
            "title": meta.get("title", "Unknown"),
            "genres": meta.get("genres", "Unknown"),
            "predicted_rating": round(float(np.clip(pred, 0.5, 5.0)), 4),
            "model_type": model_type,
            "training_status": training_status,
        }

    def recommend_movies(self, user_id, top_n=10, batch_size=2048, genre=None):
        user_id = int(user_id)
        if user_id not in self.user_to_index:
            return pd.DataFrame()

        user_idx = self.user_to_index[user_id]
        seen_set = self.user_seen_movies.get(user_id, set())
        candidate_mids = [int(m) for m in self.aligned_movie_ids if int(m) not in seen_set]

        # Optional genre filtering.
        if genre and genre != "All Genres":
            candidate_mids = [
                m
                for m in candidate_mids
                if genre.lower() in str(
                    self.movie_meta_map.get(m, {}).get("genres", "")
                ).lower().split("|")
            ]
        known_cands = [m for m in candidate_mids if m in self.movie_to_index]
        cold_cands = [m for m in candidate_mids if m not in self.movie_to_index]
        predictions = []

        if known_cands:
            u_arr = np.full((len(known_cands), 1), user_idx, dtype=np.int32)
            m_arr = np.array([[self.movie_to_index[m]] for m in known_cands], dtype=np.int32)
            mm_idxs = [self.multimodal_movie_index[m] for m in known_cands]
            c_arr = self.aligned_clip[mm_idxs]
            a_arr = self.aligned_audio[mm_idxs]
            preds = self.multimodal_model.predict([u_arr, m_arr, c_arr, a_arr], batch_size=batch_size, verbose=0).flatten()
            for m, p in zip(known_cands, preds):
                meta = self.movie_meta_map.get(m, {"title": "Unknown", "genres": "Unknown"})
                predictions.append({"movieId": m, "Movie": meta.get("title", "Unknown"), "Genres": meta.get("genres", "Unknown"), "Predicted Rating": float(np.clip(p, 0.5, 5.0)), "Model Type": "Multimodal Model", "Training Status": "Known (In Training)"})

        if cold_cands:
            u_arr = np.full((len(cold_cands), 1), user_idx, dtype=np.int32)
            m_arr = np.full((len(cold_cands), 1), self.unk_movie_index, dtype=np.int32)
            mm_idxs = [self.multimodal_movie_index[m] for m in cold_cands]
            c_arr = self.aligned_clip[mm_idxs]
            a_arr = self.aligned_audio[mm_idxs]
            preds = self.coldstart_model.predict([u_arr, m_arr, c_arr, a_arr], batch_size=batch_size, verbose=0).flatten()
            for m, p in zip(cold_cands, preds):
                meta = self.movie_meta_map.get(m, {"title": "Unknown", "genres": "Unknown"})
                predictions.append({"movieId": m, "Movie": meta.get("title", "Unknown"), "Genres": meta.get("genres", "Unknown"), "Predicted Rating": float(np.clip(p, 0.5, 5.0)), "Model Type": "Cold-Start Multimodal Model", "Training Status": "Unseen (0 Training Ratings)"})

        if not predictions:
            return pd.DataFrame()
        df = pd.DataFrame(predictions).sort_values("Predicted Rating", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", np.arange(1, len(df) + 1))
        return df.head(top_n)

    def recommend_from_movie(self, user_id, movie_id, top_n=10, batch_size=2048, personalization_weight=0.70):
        """
        Hybrid recommendation using both the user's learned preferences and
        multimodal similarity to a user-selected movie.

        The trained model supplies the personalized rating score. CLIP and
        audio embeddings supply the content-similarity score. The two scores
        are combined only at inference time; no retraining is performed.
        """
        user_id = int(user_id)
        movie_id = int(movie_id)

        if user_id not in self.user_to_index:
            return pd.DataFrame()

        if movie_id not in self.multimodal_movie_index:
            return pd.DataFrame()

        user_idx = self.user_to_index[user_id]
        seen_set = self.user_seen_movies.get(user_id, set())

        # The selected movie itself should not appear in its own recommendations.
        excluded = set(seen_set) | {movie_id}
        candidate_mids = [
            int(m) for m in self.aligned_movie_ids
            if int(m) not in excluded
        ]

        if not candidate_mids:
            return pd.DataFrame()

        # -------------------------------------------------------------
        # 1. Personalized model score
        # -------------------------------------------------------------
        predictions = []

        known_cands = [m for m in candidate_mids if m in self.movie_to_index]
        cold_cands = [m for m in candidate_mids if m not in self.movie_to_index]

        if known_cands:
            u_arr = np.full((len(known_cands), 1), user_idx, dtype=np.int32)
            m_arr = np.array([[self.movie_to_index[m]] for m in known_cands], dtype=np.int32)
            mm_idxs = [self.multimodal_movie_index[m] for m in known_cands]
            c_arr = self.aligned_clip[mm_idxs]
            a_arr = self.aligned_audio[mm_idxs]

            preds = self.multimodal_model.predict(
                [u_arr, m_arr, c_arr, a_arr],
                batch_size=batch_size,
                verbose=0
            ).flatten()

            for m, p in zip(known_cands, preds):
                predictions.append({
                    "movieId": m,
                    "personalized_rating": float(np.clip(p, 0.5, 5.0)),
                    "Model Type": "Multimodal Model",
                    "Training Status": "Known (In Training)",
                })

        if cold_cands:
            u_arr = np.full((len(cold_cands), 1), user_idx, dtype=np.int32)
            m_arr = np.full((len(cold_cands), 1), self.unk_movie_index, dtype=np.int32)
            mm_idxs = [self.multimodal_movie_index[m] for m in cold_cands]
            c_arr = self.aligned_clip[mm_idxs]
            a_arr = self.aligned_audio[mm_idxs]

            preds = self.coldstart_model.predict(
                [u_arr, m_arr, c_arr, a_arr],
                batch_size=batch_size,
                verbose=0
            ).flatten()

            for m, p in zip(cold_cands, preds):
                predictions.append({
                    "movieId": m,
                    "personalized_rating": float(np.clip(p, 0.5, 5.0)),
                    "Model Type": "Cold-Start Multimodal Model",
                    "Training Status": "Unseen (0 Training Ratings)",
                })

        if not predictions:
            return pd.DataFrame()

        df = pd.DataFrame(predictions)

        # -------------------------------------------------------------
        # 2. Multimodal similarity to the selected movie
        # -------------------------------------------------------------
        selected_idx = self.multimodal_movie_index[movie_id]
        selected_clip = self.aligned_clip[selected_idx]
        selected_audio = self.aligned_audio[selected_idx]

        candidate_indices = [self.multimodal_movie_index[m] for m in df["movieId"]]
        candidate_clip = self.aligned_clip[candidate_indices]
        candidate_audio = self.aligned_audio[candidate_indices]

        def cosine_similarity(matrix, vector):
            matrix_norm = np.linalg.norm(matrix, axis=1)
            vector_norm = np.linalg.norm(vector)
            denom = np.maximum(matrix_norm * vector_norm, 1e-12)
            return np.sum(matrix * vector, axis=1) / denom

        clip_sim = cosine_similarity(candidate_clip, selected_clip)
        audio_sim = cosine_similarity(candidate_audio, selected_audio)

        # Equal contribution from poster and trailer-audio similarity.
        multimodal_similarity = (clip_sim + audio_sim) / 2.0

        # Convert cosine similarity [-1, 1] to an interpretable [0, 5] score.
        similarity_score = np.clip((multimodal_similarity + 1.0) / 2.0 * 5.0, 0.0, 5.0)

        # 70% personalized taste + 30% similarity to selected movie.
        similarity_weight = 1.0 - personalization_weight
        df["Movie Similarity"] = multimodal_similarity
        df["Similarity Score"] = similarity_score
        df["Predicted Rating"] = df["personalized_rating"] * personalization_weight + similarity_score * similarity_weight

        # -------------------------------------------------------------
        # 3. Add movie metadata and rank
        # -------------------------------------------------------------
        titles = []
        genres = []
        for m in df["movieId"]:
            meta = self.movie_meta_map.get(int(m), {"title": "Unknown", "genres": "Unknown"})
            titles.append(meta.get("title", "Unknown"))
            genres.append(meta.get("genres", "Unknown"))

        df["Movie"] = titles
        df["Genres"] = genres

        df = df.sort_values("Predicted Rating", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", np.arange(1, len(df) + 1))

        return df[[
            "Rank", "Movie", "Genres", "Predicted Rating",
            "personalized_rating", "Similarity Score",
            "Model Type", "Training Status", "movieId"
        ]].head(top_n)

    def cold_start_movies(self, top_n=20):
        """Return catalog movies that were unseen during model training."""
        rows = []
        for mid in self.multimodal_movie_index:
            if mid not in self.movie_to_index:
                meta = self.movie_meta_map.get(mid, {})
                rows.append({"movieId": int(mid), "title": meta.get("title", "Unknown"), "genres": meta.get("genres", "Unknown")})
        return pd.DataFrame(rows).head(top_n)
