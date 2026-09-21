# Leveraging Poster and Trailer-Audio Embeddings for Cold-Start Movie Recommendation

A multimodal movie recommendation system that combines collaborative filtering with visual and audio representations to address the cold-start problem for movies with little or no historical rating data.

## Overview

Traditional recommendation systems depend heavily on user–item interaction history. This creates a problem when a movie is new or has very few ratings.

This project explores whether pretrained multimodal representations can provide useful movie information even when collaborative feedback is unavailable.

The system combines:

- MovieLens user–movie ratings
- CLIP poster embeddings
- VGGish trailer-audio embeddings
- Neural collaborative filtering
- Multimodal neural recommendation
- A dedicated cold-start recommendation model

The project was developed as a V1 research/academic implementation focused on experimentation, evaluation, and recommendation inference.

---

## Problem

A newly released or rarely rated movie has limited interaction history.

A conventional collaborative recommender may therefore struggle to estimate user preference for that movie.

This project addresses this by incorporating movie content representations:

**Poster → CLIP embedding**

**Trailer audio → VGGish embedding**

These representations are combined with user and movie representations to produce predicted ratings.

---

## Approach

The project evaluates several approaches:

1. Global Mean Baseline
2. User + Movie Bias
3. Neural Collaborative Filtering (NCF)
4. Multimodal Recommender
5. Cold-Start Multimodal Recommender

The proposed multimodal model combines:

- User embedding
- Movie embedding
- CLIP poster embedding
- VGGish trailer-audio embedding

For movies that were completely unseen during training, a dedicated cold-start model uses an unknown-movie representation together with the available visual and audio embeddings.

---

## Dataset

### MovieLens 20M

Used for:

- User ratings
- Movie metadata
- Temporal train/validation/test splitting

Dataset:

https://grouplens.org/datasets/movielens/

### M³L Multimodal Features

Used for:

- CLIP poster embeddings
- VGGish trailer-audio embeddings

Source:

https://zenodo.org/records/18499145

The project uses the provided pretrained embeddings rather than training CLIP or VGGish from scratch.

---

## Dataset Alignment

After aligning MovieLens with the multimodal features:

- Multimodal movies: **19,232**
- CLIP embedding dimension: **512**
- VGGish embedding dimension: **128**
- Movies present in training interactions: **16,517**
- Test movies completely unseen during training: **2,107**
- Cold-start test interactions: **3,630**

The rating data is highly sparse, motivating the use of content information alongside collaborative signals.

---

## Results

### Overall Evaluation

| Model | Test MAE | Test RMSE |
|---|---:|---:|
| Global Mean | 0.8267 | 1.0445 |
| User + Movie Bias | 0.6799 | 0.8926 |
| NCF | 0.8262 | 1.0438 |
| Multimodal | **0.6622** | **0.8523** |

The multimodal model improved over the User + Movie Bias baseline by approximately:

- **2.61% MAE**
- **4.52% RMSE**

### Cold-Start Evaluation

For movies completely unseen during training:

- Movies: **2,107**
- Test interactions: **3,630**
- Cold-start MAE: **0.6737**
- Cold-start RMSE: **0.8843**

Compared with the global-mean baseline evaluated on the same cold-start subset:

- MAE improvement: **19.81%**
- RMSE improvement: **21.35%**

The cold-start and overall multimodal results are evaluated on different populations and therefore should not be directly compared as equivalent metrics.

---

## Recommendation System

The final inference system supports:

### Personalized Top-N Recommendation

Given a known MovieLens user, the system:

1. Builds a candidate pool of multimodal movies.
2. Removes movies already seen by the user during training.
3. Routes known movies through the multimodal model.
4. Routes training-unseen movies through the cold-start model.
5. Ranks movies by predicted rating.
6. Returns the Top-N recommendations.

### Individual Movie Prediction

The system can also estimate a user's predicted rating for a selected movie.

The model is automatically selected based on whether the movie was present during training.

### Example

```text
User
  ↓
Movie Candidate
  ↓
Known during training?
  ├── Yes → Multimodal Model
  └── No  → Cold-Start Multimodal Model
                    ↓
             Predicted Rating
```

---

## Project Structure

```text
Leveraging-Poster-and-Trailer-Audio-Embeddings-for-Cold-Start-Movie-Recommendation/
│
├── notebooks/
│   └── 01_Notebook.ipynb
│
├── recommended_artifacts/
│   ├── multimodal_model.keras
│   ├── coldstart_multimodal_model.keras
│   ├── user_to_index.pkl
│   ├── movie_to_index.pkl
│   ├── multimodal_movie_index.pkl
│   ├── user_seen_movies.pkl
│   ├── aligned_clip.npy
│   ├── aligned_audio.npy
│   ├── aligned_movie_ids.npy
│   ├── multimodal_movies.parquet
│   └── config.json
│
├── recommendation_results/
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

The local `.venv/` environment is intentionally excluded from version control.

---

## Technologies

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* PyArrow
* MovieLens
* CLIP embeddings
* VGGish embeddings
* Jupyter Notebook

---

## Limitations

This V1 implementation has several limitations:

* Recommendations require a known MovieLens user.
* The multimodal inference catalog is limited to movies with both CLIP and VGGish embeddings.
* Missing modalities are not automatically replaced with zero vectors.
* Arbitrary external movies are outside the scope of V1.
* CLIP and VGGish embeddings are used as pretrained features.
* The project is currently implemented as a notebook-based research/academic system rather than a production API or deployed application.

---

## Data Usage & Attribution

This project uses publicly available datasets/features for academic and research purposes.

Please refer to the original sources for their respective terms of use and attribution requirements.

### MovieLens

MovieLens dataset provided by GroupLens Research.

Citation:

> Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), 19.

### M³L

Multimodal movie features are obtained from the M³L dataset/resource.

The original dataset source and associated terms should be followed when using or redistributing the data.

No raw copyrighted movie posters, trailers, or other movie media are redistributed with this repository.

---

## Project Status

**V1 — Completed**

Current version focuses on:

* Dataset integration
* Multimodal feature alignment
* Model experimentation
* Cold-start evaluation
* Recommendation inference
* Personalized Top-N recommendation

Future versions may explore production deployment, online embedding extraction, broader recommendation scenarios, and additional system engineering.