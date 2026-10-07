import os
import streamlit as st
from pathlib import Path
from recommender import MovieRecommender

# PAGE CONFIG

st.set_page_config(
    page_title="Multimodal Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# CUSTOM CSS

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #111827,
            #312e81
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: .4rem;
    }

    .movie-card {
        padding: 1.1rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
        min-height: 150px;
        margin-bottom: 1rem;
    }

    .score {
        font-size: 1.6rem;
        font-weight: 700;
    }

    .small {
        opacity: .75;
        font-size: .9rem;
    }

    .architecture-box {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(128,128,128,.08);
        border: 1px solid rgba(128,128,128,.2);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# LOAD MODEL

DEFAULT_ARTIFACT = Path(__file__).parent / "recommended_artifacts"

artifact_input = st.sidebar.text_input(
    "Artifacts folder",
    str(DEFAULT_ARTIFACT)
)

@st.cache_resource(show_spinner="Loading trained models and artifacts...")
def load_recommender(path):
    return MovieRecommender(path)

try:
    rec = load_recommender(artifact_input)
except Exception as e:
    st.error("The UI cannot load the saved model artifacts yet.")
    st.code(str(e))
    st.info(
        "Put the recommended_artifacts folder beside app.py "
        "or change the folder path in the sidebar."
    )
    st.stop()

# SIDEBAR

page = st.sidebar.radio(
    "Pages",
    [
        "🎬 Recommendations",
        "❄️ Cold Start",
        "🧠 About Model"
    ]
)

# PAGE 1 — RECOMMENDATIONS

if page == "🎬 Recommendations":
    st.markdown(
        """
        <div class="hero">
        <h1>Multimodal Movie Recommender</h1>
        <p>
        Generate recommendations using user preferences,
        movie similarity, or genre preferences.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # PROJECT STATS
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Users",
        f"{rec.stats['users']:,}"
    )

    c2.metric(
        "Multimodal Movies",
        f"{rec.stats['multimodal_movies']:,}"
    )

    c3.metric(
        "Embedding Dimensions",
        f"CLIP {rec.stats['clip_dim']} · "
        f"Audio {rec.stats['audio_dim']}"
    )

    st.divider()

    # RECOMMENDATION MODE

    st.subheader("🎯 Recommendation Mode")
    mode = st.radio(
        "Choose how you want recommendations",
        [
            "👤 General Recommendations",
            "🎬 Based on a Movie",
            "🎭 By Genre"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.divider()

    # MODE 1 — GENERAL
    if mode == "👤 General Recommendations":

        st.subheader("👤 General Personalized Recommendations")

        st.write(
            "The trained model ranks unseen movies according "
            "to the user's learned preferences."
        )

        user_id = st.number_input(
            "MovieLens User ID",
            min_value=1,
            value=1,
            step=1,
            key="general_user"
        )

        top_n = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            key="general_top_n"
        )

        if st.button(
            "✨ Generate Recommendations",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Generating personalized recommendations..."):
                results = rec.recommend_movies(
                    user_id=int(user_id),
                    top_n=top_n
                )

            if results.empty:
                st.warning(
                    "No recommendations found. "
                    "Check the User ID."
                )

            else:
                st.success(
                    f"Recommendations generated for "
                    f"User {int(user_id)}."
                )

                st.subheader("🍿 Recommended For You")

                cols = st.columns(2)
                for i, row in results.iterrows():
                    with cols[i % 2]:
                        st.markdown(
                            f"""
                            <div class="movie-card">

                            <h3>
                            #{int(row['Rank'])} · {row['Movie']}
                            </h3>

                            <p>
                            {str(row['Genres']).replace('|', ' · ')}
                            </p>

                            <div class="score">
                            ⭐ {row['Predicted Rating']:.2f} / 5
                            </div>

                            <p class="small">
                            {row['Model Type']}
                            ·
                            {row['Training Status']}
                            </p>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )


    # MODE 2 — MOVIE BASED

    elif mode == "🎬 Based on a Movie":

        st.subheader("🎬 Recommendations Based on a Movie")

        st.write(
            "Choose a movie you like. The system combines "
            "the user's learned preferences with multimodal "
            "similarity to the selected movie."
        )

        user_id = st.number_input(
            "MovieLens User ID",
            min_value=1,
            value=1,
            step=1,
            key="movie_user"
        )

        query = st.text_input(
            "🔎 Search for a movie you like",
            placeholder=(
                "Try: Interstellar, "
                "Dark Knight, Toy Story..."
            ),
            key="movie_search"
        )

        selected_movie_id = None
        movie_options = {}

        if query.strip():
            found = rec.search_movies(
                query.strip(),
                top_n=20
            )

            if found.empty:
                st.info(
                    "No matching movies found. "
                    "Try another title."
                )

            else:
                movie_options = {
                    int(row["movieId"]):
                    (
                        f"{row['title']} — "
                        f"{str(row['genres']).replace('|', ' · ')}"
                    )
                    for _, row in found.iterrows()
                }

                selected_movie_id = st.selectbox(
                    "Select the movie",
                    list(movie_options.keys()),
                    format_func=lambda x:
                        movie_options[x],
                    key="selected_movie"
                )

        top_n = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            key="movie_top_n"
        )

        if st.button(
            "✨ Recommend Based on Movie",
            type="primary",
            use_container_width=True,
            disabled=selected_movie_id is None
        ):

            with st.spinner("Combining user taste with movie similarity..."):
                results = rec.recommend_from_movie(
                    user_id=int(user_id),
                    movie_id=int(selected_movie_id),
                    top_n=top_n
                )

            if results.empty:
                st.warning(
                    "No recommendations found. "
                    "Check the User ID or selected movie."
                )

            else:
                selected_title = (
                    movie_options[selected_movie_id]
                    .split(" — ")[0]
                )

                st.success(
                    f"Recommendations generated using "
                    f"User {int(user_id)}'s taste and "
                    f"{selected_title}."
                )

                st.subheader("🍿 Recommended For You")

                cols = st.columns(2)

                for i, row in results.iterrows():
                    with cols[i % 2]:
                        st.markdown(
                            f"""
                            <div class="movie-card">

                            <h3>
                            #{int(row['Rank'])} · {row['Movie']}
                            </h3>

                            <p>
                            {str(row['Genres']).replace('|', ' · ')}
                            </p>

                            <div class="score">
                            ⭐ {row['Predicted Rating']:.2f} / 5
                            </div>

                            <p class="small">
                            Personalized:
                            {row['personalized_rating']:.2f}
                            ·
                            Movie similarity:
                            {row['Similarity Score']:.2f}
                            </p>

                            <p class="small">
                            {row['Model Type']}
                            ·
                            {row['Training Status']}
                            </p>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        st.caption(
            "Hybrid ranking: 70% learned personalized rating "
            "+ 30% multimodal similarity to the selected movie "
            "(poster/CLIP + trailer audio)."
        )

    # MODE 3 — GENRE

    else:

        st.subheader("🎭 Personalized Recommendations by Genre")

        st.write(
            "Choose a genre and the trained model will rank "
            "movies from that genre according to the user's "
            "learned preferences."
        )

        # USER
        user_id = st.number_input(
            "MovieLens User ID",
            min_value=1,
            value=1,
            step=1,
            key="genre_user"
        )

        # GENRE LIST

        genres = [
            "All Genres",
            "Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "IMAX",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western"
        ]

        selected_genre = st.selectbox(
            "🎭 Choose a genre",
            genres,
            key="selected_genre"
        )

        top_n = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=20,
            value=10,
            key="genre_top_n"
        )

        if st.button(
            "✨ Recommend in This Genre",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Finding movies matching the user's taste..."):

                if selected_genre == "All Genres":
                    results = rec.recommend_movies(
                        user_id=int(user_id),
                        top_n=top_n
                    )

                else:
                    results = rec.recommend_movies(
                        user_id=int(user_id),
                        top_n=top_n,
                        genre=selected_genre
                    )

            if results.empty:
                st.warning(
                    "No recommendations found for this "
                    "user and genre."
                )
                
            else:
                if selected_genre == "All Genres":
                    st.success(
                        f"Personalized recommendations "
                        f"generated for User {int(user_id)}."
                    )

                else:
                    st.success(
                        f"Personalized {selected_genre} "
                        f"recommendations generated for "
                        f"User {int(user_id)}."
                    )

                st.subheader("🍿 Recommended Movies")

                cols = st.columns(2)

                for i, row in results.iterrows():
                    with cols[i % 2]:

                        st.markdown(
                            f"""
                            <div class="movie-card">

                            <h3>
                            #{int(row['Rank'])} · {row['Movie']}
                            </h3>

                            <p>
                            {str(row['Genres']).replace('|', ' · ')}
                            </p>

                            <div class="score">
                            ⭐ {row['Predicted Rating']:.2f} / 5
                            </div>

                            <p class="small">
                            {row['Model Type']}
                            ·
                            {row['Training Status']}
                            </p>

                            </div>
                            """,
                            unsafe_allow_html=True
                        )

# PAGE 2 — COLD START

elif page == "❄️ Cold Start":

    st.markdown(
        """
        <div class="hero">

        <h1>Cold-Start Recommendation</h1>

        <p>
        Demonstration of the cold-start model for movies
        without historical training interactions.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    user_id = st.number_input(
        "MovieLens User ID",
        min_value=1,
        value=1,
        step=1,
        key="cold_user"
    )

    cold = rec.cold_start_movies(50)

    if cold.empty:

        st.error(
            "No unseen movies were found in the saved catalog."
        )

    else:

        labels = {
            int(r.movieId):
            (
                f"{r.title} — "
                f"{str(r.genres).replace('|', ' · ')}"
            )
            for _, r in cold.iterrows()
        }

        selected = st.selectbox(
            "Choose a catalog movie with no training interaction",
            list(labels),
            format_func=lambda x: labels[x]
        )

        if st.button(
            "❄️ Predict Cold-Start Rating",
            type="primary",
            use_container_width=True
        ):

            result = rec.predict_movie_rating(user_id, selected)

            if result["status"] == "success":
                st.success(
                    f"Predicted rating: "
                    f"{result['predicted_rating']:.2f} / 5"
                )

                a, b = st.columns(2)
                a.metric("Movie", result["title"])
                b.metric("Model", result["model_type"])

                st.write(
                    f"**Genres:** "
                    f"{str(result['genres']).replace('|', ' · ')}"
                )

                st.write(
                    f"**Training status:** "
                    f"{result['training_status']}"
                )


            else:
                st.error(result["message"])

# PAGE 3 — ABOUT MODEL

elif page == "🧠 About Model":

    st.markdown(
        """
        <div class="hero">

        <h1>🧠 How the Model Works</h1>

        <p>
        Multimodal neural architecture combining user preferences,
        movie identity, poster information and trailer-audio information.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # ARCHITECTURE

    st.header("🏗️ Multimodal Model Architecture")

    st.write("The final multimodal model combines four inputs:")


    st.code(
        """
                                    INPUTS
                                        │
        ┌──────────────┬───────────────┼───────────────┐
        │              │               │               │              
        ▼              ▼               ▼               ▼              
    User ID       Movie ID      Poster Image    Trailer Audio        
        │              │               │               │             
        ▼              ▼               ▼               ▼             
    Embedding(32)  Embedding(32)    CLIP 512-D       VGGish 128-D    
        │              │               │               │             
        │              │               ▼               ▼             
        │              │           Dense(32)       Dense(32)         
        │              │             ReLU            ReLU            
        │              │               │               │             
        └──────────────┴───────────────┴───────────────┘             
                                    │
                                    ▼
                                Concatenate
                                    │
                                    ▼
                                Dense(64)
                                  ReLU
                                    │
                                    ▼
                                Dropout(0.2)
                                    │
                                    ▼
                                Dense(32)
                                  ReLU
                                    │
                                    ▼
                                Dense(1)
                                    │
                                    ▼
                            Predicted Rating
                                0.5 – 5.0
        """,
        language="text"
    )

    st.subheader("📐 Architecture Details")

    architecture = [
        {
            "Component": "User Input",
            "Details": "User ID → Embedding",
            "Dimension": "32"
        },
        {
            "Component": "Movie Input",
            "Details": "Movie ID → Embedding",
            "Dimension": "32"
        },
        {
            "Component": "Poster Input",
            "Details": "CLIP embedding → Dense(32, ReLU)",
            "Dimension": "512 → 32"
        },
        {
            "Component": "Trailer Audio Input",
            "Details": "VGGish embedding → Dense(32, ReLU)",
            "Dimension": "128 → 32"
        },
        {
            "Component": "Fusion",
            "Details": "Concatenation",
            "Dimension": "128"
        },
        {
            "Component": "Hidden Layer 1",
            "Details": "Dense + ReLU",
            "Dimension": "64"
        },
        {
            "Component": "Regularization",
            "Details": "Dropout",
            "Dimension": "0.2"
        },
        {
            "Component": "Hidden Layer 2",
            "Details": "Dense + ReLU",
            "Dimension": "32"
        },
        {
            "Component": "Output",
            "Details": "Dense",
            "Dimension": "1 rating"
        }
    ]

    st.dataframe(
        architecture,
        use_container_width=True,
        hide_index=True
    )

    # DATA SOURCES
    st.header("🎞️ Multimodal Inputs")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("CLIP Poster", "512-D")
        st.write("Provides visual information from the movie poster.")

    with c2:
        st.metric("VGGish Audio", "128-D")
        st.write("Provides acoustic information from the movie trailer.")

    with c3:
        st.metric("MovieLens", "Ratings")
        st.write("Provides user–movie preference information.")

    st.info(
        "The project integrates MovieLens user preference data "
        "with CLIP poster embeddings and VGGish trailer-audio "
        "embeddings. The aligned multimodal catalog contains "
        "19,232 movies."
    )

    # RECOMMENDATION METHODS

    st.header("🎯 Recommendation Methods")

    st.markdown(
        """
        ### 👤 General Recommendations

        The trained multimodal model predicts ratings for unseen
        movies for a given user and ranks them from highest to lowest.

        ### 🎬 Movie-Based Recommendations

        A hybrid inference layer combines:

        **70% learned personalized rating**

        +

        **30% multimodal similarity to the selected movie**

        Poster/CLIP and trailer-audio embeddings are used for the
        movie similarity component.

        ### 🎭 Genre Recommendations

        The existing trained model is used without modification.
        Candidate movies are first filtered to the selected genre,
        and the model then ranks those movies according to the
        user's learned preferences.
        """
    )

    # COLD START

    st.header("❄️ Cold-Start Mechanism")

    st.markdown(
        """
        For movies with no training ratings, the system uses a
        dedicated cold-start multimodal model.

        An explicit **UNK movie index** represents the missing
        movie identity while the model still receives:

        - User representation
        - CLIP poster representation
        - Trailer-audio representation

        This allows the system to make predictions for movies
        with zero historical training ratings.
        """
    )

    # EVALUATION

    st.header("📊 Evaluation Results")

    results = [
        {
            "Model": "Global Mean",
            "Test MAE": 0.826679,
            "Test RMSE": 1.044473
        },
        {
            "Model": "User + Movie Bias",
            "Test MAE": 0.679932,
            "Test RMSE": 0.892608
        },
        {
            "Model": "NCF",
            "Test MAE": 0.825484,
            "Test RMSE": 1.043304
        },
        {
            "Model": "Multimodal",
            "Test MAE": 0.637302,
            "Test RMSE": 0.831881
        },
        {
            "Model": "Cold-Start Multimodal",
            "Test MAE": 0.673103,
            "Test RMSE": 0.894764
        }
    ]

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    # SYSTEM COMPONENTS

    st.header("📦 Saved System Components")

    st.markdown(
        """
        The application loads the artifacts produced by the
        training notebook:

        - `multimodal_model.keras`
        - `coldstart_multimodal_model.keras`
        - User/movie mappings
        - CLIP embeddings
        - Audio embeddings
        - Movie metadata
        - User interaction history
        - Model configuration
        """
    )