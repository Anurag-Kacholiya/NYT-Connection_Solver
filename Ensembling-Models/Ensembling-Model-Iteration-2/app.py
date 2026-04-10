import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from embedder import ConnectionsEmbedder
from evaluate import build_matrices, extract_features, compute_per_matrix_separation
from solver import InteractiveSolver
from reasoner import get_feature_reasoning, get_llm_reasoning

# --- PAGE CONFIG ---
st.set_page_config(page_title="Connections AI Solver", layout="wide")

# --- RESOURCE LOADING ---
@st.cache_resource
def load_all():
    embedder = ConnectionsEmbedder()
    model_files = {
        "rf":        "models/clf_rf.pkl",
        "gbm":       "models/clf_gbm.pkl",
        "ranker":    "models/clf_lgbm_ranker.pkl",
        "lr_fusion": "models/clf_lr_fusion.pkl",
    }
    loaded = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded[name] = pickle.load(f)
        else:
            loaded[name] = None
    return embedder, loaded

embedder, models = load_all()
clf_rf        = models["rf"]
clf_gbm       = models["gbm"]
clf_ranker    = models["ranker"]
clf_lr_fusion = models["lr_fusion"]

# --- SESSION STATE INIT ---
if "game_active" not in st.session_state:
    st.session_state.game_active    = False
    st.session_state.solver         = None
    st.session_state.mats           = None
    st.session_state.mat_names      = None
    st.session_state.history        = []
    st.session_state.mistakes       = 0
    st.session_state.solved_groups  = []
    st.session_state.suggestions    = []
    st.session_state.llm_reasoning  = {}

def restart():
    for key in ["game_active", "solver", "mats", "mat_names",
                "history", "mistakes", "solved_groups", "suggestions",
                "llm_reasoning"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- HEADER ---
st.title("Connections — AI Solver")
st.caption(f"AI SOLVER  •  {time.strftime('%B %d, %Y')}")

# Model status row
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    if clf_rf:
        st.success("RF: ONLINE")
    else:
        st.error("RF: OFFLINE")
with col_s2:
    if clf_gbm:
        st.success("GBM: ONLINE")
    else:
        st.error("GBM: OFFLINE")
with col_s3:
    if clf_ranker:
        st.success("Ranker: ONLINE")
    else:
        st.error("Ranker: OFFLINE")
with col_s4:
    if clf_lr_fusion:
        st.success("LR Fusion: ONLINE")
    else:
        st.warning("LR Fusion: OFFLINE")

st.divider()

# ==============================================================
# INPUT SCREEN
# ==============================================================
if not st.session_state.game_active:
    st.markdown("### Load New Puzzle")
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        words_text = st.text_area(
            "Paste 16 words (comma or newline separated):",
            height=160,
            value=(
                "PIPE, PANCAKE, PULPIT, LIGHT SWITCH\n"
                "SHELL, VIOLIN, MUSHROOM, ORBIT\n"
                "COIN, PASTEURIZE, NUCLEUS, MAGNIFYING GLASS\n"
                "ELECTRON, GOOGOL, DEERSTALKER, THE BIRD"
            ),
        )
    with col_in2:
        st.write("")
        st.write("")
        start_clicked = st.button(
            "SOLVE THIS PUZZLE",
            use_container_width=True,
            type="primary",
        )

    if start_clicked:
        raw   = words_text.replace("\n", ",").split(",")
        words = [w.strip().upper() for w in raw if w.strip()]
        if len(words) != 16:
            st.error(f"Need exactly 16 words. Found {len(words)}.")
        elif clf_ranker is None:
            st.error("Ranker model not found. Run train.py first.")
        else:
            with st.spinner("Building similarity matrices — this may take a minute..."):
                mat_names, mats = build_matrices(embedder, words)
                solver = InteractiveSolver(words, clf_ranker)
                st.session_state.mat_names     = mat_names
                st.session_state.mats          = mats
                st.session_state.solver        = solver
                st.session_state.game_active   = True
                st.session_state.mistakes      = 0
                st.session_state.solved_groups = []
                st.session_state.history       = []
                st.session_state.suggestions   = []
            st.rerun()

# ==============================================================
# GAME SCREEN
# ==============================================================
if st.session_state.game_active:
    solver    = st.session_state.solver
    mats      = st.session_state.mats
    mat_names = st.session_state.mat_names

    main_col, eng_col = st.columns([3, 2])

    # ----------------------------------------------------------
    # LEFT COLUMN — game board + AI controls
    # ----------------------------------------------------------
    with main_col:
        # Solved group banners
        BANNER_ICONS = ["🟡", "🟢", "🔵", "🟣"]
        for i, grp in enumerate(st.session_state.solved_groups):
            icon       = BANNER_ICONS[i % 4]
            label      = grp.get("label", "")
            prefix     = f"**{label}** — " if label else ""
            display    = "  •  ".join(grp["words"])
            words_key  = tuple(grp["words"])

            col_banner, col_reason_btn = st.columns([5, 1])
            with col_banner:
                st.success(f"{icon}  {prefix}{display}")
            with col_reason_btn:
                if st.button("Reason", key=f"reason_solved_{i}"):
                    indices = tuple(solver.words.index(w) for w in grp["words"])
                    sep = compute_per_matrix_separation(
                        list(indices), mats, set(range(16))
                    )
                    feature_reason = get_feature_reasoning(sep, mat_names)
                    with st.spinner("..."):
                        llm_reason = get_llm_reasoning(grp["words"], feature_reason)
                    st.session_state.llm_reasoning[words_key] = llm_reason

            if words_key in st.session_state.get("llm_reasoning", {}):
                st.caption(st.session_state.llm_reasoning[words_key])

        # Remaining word grid
        rem_words = [solver.words[i] for i in sorted(solver.remaining_indices)]
        if rem_words:
            cols = st.columns(4)
            for idx, w in enumerate(rem_words):
                with cols[idx % 4]:
                    st.button(w, key=f"word_{w}", disabled=True, use_container_width=True)
            st.write("")

        # Mistakes counter (plain text — no HTML dots)
        lives = 4 - st.session_state.mistakes
        st.write(f"Mistakes remaining: {'🟤 ' * lives}{'⚫ ' * st.session_state.mistakes}")
        if st.session_state.mistakes >= 4:
            st.warning("Game over — but feel free to keep exploring the remaining groups!")
        st.divider()

        # AI suggestion + feedback buttons
        if not solver.remaining_indices or len(solver.remaining_indices) < 4:
            st.success("PUZZLE COMPLETE!")
            st.button("Start New Puzzle", on_click=restart)
        else:
            suggestions = solver.get_next_suggestions(mats, mat_names, top_k=5)
            st.session_state.suggestions = suggestions

            if not suggestions:
                st.warning("No valid suggestions remaining.")
                st.button("Start New Puzzle", on_click=restart)
            else:
                top       = suggestions[0]
                guess_num = len(st.session_state.history) + 1
                st.write(f"**AI Suggestion #{guess_num}:**")
                st.info(f"### {'  •  '.join(top['words'])}")

                # --- Layer 1: Feature-driven reasoning (instant) ---
                sep = compute_per_matrix_separation(
                    list(top["indices"]), mats, solver.remaining_indices
                )
                feature_reason = get_feature_reasoning(sep, mat_names)
                st.caption(f"**Why these 4?**  {feature_reason}")

                # --- Layer 2: LLM reasoning (on-demand) ---
                words_key = tuple(top["words"])
                if st.button("Reason", key="explain_btn"):
                    with st.spinner("Generating explanation..."):
                        llm_reason = get_llm_reasoning(top["words"], feature_reason)
                        st.session_state.llm_reasoning[words_key] = llm_reason

                if words_key in st.session_state.get("llm_reasoning", {}):
                    st.info(
                        f"**AI explanation:** {st.session_state.llm_reasoning[words_key]}"
                    )

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    if st.button("Correct", use_container_width=True, type="primary"):
                        label = st.session_state.get("group_label_input", "")
                        solver.add_feedback(top["indices"], "correct")
                        st.session_state.solved_groups.append(
                            {"words": top["words"], "label": label}
                        )
                        st.session_state.history.append(
                            {"words": top["words"], "result": "Correct",
                             "score": top["score"]}
                        )
                        st.rerun()
                with c2:
                    if st.button("One Away", use_container_width=True):
                        solver.add_feedback(top["indices"], "one_away")
                        st.session_state.mistakes += 1
                        st.session_state.history.append(
                            {"words": top["words"], "result": "One Away",
                             "score": top["score"]}
                        )
                        st.rerun()
                with c3:
                    if st.button("Wrong", use_container_width=True):
                        solver.add_feedback(top["indices"], "incorrect")
                        st.session_state.mistakes += 1
                        st.session_state.history.append(
                            {"words": top["words"], "result": "Wrong",
                             "score": top["score"]}
                        )
                        st.rerun()
                with c4:
                    st.button("New Puzzle", on_click=restart, use_container_width=True)

                st.text_input(
                    "Group label (optional — fill before clicking Correct):",
                    key="group_label_input",
                    placeholder="e.g. THINGS THAT FLY",
                )

        # Guess history
        if st.session_state.history:
            st.divider()
            st.write("**Guess history:**")
            for i, h in enumerate(st.session_state.history):
                if h["result"] == "Correct":
                    icon = "✅"
                elif h["result"] == "One Away":
                    icon = "⚠️"
                else:
                    icon = "❌"
                st.write(
                    f"{icon} **{i+1}.** {', '.join(h['words'])} "
                    f"— *{h['result']}* (score: {h['score']:.3f})"
                )

    # ----------------------------------------------------------
    # RIGHT COLUMN — engineer insights
    # ----------------------------------------------------------
    with eng_col:
        st.subheader("Engineer Insights")

        suggestions = st.session_state.get("suggestions", [])

        if suggestions and solver.remaining_indices:
            top          = suggestions[0]
            cand_indices = list(top["indices"])

            # Top ranked groups table
            st.write("**Top suggested groups (ranked):**")
            rows = [
                {"Words": " · ".join(s["words"]), "Score": f"{s['score']:.4f}"}
                for s in suggestions
            ]
            st.table(pd.DataFrame(rows))

            # Per-matrix separation bar chart
            sep = compute_per_matrix_separation(
                cand_indices, mats, solver.remaining_indices
            )
            st.write("**Per-matrix separation (mean_in − mean_out) for top pick:**")
            sep_df = pd.DataFrame(
                {"Matrix": mat_names, "Separation": sep.tolist()}
            ).sort_values("Separation", ascending=False)

            fig_sep, ax_sep = plt.subplots(figsize=(5, max(3, len(mat_names) * 0.35)))
            bar_colors = ["#3366ff" if v >= 0 else "#cc3333"
                          for v in sep_df["Separation"]]
            ax_sep.barh(sep_df["Matrix"], sep_df["Separation"], color=bar_colors)
            ax_sep.axvline(0, color="black", linewidth=0.8)
            ax_sep.set_xlabel("Separation score")
            ax_sep.invert_yaxis()
            fig_sep.tight_layout()
            st.pyplot(fig_sep)
            plt.close(fig_sep)

            # LR Fusion learned weights
            if clf_lr_fusion is not None:
                st.write("**Learned LR matrix weights:**")
                coefs     = clf_lr_fusion.coef_[0]
                n         = min(len(coefs), len(mat_names))
                weight_df = pd.DataFrame(
                    {"Matrix": mat_names[:n], "Weight": coefs[:n]}
                ).sort_values("Weight", ascending=False)

                fig_w, ax_w = plt.subplots(figsize=(5, max(3, n * 0.35)))
                col_w = ["#3366ff" if v >= 0 else "#cc3333" for v in weight_df["Weight"]]
                ax_w.barh(weight_df["Matrix"], weight_df["Weight"], color=col_w)
                ax_w.axvline(0, color="black", linewidth=0.8)
                ax_w.set_xlabel("LR coefficient")
                ax_w.invert_yaxis()
                fig_w.tight_layout()
                st.pyplot(fig_w)
                plt.close(fig_w)

            # RF vs GBM agreement
            if clf_rf is not None and clf_gbm is not None:
                feat = extract_features(
                    cand_indices, mats,
                    solver.remaining_indices, solver.words, mat_names,
                )
                feat_arr = np.array([feat])
                rf_prob  = clf_rf.predict_proba(feat_arr)[:, 1][0]
                gbm_prob = clf_gbm.predict_proba(feat_arr)[:, 1][0]
                delta    = abs(rf_prob - gbm_prob)

                st.write("**Model agreement for top candidate:**")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("RF prob",  f"{rf_prob:.3f}")
                col_m2.metric("GBM prob", f"{gbm_prob:.3f}")
                col_m3.metric("Delta",    f"{delta:.3f}")

                if delta < 0.1:
                    st.success("HIGH CONSENSUS — classifiers agree")
                elif delta < 0.3:
                    st.warning("MODERATE DISAGREEMENT")
                else:
                    st.error("HIGH UNCERTAINTY — possible trap category")

            # Score distribution histogram
            all_scores = [s["score"] for s in suggestions]
            if len(all_scores) > 1:
                st.write("**Candidate score landscape:**")
                fig_h, ax_h = plt.subplots(figsize=(5, 3))
                sns.histplot(all_scores, bins=min(15, len(all_scores)),
                             ax=ax_h, color="#3366ff")
                ax_h.axvline(all_scores[0], color="red",
                             linestyle="--", label="top pick")
                ax_h.set_xlabel("Ranker score")
                ax_h.legend(fontsize=8)
                fig_h.tight_layout()
                st.pyplot(fig_h)
                plt.close(fig_h)

        else:
            st.write("Waiting for puzzle to start...")
