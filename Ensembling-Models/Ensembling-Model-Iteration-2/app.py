import streamlit as st
import numpy as np
import pandas as pd
from embedder import ConnectionsEmbedder
from solver import solve
import time
from transformers import pipeline
import torch
# Configure page
st.set_page_config(page_title="Connections NLP Solver", layout="wide")
st.title("🧩 NYT Connections NLP Solver")
st.markdown("Enter 16 words from a Connections puzzle to see how our AI groups them using Semantic Embeddings, Lexical N-Grams, and Knowledge Graphs (WordNet & WikiData).")

@st.cache_resource
def load_embedder():
    return ConnectionsEmbedder()

@st.cache_resource
def load_llm():
    # Use flan-t5-base as a lightweight CPU-friendly explanation generator
    return pipeline("text2text-generation", model="google/flan-t5-base")
def double_center(mat):
    r_mu = np.mean(mat, axis=1, keepdims=True)
    c_mu = np.mean(mat, axis=0, keepdims=True)
    g_mu = np.mean(mat)
    return mat - r_mu - c_mu + g_mu

@st.cache_data
def process_game(words):
    embedder = load_embedder()
    # Cache disabled for LLM generation inside process_game, 
    # so we load it here but Streamlit handles caching outside
    llm = load_llm()    
    # 1. Get raw matrices
    embeddings = embedder.get_embeddings(words)
    semantic_sim_raw = np.dot(embeddings, embeddings.T)
    lexical_sim_raw = embedder.get_lexical_similarity(words)
    wordnet_sim_raw = embedder.get_wordnet_similarity(words)
    wikidata_sim_raw = embedder.get_wikidata_similarity(words)
    datamuse_sim_raw = embedder.get_datamuse_similarity(words)
    conceptnet_sim_raw = embedder.get_conceptnet_similarity(words)
    wikipedia_sim_raw = embedder.get_wikipedia_categories_similarity(words)
    
    # 2. Normalize (Double Center)
    semantic_sim = double_center(semantic_sim_raw)
    lexical_sim = double_center(lexical_sim_raw)
    wordnet_sim = double_center(wordnet_sim_raw)
    wikidata_sim = double_center(wikidata_sim_raw)
    datamuse_sim = double_center(datamuse_sim_raw)
    conceptnet_sim = double_center(conceptnet_sim_raw)
    wikipedia_sim = double_center(wikipedia_sim_raw)
    
    # 3. Combine with optimized weights
    sim_matrix = (0.26 * semantic_sim + 
                  0.12 * lexical_sim + 
                  0.38 * wordnet_sim + 
                  0.01 * wikidata_sim +
                  0.17 * datamuse_sim +
                  0.01 * conceptnet_sim +
                  0.05 * wikipedia_sim)
    np.fill_diagonal(sim_matrix, 0.0)
    
    # 4. Solve exactly
    best_partition, _ = solve(sim_matrix)
    
    # 5. Determine Reasoning
    results = []
    for group_indices in best_partition:
        group_words = [words[i] for i in group_indices]
        
        # Calculate feature scores for this group based on centered matrices
        scores = {
            "Semantic": 0.0, 
            "Lexical": 0.0, 
            "WordNet": 0.0, 
            "WikiData": 0.0,
            "Datamuse": 0.0,
            "ConceptNet": 0.0,
            "Wikipedia": 0.0
        }
        pairs = 0
        for i in range(4):
            for j in range(i+1, 4):
                idx1, idx2 = group_indices[i], group_indices[j]
                
                # Using max(0, x) so negative similarities don't break percentages.
                scores["Semantic"] += max(0, semantic_sim[idx1][idx2])
                scores["Lexical"] += max(0, lexical_sim[idx1][idx2])
                scores["WordNet"] += max(0, wordnet_sim[idx1][idx2])
                scores["WikiData"] += max(0, wikidata_sim[idx1][idx2])
                scores["Datamuse"] += max(0, datamuse_sim[idx1][idx2])
                scores["ConceptNet"] += max(0, conceptnet_sim[idx1][idx2])
                scores["Wikipedia"] += max(0, wikipedia_sim[idx1][idx2])
                pairs += 1
                
        total_score = sum(scores.values()) + 1e-9 # avoid div by zero
        
        # Determine dominant reason
        reasoning_percentages = {k: (v/total_score) * 100 for k, v in scores.items()}
        dominant_feature = max(reasoning_percentages, key=reasoning_percentages.get)
        
        # Build prompt for LLM based on dominant feature
        word_list_str = ", ".join(group_words)
        
        if dominant_feature == "Semantic":
            prompt = f"What is the common broad semantic theme or category that connects these 4 words: {word_list_str}?"
        elif dominant_feature == "Lexical":
            prompt = f"What shared spelling pattern, prefix, suffix, or wordplay connects these 4 words: {word_list_str}?"
        elif dominant_feature == "WordNet":
            prompt = f"What specific type of thing are all 4 of these words: {word_list_str}?"
        elif dominant_feature == "WikiData":
            prompt = f"What specific real-world entity property or pop culture category connects: {word_list_str}?"
        elif dominant_feature == "Datamuse":
            prompt = f"What frequently associated word, phrase, or idiom connects these 4 words: {word_list_str}?"
        elif dominant_feature == "ConceptNet":
            prompt = f"What common-sense fact or relationship connects these 4 words: {word_list_str}?"
        elif dominant_feature == "Wikipedia":
            prompt = f"What extremely specific Wikipedia category or trivia fact connects: {word_list_str}?"
            
        # Generate prediction
        try:
            llm_output = llm(prompt, max_new_tokens=20, do_sample=False)[0]['generated_text']
            reason_text = f"**Generated Connection:** {llm_output.title()}\n\n*(Driven by {dominant_feature} similarity: {reasoning_percentages[dominant_feature]:.1f}%)*"
        except Exception as e:
            reason_text = f"Primary Driver: **{dominant_feature}**. (Failed to generate LLM explanation)"
            
        results.append({
            "words": group_words,
            "percentages": reasoning_percentages,
            "explanation": reason_text
        })
        
    return results

# UI
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    The system computes four $16 \\times 16$ similarity matrices:
    1. **Semantic** (MPNet) [55%]
    2. **Lexical** (Character N-Grams) [15%]
    3. **WordNet** (Wu-Palmer) [15%]
    4. **WikiData** (Knowledge Graph Properties) [15%]
    
    It then aggregates them and runs a combinatorial search to find the 4 perfectly grouped words that maximize similarity score.
    """)
    st.info("Loading the model takes ~10 seconds on the first run.")

words_input = st.text_area("Enter 16 words, separated by commas or newlines:", height=150, 
                           value="BASS, FLOUNDER, SALMON, TROUT\nGUITAR, DRUM, PIANO, FLUTE\nAPPLE, BANANA, ORANGE, GRAPE\nRED, BLUE, GREEN, YELLOW")

if st.button("Solve Game", type="primary"):
    # Parse words
    raw_words = words_input.replace("\n", ",").split(",")
    words = [w.strip().upper() for w in raw_words if w.strip()]
    
    if len(words) != 16:
        st.error(f"Please enter exactly 16 words! You entered {len(words)}.")
    else:
        with st.spinner("Embedding and computing similarities..."):
            start_time = time.time()
            try:
                results = process_game(words)
                end_time = time.time()
                
                st.success(f"Solved in {end_time - start_time:.2f} seconds!")
                
                st.subheader("Predicted Groups & Reasoning")
                
                # Use columns for nice layout
                cols = st.columns(2)
                for index, res in enumerate(results):
                    with cols[index % 2]:
                        st.markdown(f"### Group {index+1}")
                        # Display words as nice pills/buttons
                        html_words = "".join([f"<span style='background-color:#EFEFEF; color:black; padding:8px 12px; border-radius:10px; margin-right:5px; font-weight:bold;'>{w}</span>" for w in res['words']])
                        st.markdown(html_words, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Display reason
                        st.markdown(res['explanation'])
                        
                        # Show breakdown
                        st.progress(res['percentages']['Semantic'] / 100, text=f"Semantic ({res['percentages']['Semantic']:.1f}%)")
                        st.progress(res['percentages']['Lexical'] / 100, text=f"Lexical ({res['percentages']['Lexical']:.1f}%)")
                        st.progress(res['percentages']['WordNet'] / 100, text=f"WordNet ({res['percentages']['WordNet']:.1f}%)")
                        st.progress(res['percentages']['WikiData'] / 100, text=f"WikiData ({res['percentages']['WikiData']:.1f}%)")
                        st.divider()
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
