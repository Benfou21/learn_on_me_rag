# utils.py

import streamlit as st
import pathlib
import os
from rag import basic_splitter, create_faiss_index_cached, load_documents_cached, load_faiss_index, load_reader_model_cached, load_reranker_cached, save_faiss_index, get_prompt_template

# Récupération du fichier css pour le style de la page
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Fonction pour récupérer et vérifier le mot de passe
def check_password():
    def password_entered():
        if st.session_state["password"] == "ben":
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Demande du mot de passe
        st.text_input("Mot de passe", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.error("Mot de passe incorrect")
        return False
    else:
        return True
