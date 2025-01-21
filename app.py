# app.py



import streamlit as st
import pathlib
from typing import Optional, List, Tuple
import pdfplumber
import os
import torch

from utils import load_css, check_password
from rag import answer_with_rag, load_chatbot_model


# setup de la page
st.set_page_config(page_title="Mon Site Vitrine", page_icon=":rocket:", layout="wide")
  

st.title("Fourreau Benjamin")
    

css_path = pathlib.Path("styles.css")
load_css(css_path)



def main():
    
    if check_password():
            
        st.markdown(
            '<p class="custom-markdown">Je suis Benjamin Fourreau, étudiant en Data Science. Voici mon parcours et mes projets.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="image-container">
                <img src="https://dl.dropboxusercontent.com/scl/fi/u2wpzwy1l968mokt2p2hl/photo.jpg?rlkey=j50pibjq71pt700xeqvi5zxka&st=9omn8mdj alt="Image 1" width="300">
                <img src="https://dl.dropboxusercontent.com/scl/fi/2xa4lhhhxe91g4z94qpbv/Logo_P.png?rlkey=daryrsr1rhobf2f97xvgq4xdw&st=p358oio9" alt="Image 2" width="300">
            </div>
            """,
            unsafe_allow_html=True
        )
        
            # Intégration du chatbot
        st.header("Chatbot")
        st.write("Discutez avec mon chatbot pour en savoir plus sur moi ou mes projets.")

        
            
            
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
        MAX_CONVERSATION_LENGTH = 5  # max historique à 5 échanges
    
    
        #Espace pour le chatbot
        chatbot_placeholder = st.empty()
        

        st.write("Sinon vous pouvez découvrir mon parcours et projets ci-dessous !")
        
        st.header("Mon Parcours")
        st.write("Je suis étudiant en double diplôme à Polytechnique Montréal, et je me spécialise en Intelligence Artificielle.")

        # Section projets
        st.header("Mes Projets")
        st.write("Voici quelques-uns de mes projets les plus récents :")
        
        #### Sous-section : Computer Vision
        st.subheader("Mes projets en Computer Vision")
        st.markdown(
            """
            <div style='text-align: center;'>
                <img src='https://dl.dropboxusercontent.com/scl/fi/vbqnwgcwkhwdip7s85lf0/CV.webp?rlkey=s77clxl3kh29gwysnsaog9g4t&st=bn1uqmv6' width='300'>
            </div>
            """,
            unsafe_allow_html=True,
        )
        #### Projet Tracking
        st.write("#### Tracking")
        st.markdown(
            "<p class='custom-markdown'>Entraînement d'un modèle de suivi avec YOLO et Deepsort sur MOT17 et MOT20.</p>",
            unsafe_allow_html=True,
        )
        
        st.markdown(
            """
            <div class="image-container">
                <img src="https://dl.dropboxusercontent.com/scl/fi/576bbpws3e9kdk7ujwigo/tracking.jpg?rlkey=r5m3tivkqycbdyqd9tq2qvixc&st=qhyzbd9u" alt="Image 1" width="350">
                <img src="https://dl.dropboxusercontent.com/scl/fi/ll0enp83z9u1lkijn7g47/trc_1.jpg?rlkey=1n25q3iboyg665mlld228a2bj&st=3f2ry605" alt="Image 2" width="350">
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">

            <div style="text-align: center; margin-top: 20px; ">
                <a href="https://github.com/Benfou21/tracking_MOT20_MOT17" target="_blank" style="text-decoration: none;">
                    <i class="fab fa-github" style="font-size:30px;color:#007bff;"></i> 
                    <span style="font-size: 20px; vertical-align: middle; margin-left: 10px;">Lien vers le GitHub</span>
                </a>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        #### Projet Détection
        st.write("#### Détection")
        st.markdown(
            "<p class='custom-markdown'>Entraînement et quantification d'un modèle de détection de feu de forêt pour drone.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="image-container">
                <img src="https://dl.dropboxusercontent.com/scl/fi/r9tswbf3f9qeqfnqznvv2/prediction_smoke.png?rlkey=bs4f3ewjkqzj120146ro7xpkc&st=cjgqcdpp" alt="Image 1" width="300">
                <img src="https://dl.dropboxusercontent.com/scl/fi/cwim8mgw6t6fo5em9xkva/batch_smoke.jpg?rlkey=6wh9b91dnvhduhm1yjz7gjb25&st=3d3td7iw" alt="Image 2" width="350">
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">

            <div style="text-align: center; margin-top: 20px; ">
                <a href="https://github.com/Benfou21/fire_smoke_detection" target="_blank" style="text-decoration: none;">
                    <i class="fab fa-github" style="font-size:30px;color:#007bff;"></i> 
                    <span style="font-size: 20px; vertical-align: middle; margin-left: 10px;">Lien vers le GitHub</span>
                </a>
            </div>
            """, 
            unsafe_allow_html=True
        )
       
        #### Sous-section : NLP
        st.subheader("Mes projets en NLP")
        
        st.markdown(
            """
            <div style='text-align: center;'>
                <img src='https://dl.dropboxusercontent.com/scl/fi/x9je48ckpmswk0zwxiasz/NLP.webp?rlkey=u4ccnc9ca2qe701pw2qgvu8ln&st=39tznz7g' width='300'>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        
        #### Projet Transformers
        st.write("#### Transformers")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div style="text-align: justify;">
                <p class='custom-markdown'>
                <strong>1 / </strong> Développement d’un modèle Transformer de zéro pour l'analyse de sentiments textuels.
                </p>
                </div>
                """, 
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                """
                <div style="text-align: justify;">
                <p class='custom-markdown'>
                <strong>2 / </strong> Étude comparative des LSTMs et Transformers pour la tâche de classification.
                </p>
                </div>
                """, 
                unsafe_allow_html=True
            )
       
        #### Projet RAG
        st.write("#### RAG")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
                <div style="text-align: justify;">
                <p class='custom-markdown'>
                <strong>1 / </strong> Implémentation d'un chatbot RAG sur mes informations, voir chatbot ci-dessus.
                </p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.markdown(
                """
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">

                <div style="text-align: center; margin-top: 20px; ">
                    <a href="https://github.com/Benfou21/learn_on_me_rag" target="_blank" style="text-decoration: none;">
                        <i class="fab fa-github" style="font-size:30px;color:#007bff;"></i> 
                        <span style="font-size: 20px; vertical-align: middle; margin-left: 10px;">Lien vers le GitHub</span>
                    </a>
                </div>
                """, 
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                """
                <div style="text-align: justify;">
                <p class='custom-markdown'>
                <strong>2 / </strong> Chatbot RAG multi vector pour répondre à des questions basées sur les documents PDF importés.
                </p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            st.markdown(
                """
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">

                <div style="text-align: center; ">
                    <a href="https://github.com/Benfou21/multi_vector_rag" target="_blank" style="text-decoration: none;">
                        <i class="fab fa-github" style="font-size:30px;color:#007bff;"></i> 
                        <span style="font-size: 20px; vertical-align: middle; margin-left: 10px;">Lien vers le GitHub</span>
                    </a>
                </div>
                """, 
                unsafe_allow_html=True
            )
        #### Sous-section : Génération d'images
        st.subheader("Mes projets en Génération d'Images")
        st.markdown(
            """
            <div style='text-align: center;'>
                <img src='https://dl.dropboxusercontent.com/scl/fi/1wq60m2r9w5stcn01kpsu/df.webp?rlkey=ggb8u81zu28z0k0dpvyeby6ra&st=hry812p1' width='300'>
            </div>
            """,
            unsafe_allow_html=True,
        )
        #### Projet Modèle de Diffusion
        st.write("#### Modèle de Diffusion")
        st.markdown(
            "<p class='custom-markdown'>Entraînement d'un modèle de diffusion probabiliste pour générer des images de style MNIST.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="image-container">
                <img src="https://dl.dropboxusercontent.com/scl/fi/x875dy9r3ut9tysoo8e7z/sample_0.png?rlkey=65v78j4tw349whhp4xqjf7btz&st=jbet6n9m" alt="Image 1" width="300">
                <img src="https://dl.dropboxusercontent.com/scl/fi/pb22eytse4scc4ffiac05/sample_14.png?rlkey=wz8av5udevsh4oj2f5tu8s879&st=01u931nz" alt="Image 2" width="300">
            </div>
            """,
            unsafe_allow_html=True
        )
        #### Sous-section : Agent Intelligent
        st.subheader("Mes projets en Agent Intelligent")
        
        #### Projet Abalone
        st.write("#### Abalone")
        st.markdown(
            "<p class='custom-markdown'>Conception d’une IA pour jouer au jeu de société Abalone, un jeu à somme nulle et information complète.</p>",
            unsafe_allow_html=True,
        )
        
        st.markdown(
            """
            <div style='text-align: center;'>
                <img src='https://dl.dropboxusercontent.com/scl/fi/m2gsn9uviitj20dj8wk52/abalone.jpg?rlkey=5acmd5le8mv40u1riki368mqq&st=x281ho2s&dl=0' width='300'>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
                """
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">

                <div style="text-align: center; margin-top: 20px; ">
                    <a href="https://github.com/Benfou21/Abalone_sources_projet" target="_blank" style="text-decoration: none;">
                        <i class="fab fa-github" style="font-size:30px;color:#007bff;"></i> 
                        <span style="font-size: 20px; vertical-align: middle; margin-left: 10px;">Lien vers le GitHub</span>
                    </a>
                </div>
                """, 
                unsafe_allow_html=True
        )
        
        #### Sous-section : Système Embarqué
        st.subheader("Mes projets en Système Embarqué")
        
        #### Projet STM32
        st.write("#### STM32")
        st.markdown(
            "<p class='custom-markdown'>Développement d'un système embarqué sur microcontrôleur STM32.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
        """
        <div style='text-align: center;'>
            <img src='https://dl.dropboxusercontent.com/scl/fi/zdxhxgaqfid5ox58tdqrq/STMlogo.png?rlkey=nwtio3uv36an0ofyprnuxhfvj&st=5y67l2m1' width='300'>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    
        # Contact
        st.header("Me Contacter")
        st.write("Vous pouvez me contacter par mail : benjamin.fourreau.epm@outlook.fr")

        with chatbot_placeholder.container() :
            
            with st.spinner('Chargement du modèle du chatbot...'):
                
                READER_LLM, RAG_PROMPT_TEMPLATE, RERANKER, KNOWLEDGE_VECTOR_DATABASE = load_chatbot_model()

            # champ input pour la question
            question = st.text_input("Vous :", key="input")
    
            if st.button("Envoyer",key="pulse"):
                if question.strip():
                    with st.spinner('Génération de la réponse...'):
                        # Mettre à jour l'historique
                        st.session_state.conversation_history.append(f"Vous : {question}")
            
                        # Limiter la longueur de l'historique
                        if len(st.session_state.conversation_history) > MAX_CONVERSATION_LENGTH * 2:
                            st.session_state.conversation_history = st.session_state.conversation_history[-MAX_CONVERSATION_LENGTH * 2:]
            
                        answer, _ = answer_with_rag(
                            question=question,
                            llm=READER_LLM,
                            knowledge_index=KNOWLEDGE_VECTOR_DATABASE,
                            conversation_history=st.session_state.conversation_history,
                            reranker=RERANKER,
                            num_retrieved_docs=20,
                            num_docs_final=5,
                            RAG_PROMPT_TEMPLATE=RAG_PROMPT_TEMPLATE
                        )
                        st.session_state.conversation_history.append(f"Chatbot : {answer}")
                        st.write(f"Chatbot : {answer}")
                   

if __name__ == "__main__":
    main()
