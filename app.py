# app.py

import streamlit as st
import pathlib
from typing import Optional, List, Tuple
import pdfplumber
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ragatouille import RAGPretrainedModel
import torch



# setup de la page
st.set_page_config(page_title="Mon Site Vitrine", page_icon=":rocket:", layout="wide")
  

st.title("Fourreau Benjamin")
    
# Récupération du fichier css pour le style de la page
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

css_path = pathlib.Path("styles.css")
load_css(css_path)


#Fonction pour récupérer et vérifier le mdp
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





# Charger les documents
@st.cache_resource #Mise en cache 
def load_documents_cached(doc_paths):
    documents = []
    for path in doc_paths:
        if path.endswith(".pdf"):
            with pdfplumber.open(path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                documents.append(LangchainDocument(page_content=text, metadata={"source": path}))
        elif path.endswith(".md"):
            with open(path, "r", encoding="utf-8") as md_file:
                text = md_file.read()
                documents.append(LangchainDocument(page_content=text, metadata={"source": path}))
    return documents


@st.cache_resource
def create_faiss_index_cached(_documents, embedding_model_name):
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = FAISS.from_documents(
        _documents, embedding_model
    )
    print("Faiss index created")
    return vector_store

# On sauvegarde l'index FAISS
def save_faiss_index(vector_store, path="faiss_index"):
    print("saving faiss index... ")
    vector_store.save_local(path)
    print("faiss index saved")

# Chargement de l'index pour init plus rapide
def load_faiss_index(path="faiss_index"):
    print("loading faiss index...")
    return FAISS.load_local(path, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),allow_dangerous_deserialization=True)


# Division des documents
def basic_splitter(doc, chunk_size, chunk_overlap):
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "\n- ",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS
    )
    return text_splitter.split_documents([doc])

# Charger le reader 
@st.cache_resource
def load_reader_model_cached():
    READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta" #starchat-beta
    print("Loading reader model with quantization")
    
    # quantification en 4 bits
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    
    model = AutoModelForCausalLM.from_pretrained(
        READER_MODEL_NAME,
        quantization_config=bnb_config,
        #device_map="auto",  
        torch_dtype=torch.float16  # Charger en float16 pour les opérations
    )
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    
    # pipeline de génération de texte
    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=400,
    )
    
    print("Pipeline created with quantized model")
    return READER_LLM, tokenizer


# Définition du prompt
def get_prompt_template():
    RAG_PROMPT_TEMPLATE = """Vous êtes Benjamin Fourreau. 
    Le contexte décrit qui vous êtes et ce que vous avez fait. 
    En utilisant uniquement les informations fournies dans le contexte sans ajouter d'informations supplémentaires, fournissez une réponse complète à la question. 
    La réponse doit être en français si la question est en français. 
    Utilisez la première personne du singulier, ne vous référez pas à Benjamin comme quelqu'un d'autre. 
    Si la réponses indique des stages réponds dans l'ordre du plus récent au plus ancient.
    Si la réponses indique des projets énumère en priorité les projets avec l'indication (Important).
    La réponse doit être concise, structurée (utilisation de saut à la ligne et liste en -), et aborder directement la question sans préambules inutiles. 
    Commencez directement avec les informations pertinentes.

        Contexte:
        {context}
        
        Question:
        {question}
        
        Réponse:"""

    return RAG_PROMPT_TEMPLATE

# RAG
def answer_with_rag(
    question: str,
    llm: pipeline,
    knowledge_index: FAISS,
    conversation_history: List[str],
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 20,
    num_docs_final: int = 5,
    RAG_PROMPT_TEMPLATE=None
) -> Tuple[str, List[LangchainDocument]]:
    # Récupérer les documents pertinents
    relevant_docs = knowledge_index.similarity_search(
        query=question, k=num_retrieved_docs
    )
    relevant_docs = [doc.page_content for doc in relevant_docs]

    # Optionnellement, reranker les résultats
    if reranker:
        print("rerakning")
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Construire le contexte
    context = "".join(
        [f"Document {i}:\n{doc}\n" for i, doc in enumerate(relevant_docs)]
    )
    
    # Construire l'historique de conversation
    conversation = "\n".join(conversation_history)

    # Formater le prompt
    final_prompt = RAG_PROMPT_TEMPLATE.format(
        question=question, context=context, conversation_history=conversation
    )

    # Générer la réponse
    response = llm(final_prompt)[0]["generated_text"]

    return response, relevant_docs


#@st.cache_resource
def load_reranker_cached():
    RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    return RERANKER
    

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

        
            
        # lecteur et tokenizer
        READER_LLM, tokenizer = load_reader_model_cached()
            
        #  prompt
        RAG_PROMPT_TEMPLATE = get_prompt_template()
            
        # Charger le reranker
        RERANKER = load_reranker_cached()
    
    
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
    
        MAX_CONVERSATION_LENGTH = 5  # max historique à 5 échanges
    
        # chemins des documents
        doc_paths = ["infos.md"]
    
        # chargement des documents
        RAW_KNOWLEDGE_BASE = load_documents_cached(doc_paths)
    
        
        chunk_size = 400
        chunk_overlap = 40
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    

        if os.path.exists("faiss_index"):
            
            KNOWLEDGE_VECTOR_DATABASE = load_faiss_index()
            print("Finish loading faiss index")
        else:
            print("No faiss index path")
            
            docs_processed = []
            for doc in RAW_KNOWLEDGE_BASE:
                docs_processed += basic_splitter(doc, chunk_size, chunk_overlap)
            
            KNOWLEDGE_VECTOR_DATABASE = create_faiss_index_cached(docs_processed, embedding_model_name)
            
            save_faiss_index(KNOWLEDGE_VECTOR_DATABASE)
        
        
    
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
                    # affichage de l'historique
                    #for msg in st.session_state.conversation_history:
                    #    st.write(msg)
                    # Effacer l'entrée
                    #st.session_state.input = ""

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
        st.markdown(
            "<p class='custom-markdown'>Développement d’un modèle Transformer de zéro pour l'analyse de sentiments textuels.</p>",
            unsafe_allow_html=True,
        )
        
       
        #### Projet RAG
        st.write("#### RAG")
        st.markdown(
            "<p class='custom-markdown'>Implémentation d'un RAG sur mes informations, voir ci-dessus.</p>",
            unsafe_allow_html=True,
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

if __name__ == "__main__":
    main()
