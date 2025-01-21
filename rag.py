import streamlit as st
import pathlib
from typing import Optional, List, Tuple
import pdfplumber
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ragatouille import RAGPretrainedModel
import torch



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
        max_new_tokens=200,
    )
    
    print("Pipeline created with quantized model")
    return READER_LLM, tokenizer


# Définition du prompt
def get_prompt_template():
    RAG_PROMPT_TEMPLATE = """
    Vous êtes un chatbot qui répond à des questions au sujet de Benjamin Fourreau, la personne décrite dans le context.
    En utilisant uniquement les informations fournies dans le contexte sans ajouter d'informations supplémentaires, fournissez une réponse complète à la question. 
    La réponse doit être en français si la question est en français.  
    Si la réponses indique des stages réponds dans l'ordre du plus récent au plus ancient.
    Si la réponses indique des projets énumère en priorité les projets avec l'indication (Important) sans intégré le texte "(Important)" à la réponse.
    La réponse doit être concise, structurée et aborder directement la question sans préambules inutiles. 
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



# Fonction pour charger le chatbot
def load_chatbot_model():
    READER_LLM, tokenizer = load_reader_model_cached()
    RAG_PROMPT_TEMPLATE = get_prompt_template()
    RERANKER = load_reranker_cached()
    
    doc_paths = ["infos.md"]
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

    return READER_LLM, RAG_PROMPT_TEMPLATE, RERANKER, KNOWLEDGE_VECTOR_DATABASE
