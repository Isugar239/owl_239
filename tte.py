from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document as LangchainDocument
from langchain.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import pacmap
import numpy as np
import plotly.express as px

from typing import Optional, List


EMBEDDING_MODEL_NAME = "ai-forever/sbert_large_nlu_ru"
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Удалим дубликаты
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique


if __name__ == "__main__":
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=text, metadata={"source": "data.txt"})
    ]


    docs_processed = split_documents(
        chunk_size=512,
        knowledge_base=RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
    pd.Series(lengths).hist()
    plt.title("Распределение длин чанков (в токенах)")
    plt.show()

    # Создание эмбеддингов
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

 
    user_query = input()
    query_vector = embedding_model.embed_query(user_query)


    embedding_projector = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1)
    embeddings_2d = [
        list(KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0]) for idx in range(len(docs_processed))
    ] + [query_vector]
    documents_projected = embedding_projector.fit_transform(np.array(embeddings_2d), init="pca")

    df = pd.DataFrame.from_dict(
        [
            {
                "x": documents_projected[i, 0],
                "y": documents_projected[i, 1],
                "source": docs_processed[i].metadata["source"],
                "extract": docs_processed[i].page_content[:100] + "...",
                "symbol": "circle",
                "size_col": 4,
            }
            for i in range(len(docs_processed))
        ]
        + [
            {
                "x": documents_projected[-1, 0],
                "y": documents_projected[-1, 1],
                "source": "User query",
                "extract": user_query,
                "size_col": 100,
                "symbol": "star",
            }
        ]
    )

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="source",
        hover_data="extract",
        size="size_col",
        symbol="symbol",
        color_discrete_map={"User query": "black"},
        width=1000,
        height=700,
    )
    fig.update_traces(
        marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        legend_title_text="<b>Источник чанка</b>",
        title="<b>2D-проекция эмбеддингов чанков и запроса</b>",
    )
    fig.show()


    print(f"\n=== Поиск по базе для запроса: {user_query} ===")
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Результат {i + 1} ---")
        print(doc.page_content[:500], "...")
        print("Источник:", doc.metadata)
