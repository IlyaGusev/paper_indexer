from typing import Any, Dict, List, cast

import numpy as np
from sentence_transformers import SentenceTransformer


GEMMA_DEFAULT_MODEL_NAME = "unsloth/embeddinggemma-300m"
GEMMA_DEFAULT_DOCUMENT_TEMPLATE = "title: {} | text: {}"


class GemmaEmbedder:
    def __init__(
        self,
        model_name: str = "unsloth/embeddinggemma-300m",
        batch_size: int = 16,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode_documents(
        self,
        records: List[Dict[str, Any]],
    ) -> np.ndarray:
        prompts = [
            GEMMA_DEFAULT_DOCUMENT_TEMPLATE.format(r["title"], r["abstract"]) for r in records
        ]
        embeddings = self.model.encode(prompts, batch_size=self.batch_size)
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        embeddings = self.model.encode_query([query], convert_to_numpy=True)
        return cast(np.ndarray, cast(np.ndarray, embeddings)[0])
