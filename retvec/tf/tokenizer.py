from typing import List


class RetVecTokenizer():
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def tokenize():
        "Convert a sequence of tokens/words into RetVec embeddings"
        pass

    def detokenize() -> List[str]:
        "Convert a sequence of RetVec embeddings to tokens/words"
        pass

    def token_to_embedding():
        pass

    def embedding_to_token():
        pass

    def vocabulary_size(self) -> int:
        """Return RetVec vocabulary size which is all the words
        in every languages"""
        return 2**32
