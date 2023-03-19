from typing import Any, List


# TODO: Fill in the implementation. Stubbed for `mypy`.
class RetVecTokenizer():
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def tokenize(self):
        "Convert a sequence of tokens/words into RetVec embeddings"
        pass

    def detokenize(self) -> List[str]:
        "Convert a sequence of RetVec embeddings to tokens/words"
        return []

    def token_to_embedding(self):
        pass

    def embedding_to_token(self):
        pass

    def vocabulary_size(self) -> int:
        """Return RetVec vocabulary size which is all the words
        in every languages"""
        return 2**32
