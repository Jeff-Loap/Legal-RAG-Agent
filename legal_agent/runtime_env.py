from __future__ import annotations

import os


def configure_local_ml_runtime() -> None:
    # sentence-transformers in this project uses PyTorch inference only.
    # Force transformers to skip TensorFlow/Keras discovery to avoid
    # Keras 3 / tf-keras compatibility crashes during import.
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("USE_TF", "0")
