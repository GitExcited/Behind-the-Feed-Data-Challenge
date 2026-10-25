"""
Measures how good the simple approaches are.

It scores every pair in val and test with two methods:
TF-IDF: similarity based on shared words.
Pretrained MiniLM: similarity from embeddings.py, without fine-tuning.
It picks the best threshold on val (above it means “same story”), then reports precision, recall and F1 on val and test.
"""