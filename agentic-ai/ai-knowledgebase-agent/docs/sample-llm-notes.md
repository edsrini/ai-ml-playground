# Sample notes on large language models

This is invented sample text so the project runs before you add your own documents.

## What is a large language model?

A large language model (LLM) is a neural network trained on a very large amount of text to predict the next token in
a sequence. A token is a small piece of text, often a word or part of a word. By repeating that prediction many
times, a model can write paragraphs, answer questions, summarise documents and write code.

## The transformer architecture

Most LLMs are built on the transformer, a neural network architecture introduced in 2017. A transformer uses
self-attention: for every token it decides how much to focus on every other token in the input, so it can relate a
word to context far away in the text. Because attention processes all tokens in parallel, transformers train
efficiently on GPUs.

## Embeddings and vector search

An embedding turns a piece of text into a list of numbers so that texts with similar meaning get similar numbers.
Storing the embeddings of many passages in a vector database lets a program find the passages closest in meaning to
a question, even when they share no keywords.

## Retrieval-augmented generation (RAG)

RAG gives a model facts it was never trained on. The program first retrieves the passages most relevant to the
question from a document collection, then places them in the prompt, and the model writes the answer from those
passages. This reduces made-up answers, and the knowledge can be updated by changing the documents instead of
retraining the model.

## Limits and risks

LLMs can hallucinate, which means stating false things with confidence. They have a context window, a limit on how
much text they can read at once, and they do not know events after their training date. Running a model locally keeps
the data on your own machine.
