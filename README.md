# Synthetic Data Generator
Train your own GPT model on small datasets, generate infinite synthetic data

# What is this
This is the version of Andrej Karpathy's <a href="https://github.com/karpathy/nanoGPT/">
NanoGPT</a> reorganized the way all the files are merged into one. Multi GPU support
and GPT-2 tokenizer as well as the option to train from GPT-2 weights has been dropped.
It supports only char level model like in original <a href="https://www.youtube.com/watch?v=kCc8FmEb1nY">
tutorial</a> but preserves optimizations that were not introduced in the video. It is inteded to train
small models on tiny datasets, e.g. less than 10Mb, with an idea to replicate infinite data in style of
the original dataset.

# How to use it
    python gpt.py train   # to train model
    python gpt.py sample  # to sample from model

    See CONFIG section in "gpt.py" for more info
