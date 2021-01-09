# Classifying sentences into Categories

In this step we will classify each sentences obtained from ABSA Inference method into one of the pre-defined categories. This is achieved based on a Text Classification models where the sentences are our input text. Input text are first converted into vectors using [Universal Sentence Embeddings](https://tfhub.dev/google/universal-sentence-encoder-large/5) which is a pre-trained language models downloaded from [Tensforflow Hub](https://www.tensorflow.org/hub)

# What is Text Classification?

Text classification is a supervised learning method of learning and predicting the category or the class of a document given its text content. The state-of-the-art methods are based on neural networks of different architectures as well as pre-trained language models or word embeddings.

# What is Tensorflow Hub?

TensorFlow Hub is a repository of trained machine learning models ready for fine-tuning and deployable anywhere. Reuse trained models like BERT and Faster R-CNN with just a few lines of code.

## Summary

|Notebook|Environment|Description|
|---|---|---|
|[Sentence Classification](sentence_classification_USE_SVM.ipynb)|Google Colab| A notebook for training models for classifying sentences into one of the pre-defined categories |