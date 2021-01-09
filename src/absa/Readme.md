# Aspect Based Sentiment Analysis (ABSA)

This folder contains a pseudo code, written in Jupyter notebooks, for training [Aspect Based Sentiment Analysis Models using Intel's NLP Architect](https://intellabs.github.io/nlp-architect/) models with the azure machine learning service.

# What is Aspect Based Sentiment Analysis?

Aspect based sentiment analysis (ABSA) is an advanced sentiment analysis technique that identifies and provides corresponding sentiment scores to the aspects of a given text. ABSA a powerful tool for getting actionable insight from your text data.

For example consider the sentence following resturant review 

```
The ambiance is charming. Uncharacteristically, the service was DREADFUL.When we wanted to pay our bill at the end of the evening, our waitress was nowhere to be found...
```

While traditional sentiment analysis models such as [Azure Text Analytics](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/?WT.mc_id=absa-notebook-abornst) will correctly classify the sentiment of this model as negative. An aspect based model will provide more granular insight by highlighting the fact that while the **service** and **waitress** provided a negative expirence the resturants **ambiance** was indeed positive.

## Summary

|Notebook|Environment|Description|
|---|---|---|
|[Aspect based sentiment analysis](absa-news-sentiment-classifier.ipynb)|Azure ML| A notebook for training [Aspect Based Sentiment Analysis Models using Intel's NLP Architect](http://nlp_architect.nervanasys.com/absa.html) |
|[Model Deployment](absa-endpoint.ipynb)|Local & Azure ML|A notebook for deploying model as REST API |
|[Language Detection](language-detection-using-spacy.ipynb)|Google Colab|A Notebook which walks through how to detect a language of a given text using Spacy |
|[Data Import](import-data-datastore.ipynb)|Local & Azure ML|A Notebook which walks through how to import or upload local files into data store in Azure. This is required for model training using Azure ML |

