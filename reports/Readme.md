## Data Preparation
----

This folder contains a pseudo code for ABSA Inference job to get predictions from ABSA models, set of Jupyter Notebooks for categories prediction using [Text Classification](src/absa_sentence_classifier) models trained in our earlier steps and prepare the source data in order for executing YAML file as our next step. 

## What is ABSA Inference Job?

ABSA Inference is an equivalent step of obtaining predictions from any other ML models. Intel NLP Architect's ABSA library has a standalone Inference clause which can be executed seperately with inputs which we will obtain upon finishing the training of ABSA models with our own text corpus. 

## What is Azure DSVM?

Azure DSVM is a linux based virtual machine and pre-installed with most data science libraries by default. Please refer to Azure portal [here](portal.azure.com) for Azure DSVM named as **coe-dsvm** which has been provisioned for this purpose.

**WinSCP** is a file transfer software tool which can be used to transfer files from local machine to Azure DSVM and vice-versa.

## How to access Azure DSVM in Azure?

1. Logon to [Azure](portal.azure.com) and open **coe-dsvm** under all services from Dashboard view. Click on **Connect** and further click on **RDP** which will in turn download RDP file. This file can be used to connect to Azure DSVM from local machine. Below are the login credentials. 

    ```
    session: Xorg
    username: coeuser
    password: Welcome@1234
    ```
2. Once logged on to Azure DSVM (ie., target PC) and perform step 2 as per Instructions step below. Step 3 and 4 can be performed using WinSCP. 
3. Open terminal and navigate to the directory ``nlp-architect\examples\absa\inference\``
4. Type ``vi step1_code_predict_sentiment.py`` and it will open the source code in vi editor. Any other file editors can also be used instead of vi. 
5. Edit the lines highlighted below accordingly. 
    ```
    inference = SentimentInference(`<full-directory-path-of-aspect-lexicon-including-file-name-wt-extension>`, `<full-directory-path-of-opiinion-lexicon-including-file-name-wt-extension>`, parse = True)
    ```
    ```
    with open(`<full-directory-path-of-target-dataset-for-prediction-including-file-name-wt-extension>`, 'r', encoding = 'latin-1') as csv:
    ```
    ```
    with open(`<destination-directory>` + `<destination-file>` + .json, 'a') as json_file:
    ```
6. Save and run `python step1_code_predict_sentiment.py`

## Instructions

1. Install NLP Architect library in Azure DSVM or any other linux machine if prefered. 
2. Intel's NLP architect can be installed by following the link [here](https://intellabs.github.io/nlp-architect/installation.html). By default it will be installed in the following location on the target PC.
    ```
    nlp-architect\examples\absa\
    ```
3. Transfer the following files to the target PC. These files can be put in any location in the target PC. WinSCP is recommended to use for this file transfer activities.

    |File|Description|
    |---|---|
    |[Aspect Lexicon](/input/generated_aspect_lex_updated_v3.csv)|Generated aspects by ABSA model |

4. Place ``step1_code_predict_sentiment.py`` in the following directory on the target PC. 
```
    nlp-architect\examples\absa\inference\
```

|Notebook|Environment|Description|
|---|---|---|
|[Sentiment Prediction](step1_code_predict_sentiment.py)|Azure DS VM or Linux OS Machine|ABSA Inference job written in Python to get predictions for news articles from already trained ABSA models |
|[Categories Prediction](step2_code_predict_categories.ipynb)|Google Colab|A Notebook which walks through obtaining predicted categories for each sentences based on a Text Classification model previously trained|
|[Data Preparation](step3_code_prepare_data_reports.ipynb)|Local|A Notebook contains step wise procedure in order for preparing data to be used in YAML file|
