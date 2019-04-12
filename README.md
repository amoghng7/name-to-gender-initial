# Name to Gender Classifier

This repository contains a BiLSTM Recurrent Neural Network model which classifies gender using names.

## Setup

1.  [Local Environment Setup](#local-environment-setup)
2.  [Running The Application](#running-the-application)
3.  [Running The Server](#running-the-application)
3.  [Using The API](#using-the-api)
4.  [Logs](#logs)
5.  [Notes](#notes)

## Local environment setup

### Create a virtual environment

1.  `virtualenv <env-name>`
2.  `source <env-name>/bin/activate`
3.  `deactivate` to deactivate.

Creating a virtual environment helps us to make isolated python environments. You can read more about it [here](https://pythontips.com/2013/07/30/what-is-virtualenv/).

### Install requirements

1.  `pip install -r requirements.txt`

This installs all the required packages.

## Running the application

Make sure you install all the required packages before running anything. If the `requirements.txt` doesn't contain any package then you can manually do `pip install <package-name>`.

1. `python <file-name>.py`

## Running the server

To run the API server.

1. `python runserver.py [port-number]`

&nbsp;&nbsp;&nbsp;&nbsp;ex: `python runserver.py 8000`

## Using the API

After running the server you can use the following URI:

`http://localhost:5000/api/v1.0/classify?name=john`

The API should run on local machines too:
use: `http://machine-ip:5000/api/v1.0/classify?name=john`

## Logs

* Initially we used [this](https://medium.com/simpl-under-the-hood/classifying-gender-based-on-indian-names-in-82f34ce47f6d) to build the model. Although it gave me an accuracy of 84% to 85% accuracy, but it only worked on indian names.
* In this we have used features like n-gram, sonorants, ratio of syllables and boolean (last letter vowel).
* We have also considered using features like frequency or date of birth but it didn't seem like a good path.
* Even though we increased the dataset we couldn't increase the accuracy of Machine Laerning models. So we decided to use RNN.
* We used RNN because it is good for sequential data. Note that this model will only work for first names and the API cleans the data for first names before predicting.
* We have used hyperas for hyperparameter tuning.
* All the data collected, scripts used and the py files are included in this repository.

## Notes

* There was a tensorflow session problem which was solved using [this](https://stackoverflow.com/questions/47115946/tensor-is-not-an-element-of-this-graph).
* I have used flask to run the api. You can read all it [here](http://flask.pocoo.org/).
* The cleaning done for first names in the API first removes all the unicode symbols, then converts the string to lower case letters, removes surnames if any and if there are surnames not in the repository then it leaves it as it is. Finally it takes the first word if there are more than two words in the finally cleaned name.
* If you want to add more surnames, you can add it to `./data/indian_surnames.txt`.
* If the server is not accessible you can use the following to fix it. We got this fix [here](https://stackoverflow.com/questions/7023052/configure-flask-dev-server-to-be-visible-across-the-network#answer-52131348).

1. `sudo ufw enable`
2. `sudo ufw allow 5000/tcp //allow the server to handle the request on port 5000`