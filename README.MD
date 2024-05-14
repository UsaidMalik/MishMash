Instructions for running the Pytorch file

These are the instructions for running the code in the Jupyter notebooks. In this repo you will find three Jupyter notebooks. final_project baiyuo tensforflow.ipynb, final_project_pytorch_spacy.ipynb, and final_project_pytorch_bert.ipynb. Our intiial code was in tensor flow and we had based our file loading and development on that.

final_project_pytorch_spacy.ipynb is our next iteration using spacy embeddings trying to get a better representation of a mashed language and the final unmentioned file is our attempt using bert. to run any of the training files the data set must be downloaded from the link included in datasets.txt. We had utilized the French and English data sets provided by them. 

The relevant packages must be installed for each notebook and can be found in the imports section of each specific notebook.

Afterwards the model can be trained from scratch using the provided notebooks or it can be loaded from the weight files in discriminator.pt or generator.pt (spaCy embeddings) or from discriminator_bert (1).pt , generator_bert (1).pt (BERT embeddings). 

The models in both weight files for the generator generate a sequence of characters of some specified length in the model initialization and can be batched. The discriminator for the spacy embeddings takes the embedding of the tokenized tokens in the sentence of size batch_size * tokens * 300 and the BERT embedding discriminator does the same except it takes of size 768


References:

Some code is based on homework 2 question 5 and homework 3 question 3.
Datasets: https://www.statmt.org/europarl/
