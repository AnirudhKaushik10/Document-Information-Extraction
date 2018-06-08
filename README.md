# Document-Information-Extraction
Extract Date,Invoice details and total amount from invoice documents

The details are read from the invoice details images using an ocr software and converted to labelled pickle file.
The format of the data is :
(text) (co-ordiantes of the text) (label)

There are 6 types of labels:

0: It stands for other class
1: It stands for invoice date
2: It stands for invoice number
8: It stands for Buyer GST Number
14: It stands for Seller GST Number
18: It stands for Total Amount in the invoice

The text is read form the pickle file and then conerted into tokens.
It is then cleaned where only alpha-numberic text is retained.
Since class 0 dominates all other classes and we can't get relevant data, then a top-n analysis is carried out
and only top-n frequently occuring words are retained.

Then the data is split into training and validation set after normalization and one-hot encoding and fed to a 3-layer
simple artificial neural network.

Overall an accuracy of 86% was achieved when top 30 words were retained and an ccuracy of 60% for all classes apart from class 0.
