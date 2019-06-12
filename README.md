# Text Classification on SST Dataset

A simple text classification on SST dataset via PyTorch.

## Models

- Sentence CNN
- Vanilla RNN
- (Bidirectional) LSTM
- RCNN (Recurrent Convolutional Neural Network)
- Ensembling the above

## Accuracy

Model| Embedding    |Fine-tuning | SST-2 | SST-5  
--|--|--|--|--
CNN  | GloVe 6B.50d  | N | 71.95 | N/A 
CNN  | GloVe 6B.50d  | Y | 78.36 | 42.85 
CNN  | GloVe 6B.300d | Y | 78.42 | 43.10  
RNN  | GloVe 6B.50d  | Y | 73.26 | 38.78 
RNN  | GloVe 6B.300d | Y | 75.88 | 38.64  
LSTM | GloVe 6B.50d  | N | 74.21 | N/A 
LSTM | GloVe 6B.50d  | Y | 75.97 | 39.37 
LSTM | GloVe 6B.300d | Y | 78.05 | 40.54  
RCNN | GloVe 6B.50d  | N | 75.52 | N/A 
RCNN | GloVe 6B.300d | Y | **80.41** | **45.02** 
Ensemble | GloVe 6B.300d | Y | **81.36** | **46.02** 