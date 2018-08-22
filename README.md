# NLP_learning

   In this project, I apply a variety of neural sequence models to achieve NLP sequence tagging. These models include BiLSTM-CRF network, sequence-to-sequence (seq2seq) BiLSTM-CRF network, Seq2Seq-BiLSTM-Attention-CRF network and Seq+Seq-BiLSTM-CRF network. I will show that BiLSTM-CRF model can efficiently use both past and future input features thanks to a bidirectional LSTM component. It can also use sentence level tag information thanks to CRF layer. BiLSTM-CRF model can produce a relatively good accuracy on NER data sets. Thus, I use this model as my baseline for the experiments of other three models. Also I give the analysis and comparisons among these models. After experiments, the last Seq+Seq-BiLSTM-CRF model is the winner which can produce state of the art accuracy on the NER task. 

paper: https://github.com/wyk2796/NLP_learning/blob/master/doc/ner.pdf
