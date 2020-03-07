# NLP_learning

   In this project, I applied a variety of neural sequence models to achieve NLP sequence tagging. These models contained BiLSTM-CRF network, sequence-to-sequence (seq2seq) BiLSTM-CRF network, Seq2Seq-BiLSTM-Attention-CRF network and Seq+Seq-BiLSTM-CRF network. The result show that BiLSTM-CRF model can efficiently use both past and future input features with a bidirectional LSTM component. BiLSTM-CRF model can produce a relatively good accuracy on NER data sets. Thus, I used this model as my baseline for the experiments of other three models. Also I gave the analysis and comparisons among these models. After experiments, the last Seq+Seq-BiLSTM-CRF model is the best which can produce state of the art accuracy on the NER task. 

paper: https://github.com/wyk2796/NLP_learning/blob/master/doc/ner.pdf
