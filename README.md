# QA-system

## Generative Question Answering

Jun Yin, Xin Jiang, Zhengdong Lu, Lifeng Shang, Hang Li, and Xiaoming Li. [Neural Generative Question Answering](https://arxiv.org/abs/1512.01337)

### architecture
结构分为三部分，如下图所示

![结构](https://github.com/sunxs1101/QA-system/blob/master/genqa2.png) 

 - Interpreter：将问题Q转化为一个表示H，并存在short-term memory中。这里采用双向RNN来实现（Bahdanau 2015），其中的每一个RNN用GRU（Chung et al., 2014），
 
 - Enquirer:将H作为输入，它与long-term memory中的knowledge-base交互，从中检索出相关的triples，结果保存在向量rQ中。rQ表示问题Q和triple的匹配得分，值是概率。
 
![公式](https://github.com/sunxs1101/QA-system/blob/master/genqa.png) 

 - Answerer：包括Attention Model和Generator，将H和rQ作为输入，Generator用RNN生成答复。生成的答复概率定义为，

![公式](https://github.com/sunxs1101/QA-system/blob/master/genqa1.png) 

将来自common word的部分和KB-word的部分用参数概率加起来，用logistic回归实现。z=0表示词从common vacabulary中生成，z=1表示词从KB vocabulary中生成。 
### Dataset

first,extract entities and associated triples(subject,predicate,object) from the web pages. Then the extracted data is normalized and aggregated to form a knowledge-base;second, question-answer pairs by extracting from two Chinese community QA sites.
通过计算匹配得分和过滤规则来从KB中找出与QA pair真正匹配的triple，
为了测试GENQA模型的泛化能力，用triple作为分割关键词将数据随机分割成training dataset和test dateset，
### tensorflow实现

什么是双向RNN，BiRNN(Schuster and Paliwal, 1997) has been successfully used recently in speech recognition (see, e.g., Graves
et al., 2013).BiRNN通过引入第二个状态层扩展无向RNN，它包括前向和后向RNN，前向RNN顺序读取序列计算前向隐状态，后向RNN逆序读取序列计算后向隐状态，通过将前向隐状态和后向隐状态结合获得每个词的注解
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn.py 和 
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/models.py 两个BiRNN实现。论文中cell用的是GRU，而这两个里面都是调用static_rnn()。

 - get_rnn_model()参数中有cell_type和bidirectional的定义，cell_type包括rnn，gru和lstm，bidirectional是boolean，这个调用bidirectional_rnn来实现。这里采用这个方法来实现基于GRU的双向rnn
 - BiRNN+state+one-hot
  1. BiRNN 
  
  <!-- ![BiLSTM](images/illustration.png "Title" {width=40px height=400px}) -->
<img src="https://github.com/hycis/bidirectional_RNN/blob/master/item_lstm.png" height="250">
 
## 参考文献
 - [Bahdanau et al.2015] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2015. [Neural machine translation by jointly learning to align and translate](https://arxiv.org/pdf/1409.0473v7.pdf).
 - [Chung et al.2014] Junyoung Chung, Caglar Gulcehre,KyungHyun Cho, and Yoshua Bengio. 2014. [Empirical evaluation of gated recurrent neural networks on sequence modeling](https://arxiv.org/pdf/1412.3555v1.pdf).
 - Schuster, M. and Paliwal, K. K. (1997). Bidirectional recurrent neural networks.Signal Processing,IEEE Transactions on,45(11), 2673–2681
 - Graves, A., Jaitly, N., and Mohamed, A.-R. (2013).  Hybrid speech recognition with deep bidirectional LSTM.  In Automatic Speech Recognition and Understanding (ASRU), 2013 IEEE Work-shop on , pages 273–278.
 - [Bidirectional Recurrent Neural Networks as Generative Models](https://papers.nips.cc/paper/5651-bidirectional-recurrent-neural-networks-as-generative-models.pdf)
 
