# DLGrammarCorrection
A deep learning model for English grammar correction

Six error types:
- preposition
- noun
- article
- verb
- predicate
- punctuation

## Requirements
- Python 3.x
- nltk
- TensorFlow
- numpy

See also: https://github.com/xspin/deep-text-corrector

---------------

训练模型：

```bash
python3 train.py
```

使用模型：

```
python3 test.py
```

### 模型要求

纠正英文句子中的6种错误

- 介词错误
- 名词错误
- 冠词错误
- 动词错误
- 谓语错误
- 标点错误

### 运行环境

- Python 3.6
- TensorFlow 1.10
- NLTK 3.3
- NumPy 1.13

### 模型及参数

主要思想是 sequence-to-sequence + attention mechanism，细节可参见 [Bahdanau et al., 2014](http://arxiv.org/abs/1409.0473)

encoder 和 decoder 的 RNN 每层大小和层数: 512, 4

RNN 单元：LSTM

最大单词（token）数（词典大小）：2000

因使用了一种处理OOV词（词典外的词）的策略，可以使用较小的词典，可参见 [Addressing the Rare Word Problem in NMT](https://arxiv.org/pdf/1410.8206v4.pdf) 

### 数据集

- [Cornel Movie-Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
  - 电影字幕语料，可认为是没有语法错误的语料，需要生成有错误的句子以得到训练集。

- [NUCLE Release 3.2](https://www.comp.nus.edu.sg/~nlp/conll14st.html)
  - 有错误标注，可直接用于训练。

### 数据处理

基本思想是对正确的句子加入随机“噪声”来得到训练数据，分别针对下面不同情况进行处理

- 介词、冠词、谓语、标点：将其随机替换成其他相应的词或去掉
- 动词：随机改变时态

### 错误标记

对原句子和纠正后的句子分别用NLTK标记各单词的词性，然后贪心地找出不同的（被纠正的）词，最后根据其词性确定错误的类型。

### 代码结构

- corrector/corrector.py 定义了Corrector 类，封装了训练和对句子进行纠正的函数，调用方法可参考 train.py 和 test.py
- corrector/data_reader.py, text_corrector_data_readers.py 实现了对各数据集的读取和处理
- corrector/text_corrector_models.py 实现seq2seq模型
- corrector/seq2seq.py 旧版本tf中的seq2seq包
- corrector/noise.py 提供了对文本添加噪声的函数

### 参考资料

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [GRAMMATICAL ERROR CORRECTION](https://www.cse.iitb.ac.in/~krishnachaitanyagudi/btp_report.pdf)
- [Neural Language Correction with Character-Based Attention](https://arxiv.org/pdf/1603.09727.pdf)
- [Deep Context Model for Grammatical Error Correction](http://www.slate2017.org/papers/SLaTE_2017_paper_5.pdf)
- [完全图解RNN、RNN变体、Seq2Seq、Attention机制](https://zhuanlan.zhihu.com/p/28054589)
