# Project Report

# Abstract

# Introduction

The rise of the social media over the past few years has been nothing less than a revolution. Platforms developed to connect people on a local as well as a global scale through exchange of text and multimedia have lately also been exploited to manipulate public opinions through spreading of disinformation, a task usually carried by programmatically generated artificial bots that can generate fake text and multimedia that appears human-like and authentic respectively. As the technology to create such data (often referred to as deepfakes) advances, automated mechanisms that can detect such text also ought to be developed in order to effectively tackle the wrongdoings that can be effectuated using such advanced text-genrative algorithms.

Thus, for this project:
1) We utilise the TweepFake dataset, which contains human-written and various machine-generated tweets (Markov Chains, RNN, RNN+Markov, LSTM, GPT-2) to build and train numerous deep learning models for the task of **generated text detection**: classifying a piece of text as machine-generated or human-written. 
1) We inspect the performance of various state-of-the-art language models for this **generated text detection** task that undertake **text classification** to help us detect such deepfake tweets. These language models include BERT and RoBERTa.
2) We see how well a baseline GRU model fares against multiple state-of-the-art transformer language models for the **generated text detection** task.
3) We explore the implementation, training and performance of text-to-text transfer transformer model on the Tweepfake dataset, currently unprecedented in the domain of machine-generated text detection. 
4) We delve into the methodology of cross-lingual zero-shot transfer learning using English-French translated human tweets as the test-set by implementing back-translation as a data-augmentation strategy, a new and exciting path to take research in this field further.

# Motivation and Contributions/Originality

GPT-2, a pre-trained language model that was released by OpenAI in 2019, can generate coherent, non-trivial and human-like text samples. There is a growing evidence that the adversaries are exploiting the tremendous generative capabilities from GPT-2 to write deepfake messages to contaminate public debate. <br>

This is especially rampant on social medias where the adversaries have the motivation to spread the disinformation through GPT-2 and make such deliberate attempts to seed false narratives or to achieve certain political and social agenda. Therefore, it is essential to develop the detection system to distinguish the messages from human and GPT-2 and mitigate the misuse of text generative models (TGMs) <br>

There are not many deepfake detection systems being conducted over social media texts. TweepFake is the first deepfake tweets dataset that collected tweets from a total of 23 bots, imitating 17 human accounts, which are based on various generation techniques. The paper claims that the transformer architecture could produce high-quality short texts that are even difficult for expert human annotators to detect and the paper useed transformer-based language models which could achieve nearly 90% accuracy. <br>

For this project, we aim to use other techniques to investigate if other models could achieve similar or even better accuracy. In details, we plan to use advanced contextualized word representation model like BERT to detect the deepfake tweets. [reference](https://arxiv.org/abs/2008.00036) We also plan to use Hugging Face M2M100 to translate the English text to other languages such as French. Since all the translated texts are technically machine generated, we could use our models to test how it performs the classification on other languages and see if it could differentiate the bot from human texts. (all the texts should be bot in this case) [reference](https://huggingface.co/facebook/m2m100_418M)

# Previous works

Our undertaking here is not unprecedented. Extensive research in the field of artificial text generation and detection has been done for a few years now. For instance, \[1\] and \[2\] propose a large-scale unsupervised language model, that generates coherent and human-like sequence of texts, GPT-2. It shows incredible text-generation capability that has become both a source of amazement and concern in the deep learning community, since even humans seem unable to detect automatically generated text. Thus, to counter these models' ability to produce deceiving multimedia, various deepfake detecting strategies are continuously being developed and researched, from video to audio to text detectors. While the majority of undertaken studies focus on text sequences outside the social media domain, like \[2\] (in-house research by OpenAI) and \[3\], some have also focussed on social media posts, like \[4\] and \[5\]. Additionally, TweepFake - A Twitter Deep Fake Dataset \[6\] provides the first properly annotated dataset of human and *real* machine-generated social media posts (specifically from Twitter). For this project, we have used this dataset and build upon and benefited from these previous studies, innovations and research.

As meticulously elaborated in \[7\], the most widespead methodology to confront the potential downsides to various text-generative models is to formulate the problem as a classification task, with classes being, in our case, machine-generated and human-written. Here's a categorised (based on underlying mechanics) description of some of the detectors that have been tried thus far:

1) Simple classifiers trained from scratch like bag-of-words \[2\] and detecting machine configuration \[8\].
2) Zero-shot detection like Total log probability \[2\] and GLTR \[3\]
3) Fine-tuning language models as explored in \[9\], \[2\] and \[6\].

Our application borrows a lot of ideas from the third strategy as discussed above. However, by introducing multilinguality to the equation, we've combined elements of fine-tuning  language models and cross-lingual zero-shot transfer learning to explore new avenues in this domain.

Furthermore, we also look into some of the previous works in the domain of multilingual machine translation models. MMTs aim to build a single model to translate between multiple pairs of languages. Models based on bilingual translation have been largely successful thus far \[10\]. Moreover, many neural MMT models have also had impressive perfomance. Nonetheless, it is noteworthy to note that many of these MMT models have had worse results than a bilingual translation model \[11\] even though most of these models operate at high capacity \[12\]. This is partially because most of these models have been trained on English-Centric datasets which translate from or to English but not between the combination of non-English languages. This English-centric bias in the data has evidently led to lower performance, specifically for non-English translations. This is where the Many-to-Many model for 100 languages comes into the picture \[13\]. It is trained on an improved dataset that comprises of 7.5B training sentences for 100 languages and is not English-centric and subsequently performs competitively with the state-of-the-art bilingual translation models. We've used Facebook's M2M100 model to translate the human-written English tweets into French to test our cross-lingual zero-shot transfer learning approach.

Previously, there has been strenous work to improve models with monolingual data, which includes language model fusion \[14\];\[15\], dual-learning \[16\];\[17\] and back-translation \[18\]. Back-translation has been reasonably successful to tackle phrase-based translation \[19\], NMT \[18\];\[21\] as well as unsupervised MT \[22\]. Persistent upon sticking to the cross-lingual zero-shot transfer learning strategy for French text, and rightfully so since it allows us to explore new domains, we decided to apply back-translation to augment our training data. However, contrary to the more conventional method of back-translation as explained in \[22\], we did not have any parallel data (French) and hence translated English tweets to French and then translated them back to English again as a data-augmentation process.

## References

1)  [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
2)  [Release Strategies and the Social Impacts of Language Models](https://arxiv.org/pdf/1908.09203.pdf)
3)  [GLTR: Statistical Detection and Visualization of Generated Text](https://arxiv.org/pdf/1906.04043.pdf)
4)  [An Empirical Study on Pre-trained Embeddings and Language Models for Bot Detection](https://aclanthology.org/W19-4317.pdf)
5)  [On-the-fly Detection of Autogenerated Tweets](https://arxiv.org/pdf/1802.01197.pdf)
6)  [TweepFake: about detecting deepfake tweets](https://arxiv.org/pdf/2008.00036.pdf)
7)  [Automatic Detection of Machine Generated Text: A Critical Survey](https://arxiv.org/pdf/2011.01314.pdf)
8)  [Reverse Engineering Configurations of Neural Text Generation Models](https://arxiv.org/abs/2004.06201)
9)  [Defending Against Neural Fake News](https://proceedings.neurips.cc/paper/2019/hash/3e9f0fc9b2f89e043bc6233994dfcf76-Abstract.html)
10)  [Attention is All you need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
11)  [Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation](https://watermark.silverchair.com/tacl_a_00065.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAswwggLIBgkqhkiG9w0BBwagggK5MIICtQIBADCCAq4GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMr-oTpvxzYPliGt5NAgEQgIICf_AF-p0__9h-1cl46kNhCErpM1DEKgSdsv552Yq6GGomeWqubxJmi83sRCyn8f5qHTeMIU5F_DXqBNRE66lHA_wz9V_bMDkqQnMXNzl3IzZ2QyXuHUXr3vcmHqgr_ywzj5Br2u2BT-gCKjoWpuexfanuFxxzakzK8x_kfn_v3fcPi-coa75f8piOhP1Mzn4Lmq4gkrOT7MOrmri4EnT0jflCHOX68YrXXhA5RZYPSDBL2E4ueTlH9zIbNTI4YWRMTabRkVP36hzAHGHwhVAPMvMlrcqohoREX4ByRl16DKkWFFxXT12CXMyaJcL0u5I6UYLi5fmZSNeTSN4VlE59qEAqglhDvfpf715x_g8-yec_uR0okjZPjq5jmny4KtUjVUFdPmOq2sPZ4aRS0Iwap7eenV43_UsMJqHWf9FR7MjeHRCm5TTvViWGkKVhq1Mdj44dPZdTgP0_n9vkTW2CWGRqX7JId9cZITNCrRQ_H9OgfCHqzUvQeXTFWxmxMmxor3zZtfWtuCXS6HV8QDB4bfLBh6YMknIccKMfts2H_GZFc7f0boIJYJC70hMqkx1C0ojdxnbH-cImUz6Bicsh1zK-yeGjG6zWrMAVAht3LxdeviXufEQWd4NjiClOcNX7_mNu4nbLtwpyXSvIRPXuh2M1sKAniSKCYF_rZpIdGEcrYdWm1QIhaEONsZ6ISvECo1ie8sHhEDH94kUJ0cznmA2fwvVwycOK_zVSbmqvKPzmirD_K0IYy8zjA6B47le4mhqG3frIdKrGbgVBdMWBXnmCvuvFVIwug_Dc0GIxjccCNri77bvJ4tffZninAJ7zr6imqnaXvQFR941-xNkcxw)
12)  [Improving Massively Multilingual Neural Machine Translation and Zero-Shot Translation](https://arxiv.org/pdf/2004.11867.pdf)
13) [Beyond English-Centric Multilingual Machine Translation](https://arxiv.org/pdf/2010.11125.pdf)
14) [On using monolingual corpora in neural machine translation](https://arxiv.org/pdf/1503.03535.pdf)
15) [On integrating a language model into neural machine translation](https://www.sciencedirect.com/science/article/pii/S0885230816301395?casa_token=U3OMoT3BOcoAAAAA:dgFd0l5H4l7Q7_lym6eTYTP2ckpD1zBTPp54X7GQq0AbGbFemEfBaapBkE1bGM5PUYaOS8E4CQ)
16) [Semi-supervised Learning for Neural Machine Translation](https://link.springer.com/chapter/10.1007/978-981-32-9748-7_3)
17) [Dual Learning for Machine Translation](https://proceedings.neurips.cc/paper/2016/hash/5b69b9cb83065d403869739ae7f0995e-Abstract.html)
18) [Improving Neural Machine Translation Models with Monolingual Data](https://arxiv.org/pdf/1511.06709.pdf)
19) [Improving Translation Model by Monolingual Data∗](https://aclanthology.org/W11-2138.pdf)
20) [Investigating Backtranslation in Neural Machine Translation](https://arxiv.org/pdf/1804.06189.pdf)
21) [UNSUPERVISED MACHINE TRANSLATION
USING MONOLINGUAL CORPORA ONLY](https://arxiv.org/pdf/1711.00043.pdf)
 
 ### Additional references
-   [Hugging Face m2m100](https://huggingface.co/facebook/m2m100_418M)
-   [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)

# Datasets


In this project, we have used the [‘TweepFake - Twitter deep Fake text Dataset’](https://www.kaggle.com/datasets/mtesconi/twitter-deep-fake-text) from Kaggle. The dataset consists of human and deepfake tweets in English, the tweets were collected from 23 bots and 17 human accounts via the Twitter REST API.     

The whole dataset is 353 kB, in total, 25,836 tweets were collected, half of them were human-written, half of them were generated using one of the following generation techniques: GPT-2,  RNN, Torch RNN, Markov Chain and CharRN. The data is split into training, validation and test sets.     


| Split          | \# bot tweets | \# human tweets | total |
|----------------|---------------|-----------------|-------|
| Training set   | 10354         | 10358           | 20712 |
| Validation set | 1152          | 1150            | 2302  |
| Test set       | 1280          | 1278            | 2558  |

The data will be stored on Google Drive.

# Methods
We chose 4 models as baselines for detecting deepfake tweets. All the models are run on the computing infrastructure Google Colab and created by PyTorch framework.


| Model Name| Parameters| Epochs| Evaluations   |
|----------------|---------------|-----------------|-------|
| BERT | default config:  vocab_size = 30522, hidden_size = 768, num_hidden_layers = 12 , num_attention_heads = 12, intermediate_size = 3072, hidden_act = 'gelu', hidden_dropout_prob = 0.1, attention_probs_dropout_prob = 0.1, max_position_embeddings = 512,type_vocab_size = 2, initializer_range = 0.02, layer_norm_eps = 1e-12, pad_token_id = 0, position_embedding_type = 'absolute', use_cache = True, classifier_dropout = None    | 3 | evaluation: accuracy used by sklearn.metrics.accuracy_score,  f1 used by sklearn.metrics.f1_score  |
| RoBERta         | vocab_size = 30522, hidden_size = 768, num_hidden_layers = 12 , num_attention_heads = 12, intermediate_size = 3072, hidden_act = 'gelu', hidden_dropout_prob = 0.1, attention_probs_dropout_prob = 0.1, max_position_embeddings = 512,type_vocab_size = 2, initializer_range = 0.02, layer_norm_eps = 1e-12, pad_token_id = 0, position_embedding_type = 'absolute', use_cache = True, classifier_dropout = None, pad_token_id = 1, bos_token_id = 0, eos_token_id = 2  | 3   |  evaluation: accuracy used by sklearn.metrics.accuracy_score,  f1 used by sklearn.metrics.f1_score |                                                                         
| T5 | vocab_size = 32128, d_model = 512, d_kv = 64, d_ff = 2048, num_layers = 6, num_decoder_layers = None, num_heads = 8, relative_attention_num_buckets = 32, relative_attention_max_distance = 128, dropout_rate = 0.1, layer_norm_epsilon = 1e-06, initializer_factor = 1.0, feed_forward_proj = 'relu', is_encoder_decoder = True, use_cache = True, pad_token_id = 0, eos_token_id = 1                   | 3               | evaluation: F1 score |                                                                                             
| GRU             | embedding_size = 300, activation = Tanh, num_layers = 2; hidden_size = 512; learning rate = 0.0003; optimizer: Adam    | 25              | evluation: classification_report from sklearn.metrics which include precision, recall, and f1_score |                               




For the 4 basesline models, Roberta has the best performance so fine-tuned Roberta for testing on cross-lingual performance.


| Model            | Parameters                                         | Accuracy     | F1   |     
|----------------|---------------|-----------------|-------|
| roberta-base     | lr=4e-5 batch_size=8 early_stopping=False epochs=3 | 0.9077       | 0.9071 |
| xlm-roberta-base | lr=4e-5 batch_size=8 early_stopping=False epochs=3 | 0.8846       | 0.8822   |
| xlm-roberta-base | lr=1e-5 batch_size=16 early_stopping=True epochs=3 | 0.8968       | 0.8977   |
| xlm-roberta-base | lr=1e-5 batch_size=32 early_stopping=True epochs=3 | 0.8819       | 0.8821   |
| xlm-roberta-base | lr=2e-5 batch_size=16 early_stopping=True epochs=3 | 0.8999       | 0.8998   |
| xlm-roberta-base | lr=2e-5 batch_size=32 early_stopping=True epochs=3 | 0.9023       | 0.9030   |
| xlm-roberta-base | lr=3e-5 batch_size=16 early_stopping=True epochs=3 | 0.8987       | 0.8973   |
| xlm-roberta-base | lr=3e-5 batch_size=32 early_stopping=True epochs=3 | 0.9077       | 0.9077   |





# Engineering

For the baseline models, we plan to reproduce some of existing models from the paper and add one more model with T5 as embedding as the initial baseline models. From the original paper, all the methods (except Random Forest Bag of Words) perform very well in identifying the tweets as bot on both RNN and other accounts. However, for the human accounts, only fine tuning models such as RoBERTa performs good in identifying them. For the tweets generated by GPT-2 models, all the methods have difficulties but RNN based models perform relatively better. Therefore, we chose 4 models as baselines (BERT, RoBERTa, T5, and GRU) for this week and will pick the ones performing well for fine tuning in further steps.

-   BERT: First of all, we use a pretrained BERT model to provide word encoding, and apply non-deep learning classifiers including logistic regression, random forest and support vector machine. These models did not report good performance in the original paper and did not use deep learning other than the pretrained encodings. We would not focus on these models. And then we use a BERT language model with fine-tuning. The classification model is imported from `simpletransformers` module and trained for 3 epochs.

-   RoBERTa: We used Apex from NVIDIA to train with RoBERTa. Apex is a extension tool that could help with mixed precision and distributed training in Pytorch. After git cloning from the NVIDIA apex repository, We used setup.sh to write the file and then installed transformers simpletransformers==0.41. Later, We used GridSearchCSV, classification_report and PredifinedSplit from Sklearn library to define the DataHandler class that could read the files and fine-tuning the model with different features and parameters. I processed the data with pandas by dropping the irrelevant columns and create dictlabels and the dictlabelsReverse dictionaries to the target of out training. Later, I defined the roberta model by using ClassificationModel from simpletransformer library and use F1 metrics as evaluation for the model.

-   T5: The T5-base model is pre-trained on a multi-task mixture of unsupervised and supervised tasks with datasets like C4, CoLA, MNLI, CB, ReCorD and many more. To the existing codebase, we added the T5-base model to explore how well it can perform with the Tweepfake dataset. After all the required imports and installations, the model was fine-tuned with mostly default hyperparameter selection, an exception being `num_epochs` (set to 3 considering computational constraints). The task was set to `translation_en_to_en`, as this argument is irrelevant for the classification task. Furthermore, the corresponding files for train, val and test splits of the data were also selected, with `text_column` being set to the `text` column in the dataset and the `summary_column` (target-column) set to `label` column in the dataset ('account.type' originally). A `run_seq2seq.py` script was also used to fine-tune the model to undertake this classification task, which is a T5 implementation provided by HuggingFace.

-   GRU: To start with, I passed the white space tokenizer function to TorchText 's Fields to tokenize input string text to the list of tokens. For the tweet text, we convert the input to lowercase and sequential. The input label is not sequential and doesn't need the unknown token for OOV words. Then we use the TabularDataset class and Fields to process the .tsv files (train, dev, and test). We have built the vocabulary to map words to integers using .build_vocab() with argument min_freq=2 to only keep words that occur more than once in the data set. We constructed the iterator using BucketIterator, and the batch size for train, dev and test is 32. 
    Our model is a 2-layered GRU model with a word embedding size of 300, a hidden size of 512, and Tanh activation. We have used the Adam optimizer with a learning rate of 0.0003 and trained the model for 25 epochs. 

During the engineering process, we only face some problems for training the baseline models.

For BERT, In the provided notebook, Apex is used but the script `setup.sh` could not run in Colab but the training can be done without the tool. We also need to investigate whether it is possible to tune the hyperparameters of BERT fine-tuning models.

For RoBERTa, we tried to use the Roberta_adapter to train the dataset but encoutered caught keyerror in the training process. The error happens when fetching particular indexes from the training file. I'll investigate the error next week if we need to fine-tuning the hyper-parameters from the original models as the paper used the default hyper-parameters from the model.

For T5, one problem that I did face while fine-tuning the model was that I was constantly running out of memory when trying to train the model for more number of epochs (initially I tried `num_epochs=10`). In-fact, even with `num_epochs=3`, I could only complete around 2200-2300 of the 7767 total optimisation steps. Though I could try and reduce the batch-size, it was only 8 to begin with and reducing it further could lead to an increase in noise while training. The team plans to take it up with the teaching staff to understand and resolve the issue for future engineering associated with the T5-base model.

For GRU, the original paper used the Keras Python library to implement the bidirectional GRU model followed by a dropout layer. I have used the same model structure and hyperparameters to implement the GRU model with the PyTorch framework to adhere to the instructions. However, I was unable to reproduce exactly the same results, the test accuracy and F1 were only 0.74 from using the original hyperparameters from the paper. After some hyperparameter selection, I was able to get a test accuracy and F1 of 0.79, but it is still lower than the results obtained in the original paper (0.83).

After selecting Roberta as our best performing model, we utilized the current test-set we have (in English) to create test sets in other languages using the M2M100 model by Facebook. These translated test-sets can be used as to determine how accurate a multilingual model can be to detect these artificially generated texts. For this week, we create a test-set for the French language. We use the simple implementation of the M2M100 model available on the huggingface to translate the English test-set into French by using [EasyNMT](https://github.com/UKPLab/EasyNMT) wrapper around the model. We write the translated data into a csv file and save it as `translation.csv`.

One problem we encountered with M2M100 is that both its pretrained model as well as the source code is written in such a way that it needs many high-end GPUs to work. This environemnt is quite hard to simulate in the Google Colab notebook and our model simply used CPU for translating the text in the test-set which was taking around a minute to translate a single example. To circumvent this challenge, we made use of the EasyNMT wrapper and as a result were able to translate all of our test data (2500 examples) in around 30 minutes.

The best model suggested by the original paper and verified by us is a classifier fine-tuned on `roberta-base`. The parameters are shown above and the model achieved an accuracy of 0.9077 and an F1 score of 0.9071. To test a roberta model on French data, we switched the base model to a multiple-lingual model `xlm-roberta-base`. The multi-lingual model performs worse on the English task, dropping the accuracy to 0.8788.

We performed hyperparameter tuning on the multilingual model by doing grid search on the learning rate and batch size as suggested by the RoBERTa paper (Liu et al., 2019) for `roberta-base`. We also enabled early stopping. We attempted learning rates of {1e-5, 2e-5, 3e-5} and batch sizes of {16, 32}. The accuracy and f1 scores are shown above.

After hyperparemeter tuning, we take the best performing multilingual model and test its performance translated French data with `M2M_100`. We treated the translated French data in two ways: as all bot created data as `M2M_100` is a neural model, and with original labels. We also trained another model with the original training set, and the training set that was first translated to French then back to English with `M2M_100`. The reasoning is that `M2M_100` is an encoder-decoder model and adding more data from backtranslation could improve cross-lingual performance.


# Results:

For the baselines models, these are the results:

-   BERT: the BERT language model with fine-tuning achieves an f1 score of 0.893 and accuracy of 0.893, which are consistent with the original paper. The model also shows the highest error rates with GPT2 based bots as suggested by the original paper.

-   RoBERTa: Overall, the model has about 0.9077 accuracy which is consistent with 0.904 from the original paper. The evaluation loss is 0.444 and it has F1 score about 0.907. However, for different accounts, the model has very different error_ratio. For example, it has very low error ratio for accounts like Justin Truedeau or Donald Trump, which indicate that it is very accurate identifying the tweets from real humans. However, it performs poorly on gpt2 accounts. For instance, the error rate for one GPT2 account Gpt2Wint is about 50% and 30% for nsp_gpt2. It might illustrate that RoBERTa is not very good at the tweets generated from GPT2.

-   T5: For the test set, the model achieved an F1 micro score of 0.8835 and a F1 macro score of 0.8834. An observable trend was its extreme accuracy with the tweets originating from human accounts but not from the artifically created tweets by the bots. To delve into the specifics, while it performed relatively well on tweets originating from artifical bots (other than based on GPT-2), it performed the worst on the GPT-2 generated tweets. However, it is interesting to note that it performed much better than the BERT model originally implemented in the paper, though a little worse than the RoBERTa model as specified in the paper and re-implemented by one of the team-members.

-   T5: For the test set, the model achieved an F1 micro score of 0.8835 and a F1 macro score of 0.8834. An observable trend was its extreme accuracy with the tweets originating from human accounts but not from the artifically created tweets by the bots. To delve into the specifics, while it performed relatively well on tweets originating from artifical bots (other than based on GPT-2), it performed the worst on the GPT-2 generated tweets. However, it is interesting to note that it performed only a little worse than the RoBERTa (0.907) model as well as the BERT (0.893) model as re-implemented by two of the team-members.

-   GRU: The model achieved an accuracy and f1 score of 0.79 on the test set. The trend of error rate is consistent with the original paper for accounts such as AINarendraModi and JustinTrudeau. However, for the GPT-2 accounts such as dril_gpt2 and GenePark_GPT2, the error rate is significantly higher compared to the original finding. On average, the model performs better with human accounts.

As we could see, Roberta has the best performance for classifying the text produced by human and bots and then we did fine-tuning on Robert to further improve the performance. The parameters are shown above and the model achieved an accuracy of 0.9077 and an F1 score of 0.9071. To test a roberta model on French data, we switched the base model to a multiple-lingual model `xlm-roberta-base`. The multi-lingual model performs worse on the English task, dropping the accuracy to 0.8788.The best performance (accuracy=0.9077 F1=0.9077) was achieved with a learning rate 3e-5 and batch size of 32.

| Model|Translated Human tweets in French test set labeled all as bot|Translated French test set with original labels|   
|----------------|---------------|-----------------|
|original train set|accuracy=0.2019 f1=0.3359|accuracy=0.8526 f1=0.8603|
|data augmentation|accuracy=0.0665 f1=0.1247|accuracy=0.7619 f1=0.7129|

Later, We then evaluate the best model on a French test set of 2558 tweets translated from the English test set with `M2M_100`. Since all texts were translated by a neural model, we marked all tweets from human as bot after translation and only tested our model on the test set. Our best model only achieved an accuracy of 0.2019 as shown above. The data augmented model achieves an even lower score of 0.0665.

When using the original labels (keeping the bot/human label from the original English test set) and testing on the entire French test set, surprisingly the model trained with the original train set performs similarly on the translated French test set compared to the original English test set, acheiving an accuracy of 0.8526 while the model augmented with back translation performs worse.

# Conclusion 

\[summary\]

Even though overall we have high accuracy for detecting the text generated by bots and humans, we have a hard time detecting the text generated by GPT-2 specifically. The accuracy is slightly above the chance. This is probably partly because GPT-2 is a general-purpose learner that could be trained to do many tasks and has the ability to perform tee extension of synthesizing the next item in arbitrary sequence. Another reason is that the architecture implements a transformer model that uses attention and thus allows the model to selectively focus on segments of input text that predicts to be the most relevant, it as increased parallelization and outperforms many other models. Since our baselines models except GRU are all transformer related, it might have the bias so it has hard time to detect it.

Another reason is that we are detecting the texts from Twitter data. The models we used so far are not trained on such data so it has hard time to detect those text from Twitter. Moreover, GPT-2's ability to generate plausible passages of natural language text are limited by the length of the text. In details, if thee text is long enough like a couple of paragraphs, we could easily detect the text generated by GPT-2 because it is prone to logical errors and etc. However, since tweet is usually short, it could be one of the reasons for the poor performance.

A future direction for us to investigate if we have more time and resources is that we could train some some models on the tweets such as TweetBert which has language representation models to extract information from Twitter and has the text analysis specific to social media domain. It could be possible that TweetBert would be better detecting the tweets from bots to humans. Another thing we could try is to improve the accuracy of GRU because it is a RNN based model rather than transformer related, so, it has less bias to detect the tweets generated by GPT-2. However, we do not have good accuracy for GRU. If we have more time and resources, we could also train a TweetGRU and improve the accuracy for RNN-based models. 

In our cross-lingual experiments, we found that the multi-lingual roBERTa model does not recognize human tweets passed through a translation model as bot generated texts. Instead, the model could still distinguish between human and bot generated tweets even after translation with an encoder-decoder model. Part of the reason could be that `M2M_100` did not translate some of the texts due to unrecognized syntactical structure. One limitation of our study is that we do not have access to a dataset of real human vs bot tweets in French and cannot effectively evaluate the cross-lingual performance of our models. A Tweepfake dataset collected from French tweets could help further study the cross-lingual transfer learning of bot detection. Our study also reveals that data augmentation with back translation does not improve cross-lingual bot dectection.

