# Language Models

- [Language Models](#language-models)
  - [Representing Language in Structured Form](#representing-language-in-structured-form)
    - [Bag of Words (BoW)](#bag-of-words-bow)
    - [Dense Vector Embeddings word2vec](#dense-vector-embeddings-word2vec)
    - [Types of Embeddings](#types-of-embeddings)
    - [Encoding and Decoding Context with Attention](#encoding-and-decoding-context-with-attention)
    - [Attention is all you need](#attention-is-all-you-need)


Language models or large language models (LLMs) are a type of artificial intelligence model designed to understand and generate human language.

- **Representation model**s: LLMs that do not generate text but are used for task specific use cases, like classification, clustering, etc.
- **Generative model**s: LLMs that can generate text, such as ChatGPT, Bard, etc.

## Representing Language in Structured Form

Language is unstructured data and difficult to process for computers. Therefore, we need to represent language in a structured form. There are several ways to represent language in a structured form

### Bag of Words (BoW)
- A simple representation model of text data.
- Sentences are split by whitespace.
- A vocabulary is created from the unique words in the text.
- Each sentence is represented as a vector of word counts.
- Example:
  - Sentence: "that is a cute dog my cat is cute"
  - Vocabulary: ["that", "is", "a", "cute", "dog", "my", "cat"]
  - Vector: [1, 2, 1, 2, 1, 1, 1]

![alt text](images/llm/bag-of-words.png)

bag of words is a simple representation model but has several limitations:
- It does not consider the order of words.
- it does not consider the semantic nature of words.

### Dense Vector Embeddings word2vec
- A more advanced representation model of text data.
- Word2vec learns semantic representations of words by training on vast amounts of text data. E.g. Wikipedia
- It uses a neural network to learn the relationships between words in a corpus.
- word2vec generates word embeddings by looking at which other words appear next to in a sentence and learns the relationship between words.

![alt text](images/llm/neural-network-word-embedding.png)

For instance, the word “baby” might score high on the properties “newborn” and “human” while the word “apple” scores low on these properties.

![alt text](images/llm/values-of-embeddings.png)

In practice, these properties are often obscure and do not relate to single entity or humanly identifiable concepts.

Embeddings are fixed-length vectors that represent words in a continuous vector space. The distance between two word embeddings indicates the semantic similarity between the words. For example, the word embeddings for "king" and "queen" are closer together than the word embeddings for "king" and "carrot".

Embeddings of words that are similar will be close to each other in the vector space:

![alt text](images/llm/embeddings-vectore-space.png)

### Types of Embeddings
- **Word Embeddings**: Represent individual words in a continuous vector space. Examples include Word2Vec, GloVe, and FastText.
- **Sentence Embeddings**: Represent entire sentences or phrases in a continuous vector space.
- **Document Embeddings**: Represent entire documents in a continuous vector space. e.g. Bag of Words
- **Token Embeddings**: Represent individual tokens (words, subwords, or characters) in a continuous vector space. e.g. BERT, GPT-2, and RoBERTa

### Encoding and Decoding Context with Attention

Word2Vec creates static representations of words, meaning that the same word will always have the same representation, regardless of its context. This is a limitation because the meaning of a word can change depending on its context. E.g. bank can refer to a financial institution or the side of a river.

To address this and generate embeddings that are contextually aware, wei can use recurrent neural networks (RNNs). RNNs can process sequences of data. They are similar to traditional neural networks but have a feedback loop that allows them to maintain state. 

The RNNs are used for two tasks:

- **Encoding**: The RNN processes the input sequence and generates a fixed-length vector representation of the entire sequence. This vector captures the context of the entire sequence and is the input for the decoder.
- **Decoding**: The RNN generates the output sequence based on the encoded vector. This is typically done using a separate RNN that takes the encoded vector as input and generates the output sequence one token at a time.

![alt text](images/llm/encoder-decoder.png)

**Attention**: Attention mechanisms allow the model to focus on specific parts of the input sequence when generating the output sequence. 

E.g. when generating a translation, the model can focus on the relevant words in the input sentence rather than treating all words equally. 

![alt text](images/llm/attention.png)

By adding the attention mechanism to the encoder-decoder architecture, we can create a more powerful model that can generate contextually aware embeddings. 

E.g. durning the generation of "Ik hou van lamas's" the RNN keeps track of the words it mostly attends to perform the translation. After generating the words “Ik,” “hou,” and “van,” the attention mechanism of the
decoder enables it to focus on the word “llamas” before it generates the Dutch translation

![alt text](images/llm/encoder-decoder-with-attention.png)

This architecture is autoregressive. When generating the next
word, this architecture needs to consume all previously generated words.

The sequential nature of RNNs makes them difficult to train, especially for long sequences. 

### Attention is all you need

Attention is all you need is a well-known paper where the authors proposed a new architecture called the Transformer. Compare to RNNs, the Transformer architecture is more efficient an can be trained in parallel. The Transformer architecture is based on the attention mechanism and does not use RNNs.

**Transformer encoder**

The encoder consists of a stack of identical layers. Each layer has two sub-layers:
1. A self-attention mechanism
2. A feed-forward neural network

![alt text](images/llm/transformer-encoder.png)

**Transformer decoder**

The decoder also consists of a stack of identical layers. Each layer has three sub-layers:
1. A masked self-attention mechanism
2. Encoder attention mechanism that attends to the encoder's output
3. A feed-forward neural network

![alt text](images/llm/transformer-decoder.png)

- Encoder-decoder together builds the transformer architecture.
- BERT is a transformer-based model that only uses the encoder part of the transformer architecture.
- Decoder-only models (generative models) like GPT-2 and GPT-3 use the decoder part of the transformer architecture

