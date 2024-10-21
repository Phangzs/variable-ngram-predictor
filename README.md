# variable-ngram-predictor

A next-word predictor based on variable n-gram contexts.

# Data
I used [AllenAI's c4 dataset on HuggingFace](https://huggingface.co/datasets/allenai/c4/tree/main) to train the model. The code supports multiple files, but on a laptop 1 json file takes enough time to train.

Specifically, here is the download link to the file I used to train the model: <br /> [https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz)

# Methods
This algorithm predicts the next word based on the previous occurances on what the next word is based on the past n words (an n-gram).

By creating a matrix with each row corersponding to a word, we can find and count every occurence of the next word by increasing the column corresponding to the word to 1. That is, for every word $$w_n$$ in our training set, we increase the count of the next word $$w_{n+1}$$ in our word matrix $$M_{x_n,x_{n+1}}$$ by 1, where $$x_n$$ represents the column/row number corresponding to $$w_n$$. More formally, if we let $$(a_n)$$ be the sequence of words in our training set, $$M$$ be the $$0$$ matrix, and $$n \leq{N}$$), we have the algorithm defined by $$\forall n < N \quad M_{x_n,x_{n+1}} \texttt{+= 1}$$ to produce our $$V\times{V}$$ (where $$V$$ is the size of our vocubulary or unique words) matrix $$M$$ with each $$M_{i}{j}$$ representing the number of times the word corresponding to column $$j$$ appeared after row $$i.$$ 

This is a uni-gram matrix. We then increase the context length by $$1$$ until hitting the maximum context as set by the variable $$\texttt{max}$$ _ $$\texttt{n}.$$ The nth-gram matrix simply contains an array of words as the rows, while still retaining a single word as the column. In this implementation a 3D matrix is used, stacking all nth-gram matrices on top of eachother.

Next, the code finds the _best_ n-gram in order to predict the next single word. This is done by computing the entropy of the probability distribution of an n-gram to find the uncertainty in its probability distribution (and therefore its predictions), followed by using a scoring algorithm to favor more predictable contexts with strong candidates for the next word.

<br /> <br />
$$\textbf{entropy}=-\sum{p\times \log_2{p}}$$ <br /> <br />
$$\textbf{score}=\frac{\log{(\text{total count}+1)}\times{\text{max probability}}}{\text{entropy}+10^{-6}}$$ <br /> <br />

After the max score for the context for each n-gram model is found and the associated probabilities per word are captured, we raise every probability to the power of $$\alpha > 1,$$ a variability restrictor that punishes lower probabilities the higher $\alpha$ is. We then finally normalize the probabilities in order to create a distribution from which to randomly pick a word from.

# Time-Complexity
Suppose we have the following: <br /><br /> $$T=\text{Total lines of text}$$ <br /> $$n=\text{Maximum n-gram order},$$ <br /> <br />Then we have a time complexity of $$\bf O(Tn^2).$$

# Space-Complexity
Suppose we have the following: <br /><br /> $$V=\text{Vocabulary size}$$ <br /> $$n=\text{Maximum n-gram order},$$ <br /> <br />Then we have a space complexity of $$\bf O(V^{n}).$$

# Performance
After training the model of 200,000 lines of text with a max n-gram of 5, it produces the following outputs:

`Given the context 'how is the best', the next 20 words predicted are:
possible outcome of any event is the sound sytem that makes the moment live and love in the book she`

`Given the context 'let a sequence of real numbers', the next 20 words predicted are:
for which there is no other custom writing service as flexible and convenient as this one us and uk writers`

`Given the context 'it will only take', the next 20 words predicted are:
a girl out for drinks because it allows them to you are you to implement the best solution to your`

`Given the context 'my water bottle is', the next 20 words predicted are:
designed to be the weakest link in chapter two 20040924 the qt widget gallery has been added to the with`

`Given the context 'the ipad is', the next 20 words predicted are:
a tremendous tool but the fear of stating who can or should be a flat out no absolutely not i`

`Given the context 'suppose that i said that', the next 20 words predicted are:
we dont really specifically celebrate the day anymore valentines day has really turned into a day to simply celebrate our`
