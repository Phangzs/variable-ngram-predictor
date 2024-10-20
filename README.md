# variable-ngram-predictor

A next-word predictor based on variable n-gram contexts.

# Data
I used [AllenAI's c4 dataset on HuggingFace]([url](https://huggingface.co/datasets/allenai/c4/tree/main)) to train the model. The code supports multiple files, but on a laptop 1 json file takes enough time to train.

Specifically, here is the download link to the file I used to train the model: [https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz]([url](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz))

# Methods
Instead of using unigrams or bigrams, the code finds the _best_ n-gram in order to predict the next single word. This is done by computing the entropy of the probability distribution of an n-gram and using a scoring algorithm to favor more predictable contexts with strong candidates for the next word.

<br /> <br />
$$\textbf{entropy}=-\sum{p\times \log_2{p}}$$ <br /> <br />
$$\textbf{score}=\frac{\log{(\text{total count}+1)}\times{\text{max probability}}}{\text{entropy}+10^{-6}}$$

# Time-Complexity
Suppose we have the following: <br /><br /> $$T=\text{Total lines of text}$$ <br /> $$n=\text{Maximum n-gram order},$$ <br /> <br />Then we have a time complexity of $$\bf O(Tn^2).$$

# Space-Complexity
Suppose we have the following: <br /><br /> $$V=\text{Vocabulary size}$$ <br /> $$n=\text{Maximum n-gram order},$$ <br /> <br />Then we have a space complexity of $$\bf O(V^{n}).$$

# Performance
Based on training the model on 200,000 lines of text, here are some of it's outputs.

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
