Representation Learning of Very Short Texts
===========
Code base for representation learning of very short texts, such as tweets.
By Cedric De Boom, IBCN, Ghent University, Belgium.

Most of the usable code is in the `Embeddings/vectors` directory. There is a lot of old stuff in the base directory.

The `NN_train_x.py` scripts are used to train word embedding weights. There is always an explanation in comments about what the purpose of each script is immediately after loading the necessary modules. In the `run()` method the word2vec model to be used can be specified, along with the input text couples. Each line in a text file should be formatted as follows: `"text_1;text_2\n"`. The word2vec model is a file that points to a model trained with the `w2v.py` script. `NN_layers.py` contains most of the implemented logic and Theano code, and is used in the `NN_train_x.py` scripts.

In `similarity_plots.py` and `metrics.py` there is a bunch of code that can be used to evaluate trained models and baselines (see commented code in the main method of `similarity_plots.py` for some examples).

Please send me a message if you have any questions regarding the code!