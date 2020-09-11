# Part-of-Speech tagger

This project builds a [part-of-speech tagger](https://en.wikipedia.org/wiki/Part-of-speech_tagging) built using the [Conditional Random Fields](https://en.wikipedia.org/wiki/Conditional_random_field) probabilistic method. The projects contains all files you need to train, evaluate and make individual predictions on new observations.

## How to run
1. (Create new virtual env using `requirements.txt` )
2.  Train the model by running:

```bash
python3 train.py "path_to_train_file.conllu" "path_to_test_file.conllu"
```
(on my machine this should take around 40s. You will get some visual feedback when it's done)
3. Evaluate the model by running:

```bash
python3 eval.py "path_to_test_file.conllu "
```
(_note_: you must first train the model, so a pickled model file is generated in the directory)

4. Generate new predictions by running:
```bash
python3 generate.py "text_file.txt"
```
_Note_: I assume that the `test`, `dev` and `train` files would be in Conllu format, as provided by the [Universal Dependencies Project](https://github.com/UniversalDependencies/UD_English-GUM/tree/master), while the simple text file to generate new predictions a text file with one sentence per line, _already tokenized_. For example:

> I love food
>
> Eating healthy food is important for your body
>
> the word " blue " has 4 letters
>
> Once in a blue moon
>
> I 'm a lone wolf .


## Discussion
### Assumptions
In building this tagger, I had to make a set of assumptions that would allow me to choose the model, architecture, testing strategy, etc.
- First of all, I had to make an assumption about the use case for which we're building the POS tagger. I assumed that we would be training a custom tagger to generate tags for new, unseen text, to then use potentially as features for another NLP task, such as topic classification.
- Given that, I had to also make some assumptions about the future requirements of this model. For example,
  - the texts we would be dealing with in the future could be in different languages
  - there exists domain-specific vocabulary that we will need to capture in the future (e.g. the name of a specific tool)
  - full-sentence accuracy is not important in the task (i.e. % of all sentences that are 100% correctly tagged)
  - we will want to make predictions on-the-go through streaming
  - algorithm training is offline, because new tagged sentences are hard/expensive to get, but we might want to retrain a model given new features, or train a model on a new language
  - We care about accuracy for all tags equally, i.e. the correct prediction rate for `NOUN` is equally important as for the tag `AUX`.
- Techical assumptions:
  - Data formats for the evaluation are as described in section `How to run` ☝️
  - In production, the data would be coming in batches for both training and testing (in reality, this doesn't need to be true for training since we only want to train once, but it helps to make the implementation scalable for both parts)
  - Incremental learning is not a hard requirement for the project.
  - We want the model to be stored and to be able to reload.

### Model choice
For this problem, I chose the probabilistic Conditional Random Fields, which is a sequence modelling algorithm, which excels in tasks of pattern recognition. Generally speaking, Conditional Random Fields (CRFs) are a generalisation of log-linear models. The main difference between log-linear models and CRFs is that the output set Y of CRFs is generalised to encompass structured data, such as sequences (which a sentence is). I chose this model after research on POS taggers, for the following reasons:

#### Pros
- Achieves state-of-the-art ([compare here](https://aclweb.org/aclwiki/POS_Tagging_(State_of_the_art))) accuracy results without consuming resources or needing a lot of tuning.
- Picks up contextual relationships within the sentence, such as previous/next word, position in sentence, etc, which makes the algorithm superior to let's say a bag-of-words approach.
- By using weights of features, CRF allows us to construct an arbitrary number of features for each word, which can be for example _is this word capitalized?_ , but can also allow for special rule-based features (perhaps if the new vocabulary is highly domain-specific or we discover a new pattern in the data we see)
- By extracting not only the word + its sequence, but also additional information about the sequence and its elements, CRF also performs well in POS tagging for morphologically rich languages, such as German, Russian (see example performance for Amharic: [Adafre, Sisay. (2005). Part of speech tagging for Amharic using conditional random fields](https://www.researchgate.net/publication/271453070_Part_of_speech_tagging_for_Amharic_using_conditional_random_fields))
- Interpretability: As the model learns weight of each feature in addition to state transition probabilities, it is possible to understand how each prediction is made (especially by using a library like `ELI5`)
- CRF allows for incremental learning as it uses stochastic gradient descent for parameter estimation, albeit that has some negatives (discussed below).
- Since the algorithm normalizes over all Ys, it avoids a local bias problem (read more about it here: [Batista, David, Conditional Random Fields for Sequence Prediction](http://www.davidsbatista.net/blog/2017/11/13/Conditional_Random_Fields/))
- It does not assume independence of the observations, as other similar models do (e.g. Hidden Markov Model)

#### Cons
- Although the model allows for incremental learning (adding one or a mini-batch of observations at a time), this makes obtaining a cross-validation accuracy score quite tricky. Also, P(Y|x) includes a normalization constant Z, which is computed over all possible state sequences. So, normalization (to a value between 0-1, as we are interested in probabilities) is done automatically as part of the algorithm, however at the expense of efficiency.
- There are very few implementations of the CRF model in Python, so the one I chose to use (`sklearn-crfsuite`) is one of the few that has a `scikit-learn` interface.

### Testing strategy
My testing strategy is the following, acknowledging the relevant trade-offs:
- Per the literature, the prediction task aims to optimize the token-wise accuracy across _all_ tokens, i.e. the % of all tokens in a holdout set (unseen data) that was correctly classified to its appropriate tag. Since we are dealing with 17 tags/classes, we use the overall accuracy rate as the most important metric, as well as sentence-wise accuracy (what % of all sentences were predicted 100% correctly?) and F1 score (harmonic mean of precision and recall) as monitoring metrics.
- There are generally speaking two methods for model evaluation: holdout set and cross-validation. My strategy for the implementation is to use the `eval.py` script to evaluate the model on unseen, tagged test data as a "holdout set", whereas I use all the data provided for the `train.py` model to train the model. That means that while training the model, all data in both the `train` and `dev` files is consumed into training, and none is set aside for evaluation. The reasons are the following:
  - The dataset is relatively small, so we want to maximize the amount of data/sentences we use to train the model.
  - The step of training needs NOT necessarily provide an evaluation of the model. Since the model does not attempt to minimize the dev-set errors, it is of no particular use to sacrifice this data for evaluation.
  - I chose the parameters that go into the model via hyperparameter optimization and 5-fold cross-validation ([`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)), which took over 20 minutes to run, providing a mean cross-validation accuracy score of around 93%. If we were to use the `train.py` script to train a new model on new data or new languages, I would recommend to implement again a `RandomizedSearchCV` or `GridSeachCV` for hyperparameter optimization, however for the sake of this project, I decided to not include cross-validation in training the model.
  - It could be theoretically possible to choose the hyperparameters statically (as is done now), yet to evaluate the model on a cross-validated accuracy score. However, this would retraining the algorithm on each test-train-split segment, which would require time (patience), as well as the entire data set. Since I decided to feed all the data as a stream through a generator, it is not possible in the current implementation to obtain a cross-validation score over multiple folds.
  - It is also important to see how the model performs against a **naive baseline**.
    - While exploring the solution (not included in the project), I constructed a naive baseline by mapping each word in the training set to its most frequently appearing tag (e.g. if `Apple -> NOUN` in 60% of occurences of the word in the trainingd ata, then we map `{'Apple': 'NOUN'}`). Any previously unseen words are mapped to `NOUN`. This naive strategy already achieves **>81% accuracy** in an unseen dataset. It is important to keep this baseline in mind when evaluating more complex models.
  - Finally, it should be noted that the topic of standard splits is quite important and often overlooked in NLP, as discussed in [Gorman, Bedrick (2019). We need to talk about standard splits](https://www.aclweb.org/anthology/P19-1267/). The paper demonstrates that the majority of POS tagger state-of-the art results published between 2000 and 2018 are all based on the same standard split of popular corpora, and when the researchers tried to validate the score on a different split, they were unable to. If I had more time, and we were not interested in streaming data into testing, I would definitely implement cross-validation to both tune and evaluate the performance of the model.

>Overall, the model I trained on the UD corpus achieves an average accuracy score of 93% in cross-validation, and 94% with the standard split provided. I am certainly confident that it performs better than the naive baseline seen above (81%), but I would be more confident if cross-validation is built into the training phase itself.

### Trade-offs

Some of the trade-offs I had to make:
- If data is to be consumed in mini-batches for both training/eval and prediction generation, it is quite tricky(or even impossible) to implement cross-validation (the generator is exhausted). I chose to go with batches to account for the scenario of "live" prediction, as well as to minimize memory load in the training phase (the feature set is quite large)
- If I had more time I would look into vectorizing or hashing the feature set (dictionary)
- Cross validation and Hyperparameter optimization with `RandomizedSearchCV` takes a long time (over 20' on my machine) so I decided to not implement it in this project for simplicity's sake. However, if we were truly interested in the prediction accuracy of the model (and wanted to train it on a different corpus or language), I would certainly build those two into the training pipeline.
- In choosing the model, I had to make a choice between capacity for out-of-core training (as well as ease of implementation) and performance in sequence modelling tasks specifically. If we know that the model needs to be retrained online as new observations come in, I would choose a model that implements incremental learning ([`.partial_fit()` method in scikit-learn](https://scikit-learn.org/stable/modules/computing.html)). However, none of those models are particularly good at solving Sequence tasks, at least out-of-the-box. Assuming that tagged POS data is hard to get and the need for out-of-core retraining of the model is unlikely, I decided to go with an algorithm that has proven good at the particular task, to the detriment of potential out-of-core learning.

### Time spent
I spent around 15 hours on this project, approximately half of it on research on the problem and half coding.
