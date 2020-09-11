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


## Assumptions