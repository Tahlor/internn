Title
Towards Inter— Neural Network Communication and Resolution

Abstract
Visual handwriting recognition systems are often supplemented with a language model to further improve predictions. Traditionally, the visual system outputs some probability for each possible token conditioned on an image, which is then multiplied by some prior probability of that token conditioned on the context and chosen language model, and the token with the highest joint probability is selected. We present a method for separately trained visual and linguistic neural networks to converse with each other, which enables the system to provide predictions that involve higher-order interactions between the visual and linguistic component models.

# Plan
    * Create corrupted digits
      * Use fonts
      * Occlusion, gaussian noise, warping, etc.
    * Get a CNN
    * Get a language model
      * Find one with character tokens
      * Think of how to do this with words?
    * Consideration:
      * Where was the language model trained?

* Experiment 1:
    * Just backprop through the language model to the CNN so the CNN learns
    * Control: Train a CNN, then pass predictions to Kaldi LM

* VilBERT


# Analyses / Research
    * Comparison of visual vs. linguisitc embeddings
    * how/why

* Experiment X: (more thinking)
    * [Design an RNN cell that keeps going until it settles on something]
        * Like an RNN that makes an external call to the language model
    * GLOM
    *


** DEPRECATED EXPERIMENTS

* Experiment 2:
    * Deconvolve language model outputs and feed back into the CNN
    * Loop
    * Backprop at each language model output prediction
        * Include a variable for "certainty"
            * measure the similarity between visual portion outputs and the language model ones
                * E.g., each produces vector, measure cosine similarity
            * estimate the "energy" of the outputs?

* Experiment 3:
    * Use an RNN after the CNN
        * i.e. visual letters probably cluster differently than language model ones
        * may need some intermediate structure to go between visual clusters / lexical ones, maybe ideally some joint space is discovered
    * Pass RNN outputs to the language model
        * pass language model outputs back to the RNN
    * Is there a way to do this with CTC?

* Experiment 4:
    * Concatenate what comes out of the language model
        * E.g. you deconvolve what comes out of the language model and add it as a channel to the CNN
        * You concatenate the language model output to the RNN input
