## Goals - TODAY
* train CNN alone
    # Almost there!
* train CNN with NN language model
    * fine tune language model
* test accuracy giving the GT letters to the language model directly

## Later
* warp / noise inputs
* deconvolve

## VILBERT

#### BIG PICTURE
    # we are not simply uniting language with images
    # we are finding a way to make limited capacity networks work harder about harder problems
    # one way to do this is to have different expert modules communicate until they come to a consensus
        # not simply voting on it
    ### NEED TO SHOW YOU'RE MORE PARAMETER EFFICIENT!
        # Conversing is more efficient than just a million shared attention layers

"Hard" problems should take longer to solve (rather than "hard" simply referring to the problems the model gets incorrect).

The usual paradigm is to treat all inputs exactly the same; maybe the idea is we just focus on building the biggest model to get the hardest examples correct, and even though the easier ones could have been solved with less computation, it's not a huge computational cost to us at inference. Still, it would be nice if we could re-use weights to "think harder" about some problems than expand the network indefinitely (humans seem to do this, but maybe you can view it as an inefficient/stochastic tree search).