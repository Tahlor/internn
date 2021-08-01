# internn


## Environment
Install anaconda. From the root folder, run:

`conda env create -f environment.yaml --name internn`

(or if you have a specific path)
`conda env create -f environment.yaml --prefix ../env/internn`

::cross fingers::

To activate, run: 
`conda activate internn`

If using PyCharm, update your interpreter with the path to this environment (which you can get by running `which python` after activating it): `Ctrl+Alt+S` (settings) then `Project -> Python Interpretter`.



Ben and I integrated the BERT model I set up into the main file on the git repo on the branch emb_loader. 
* The loss is STILL doing that dumb thing where it increases right off the bat and then decreases again so we need to figure out what’s going on there.
* That being said the data is correctly being passed from the image model to the language model without breaking. I suspect there’s something going on within the model it’s self but can’t figure out what.

* I also worked on finishing up the base model we talked about that predicts on the character level from a single string of text instead of through embeddings. 
* I just realized I forgot to upload it to github so let me do that now. It’s loss is going down over time (which is odd because it’s almost the same model as our embedding one) but the accuracy from masking is super low. This may be because it needs to train longer or because of bigger issues.
