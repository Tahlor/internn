# IDIOMS

* Have dictionary of idioms with example sentences
* Replace every idiom with ever other idiom to get an initial score
* Then add a 1 word buffer
* Get another score
* Compute the best score

* you'd probably have to normalize it somehow to control for common vs. less common idioms.
* It also might be tricky to control for variable length--like maybe you replace "top gun" with "out of the frying pan and into the fire", maybe it gets a high score since the second half of that phrase is very likely given the first half
* also control for non-idiomatic usage, i.e., is it a literal "top dog"?
* I mean maybe you can use the grammarly API and change a word or two in the original sentence to make it work better, IDK

## Need to look up papers of how people build thesauruses -- how was wordnet built?
* Examples: show the ropes, humble pie, smell a rat, kick the bucket
* The Xer the Yer
* Can't do X, let alone Y

http://wordnetweb.princeton.edu/perl/webwn?s=smell+a+rat&sub=Search+WordNet&o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=&o3=&o4=&h=
