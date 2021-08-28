#!/bin/bash

map_galois
#rsync -av ~/shares/galois/media/data/GitHub/internn/data/
rsync -vurlt --size-only  '/home/taylor/shares/galois/media/data/GitHub/internn/data/' '/home/taylor/github/internn/data' --info=progress
