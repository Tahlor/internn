#/bin/bash

local_root="/media/data/GitHub/internn/lm/kaldi"
fsl_root="/home/taylor/shares/SuperComputerRoot/lustre/scratch/grp/fslg_hwr/other_repos/lm_adaptation/handwriting-net5"
#rsync -rutvm --update $fsl_root $local_root  --info=progress --include="*/" --include "*.py" --exclude 'archive*' --exclude="*"
rsync -rutv --update $fsl_root $local_root  --info=progress --exclude 'archive*'

# rsync -vurlt --size-only  '/home/pi/Downloads/external/' '/mnt/theserve/Media/Library/Audio Books/PodcastServer' --info=progress
# v: verbose
# u: update - only update when size/time is mismatched/newer
# r: recursive
# l: copy symlinks
# m: don't copy empty folders
# t: preserve modification times
# --size-only - only use size
# --modify-window=3660 - modified times can be off this much; TIMEZONE ISSUES SOMEHOW!!
