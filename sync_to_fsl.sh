#/bin/bash

local_root="/media/data/GitHub/internn/"
fsl_root="/home/taylor/shares/SuperComputerRoot/lustre/scratch/grp/fslg_internn/internn"
rsync -rutv --update $local_root/data $fsl_root  --info=progress --exclude 'archive*'

# rsync -vurlt --size-only  '/home/pi/Downloads/external/' '/mnt/theserve/Media/Library/Audio Books/PodcastServer' --info=progress
# v: verbose
# u: update - only update when size/time is mismatched/newer
# r: recursive
# l: copy symlinks
# t: preserve modification times
# --size-only - only use size
# --modify-window=3660 - modified times can be off this much; TIMEZONE ISSUES SOMEHOW!!
