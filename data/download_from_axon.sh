ROOT="$( cd -- "$(dirname $(dirname "$0"))" >/dev/null 2>&1 ; pwd -P )"

wget axon.cs.byu.edu/media/data.zip -O $ROOT/data.zip --show-progress

# use unzip -o to overwrite without prompting
unzip $ROOT/data.zip -d $ROOT/
