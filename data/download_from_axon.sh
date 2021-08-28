ROOT="$(dirname $( cd -- "$(dirname "$0"))" >/dev/null 2>&1 ; pwd -P ))"
echo $ROOT
#wget axon.cs.byu.edu/media/data.zip -O $ROOT/data.zip --show-progress
wget axon.cs.byu.edu/media/data.zip -O $ROOT/data.zip

# use unzip -o to overwrite without prompting
unzip $ROOT/data.zip -d $ROOT/
