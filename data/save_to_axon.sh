ROOT="$(dirname $( cd -- "$(dirname "$0"))" >/dev/null 2>&1 ; pwd -P ))"

map_axon
zip $ROOT/data.zip $ROOT/data -r
cp $ROOT/data.zip ~/shares/axon.cs.byu.edu/var/www/media/
