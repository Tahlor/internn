#!/bin/bash

### USAGE:
# wb_prep - add lines of text to file if they are not there
# wb_prep DELETE - undo

# RUN PORT COMMAND ON LOGIN NODE
# if ! pgrep -f "$(hostname)" > /dev/null ; then
# ssh -fNt -D 41080 tarch@login04.rc.byu.edu;
# fi;
#

### ADD SOCKS5 TO SOCKET
TEXT="# BEGIN ADDITION"
FILE="../env/internn/lib/python3.8/socket.py"

remove_additions() {
sed -i '/# BEGIN ADDITION/,/# END ADDITION/d' $FILE
}

add_to_file_if_missing() {
    # Example ./add_to_file_if_missing.sh "this" "this_file.txt"
    TEXT=${1:-$TEXT}
    FILE=${2:-$FILE}
    echo $FILE
    echo "Looking for $TEXT"
    cat "$FILE" | grep -F "$TEXT"

    # If grep doesn't find it
    if [ $? -ne 0 ]; then
        if [ -f "$FILE" ]; then
            echo "$FILE exists, appending"

            cat << 'EOF' >> $FILE
# BEGIN ADDITION

import socks

def internet(host="8.8.8.8", port=53, timeout=3):
    """
    Host: 8.8.8.8 (google-public-dns-a.google.com)
    OpenPort: 53/tcp
    Service: domain (DNS/TCP)
    """
    try:
        setdefaulttimeout(timeout)
        socket(AF_INET, SOCK_STREAM).connect((host, port))
        return True
    except error as ex:
        print(ex)
        return False

if not internet():
  socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 41080)
  socket = socks.socksocket

# END ADDITION
EOF
        else
            echo "$FILE does not exists"
        fi
    else
        echo "$FILE contains $TEXT, not adding"
    fi
}

if [[ $1 == "DELETE" ]]; then
    remove_additions
else
    add_to_file_if_missing
fi