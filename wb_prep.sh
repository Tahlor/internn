#!/bin/bash

### ADD SOCKS5 TO SOCKET

add_to_file_if_missing() {
    # Example ./add_to_file_if_missing.sh "this" "this_file.txt"
    TEXT=${1}
    FILE=${2}

    TEXT="socks.set_default_proxy"
    FILE=${2-"../env/internn/lib/python3.8/socket.py"}

    echo "Looking for $TEXT"
    cat "$FILE" | grep -F "$TEXT" 

    # If grep doesn't find it
    if [ $? -ne 0 ]; then
        if [ -f "$FILE" ]; then
            echo "$FILE exists, appending"

            cat << 'EOF' >> $FILE

import socks
socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 41080)
socket = socks.socksocket
EOF
        else
            echo "$FILE does not exists"
        fi
    else
        echo "$FILE contains $TEXT, not adding"
    fi
}

add_to_file_if_missing
