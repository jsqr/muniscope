#!/bin/bash

# Check for required arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <host> <database> <user>" >&2
    echo "You may also need to set up the postgress password in ~/.pgpass" >&2
    exit 1
    read -n 1 -s -r -p "Press any key to continue"
fi

# Get arguments
host="$1"
database="$2"
user="$3"

script_dir=$(dirname "$0")
sql_file="$script_dir/reset.sql"

psql -h "$host" -d "$database" -U "$user" -f "$sql_file"

if [ $? -eq 0 ]; then
    echo "reset.sql executed successfully."
else
    echo "error executing reset.sql."
    exit 1
    read -n 1 -s -r -p "Press any key to continue"
fi
