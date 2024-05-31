#!/bin/bash

# Check for required arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <host> <database> <user>" >&2
    echo "You should also set up the postgress password in ~/.pgpass" >&2
    exit 1
fi

# Get arguments
host="$1"
database="$2"
user="$3"

sql_file="reset.sql"
psql -h "$host" -d "$database" -U "$user" -f "$sql_file"

if [ $? -eq 0 ]; then
    echo "reset.sql executed successfully."
else
    echo "error executing reset.sql."
fi
