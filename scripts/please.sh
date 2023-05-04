#!/bin/bash

set -e

# Load the baseball database from the SQL file
echo "Loading database..."
if ! mysql -u root -proot baseball < ./sql/baseball.sql; then
    echo "Error: Failed to load database from ./sql/baseball.sql" >&2
    exit 1
fi

# Run the SQL query and store the results in a file
echo "Running SQL query..."
if ! mysql -u root -proot baseball < ./sql/joshi.sql > ./output/results.txt; then
    echo "Error: Failed to run SQL query from ./sql/joshi.sql" >&2
    exit 1
fi

echo "Result data stored in output/results.txt"
