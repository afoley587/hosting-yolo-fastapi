#!/bin/bash

TEST_IMG_DIR=$HOME/Desktop/tests

for file in $(find $TEST_IMG_DIR -name "*.jpg"); do
    j=$(curl --silent -X POST -F"file=@$file" http://localhost/yolo/)
    
    echo $j | jq

    if [[ $(echo $j | jq '.labels | length == 0') == 'false' ]]; then
        id=$(echo $j | jq '.id')
        echo "fetching image with id $id"
        curl -o ~/Desktop/trash-$id-LABELED.png http://localhost/yolo/$id
    fi
done

read  -n 1 -p "Delete:" delete

rm ~/Desktop/trash-*-LABELED.png