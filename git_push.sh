#!/bin/bash


if [ ! "$1" ]; then
    echo "commit detail please ~"
    return
fi
echo "commit: $1"

COMMIT=$1
BRANCH=master

source git_pull.sh $BRANCH

if [ ! -z "$2" ]; then
    echo "operator on branch: $2"
    BRANCH=$2
fi

git add -A
git commit -m "$1 on core"
git push