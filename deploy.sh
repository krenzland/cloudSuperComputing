#!/bin/bash

set -o errexit -o nounset

if [ "$TRAVIS_BRANCH" != "master" ]
then
  echo "This commit was made against the $TRAVIS_BRANCH and not the master! No deploy!"
  exit 0
fi

rev=$(git rev-parse --short HEAD)

cp writing/thesis/main.pdf writing/thesis/build/
cd writing/thesis/build

git init
git config user.name "Lukas Krenz"
git config user.email "lukas@krenz.land"

git remote add upstream "https://$GH_TOKEN@github.com/krenzland/cloudSuperComputing.git"
git fetch upstream
git reset upstream/pdf

touch main.pdf

git add main.pdf
git commit --amend -m "rebuild pages at ${rev}"
git push --force-with-lease -q upstream HEAD:pdf
