#! /usr/bin/env bash

make all

echo "---- ./with-user-cast --------------------------"
./with-user-cast
echo "---- ./no-user-cast ----------------------------"
./no-user-cast
