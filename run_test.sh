#! /usr/bin/env bash

log="cuda-crtp.log"

make all

nvcc --version   | tee ${log}
nvidia-smi       | tee --append ${log}

echo "---- ./with-user-cast --------------------------"   | tee --append ${log}
./with-user-cast            | tee --append ${log}
echo "---- ./no-user-cast ----------------------------"   | tee --append ${log}
./no-user-cast              | tee --append ${log}
