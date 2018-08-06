#!/usr/bin/env bash
# Copyright 2018 The Tefla Authors. All Rights Reserved.

apt-get update && apt-get install -y --no-install-recommends apt-utils
apt-get install pkg-config
apt-get update && apt-get install -y --no-install-recommends apt-utils
apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
apt-get update
apt-get install -y software-properties-common vim
add-apt-repository ppa:jonathonf/python-3.6
apt-get update

apt-get install -y build-essential python3.6 python3.6-dev python3-pip python3.6-venv 
apt-get install -y --no-install-recommends python3-tk
apt-get install -y git
apt-get install -y libsm6 libxext6
apt-get install -y libxrender-dev

# update pip
python3.6 -m pip install pip --upgrade
python3.6 -m pip install wheel
