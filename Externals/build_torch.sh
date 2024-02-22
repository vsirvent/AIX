#!/bin/bash

#
#  Copyright (c) 2024-Present, Arkin Terli. All rights reserved.
#

# TODO: This script installs only arm64 version of libtorch.

lib_name=libtorch
lib_version=2.2.0   # Make sure the url has the same version too.
lib_url=https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.2.0.zip
tmp_dir=$(mktemp -d)

pushd .

rm -rf $lib_name/$lib_version
mkdir -p $lib_name/$lib_version

# Download to tmp folder
curl -L $lib_url --output $tmp_dir/lib.zip

# Change to the target directory
cd $lib_name/$lib_version

# Unzip files and folders from the specific folder in the zip file
unzip $tmp_dir/lib.zip $lib_name/*

# Move the extracted specific folder contents to the current directory
mv $lib_name/* .

# Cleanup
rm -rf $lib_name
rm -rf $tmp_dir

popd
