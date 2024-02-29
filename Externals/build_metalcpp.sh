#!/bin/bash

#
#  Copyright (c) 2024-Present, Arkin Terli. All rights reserved.
#

# Do not perform installation if it's not macOS and Apple Silicon.
[[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]] || exit 1

lib_name=metal-cpp
lib_version=14.2  # Make sure the url has the same version too.
lib_url=https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14.2_iOS17.2.zip
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
