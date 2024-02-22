#!/bin/bash

#
#  Copyright (c) 2024-Present, Arkin Terli. All rights reserved.
#

lib_name=doctest
lib_version=2.4.11
lib_url=https://github.com/doctest/doctest.git

pushd .

rm -rf $lib_name

git clone --recurse-submodules -b v$lib_version $lib_url ./$lib_name/$lib_version
cd $lib_name/$lib_version

rm -rf build
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=FALSE -DCMAKE_INSTALL_PREFIX=../installed
cmake --build . --target install -- -j `nproc --all`

cd ..
rm -rf .git .gitattributes .github .gitignore build

popd
