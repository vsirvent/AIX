#!/bin/bash

#
#  Copyright (c) 2024-Present, Arkin Terli. All rights reserved.
#

lib_name=yaml-cpp
lib_version=0.8.0
lib_url=https://github.com/jbeder/yaml-cpp.git
logical_core_count=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)

pushd .

git clone --recurse-submodules -b $lib_version $lib_url ./$lib_name/$lib_version
cd $lib_name/$lib_version

rm -rf build
rm -rf installed
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../installed
cmake --build . --target install -- -j $logical_core_count

cd ..
rm -rf .git .gitattributes .github .gitignore build

popd
