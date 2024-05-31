#!/bin/bash

#
#  Copyright (c) 2024-Present, Arkin Terli. All rights reserved.
#

function showHelp()
{
    echo ""
    echo "Usage:"
    echo "    $0 <build_type> <install_dir> [<build_options>...]"
    echo ""
    echo "Example:"
    echo "    $0 release product-rel"
    echo ""
    echo "Options:"
    echo "    build_type       Valid build types: release, debug, ccov, asan, tsan"
    echo "    install_dir      Product installation directory name."
    echo "    build_options    CMake build options. Can be multiple. i.e. -DAIX_BUILD_STATIC=ON"
    echo ""
}

function checkBuildType()
{
    arr=("release" "debug" "ccov" "asan" "tsan")
    build_type="$(tr [A-Z] [a-z] <<< "$1")"   # to lower-case.
    # if build type is not valid then exit.
    if [[ ! " ${arr[*]} " == *" ${build_type} "* ]]; then
        echo "Error: Invalid build type: ${type}"
        exit 1
    fi
}

function main()
{
    checkBuildType $1
    pushd .
    rm -rf $2
    rm -rf build-$1
    mkdir build-$1
    cd build-$1

    cmake .. -DCMAKE_BUILD_TYPE=$1 -DAIX_BUILD_EXAMPLES=ON -DAIX_BUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX="../$2" $3 $4 $5 $6 $7 $8 $9 ${10}
    cmake --build . --target install -- -j

    popd
}

if [ "$#" -ge 2 ]; then
    main "$@"  # Pass all parameters
else
    showHelp
fi
