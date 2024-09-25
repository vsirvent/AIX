#
#  Copyright © 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

cmake_minimum_required(VERSION 3.24)

project(AIX)

# Set default build options

option(AIX_BUILD_TESTS    "Build AIX test application"     OFF)
option(AIX_BUILD_EXAMPLES "Build AIX example applications" OFF)
option(AIX_BUILD_STATIC   "Build static AIX library"       OFF)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Prof;CCov;ASan;TSan" CACHE STRING "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH .)

# Set default build type as Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-function")   # Common parameters for debug and release
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto")
set(CMAKE_CXX_FLAGS_PROF "${CMAKE_CXX_FLAGS_RELEASE} -O1 -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer")
set(CMAKE_CXX_FLAGS_CCOV "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS} -g -O1 -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS} -g -O2 -fsanitize=thread -fPIE")

# Set external library versions
set(DOCTEST_VERSION 2.4.11)
set(DOCOPT_VERSION 0.6.3)
set(YAML_VERSION 0.8.0)

# Include folders
include_directories(Externals/docopt/${DOCOPT_VERSION}/installed/include)
include_directories(Externals/doctest/${DOCTEST_VERSION}/installed/include)
include_directories(Externals/yaml-cpp/${YAML_VERSION}/installed/include)
include_directories(Targets/AIXLib)

# IMPORTANT: Read README.MD to install dependencies.

# Lib folders
link_directories(Externals/docopt/${DOCOPT_VERSION}/installed/lib)
link_directories(Externals/yaml-cpp/${YAML_VERSION}/installed/lib)

# Target folders
add_subdirectory(Targets/AIXLib)

if (AIX_BUILD_TESTS)
    add_subdirectory(Targets/AIXBenchmarks)
    add_subdirectory(Targets/AIXTests)
endif ()

if (AIX_BUILD_EXAMPLES)
    add_subdirectory(Targets/AIXExamples/GPT2)
    add_subdirectory(Targets/AIXExamples/XORApp)
    add_subdirectory(Targets/AIXExamples/XORLayerApp)
    add_subdirectory(Targets/AIXExamples/XORSequentialApp)

    # Build the following targets only on macOS with Apple Silicon.
    if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        add_subdirectory(Targets/AIXExamples/XORMetalApp)
    endif()
endif ()

if (NOT AIX_BUILD_STATIC AND (AIX_BUILD_TESTS OR AIX_BUILD_EXAMPLES))
    install(TARGETS AIXLib DESTINATION ${CMAKE_INSTALL_PREFIX})
endif ()
