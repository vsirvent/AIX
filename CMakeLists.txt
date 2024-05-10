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

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;CCov;ASan;TSan" CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 20)

# Set default build type as Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")    # Common parameters for debug and release
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_CCOV "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS} -g -O1 -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS} -g -O2 -fsanitize=thread -fPIE")

# Set external library versions
set(DOCTEST_VERSION 2.4.11)

# Include folders
include_directories(Externals/doctest/${DOCTEST_VERSION}/installed/include)
include_directories(Targets/AIXLib)

# IMPORTANT: Read README.MD to install dependencies.

# Lib folders
#link_directories(...)

# Target folders
add_subdirectory(Targets/AIXTests)
add_subdirectory(Targets/AIXExamples/XORApp)
add_subdirectory(Targets/AIXExamples/XORLayerApp)
add_subdirectory(Targets/AIXExamples/XORSequentialApp)
add_subdirectory(Targets/AIXExamples/XOROpenMPApp)

# Build the following targets only on macOS with Apple Silicon.
if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    add_subdirectory(Targets/AIXExamples/XORMetalApp)
    add_subdirectory(Targets/AIXExamples/XORMPSApp)
endif()
