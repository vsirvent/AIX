#
#  Copyright © 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

set(TARGET_NAME XORMetalApp)

# External library versions
set(METAL_CPP_VERSION 14.2)

# Include folders
include_directories(${CMAKE_SOURCE_DIR}/Externals/metal-cpp/${METAL_CPP_VERSION})

set_source_files_properties(default.metal PROPERTIES LANGUAGE METAL
        COMPILE_FLAGS "-emit-llvm"
        MTL_ENABLE_DEBUG_INFO YES
)

# TODO: Compile all compute shaders into a single lib.

add_custom_command(
        OUTPUT default.air
        COMMAND xcrun -sdk macosx metal
        -fno-fast-math
        -c ${CMAKE_CURRENT_SOURCE_DIR}/default.metal
        -o default.air
        MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/default.metal
)

add_custom_command(
        OUTPUT default.metallib
        COMMAND xcrun -sdk macosx metallib default.air -o default.metallib
        MAIN_DEPENDENCY default.air
)

add_custom_target(default${TARGET_NAME} ALL
        DEPENDS default.metallib
)

add_executable(${TARGET_NAME}
        main.cpp
)

add_dependencies(${TARGET_NAME} default${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE
        "-framework Foundation"
        "-framework Metal"
)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION .
)

# Install the default.metallib file to the same directory as the executable
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/default.metallib"
        DESTINATION .
)
