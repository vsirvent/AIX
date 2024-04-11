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

add_executable(${TARGET_NAME}
        main.cpp
)

target_link_libraries(${TARGET_NAME} PRIVATE
        "-framework Foundation"
        "-framework Metal"
)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION .
)
