#
#  Copyright © 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

set(TARGET_NAME XOROpenMPApp)

find_package(OpenMP)

# Add OpenMP manually if cmake can't find the package in case user uses MacPorts or Brew.
if (NOT OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")    # Enable openmp support
    # MacPorts
    if (EXISTS /opt/local/include/libomp)
        include_directories(/opt/local/include/libomp)
        link_directories(/opt/local/lib/libomp)
    endif()
    # Brew
    if (EXISTS /opt/homebrew/opt/libomp/include)
        include_directories(/opt/homebrew/opt/libomp/include)
        link_directories(/opt/homebrew/opt/libomp/lib)
    endif()
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")    # Enable openmp support
endif ()

add_executable(${TARGET_NAME}
        main.cpp
)

target_link_libraries(${TARGET_NAME} PRIVATE
        omp
)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION .
)
