#
#  Copyright © 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

set(TARGET_NAME XORMPSApp)

set(CMAKE_OBJC_FLAGS "${CMAKE_OBJC_FLAGS} -fobjc-arc -Wall -Wextra")
set(CMAKE_OBJC_FLAGS_DEBUG "${CMAKE_OBJC_FLAGS_DEBUG} -g")
set(CMAKE_OBJC_FLAGS_RELEASE "${CMAKE_OBJC_FLAGS_RELEASE} -O3")

add_executable(${TARGET_NAME}
        main.cpp
        mps.mm      # Only this file compiled as Objective-C module. Application is still a C++ application.
)

target_link_libraries(${TARGET_NAME} PRIVATE
        "-framework Foundation"
        "-framework Metal"
        "-framework QuartzCore"
        "-framework MetalPerformanceShaders"
)

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION .
)
