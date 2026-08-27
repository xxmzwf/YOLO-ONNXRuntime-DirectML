cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(YoloOrtCml VERSION 1.0.0 LANGUAGES CXX)

set(ONNXRUNTIME_PATH "/Users/wufeng/Code/Libs/ONNXRuntime-Shared" CACHE PATH "ONNX Runtime installation prefix")

list(APPEND CMAKE_PREFIX_PATH ${ONNXRUNTIME_PATH})
find_package(onnxruntime REQUIRED)
include(CMakePackageConfigHelpers)

file(GLOB YOLOORTCML_SRC_FILES "YoloOrtCml/src/*.cpp")
file(GLOB YOLOORTCML_HEADER_FILES "YoloOrtCml/include/*.h")

add_library(YoloOrtCml SHARED)
add_library(YoloOrtCml::YoloOrtCml ALIAS YoloOrtCml)
target_sources(YoloOrtCml
    PRIVATE
        ${YOLOORTCML_SRC_FILES}
        ${YOLOORTCML_HEADER_FILES}
)
target_include_directories(YoloOrtCml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/YoloOrtCml/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_definitions(YoloOrtCml
    PRIVATE
        YOLOORTCML_BUILD_SHARED
    INTERFACE
        YOLOORTCML_USE_SHARED
)
target_link_libraries(YoloOrtCml
    PRIVATE
        onnxruntime::onnxruntime
)
if(APPLE)
    set_target_properties(YoloOrtCml PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_PATH}/lib"
        INSTALL_RPATH "@loader_path"
    )
endif()

install(TARGETS YoloOrtCml
    EXPORT YoloOrtCmlTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES YoloOrtCml/include/YoloOrtCml.h DESTINATION include)
if(APPLE)
    file(GLOB ONNXRUNTIME_DYLIB_FILES "${ONNXRUNTIME_PATH}/lib/libonnxruntime*.dylib")
    install(FILES ${ONNXRUNTIME_DYLIB_FILES} DESTINATION lib)
else()
    install(FILES
        "${ONNXRUNTIME_PATH}/bin/onnxruntime.dll"
        "${ONNXRUNTIME_PATH}/bin/onnxruntime_providers_shared.dll"
        DESTINATION bin
    )
endif()
install(EXPORT YoloOrtCmlTargets
    FILE YoloOrtCmlTargets.cmake
    NAMESPACE YoloOrtCml::
    DESTINATION lib/cmake/YoloOrtCml
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/YoloOrtCmlConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/YoloOrtCmlConfig.cmake"
    INSTALL_DESTINATION lib/cmake/YoloOrtCml
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/YoloOrtCmlConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/YoloOrtCmlConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/YoloOrtCmlConfigVersion.cmake"
    DESTINATION lib/cmake/YoloOrtCml
)
