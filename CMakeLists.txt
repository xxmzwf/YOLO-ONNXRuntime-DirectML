cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
project(YoloOrtDml VERSION 1.0.0 LANGUAGES CXX)

set(ONNXRUNTIME_PATH "D:/CodeLibraries/ONNXRuntime-1.28.0-Shared")
set(DIRECTML_IMPORT_PATH "D:/CodeLibraries/ONNXRuntime-1.28.0-Shared/lib/DirectML.lib")

list(APPEND CMAKE_PREFIX_PATH ${ONNXRUNTIME_PATH})
find_package(onnxruntime REQUIRED)
include(CMakePackageConfigHelpers)

file(GLOB YOLOORTDML_SRC_FILES "YoloOrtDml/src/*.cpp")
file(GLOB YOLOORTDML_HEADER_FILES "YoloOrtDml/include/*.h")

add_library(YoloOrtDml SHARED)
add_library(YoloOrtDml::YoloOrtDml ALIAS YoloOrtDml)
target_sources(YoloOrtDml
    PRIVATE
        ${YOLOORTDML_SRC_FILES}
        ${YOLOORTDML_HEADER_FILES}
)
target_include_directories(YoloOrtDml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/YoloOrtDml/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_definitions(YoloOrtDml
    PRIVATE
        YOLOORTDML_BUILD_SHARED
    INTERFACE
        YOLOORTDML_USE_SHARED
)
target_link_libraries(YoloOrtDml
    PRIVATE
        onnxruntime::onnxruntime
        d3d12
        dxgi
        ${DIRECTML_IMPORT_PATH}
)

install(TARGETS YoloOrtDml
    EXPORT YoloOrtDmlTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES YoloOrtDml/include/YoloOrtDml.h DESTINATION include)
install(EXPORT YoloOrtDmlTargets
    FILE YoloOrtDmlTargets.cmake
    NAMESPACE YoloOrtDml::
    DESTINATION lib/cmake/YoloOrtDml
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/YoloOrtDmlConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/YoloOrtDmlConfig.cmake"
    INSTALL_DESTINATION lib/cmake/YoloOrtDml
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/YoloOrtDmlConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/YoloOrtDmlConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/YoloOrtDmlConfigVersion.cmake"
    DESTINATION lib/cmake/YoloOrtDml
)