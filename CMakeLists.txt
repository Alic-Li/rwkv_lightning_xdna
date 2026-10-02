cmake_minimum_required(VERSION 3.24)
project(rwkv_lightning_xdna VERSION 0.1.0 LANGUAGES CXX)
include(GNUInstallDirs)
include(CTest)
include(CMakePackageConfigHelpers)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(Threads REQUIRED)
find_package(XRT REQUIRED)
find_path(UUID_INCLUDE_DIR uuid/uuid.h HINTS "${CMAKE_CURRENT_SOURCE_DIR}/.cache/sdk/usr/include" REQUIRED)
add_library(rwkv_xdna src/runtime/session.cpp)
add_library(rwkv::xdna ALIAS rwkv_xdna)
set_target_properties(rwkv_xdna PROPERTIES EXPORT_NAME xdna)
target_compile_features(rwkv_xdna PUBLIC cxx_std_17)
target_include_directories(rwkv_xdna PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${XRT_INCLUDE_DIR} ${UUID_INCLUDE_DIR})
target_link_libraries(rwkv_xdna PRIVATE XRT::Coreutil Threads::Threads)
target_compile_options(rwkv_xdna PRIVATE -Wall -Wextra -Wpedantic)
add_executable(xdna-run apps/xdna_run.cpp)
target_include_directories(xdna-run PRIVATE third_party/nlohmann/include)
target_link_libraries(xdna-run PRIVATE rwkv::xdna)
target_compile_options(xdna-run PRIVATE -Wall -Wextra -Wpedantic)
install(TARGETS rwkv_xdna xdna-run EXPORT rwkvXdnaTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT rwkvXdnaTargets NAMESPACE rwkv:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rwkvXdna)
configure_package_config_file(cmake/rwkvXdnaConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/rwkvXdnaConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rwkvXdna)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/rwkvXdnaConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES cmake/FindXRT.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/rwkvXdnaConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rwkvXdnaConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rwkvXdna)
option(RWKV_XDNA_HARDWARE_TESTS "Register compiled hardware fixtures with CTest" OFF)
if(BUILD_TESTING AND RWKV_XDNA_HARDWARE_TESTS)
    file(GLOB manifests "${CMAKE_CURRENT_SOURCE_DIR}/build/kernels/*/manifest.json")
    foreach(manifest IN LISTS manifests)
        get_filename_component(directory "${manifest}" DIRECTORY)
        get_filename_component(case_id "${directory}" NAME)
        add_test(NAME "dispatch.${case_id}" COMMAND xdna-run "${manifest}")
        set_tests_properties("dispatch.${case_id}" PROPERTIES
            RUN_SERIAL TRUE TIMEOUT 60 FIXTURES_SETUP "result.${case_id}")
        add_test(NAME "check.${case_id}" COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/validation/check_case.py" "${case_id}")
        set_tests_properties("check.${case_id}" PROPERTIES
            TIMEOUT 60 FIXTURES_REQUIRED "result.${case_id}")
    endforeach()
endif()
