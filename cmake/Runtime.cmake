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
