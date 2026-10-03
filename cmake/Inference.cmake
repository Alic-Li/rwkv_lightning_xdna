# Standalone model API; tokenization/sampling are private CLI concerns.
add_library(rwkv_inference
    src/inference/model.cpp src/inference/weights.cpp src/inference/backend.cpp
    src/inference/ops.cpp src/inference/graph.cpp
    src/inference/resident_plan.cpp src/inference/graph_execution.cpp
    src/inference/artifacts.cpp src/inference/weight_layout.cpp
    third_party/rwkv_cuda_io/src/pth_archive.cpp third_party/rwkv_cuda_io/src/pth_tensor.cpp)
add_library(rwkv::inference ALIAS rwkv_inference)
set_target_properties(rwkv_inference PROPERTIES EXPORT_NAME inference)
target_compile_features(rwkv_inference PUBLIC cxx_std_17)
target_include_directories(rwkv_inference PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE third_party/nlohmann/include third_party/rwkv_cuda_io/include)
target_link_libraries(rwkv_inference PRIVATE rwkv::xdna)
target_compile_options(rwkv_inference PRIVATE -Wall -Wextra -ffp-contract=off)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rwkv_inference PRIVATE OpenMP::OpenMP_CXX)
endif()
add_library(rwkv_text STATIC src/utils/tokenizer.cpp src/utils/sampler.cpp)
target_include_directories(rwkv_text PUBLIC src/utils)
target_compile_features(rwkv_text PUBLIC cxx_std_17)
target_link_libraries(rwkv_text PRIVATE Threads::Threads)
add_executable(rwkv-cli apps/rwkv_cli.cpp)
target_link_libraries(rwkv-cli PRIVATE rwkv::inference rwkv_text)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rwkv-cli PRIVATE OpenMP::OpenMP_CXX)
endif()
