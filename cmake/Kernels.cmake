# Explicit offline targets: ordinary host builds never compile device kernels.
add_custom_target(kernels-release
    COMMAND "${RWKV_XDNA_PYTHON}" "${PROJECT_SOURCE_DIR}/tools/compile/rwkv7_optimized.py"
        --output "${RWKV_XDNA_KERNEL_DIR}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" USES_TERMINAL VERBATIM)
add_custom_target(rwkv-release DEPENDS rwkv-cli kernels-release)
if(BUILD_TESTING)
    foreach(tier smoke all)
        add_custom_target(kernels-test-${tier}
            COMMAND "${CMAKE_COMMAND}" -E env "RWKV_XDNA_TEST_KERNEL_DIR=${RWKV_XDNA_TEST_KERNEL_DIR}"
                "${RWKV_XDNA_PYTHON}" "${PROJECT_SOURCE_DIR}/tools/validation/sweep.py"
                --tier ${tier} --jobs "${RWKV_XDNA_COMPILE_JOBS}" --compile-only
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" USES_TERMINAL VERBATIM)
    endforeach()
endif()
