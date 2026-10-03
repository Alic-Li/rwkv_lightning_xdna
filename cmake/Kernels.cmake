# Host builds never compile kernels or dispatch the device implicitly.
function(rwkv_kernel_target name)
    add_custom_target(${name}
        COMMAND "${RWKV_XDNA_PYTHON}" "${PROJECT_SOURCE_DIR}/tools/compile/rwkv7_optimized.py"
            --output "${RWKV_XDNA_KERNEL_DIR}" ${ARGN}
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" USES_TERMINAL VERBATIM)
endfunction()
rwkv_kernel_target(kernels-release)
rwkv_kernel_target(kernels-release-int8-ffn --int8-ffn)
rwkv_kernel_target(kernels-release-prefill-batch2 --prefill-batch2)
rwkv_kernel_target(kernels-release-prefill-chunk4 --prefill-chunk4)
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
