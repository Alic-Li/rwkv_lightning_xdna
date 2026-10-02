find_path(XRT_INCLUDE_DIR xrt/xrt_kernel.h HINTS /opt/xilinx/xrt/include)
find_library(XRT_COREUTIL_LIBRARY xrt_coreutil HINTS /opt/xilinx/xrt/lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XRT REQUIRED_VARS XRT_INCLUDE_DIR XRT_COREUTIL_LIBRARY)
if(XRT_FOUND AND NOT TARGET XRT::Coreutil)
    add_library(XRT::Coreutil UNKNOWN IMPORTED)
    set_target_properties(XRT::Coreutil PROPERTIES
        IMPORTED_LOCATION "${XRT_COREUTIL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${XRT_INCLUDE_DIR}")
endif()
