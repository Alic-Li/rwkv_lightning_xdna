install(TARGETS rwkv_xdna rwkv_inference rwkv-cli EXPORT rwkvXdnaTargets)
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

install(FILES assets/rwkv_vocab_v20230424.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/rwkv_lightning_xdna)
