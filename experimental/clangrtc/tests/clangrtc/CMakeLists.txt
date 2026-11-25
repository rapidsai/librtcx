# @file tests/clangrtc/CMakeLists.txt
#
# @brief Build and register compiler, artifact, host-execution, and compiler-linkage tests.
#
# Discovers individual GoogleTest cases and configures the external consumer and compiler- linkage
# fixtures.

add_executable(
    clangrtc_tests
    c_header.c
    "${CMAKE_CURRENT_SOURCE_DIR}/../support.cpp"
    host_test.cpp
    options_test.cpp
    compilation_test.cpp
    profiles_test.cpp
    header_modes_test.cpp
    diagnostics_test.cpp
    toolchain_test.cpp
    expressions_test.cpp
    artifact_fixture.cpp
    artifacts_test.cpp
    package_test.cpp
    prepared_diagnostics_test.cpp
    concurrency_test.cpp
    error_macros_test.cpp)
target_compile_features(clangrtc_tests PRIVATE cxx_std_20)
target_include_directories(
    clangrtc_tests PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/.." "${CMAKE_CURRENT_SOURCE_DIR}/.."
                           "${PROJECT_SOURCE_DIR}/src")
target_include_directories(clangrtc_tests SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS}
                                                         "${clangrtc_xxhash_SOURCE_DIR}")
target_link_libraries(clangrtc_tests PRIVATE clangrtc GTest::gtest_main ${_clangrtc_llvm_libraries}
                                             Threads::Threads)
gtest_discover_tests(clangrtc_tests DISCOVERY_MODE PRE_TEST PROPERTIES LABELS unit)

add_test(
    NAME clangrtc_host_execution
    COMMAND
        "${CMAKE_COMMAND}" "-DCLI=$<TARGET_FILE:clangrtc_cli>" "-DCXX=${CMAKE_CXX_COMPILER}"
        "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/host-execution" -P
        "${CMAKE_CURRENT_SOURCE_DIR}/host_execution.cmake")
set_tests_properties(clangrtc_host_execution PROPERTIES LABELS unit)

foreach(
    _case IN
    ITEMS static
          shared_llvm
          shared_clang
          transitive_shared
          wrapped_shared
          raw_shared
          disguised_static
          missing_archive)
    add_test(
        NAME clangrtc_compiler_linkage_${_case}
        COMMAND
            "${CMAKE_COMMAND}" "-DCASE=${_case}"
            "-DSOURCE=${CMAKE_CURRENT_SOURCE_DIR}/compiler_linkage"
            "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/compiler-linkage/${_case}"
            "-DMODULE=${PROJECT_SOURCE_DIR}/cmake/Linkage.cmake" -P
            "${CMAKE_CURRENT_SOURCE_DIR}/compiler_linkage.cmake")
    set_tests_properties(clangrtc_compiler_linkage_${_case} PROPERTIES LABELS unit)
endforeach()
