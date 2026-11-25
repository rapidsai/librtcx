# SPDX-License-Identifier: Apache-2.0

# LLVM and Clang are always embedded from component archives. Never mix these archives with a
# compiler dylib: LLVM's global registries would be duplicated.
llvm_map_components_to_libnames(
    _clangrtc_llvm_libraries
    Core
    Support
    BitReader
    BitWriter
    AsmParser
    Linker
    Passes
    Target
    nativecodegen)
set(_clangrtc_clang_libraries clangCodeGen clangFrontend clangDriver clangSerialization clangSema)

# Check transitive compiler dependencies without modifying imported targets. Follow other target
# wrappers too, but leave their own linkage choices intact.
set(_clangrtc_pending ${_clangrtc_clang_libraries} ${_clangrtc_llvm_libraries})
set(_clangrtc_visited)
set(_clangrtc_compiler_files)
while(_clangrtc_pending)
    list(POP_FRONT _clangrtc_pending _target)
    if(NOT TARGET "${_target}")
        message(
            FATAL_ERROR
                "Required static compiler target ${_target} is unavailable in this LLVM/Clang installation"
        )
    endif()
    get_target_property(_aliased_target "${_target}" ALIASED_TARGET)
    if(_aliased_target)
        set(_target "${_aliased_target}")
    endif()
    if(_target IN_LIST _clangrtc_visited)
        continue()
    endif()
    list(APPEND _clangrtc_visited "${_target}")
    get_target_property(_dependencies "${_target}" INTERFACE_LINK_LIBRARIES)
    if(NOT _dependencies)
        set(_dependencies)
    endif()
    foreach(_dependency IN LISTS _dependencies)
        while(_dependency MATCHES "^\\$<LINK_ONLY:(.+)>$")
            set(_dependency "${CMAKE_MATCH_1}")
        endwhile()
        if(_dependency MATCHES "^\\$<TARGET_NAME_IF_EXISTS:([^>]+)>$")
            set(_dependency "${CMAKE_MATCH_1}")
            if(NOT TARGET "${_dependency}")
                continue()
            endif()
        endif()
        if(TARGET "${_dependency}")
            list(APPEND _clangrtc_pending "${_dependency}")
        elseif(IS_ABSOLUTE "${_dependency}")
            get_filename_component(_filename "${_dependency}" NAME)
            if(_filename MATCHES "^(lib)?(LLVM|[Cc]lang)")
                if(NOT EXISTS "${_dependency}" OR NOT _dependency MATCHES
                                                  "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
                    message(
                        FATAL_ERROR
                            "Compiler dependency ${_dependency} must be an existing static archive")
                endif()
                list(APPEND _clangrtc_compiler_files "${_dependency}")
            endif()
        elseif(_dependency MATCHES "LLVM|[Cc]lang")
            message(
                FATAL_ERROR
                    "Cannot verify static compiler dependency ${_dependency}; use a package exporting static compiler targets"
            )
        endif()
    endforeach()
    get_target_property(_type "${_target}" TYPE)
    if(_type STREQUAL "INTERFACE_LIBRARY")
        continue()
    endif()
    if(_target MATCHES "^(LLVM|[Cc]lang)" AND NOT _type STREQUAL "STATIC_LIBRARY")
        message(
            FATAL_ERROR
                "clangrtc requires static LLVM/Clang libraries, but ${_target} is ${_type}. Build LLVM/Clang with BUILD_SHARED_LIBS=OFF, LLVM_LINK_LLVM_DYLIB=OFF, and CLANG_LINK_CLANG_DYLIB=OFF, then select that installation."
        )
    endif()
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _configuration)
    get_target_property(_library "${_target}" "IMPORTED_LOCATION_${_configuration}")
    if(NOT _library)
        get_target_property(_library "${_target}" IMPORTED_LOCATION)
    endif()
    if(NOT _library)
        get_target_property(_configurations "${_target}" IMPORTED_CONFIGURATIONS)
        foreach(_configuration IN LISTS _configurations)
            get_target_property(_library "${_target}" "IMPORTED_LOCATION_${_configuration}")
            if(_library)
                break()
            endif()
        endforeach()
    endif()
    if(NOT EXISTS "${_library}")
        message(FATAL_ERROR "Required dependency ${_target} is not installed: ${_library}")
    endif()
    if(_target MATCHES "^(LLVM|[Cc]lang)")
        if(NOT _library MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
            message(
                FATAL_ERROR
                    "Compiler target ${_target} must reference a static archive: ${_library}")
        endif()
        list(APPEND _clangrtc_compiler_files "${_library}")
    endif()
endwhile()
list(REMOVE_DUPLICATES _clangrtc_compiler_files)
